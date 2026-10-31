"""Applicatiegrens: bronnen buiten het slot, registratie via guard, geld via CAS.

De 48h-terugblik verzamelt externe correcties vóór recorderretentie; bekende
dirty geschiedenis wordt onafhankelijk daarvan uit het archief herberekend.
Dit bewijst geen detectie van oudere, nog niet opgehaalde externe revisies.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from datetime import time as midnight
from decimal import Decimal
from time import time
from zoneinfo import ZoneInfo

from sqlalchemy import MetaData, Table, select

from dao.prog.fastctrl.plan import FinancialBasis
from dao.prog.settlement.engine import delivery_year_bounds, settle_series
from dao.prog.settlement.metering import merge_inputs, read_ha_energy, reprice_inputs, resolve_spot
from dao.prog.settlement.publication import FinancialPlanBlocked, SettlementPublication
from dao.prog.settlement.repository import ConcurrentSettlementError
from dao.prog.settlement.tariffs import quote, resolve_snapshot
from dao.prog.settlement.types import (
    MeterInterval,
    PriceBatch,
    PriceQuote,
    SettlementResult,
    SettlementState,
    SettlementView,
    SpotInterval,
)


@dataclass(frozen=True, slots=True)
class PlanningContext:
    basis: FinancialBasis
    state: SettlementState
    quotes: tuple[PriceQuote, ...]
    price_batch: PriceBatch


def _bounds(start, end):
    if type(start) is not int or type(end) is not int or end <= start:
        raise ValueError("Verwacht halfopen UTC-epochgrenzen")


def _unknown_tail(state, at):
    return SettlementState(at, None if state.bonus is None else replace(
        state.bonus, asof_ts=at, total_export_kwh=None), None if state.month is None else replace(
            state.month, spot_export_eur=None, complete=False))


class SettlementService:
    """Adapters zijn read-only callables(start,end); de meter levert expliciet meter_id."""

    def __init__(self, repository, metering_adapter, price_adapter, snapshots, *, publication: SettlementPublication):
        self.repository = repository
        self.metering_adapter = metering_adapter
        self.price_adapter = price_adapter
        self.snapshots = tuple(sorted(snapshots, key=lambda item: item.valid_from_ts))
        self.publication = publication
        self.meter_id = metering_adapter.meter_id
        if not isinstance(self.meter_id, str) or not self.meter_id.strip():
            raise ValueError("Een expliciete meteridentiteit is vereist")

    def _prefix_start(self, at, snapshots=None):
        selected = self.snapshots if snapshots is None else snapshots
        try:
            snapshot = resolve_snapshot(selected, at, at + 1)
        except ValueError:
            return at
        zone = ZoneInfo(snapshot.timezone)
        day = datetime.fromtimestamp(at, zone).date()
        starts = []
        if snapshot.monthly_export_floor != "none":
            starts.append(max(day.replace(day=1), snapshot.supply_start or day.replace(day=1)))
        if snapshot.bonus is not None and snapshot.bonus.enabled and snapshot.supply_start is not None:
            start, _ = delivery_year_bounds(snapshot.supply_start, day)
            starts.append(max(start, snapshot.bonus.start_date, snapshot.supply_start))
        return min((int(datetime.combine(opening, midnight.min, zone).timestamp())
                    for opening in starts if opening <= day), default=at)

    def _pinned_state(self, asof, revision):
        selection = self.repository.read_inputs(self.meter_id, asof, asof + 1,
                                                expected_input_revision=revision)
        start = self._prefix_start(asof, selection.snapshots if revision else self.snapshots)
        if start == asof:
            reference = self.repository.read_book_reference(self.meter_id)
            if reference.input_revision != revision:
                raise ConcurrentSettlementError("Boek en beginstaat hebben verschillende revisies")
            if reference.dirty_from_ts is not None and reference.dirty_from_ts < asof:
                dirty = self.repository.read(self.meter_id, reference.dirty_from_ts, asof)
                if dirty.lines:
                    raise ConcurrentSettlementError("Dirty gepubliceerde prefix vereist expliciete consolidatie")
            # Zonder periodeafhankelijkheid is geen onbekende geschiedenis nodig.
            return SettlementState(asof, None, None)
        batch = self.repository.read_inputs(self.meter_id, start, asof, expected_input_revision=revision)
        view = self.repository.read(self.meter_id, start, asof)
        if view.current_input_revision != revision:
            raise ConcurrentSettlementError("Boek en beginstaat hebben verschillende revisies")
        lines = tuple(line for line in view.lines if line.end_ts <= asof)
        published_end = max((line.end_ts for line in lines), default=start)
        if view.dirty_from_ts is not None and view.dirty_from_ts < published_end:
            raise ConcurrentSettlementError("Dirty gepubliceerde prefix vereist expliciete consolidatie")
        state = lines[-1].after_state if lines else SettlementState(start, None, None)
        tail = tuple(item for item in batch.items if state.asof_ts <= item.start_ts and item.end_ts <= asof)
        if tail:
            state = settle_series(tail, batch.snapshots, state).end_state
        return _unknown_tail(state, asof) if state.asof_ts != asof else state

    def state_at(self, asof_ts: int) -> SettlementState:
        if type(asof_ts) is not int:
            raise ValueError("Verwacht een exact UTC-epochmoment")
        revision = self.repository.read_book_reference(self.meter_id).input_revision
        state = self._pinned_state(asof_ts, revision)
        if self.repository.read_book_reference(self.meter_id).input_revision != revision:
            raise ConcurrentSettlementError("Invoer wijzigde tijdens de beginstaatlezing")
        return state

    @staticmethod
    def _cover(readings, start, end):
        """Alleen expliciet onbekende gaten; nooit energie splitsen of nulvullen."""
        result, cursor = [], start
        for item in readings:
            if item.start_ts < cursor or item.end_ts > end:
                raise ValueError("Meetadapter leverde overlap of niet-gedekte grenzen")
            while cursor < item.start_ts:
                stop = min(item.start_ts, (cursor // 900 + 1) * 900)
                result.append(MeterInterval(cursor, stop, None, None, None, "unknown", "incomplete", ()))
                cursor = stop
            result.append(item)
            cursor = item.end_ts
        while cursor < end:
            stop = min(end, (cursor // 900 + 1) * 900)
            result.append(MeterInterval(cursor, stop, None, None, None, "unknown", "incomplete", ()))
            cursor = stop
        return tuple(result)

    def preview(self, start_ts: int, end_ts: int) -> SettlementResult:
        _bounds(start_ts, end_ts)
        opening = self.state_at(start_ts)
        prices = self._read_prices(start_ts, end_ts)
        readings = self._cover(self.metering_adapter(start_ts, end_ts), start_ts, end_ts)
        inputs = merge_inputs(readings, prices, self.snapshots, meter_id=self.meter_id)
        return settle_series(inputs, self.snapshots, opening)

    def scheduled_range(self):
        end = int(time()) // 900 * 900
        batch = self.repository.read_inputs(self.meter_id, end - 1, end)
        watermark = batch.source_watermark_ts
        first = min((item.valid_from_ts for item in self.snapshots), default=end)
        return max(first, min(end, watermark if watermark is not None else end) - 48 * 3600), end

    def _read_prices(self, start, end):
        lower, upper = start // 3600 * 3600, (end + 3599) // 3600 * 3600
        raw = list(self.price_adapter(lower, upper))
        keys = {(item.start_ts, item.end_ts) for item in raw}
        for snapshot in self.snapshots:
            step = 900 if snapshot.hour_price_rule == "mean_quarters" else snapshot.price_interval_seconds
            for at in range(lower, upper, step):
                if snapshot.valid_from_ts >= at + step or (snapshot.valid_to_ts is not None and at >= snapshot.valid_to_ts):
                    continue
                if (at, at + step) not in keys:
                    raw.append(SpotInterval(at, at + step, None, "missing:price-adapter", "incomplete"))
                    keys.add((at, at + step))
        return tuple(sorted(raw, key=lambda item: (item.end_ts - item.start_ts, item.start_ts)))

    def _refresh(self, start, closed_end, price_end, *, reason):
        if any(item.status != "confirmed" for item in self.snapshots):
            self.publication.block(now_ts=int(time()), reason="contract_unconfirmed")
            raise FinancialPlanBlocked("contract_unconfirmed")
        if closed_end > int(time()) // 900 * 900:
            raise ValueError("Alleen gesloten meetintervallen publiceren")
        # Alle bron-I/O gebeurt vóór de korte registratie-/CAS-sloten.
        readings = self.metering_adapter(start, closed_end) if start < closed_end else ()
        raw = self._read_prices(start, price_end)
        revision = self.repository.read_book_reference(self.meter_id).input_revision
        revision = self.repository.register_inputs(self.meter_id, (), self.snapshots, revision)
        revision = self.repository.register_prices(self.meter_id, raw, revision)
        reference = self.repository.read_book_reference(self.meter_id)
        if reference.input_revision != revision:
            raise ConcurrentSettlementError("Bronregistratie is gelijktijdig gewijzigd")
        lower = min(start, reference.dirty_from_ts if reference.dirty_from_ts is not None else start)
        lower = min(lower, self._prefix_start(lower))
        archived = self.repository.read_inputs(self.meter_id, lower, max(lower + 1, closed_end),
                                               expected_input_revision=revision)
        watermark = archived.source_watermark_ts
        if watermark is not None and watermark > int(time()) // 900 * 900:
            raise ValueError("Bestaand toekomst-meetwatermerk vereist afzonderlijk herstel")
        end = max(closed_end, watermark or closed_end, lower + 1)
        archived = self.repository.read_inputs(self.meter_id, lower, end, expected_input_revision=revision)
        prices = self.repository.read_prices(self.meter_id, lower, max(end, price_end),
                                             expected_input_revision=revision)
        candidates = {item.start_ts: item for item in reprice_inputs(archived, prices)}
        # Verdwenen recorderdata overschrijven geen eerder gearchiveerde kWh.
        fresh = merge_inputs(self._cover(readings, start, closed_end), tuple(row.spot for row in prices.items), prices.snapshots,
                             meter_id=self.meter_id)
        for item in fresh:
            old = candidates.get(item.start_ts)
            if old is not None and all(getattr(item, key) is None for key in
                                       ("import_kwh", "export_kwh")):
                continue
            overlaps = [value for value in candidates.values()
                        if value.start_ts < item.end_ts and item.start_ts < value.end_ts]
            if overlaps and any((value.start_ts, value.end_ts) != (item.start_ts, item.end_ts) for value in overlaps):
                continue  # Geen oude uurenergie in kwartieren verdelen of intervalidentiteit herschrijven.
            candidates[item.start_ts] = item
        ordered = tuple(sorted(candidates.values(), key=lambda item: item.start_ts))
        if ordered:
            revision = self.repository.register_inputs(self.meter_id, ordered, prices.snapshots, revision,
                                                        price_batch=prices, closed_before_ts=int(time()) // 900 * 900)
            batch = self.repository.read_inputs(self.meter_id, lower, end, expected_input_revision=revision)
            # Niet een oude PriceBatch herlabelen; de registratie heeft haar eigen revisie.
            self.repository.read_prices(self.meter_id, lower, max(end, price_end), expected_input_revision=revision)
            view = self.repository.read(self.meter_id, lower, end)
            if view.current_input_revision != revision:
                raise ConcurrentSettlementError("Generatie en invoer zijn niet samen gepind")
            result = settle_series(batch.items, batch.snapshots, SettlementState(lower, None, None))
            self.repository.commit(self.meter_id, result, batch, view.generation, reason)
        return revision

    def consolidate(self, start_ts: int, end_ts: int, *, publish: bool = False,
                    reason: str) -> SettlementView | SettlementResult:
        _bounds(start_ts, end_ts)
        if not publish:
            return self.preview(start_ts, end_ts)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Publicatie vereist een expliciete reden")
        self._refresh(start_ts, end_ts, end_ts, reason=reason)
        return self.repository.read(self.meter_id, start_ts, end_ts)

    @staticmethod
    def _quotes(start, end, snapshots, spots):
        result = []
        while start < end:
            stop = min(end, (start // 900 + 1) * 900)
            snapshot = resolve_snapshot(snapshots, start, stop)
            resolved = resolve_spot(snapshot, start, stop, spots)
            if resolved.price_ex_vat is None or resolved.quality == "incomplete":
                raise ValueError("missing_base_price")
            result.append(quote(snapshot, start, stop, resolved.price_ex_vat))
            start = stop
        return tuple(result)

    def price_quotes(self, start, end):
        """Alleen-lezen prijsadapter; geen financieel plan of afrekening."""
        return self._quotes(start, end, self.snapshots, self._read_prices(start, end))

    def prepare_planning(self, start_ts: int, end_ts: int) -> PlanningContext:
        _bounds(start_ts, end_ts)
        prefix = self._prefix_start(start_ts)
        closed = min(start_ts // 900 * 900, int(time()) // 900 * 900)
        first = min((item.valid_from_ts for item in self.snapshots if item.valid_from_ts <= closed), default=closed)
        refresh = max(first, closed - 48 * 3600)
        revision = self._refresh(refresh, closed, end_ts, reason="planning-refresh")
        prices = self.repository.read_prices(self.meter_id, min(prefix, start_ts), end_ts,
                                             expected_input_revision=revision)
        state = self._pinned_state(start_ts, revision)
        try:
            quotes = self._quotes(start_ts, end_ts, prices.snapshots, tuple(row.spot for row in prices.items))
        except ValueError as exc:
            self.publication.block(now_ts=int(time()), reason=f"base_price_unavailable: {exc}")
            raise FinancialPlanBlocked(f"base_price_unavailable: {exc}") from exc
        basis = FinancialBasis(self.meter_id, revision, tuple(item.snapshot_hash for item in prices.snapshots),
                               prefix, start_ts, end_ts, prices.captured_ts, prices.source_watermark_ts)
        self.publication.check_basis(basis, now_ts=int(time()))
        return PlanningContext(basis, state, quotes, prices)


class HAMeteringAdapter:
    def __init__(self, db, selection):
        self.db, self.selection = db, selection
        self.meter_id = selection.meter_id

    def __call__(self, start, end):
        # Een niet-gemeten reststuk wordt niet via recorderinterpolatie bewezen.
        lower, upper = (start + 899) // 900 * 900, end // 900 * 900
        return () if lower >= upper else read_ha_energy(self.db, self.selection, lower, upper, 900)


class DatabasePriceAdapter:
    """Raw DAO-prijswaarden op expliciete contractvensters, nooit solvermiddelen.

    De oude tabel bewaart geen leveranciersbewijs. De bronverwijzing benoemt
    daarom alleen de gelezen lokale tabel, niet een geraden historische provider.
    Native uurselectie weigert een uur met meerdere opgeslagen marktkwartieren.
    """

    def __init__(self, db, snapshots):
        self.db, self.snapshots = db, tuple(snapshots)

    def __call__(self, start, end):
        with self.db.engine.connect() as connection:
            metadata = MetaData()
            values = Table("values", metadata, autoload_with=connection)
            variables = Table("variabel", metadata, autoload_with=connection)
            rows = connection.execute(select(values.c.time, values.c.value).where(
                values.c.variabel == variables.c.id, variables.c.code == "da",
                values.c.time >= start, values.c.time < end,
            ).order_by(values.c.time)).all()
        result = []
        by_time = {int(at): value for at, value in rows}
        for at, value in by_time.items():
            try:
                snapshot = resolve_snapshot(self.snapshots, at, at + 1)
            except ValueError:
                continue
            step = 900 if snapshot.hour_price_rule == "mean_quarters" else snapshot.price_interval_seconds
            if at % step:
                continue
            conflicting = any(at < other < at + step for other in by_time)
            rate = None if value is None or conflicting else Decimal(str(value))
            result.append(SpotInterval(at, at + step, rate, "dao:values:da",
                                       "incomplete" if rate is None else "measured"))
        return tuple(result)
