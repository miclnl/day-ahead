"""Alleen-lezen meetadapters; geen optimizerresolutie, interpolatie of nulvulling.

HA-statistieken beschrijven een bucket, geen exacte grensstand: ``start_ts``
is het begin van de 300s/3600s-bucket, ``sum``/``state`` de laatste waarneming
erin. Voor [a,b) gebruiken we dus de buckets [a-step,a) t/m [b-step,b).
Alle tussenliggende buckets moeten aanwezig zijn; resetdetectie gebruikt ook
de ruwe ``state`` en ``last_reset_ts``. Deze conventie blijft een *schatting*:
de laatste waarneming bewijst niet de stand op de exacte intervalgrens.

Genormaliseerde CSV (UTF-8, komma, header): verplichte kolommen ``start_ts``,
``end_ts`` (integer UTC epochs), ``import_kwh``, ``export_kwh``, ``unit``
(uitsluitend kWh), ``quality`` (measured|estimated|incomplete), ``source``.
Optioneel: ``battery_discharge_kwh`` (bruto, niet gesaldeerd) en ``origin``
(verified_pv|unknown; standaard unknown). Energie is decimale tekst met hoogstens
9 decimalen; leeg betekent onbekend, ook voor ontbrekende batterijontlading.
Een gemeten CSV-regel verklaart expliciete intervalgrenzen, geen HA-bucketstand.
Overlap, ongesorteerde regels en onverklaarde bronwissels worden afgewezen.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import MetaData, Table, inspect, select

from dao.lib.db_manager import DBmanagerObj
from dao.prog.settlement.tariffs import quote, resolve_snapshot
from dao.prog.settlement.types import (
    InputInterval,
    MeterInterval,
    MeterSelection,
    SpotInterval,
    TariffSnapshot,
)

_TABLE_STEPS = {"statistics_short_term": 300, "statistics": 3600}
_UNIT_FACTORS = {"Wh": Decimal("0.001"), "kWh": Decimal("1"), "MWh": Decimal("1000")}
_SENSOR_BATCH = 100
_TIME_BATCH = 86400
_CSV_REQUIRED = {"start_ts", "end_ts", "import_kwh", "export_kwh", "unit", "quality", "source"}
_CSV_OPTIONAL = {"battery_discharge_kwh", "origin"}


def _check_meter_id(meter_id: str) -> None:
    if not isinstance(meter_id, str) or not meter_id.strip():
        raise ValueError("meter_id must be an explicit nonempty identifier")


def _check_order(items: Sequence, label: str) -> None:
    for previous, current in zip(items, items[1:], strict=False):
        if current.start_ts < previous.end_ts:
            raise ValueError(f"{label}: unordered or overlapping intervals")


def _decimal(value) -> Decimal | None:
    """Floatbronnen uitsluitend via hun decimale tekst, nooit Decimal(float)."""
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _energy_precision(value: Decimal) -> Decimal:
    # Alleen overbodige nullen verwijderen; nooit meetprecisie stil afronden.
    if value.as_tuple().exponent < -9:
        normalized = value.normalize()
        if normalized.as_tuple().exponent < -9:
            raise ValueError("Energy precision exceeds 9 decimal places")
        return normalized
    return value


def _read_recorder(db: DBmanagerObj, sensors: tuple[str, ...], start: int, end: int):
    """Kleine sensor-/tijdqueries met uitsluitend gebonden SQLAlchemy-waarden."""
    metas, data = {}, {name: {} for name in _TABLE_STEPS}
    with db.engine.connect() as connection:
        inspector = inspect(connection)
        if not inspector.has_table("statistics_meta") or not sensors:
            return metas, data
        metadata = MetaData()
        meta = Table("statistics_meta", metadata, autoload_with=connection)
        for offset in range(0, len(sensors), _SENSOR_BATCH):
            batch = sensors[offset:offset + _SENSOR_BATCH]
            rows = connection.execute(select(
                meta.c.id, meta.c.statistic_id, meta.c.unit_of_measurement, meta.c.has_sum,
            ).where(meta.c.statistic_id.in_(batch))).mappings()
            for row in rows:
                sensor = row["statistic_id"]
                if sensor in metas:
                    metas[sensor] = None  # conflicterende metadata niet willekeurig kiezen
                else:
                    if row["unit_of_measurement"] not in _UNIT_FACTORS or not row["has_sum"]:
                        raise ValueError("Selected recorder source has an unsupported energy unit/kind")
                    metas[sensor] = row
        by_id = {row["id"]: sensor for sensor, row in metas.items() if row is not None}
        ids = tuple(by_id)
        for name, step in _TABLE_STEPS.items():
            if not inspector.has_table(name):
                continue
            table = Table(name, metadata, autoload_with=connection)
            for offset in range(0, len(ids), _SENSOR_BATCH):
                batch = ids[offset:offset + _SENSOR_BATCH]
                for lower in range(start - step, end, _TIME_BATCH):
                    query = select(
                        table.c.metadata_id, table.c.start_ts, table.c.sum, table.c.state,
                        table.c.last_reset_ts,
                    ).where(
                        table.c.metadata_id.in_(batch), table.c.start_ts >= lower,
                        table.c.start_ts < min(lower + _TIME_BATCH, end),
                    ).order_by(table.c.metadata_id, table.c.start_ts)
                    rows = connection.execute(query).mappings()
                    for chunk in rows.partitions(500):
                        for row in chunk:
                            ts = row["start_ts"]
                            if ts != int(ts):
                                raise ValueError("Recorder bucket starts must be integer UTC epochs")
                            points = data[name].setdefault(by_id[row["metadata_id"]], {})
                            value = (row["sum"], row["state"], row["last_reset_ts"])
                            key = int(ts)
                            if key in points and points[key] != value:
                                points[key] = None
                            elif key not in points:
                                points[key] = value
    return metas, data


def _sensor_energy(points: dict, start: int, end: int, step: int, factor: Decimal) -> Decimal | None:
    rows = [points.get(ts) for ts in range(start - step, end, step)]
    if len(rows) < 2 or any(row is None for row in rows):
        return None
    total = Decimal("0")
    for before, after in zip(rows, rows[1:], strict=False):
        previous_sum, previous_state = _decimal(before[0]), _decimal(before[1])
        current_sum, current_state = _decimal(after[0]), _decimal(after[1])
        if any(value is None or value < 0 for value in (previous_sum, previous_state, current_sum, current_state)):
            return None
        if current_sum < previous_sum or current_state < previous_state or before[2] != after[2]:
            return None
        if current_sum - previous_sum != current_state - previous_state:
            return None  # Ook een reset/correctie binnen één bucket blijft onvolledig.
        total += (current_sum - previous_sum) * factor
    return _energy_precision(total)


def _has_short_term(data: dict, sensors: tuple[str, ...], start: int, end: int) -> bool:
    # Eén overgebleven eindbucket bewijst geen fijnere dekking (retentiegrens).
    for sensor in sensors:
        points = data["statistics_short_term"].get(sensor, {})
        if any(ts in points and ts + 300 in points for ts in range(start - 300, end - 300, 300)):
            return True
    return False


def _has_hour(data: dict, sensors: tuple[str, ...], start: int) -> bool:
    return any(start - 3600 in data["statistics"].get(sensor, {})
               and start in data["statistics"].get(sensor, {}) for sensor in sensors)


def _group_energy(sensors: tuple[str, ...], metas: dict, points: dict, name: str, start: int, end: int):
    sources, values = [], []
    step = _TABLE_STEPS[name]
    for sensor in sensors:
        meta = metas.get(sensor)
        if meta is None:
            sources.append(f"ha:missing-or-conflicting-source:{sensor}")
            values.append(None)
            continue
        unit = meta["unit_of_measurement"]
        sources.append(f"ha:{name}:{sensor}:{unit}:{step}s:bucket_end_estimate")
        value = _sensor_energy(points.get(sensor, {}), start, end, step, _UNIT_FACTORS[unit])
        if value is None:
            sources.append(f"issue:missing-or-invalid-buckets:{sensor}")
        values.append(value)
    if not values or any(value is None for value in values):
        return None, sources
    return _energy_precision(sum(values, Decimal("0"))), sources


def read_ha_energy(
    db: DBmanagerObj, selection: MeterSelection, start_ts: int, end_ts: int, resolution_s: int,
) -> tuple[MeterInterval, ...]:
    """Lees echte bronresolutie; uurretentie is hoogstens een basisraming.

    ``resolution_s`` is de gevraagde meetresolutie (900/3600), geen solverstap.
    Zodra een uur gedeeltelijke korte statistieken heeft, blijven gaten daarin
    zichtbaar; de uurstatistiek maskeert die niet. Zonder fijne brondata kan een
    volledig uur als één geschatte regel terugkomen, nooit als vier verdelingen.
    Geen batterijbewijs betekent onbekende ontlading. Ook no-battery-bewijs op
    zichzelf bewijst niet dat alle export afkomstig is van PV.
    """
    _check_meter_id(selection.meter_id)
    if type(start_ts) is not int or type(end_ts) is not int or type(resolution_s) is not int:
        raise TypeError("Meter boundaries and resolution must be integer UTC seconds")
    if resolution_s not in (900, 3600) or end_ts <= start_ts:
        raise ValueError("Meter interval/resolution must be positive, 900s or 3600s")
    if start_ts % resolution_s or end_ts % resolution_s:
        raise ValueError("Meter boundaries must align with the requested resolution")
    sensors = (*selection.import_sensors, *selection.export_sensors, *selection.battery_sources)
    if len(set(sensors)) != len(sensors) or any(not sensor.strip() for sensor in sensors):
        raise ValueError("Selected sources must be distinct nonempty energy sensors")
    result = []
    with localcontext() as ctx:
        ctx.prec = 36
        metas, data = _read_recorder(db, sensors, start_ts, end_ts)
        start = start_ts
        while start < end_ts:
            end = start + resolution_s
            name = "statistics_short_term"
            hour_end = start + 3600
            full_hour = start % 3600 == 0 and hour_end <= end_ts
            if full_hour and not _has_short_term(data, sensors, start, hour_end) and _has_hour(data, sensors, start):
                name, end = "statistics", hour_end
            imports, import_sources = _group_energy(selection.import_sensors, metas, data[name], name, start, end)
            exports, export_sources = _group_energy(selection.export_sensors, metas, data[name], name, start, end)
            if selection.no_battery_evidence is not None:
                discharge = Decimal("0")
                battery_sources = [f"no-battery:{selection.no_battery_evidence}"]
            else:
                discharge, battery_sources = _group_energy(selection.battery_sources, metas, data[name], name, start, end)
            quality = "incomplete" if any(value is None for value in (imports, exports, discharge)) else "estimated"
            result.append(MeterInterval(
                start, end, imports, exports, discharge, "unknown", quality,
                tuple(import_sources + export_sources + battery_sources),
            ))
            start = end
    return tuple(result)


def read_meter_csv(path: Path, meter_id: str) -> tuple[MeterInterval, ...]:
    """Lees het hierboven vastgelegde schema; de aanroeper kiest de meter-id."""
    _check_meter_id(meter_id)
    result, previous_source = [], None
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not _CSV_REQUIRED.issubset(columns):
            raise ValueError("CSV requires unique normalized meter columns")
        if set(columns) - (_CSV_REQUIRED | _CSV_OPTIONAL):
            raise ValueError("CSV contains unsupported columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV row has missing or surplus columns")
            if row["unit"] != "kWh":
                raise ValueError("Normalized CSV unit must be kWh")
            source = row["source"].strip()
            if not source or (previous_source is not None and source != previous_source):
                raise ValueError("CSV requires one explicit source; unexplained source switch")
            values = {}
            for key in ("import_kwh", "export_kwh", "battery_discharge_kwh"):
                text = row.get(key, "").strip()
                values[key] = text if text else None
            quality = row["quality"]
            # De DTO valideert ook de aangeleverde kwaliteit vóór het afwaarderen.
            item = MeterInterval(
                start_ts=int(row["start_ts"]), end_ts=int(row["end_ts"]), **values,
                origin=row.get("origin", "unknown"), quality=quality,
                sources=(f"csv:{meter_id}:{source}:kWh:{int(row['end_ts']) - int(row['start_ts'])}s:explicit_bounds",),
            )
            if any(getattr(item, key) is None for key in values):
                item = MeterInterval(**{**asdict(item), "quality": "incomplete"})
            result.append(item)
            previous_source = source
    _check_order(result, "CSV meter")
    return tuple(result)


def _price_for_interval(item: MeterInterval, snapshot: TariffSnapshot, spots: dict[int, SpotInterval]):
    duration = snapshot.price_interval_seconds
    local = datetime.fromtimestamp(item.start_ts, ZoneInfo(snapshot.timezone))
    minute = 0 if duration == 3600 else local.minute // 15 * 15
    start = int(local.replace(minute=minute, second=0, microsecond=0).timestamp())
    end = start + duration
    if item.end_ts > end:
        return None, None, "incomplete"
    if duration == 3600 and snapshot.hour_price_rule == "unconfirmed":
        return None, None, "incomplete"
    step = 900 if duration == 3600 and snapshot.hour_price_rule == "mean_quarters" else duration
    prices = [spots.get(ts) for ts in range(start, end, step)]
    rule = "mean_quarters" if step != duration else "native"
    source = f"{rule}:{duration}s:" + json.dumps([
        None if price is None else {
            "source": price.source, "start_ts": price.start_ts, "end_ts": price.end_ts,
            "quality": price.quality,
        } for price in prices
    ], sort_keys=True, separators=(",", ":"))
    if any(price is None or price.end_ts != ts + step or price.price_ex_vat is None
           or price.quality == "incomplete" for ts, price in zip(range(start, end, step), prices, strict=True)):
        return None, "unresolved:" + source if any(price is not None for price in prices) else None, "incomplete"
    # Tijdgewogen, nooit met import/export gewogen; rekenen onder precisie 36.
    value = sum((price.price_ex_vat * (price.end_ts - price.start_ts) for price in prices), Decimal("0")) / duration
    if value.as_tuple().exponent < -12:
        value = value.normalize()
        if value.as_tuple().exponent < -12:
            raise ValueError("Derived spot precision exceeds 12 decimal places")
    quality = "estimated" if any(price.quality == "estimated" for price in prices) else "measured"
    # Bestaande tarieflaag bewaakt ook bonus-/leverings-/venstergrenzen.
    try:
        quote(snapshot, item.start_ts, item.end_ts, value)
    except ValueError:
        return None, "unresolved:" + source, "incomplete"
    return value, source, quality


def _canonical_decimal(value: Decimal) -> str:
    if not isinstance(value, Decimal):
        raise TypeError("Only Decimal values need canonical input serialization")
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def merge_inputs(
    meter: Sequence[MeterInterval], spots: Sequence[SpotInterval], snapshots: Sequence[TariffSnapshot],
    *, meter_id: str,
) -> tuple[InputInterval, ...]:
    """Koppel één meter zonder energie te verdelen of ontbrekende prijzen te raden.

    Meteridentiteit komt uitsluitend uit de expliciete parameter. Revisie 0 is
    ongeregistreerde invoer; uitsluitend A6 kent bij registratie een positieve
    revisie toe. De inhoudshash bevat genormaliseerde meet-/prijsdata, herkomst,
    meter-id en gekozen snapshothash, maar is zelf geen geregistreerde revisie.
    Interne gaten blijven expliciet; dekking buiten de aangeleverde grenzen is
    onbekend aan deze functie en wordt niet verzonnen.
    """
    _check_meter_id(meter_id)
    _check_order(meter, "Meter")
    _check_order(spots, "Spot")
    by_start = {price.start_ts: price for price in spots}
    readings = []
    for item in meter:
        if readings and readings[-1].end_ts < item.start_ts:
            readings.append(MeterInterval(
                readings[-1].end_ts, item.start_ts, None, None, None,
                "unknown", "incomplete", ("issue:missing-meter-interval",),
            ))
        readings.append(item)
    result = []
    with localcontext() as ctx:
        ctx.prec = 36
        for item in readings:
            snapshot, price, source, price_quality = None, None, None, "incomplete"
            try:
                snapshot = resolve_snapshot(snapshots, item.start_ts, item.end_ts)
            except ValueError:
                pass  # Geen enkele geldige dekking: niet splitsen met verzonnen energie.
            if snapshot is not None:
                price, source, price_quality = _price_for_interval(item, snapshot, by_start)
            quality = item.quality
            if price_quality == "incomplete" or any(value is None for value in
                                                     (item.import_kwh, item.export_kwh, item.battery_discharge_kwh)):
                quality = "incomplete"
            elif quality != "incomplete" and (price_quality == "estimated" or item.end_ts - item.start_ts > 900):
                quality = "estimated"
            content = {
                **asdict(item), "quality": quality, "spot_ex_vat": price,
                "spot_source": source, "meter_id": meter_id,
            }
            canonical = json.dumps({
                **content, "adapter_version": "metering-v1",
                "snapshot_hash": None if snapshot is None else snapshot.snapshot_hash,
            }, default=_canonical_decimal, sort_keys=True, separators=(",", ":"))
            result.append(InputInterval(
                **content, input_hash=sha256(canonical.encode("utf-8")).hexdigest(), input_revision=0,
            ))
    return tuple(result)
