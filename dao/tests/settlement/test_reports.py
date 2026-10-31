"""Stored amounts, not repriced history or energy times a mean tariff."""

import datetime as dt
import json
import time
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dao.prog.da_report import Report
from dao.tests.prog.test_consolidate import db as db
from dao.tests.prog.test_da_report_frame_building import make_report_with_prices
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_cli import local_config as local_config
from dao.tests.settlement.test_metering import recorder as recorder
from dao.tests.settlement.test_publication import pricing_config
from dao.tests.settlement.test_repository import terms as terms
from dao.tests.settlement.test_service import system as system


@pytest.fixture
def report(system):
    instance = Report.__new__(Report)
    instance.config = SimpleNamespace(prices=system.pricing)
    instance.prices_options = system.pricing
    instance.time_zone = "Europe/Amsterdam"
    instance.db_da = SimpleNamespace(epoch=lambda value: int(value.replace(tzinfo=ZoneInfo(instance.time_zone)).timestamp())
                                    if value.tzinfo is None else int(value.timestamp()))
    instance.settlement_service = lambda: system.service
    return instance


@pytest.fixture
def reference_energy(db, report):
    """Stored components, independent of the report's presentation frame."""
    from sqlalchemy import text

    db.TARGET_TIMEZONE = report.time_zone
    report.db_da = db
    with db.engine.begin() as connection:
        connection.execute(text("INSERT INTO variabel (id,code,name,dim,aggregate) VALUES (6,'bat_in','Charge','kWh','sum'),(7,'bat_out','Discharge','kWh','sum')"))

    def store(cons, prod, charge, discharge):
        with db.engine.begin() as connection:
            connection.execute(text('INSERT INTO "values" (variabel,time,value) VALUES (:code,:at,:amount)'), [
                {"code": code, "at": START_TS, "amount": amount}
                for code, amount in zip((1, 2, 6, 7), (cons, prod, charge, discharge), strict=True)])
    return store


def test_report_cost_not_energy_times_mean_price(system, report):
    system.meter.items = [system.meter.items[0], system.meter.items[4]]
    system.service.consolidate(START_TS, START_TS + 4500, publish=True, reason="prices")
    result = report.get_settlement_data(START_TS, START_TS + 4500)
    assert result.import_cost.dropna().sum() == Decimal("0.50")


def test_report_keeps_frozen_tariff_after_config_edit(system, report):
    view = system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="frozen")
    before = report.get_settlement_data(START_TS, START_TS + 900, generation=view.generation)
    report.prices_options = system.pricing.model_copy(update={"vat_consumption": {"2020-01-01": 99.0}})
    after = report.get_settlement_data(START_TS, START_TS + 900, generation=view.generation)
    assert after.import_cost.iloc[0] == before.import_cost.iloc[0] == Decimal("0.10")
    assert set(after.bonus_kind) == {"calculated"}


def test_unknown_pre2027_history_stays_legacy(report, monkeypatch):
    calls = []
    def legacy(*args, **kwargs):
        calls.append(args)
        return pd.DataFrame({"cost": [7]})
    monkeypatch.setattr(report, "_get_legacy_grid_data", legacy, raising=False)
    result = report.get_grid_data("", dt.datetime(2026, 12, 1), dt.datetime(2026, 12, 2))
    assert result.cost.iloc[0] == 7 and len(calls) == 1


def test_active_calc_cost_and_recalc_use_amounts_not_mutable_prices(system, report):
    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="report")
    start = dt.datetime.fromtimestamp(START_TS, ZoneInfo("Europe/Amsterdam"))
    cost = report.calc_cost(start, start + dt.timedelta(minutes=15), "cost")
    assert cost.value.iloc[0] == Decimal("0.10")
    frame = pd.DataFrame({"tijd": [start], "consumption": [1.0], "production": [0.0],
                          "da_cons": [999.0], "da_prod": [999.0], "tot": [start + dt.timedelta(minutes=15)],
                          "datasoort": ["recorded"]})
    result = report.recalc_df_ha(frame, "uur")
    assert result.cost.iloc[0] == Decimal("0.10")


def test_legacy_profit_sign_conventions(system, report):
    from dataclasses import replace
    system.meter.items[0] = replace(system.meter.items[0], export_kwh="2")
    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="sign")
    frame = report.get_settlement_data(START_TS, START_TS + 900)
    assert frame.base_export_revenue.iloc[0] == Decimal("0.20")
    from dao.prog.da_report import legacy_prognosis_revenue
    assert legacy_prognosis_revenue(Decimal("-0.20")) == Decimal("0.20")
    assert legacy_prognosis_revenue(Decimal("0.20")) == Decimal("-0.20")


def test_saving_settles_reference_independently_not_at_mean_or_bonus_price(system, report, reference_energy):
    reference_energy(1.0, 0.0, 1.0, 0.0)
    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="actual")
    start = dt.datetime.fromtimestamp(START_TS, ZoneInfo("Europe/Amsterdam"))
    report.periodes = {"synthetic": {"vanaf": start, "tot": start + dt.timedelta(minutes=15), "interval": "uur"}}
    report.saving_cost_dict = {"series": {}}
    report.get_columns = lambda *args: pd.DataFrame({"tijd": [start], "cons": [1.0], "prod": [0.0],
        "bat_in": [1.0], "bat_out": [0.0], "da_cons": [999.0], "da_prod": [999.0], "da_bonus": [999.0]})
    report.clean_df = lambda config, frame, *args: frame
    result = report.calc_saving_cost("synthetic")
    assert result.cost.iloc[0] == Decimal("0.10")
    assert result.cost_zonder.iloc[0] == Decimal("0")
    assert result.saving.iloc[0] == Decimal("-0.10")


def test_unpublished_part_of_report_bucket_stays_unknown(system, report):
    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="partial")
    frame = report.get_grid_data("", START_TS, START_TS + 7200)
    assert len(frame) == 2
    assert frame.cost.isna().all() and frame.netto_cost.isna().all()
    assert set(frame.quality) == {"incomplete"}
    assert set(frame.quality_label) == {"onvolledig"}


def test_savings_reference_uses_registered_prices_after_source_or_config_edit(system, report, reference_energy):
    from dataclasses import replace
    reference_energy(2.0, 0.0, 1.0, 0.0)
    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="frozen-reference")
    start = dt.datetime.fromtimestamp(START_TS, ZoneInfo("Europe/Amsterdam"))
    report.periodes = {"synthetic": {"vanaf": start, "tot": start + dt.timedelta(minutes=15), "interval": "uur"}}
    report.saving_cost_dict = {"series": {}}
    report.get_columns = lambda *args: pd.DataFrame({"tijd": [start], "cons": [2.0], "prod": [0.0],
        "bat_in": [1.0], "bat_out": [0.0]})
    report.clean_df = lambda config, frame, *args: frame
    system.spots[0] = replace(system.spots[0], price_ex_vat="0.90")
    system.service.snapshots = ()  # Current options are not the historical financial selection.
    result = report.calc_saving_cost("synthetic")
    assert result.cost_zonder.iloc[0] == Decimal("0.10")
    assert result.saving.iloc[0] == Decimal("0.00")


def test_grid_report_does_not_round_decimal_amounts_at_default_context(system, report):
    system.meter.items[0] = replace(
        system.meter.items[0], import_kwh="12345678901234567890.123456789")
    system.spots[0] = replace(system.spots[0], price_ex_vat="0.123456789012")
    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="precision")
    frame = report.get_grid_data("", START_TS, START_TS + 900)
    assert frame.netto_cost.iloc[0] == Decimal("1524157875319616034.33196160343167200")
    at = dt.datetime.fromtimestamp(START_TS, ZoneInfo("Europe/Amsterdam"))
    source = pd.DataFrame({"tijd": [at], "tot": [at + dt.timedelta(minutes=15)],
                           "consumption": [1.0], "production": [0.0], "datasoort": ["recorded"]})
    recalculated = report.recalc_df_ha(source, "uur")
    assert recalculated.cost.iloc[0] == Decimal("1524157875319616034.33196160343167200")


def test_active_grid_presentation_keeps_unknown_buckets_and_decimal_totals(system, report):
    system.service.consolidate(START_TS, START_TS + 3600, publish=True, reason="presentation")
    grid = report.get_grid_data("", START_TS, START_TS + 7200)
    table = report.calc_grid_columns(grid, "uur", "tabel")
    assert pd.isna(table[("Kosten", "eur")].iloc[-1])
    assert table[("Kwaliteit", "")].iloc[-1] == "onvolledig"
    complete = report.calc_grid_columns(grid.iloc[:1], "uur", "tabel")
    assert complete[("Kosten", "eur")].iloc[-1] == Decimal("0.40")


def test_active_savings_presentation_sums_decimal_amounts(system, report, reference_energy):
    reference_energy(1.0, 0.0, 1.0, 0.0)
    system.service.consolidate(START_TS, START_TS + 900, publish=True, reason="savings-table")
    start = dt.datetime.fromtimestamp(START_TS, ZoneInfo("Europe/Amsterdam"))
    report.periodes = {"synthetic": {"vanaf": start, "tot": start + dt.timedelta(minutes=15), "interval": "uur"}}
    fields = ("cost", "profit", "netto_cost", "cost_zonder", "profit_zonder", "netto_cost_zonder", "saving")
    report.saving_cost_dict = {"series": {key: {"header": "Bedragen", "name": key, "dim": "eur", "agg": "sum"}
                                            for key in fields}}
    report.get_columns = lambda *args: pd.DataFrame({"tijd": [start], "uur": ["00:00"], "cons": [1.0],
        "prod": [0.0], "bat_in": [1.0], "bat_out": [0.0]})
    table = report.calc_saving_cost("synthetic", active_view="tabel")
    assert table[("Bedragen", "cost", "eur")].iloc[-1] == Decimal("0.10")
    assert table[("Bedragen", "saving", "eur")].iloc[-1] == Decimal("-0.10")


def test_mixed_legacy_active_prices_use_each_adapter_without_decimal_float_error(local_config):
    from sqlalchemy import text

    from dao.prog.da_settlement import load_service
    service = load_service(local_config.path)
    with local_config.db.engine.begin() as connection:
        connection.execute(text('INSERT INTO "values" (variabel,time,value) VALUES (5,:at,0.1)'),
                           {"at": START_TS - 3600})
    report = make_report_with_prices(local_config.db)
    report.prices_options = pricing_config()
    report.time_zone = "Europe/Amsterdam"
    report.settlement_service = lambda: service
    frame = report.get_price_data(dt.datetime(2026, 12, 31, 23), dt.datetime(2027, 1, 1, 1))
    assert list(frame.da_cons) == [Decimal("0.242"), Decimal("0.1")]
    assert set(frame.bonus_kind) == {"potential"}


def test_mixed_grid_range_uses_stored_legacy_then_book_never_both(local_config):
    from sqlalchemy import text

    from dao.prog.da_settlement import load_service
    service = load_service(local_config.path)
    service.consolidate(START_TS, START_TS + 900, publish=True, reason="mixed-report")
    with local_config.db.engine.begin() as connection:
        connection.execute(text('INSERT INTO "values" (variabel,time,value) VALUES (:code,:at,:amount)'), [
            {"code": code, "at": at, "amount": amount}
            for at, cost in ((START_TS - 3600, 7.0), (START_TS, 999.0))
            for code, amount in ((1, 1.0), (2, 0.0), (3, cost), (4, 0.0))])
    report = make_report_with_prices(local_config.db)
    report.prices_options = pricing_config()
    report.time_zone = "Europe/Amsterdam"
    report.settlement_service = lambda: service
    frame = report.get_grid_data("", dt.datetime(2026, 12, 31, 23), dt.datetime(2027, 1, 1, 0, 15))
    table = report.calc_grid_columns(frame, "uur", "tabel")
    assert len(frame) == 2 and list(frame.cost) == [7.0, Decimal("0.03")]
    assert table[("Kosten", "eur")].iloc[-1] == Decimal("7.03")


def test_settlement_api_preserves_old_components_and_separate_month_credit(system, report, terms):
    from dao.prog.settlement.engine import settle_series
    from dao.prog.settlement.metering import merge_inputs
    from dao.prog.settlement.types import MeterInterval, MonthState, SettlementState, SpotInterval
    from dao.tests.settlement.conftest import MONTH_END_TS
    from dao.tests.settlement.test_repository import METER

    at = MONTH_END_TS - 900
    snapshot = terms(monthly_export_floor="spot_zero")
    reading = MeterInterval(at, MONTH_END_TS, "0", "1", "0", "verified_pv", "measured", ("synthetic-close",))
    prices = (SpotInterval(at // 3600 * 3600, MONTH_END_TS, "-0.20", "synthetic-hour", "measured"),)
    items = merge_inputs((reading,), prices, (snapshot,), meter_id=METER)
    system.repo.register_inputs(METER, items, (snapshot,), 0)
    batch = system.repo.read_inputs(METER, at, MONTH_END_TS)
    opening = SettlementState(at, None, MonthState(snapshot.contract_id, START_TS, MONTH_END_TS, "0", (), True))
    result = settle_series(batch.items, batch.snapshots, opening)
    system.repo.commit(METER, result, batch, 0, "synthetic-close")
    report.periodes = {"synthetic": {"vanaf": at, "tot": MONTH_END_TS, "interval": "uur"}}
    payload = json.loads(report.get_api_data("settlement", "synthetic", generation=1))
    assert payload["net_cost"] == "0" and len(payload["data"]) == 2
    interval, adjustment = payload["data"]
    assert interval["base_export_revenue"] == "-0.2" and interval["bonus_revenue"] == "0"
    assert adjustment["credit_total"] == "0.2" and adjustment["time"] == MONTH_END_TS - 1
    assert adjustment["period_start_ts"] == START_TS and adjustment["period_end_ts"] == MONTH_END_TS
    assert adjustment["quality_label"] == "geschat"  # VAT allocation is a model estimate, not supplier evidence.
    corrected = merge_inputs((reading,), (replace(prices[0], price_ex_vat="0.10"),), (snapshot,), meter_id=METER)
    system.repo.register_inputs(METER, corrected, (snapshot,), batch.input_revision)
    revised = system.repo.read_inputs(METER, at, MONTH_END_TS)
    system.repo.commit(METER, settle_series(revised.items, revised.snapshots, opening), revised, 1, "revision")
    historical = json.loads(report.get_api_data("settlement", "synthetic", generation=1))
    assert historical["data"][0]["base_export_revenue"] == "-0.2"
    assert historical["data"][1]["credit_total"] == "0.2" and historical["freshness"] == "outdated"


def test_savings_report_aborts_instead_of_mixing_generations(system, report, monkeypatch):
    from dao.prog.settlement.repository import ConcurrentSettlementError
    system.service.consolidate(START_TS, START_TS + 7200, publish=True, reason="report-prefix")
    start = dt.datetime.fromtimestamp(START_TS, ZoneInfo("Europe/Amsterdam"))
    report.periodes = {"synthetic": {"vanaf": start, "tot": start + dt.timedelta(hours=2), "interval": "uur"}}
    report.saving_cost_dict = {"series": {}}
    report.get_columns = lambda *args: pd.DataFrame({"tijd": [start, start + dt.timedelta(hours=1)],
        "cons": [4.0, 4.0], "prod": [0.0, 0.0], "bat_in": [0.0, 0.0], "bat_out": [0.0, 0.0]})
    report.clean_df = lambda config, frame, *args: frame
    read = report._book_grid
    calls = []
    def interleaved(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            system.spots[1] = replace(system.spots[1], price_ex_vat="0.8")
            system.service.consolidate(START_TS, START_TS + 7200, publish=True, reason="interleaved")
        return read(*args, **kwargs)
    monkeypatch.setattr(report, "_book_grid", interleaved)
    with pytest.raises(ConcurrentSettlementError):
        report.calc_saving_cost("synthetic")


@pytest.mark.parametrize("consumer", ["grid", "recalc", "cost"])
def test_composite_report_pins_one_basis_across_contract_ranges(system, report, monkeypatch, consumer):
    from datetime import date

    from dao.prog.settlement.repository import ConcurrentSettlementError
    from dao.prog.settlement.tariffs import make_snapshot

    first = system.pricing.settlement.contracts[0].model_copy(update={"valid_to": date(2027, 1, 2)})
    second = first.model_copy(update={"id": "synthetic-second", "valid_from": date(2027, 1, 2), "valid_to": None})
    config = system.pricing.settlement.model_copy(update={"contracts": [first, second]})
    report.prices_options = system.pricing.model_copy(update={"settlement": config})
    system.service.snapshots = tuple(make_snapshot(report.prices_options, contract) for contract in config.contracts)
    later = START_TS + 86400
    system.clock[0] = later + 86400
    system.meter.items = [system.meter.items[0], replace(system.meter.items[0], start_ts=later, end_ts=later + 900)]
    system.spots[:] = [system.spots[0], replace(system.spots[0], start_ts=later, end_ts=later + 3600)]
    system.service.consolidate(START_TS, later + 900, publish=True, reason="two-contracts")
    entry = "get_settlement_data" if consumer == "cost" else "_book_grid"
    read = getattr(report, entry)
    changed = []

    def interleaved(*args, **kwargs):
        result = read(*args, **kwargs)
        if not changed:
            changed.append(True)
            system.spots[1] = replace(system.spots[1], price_ex_vat="0.30")
            system.service.consolidate(later, later + 900, publish=True, reason="interleaved-contract")
        return result

    monkeypatch.setattr(report, entry, interleaved)
    with pytest.raises(ConcurrentSettlementError):
        if consumer == "grid":
            report.get_grid_data("", START_TS, later + 900)
        elif consumer == "cost":
            report.calc_cost(START_TS, later + 900, "cost")
        else:
            moments = [dt.datetime.fromtimestamp(at, ZoneInfo(report.time_zone)) for at in (START_TS, later)]
            report.recalc_df_ha(pd.DataFrame({"tijd": moments,
                "tot": [at + dt.timedelta(minutes=15) for at in moments], "consumption": [1.0, 1.0],
                "production": [0.0, 0.0], "datasoort": ["recorded", "recorded"]}), "dag")


def stored_savings_report(db, service, pricing):
    """Real DB energy/price adapters; expose unrounded amounts before presentation."""
    instance = make_report_with_prices(db)
    instance.prices_options = pricing
    instance.time_zone = "Europe/Amsterdam"
    db.TARGET_TIMEZONE = instance.time_zone
    instance.settlement_service = lambda: service
    instance.saving_cost_dict = {"calc_interval": "uur", "series": {
        **{key: {"source": "db", "sensors": []} for key in ("cons", "prod", "bat_in", "bat_out")},
        **{key: {"source": "prices"} for key in ("da_cons", "da_prod")},
    }}
    instance.clean_df = lambda config, frame, *args: frame
    return instance


@pytest.fixture
def amsterdam_time(monkeypatch):
    # SQLite's legacy localtime adapter follows the container timezone.
    with monkeypatch.context() as local:
        local.setenv("TZ", "Europe/Amsterdam")
        time.tzset()
        yield
    time.tzset()


def test_savings_crossing_contract_start_normalizes_legacy_energy_and_prices(local_config, amsterdam_time):
    from sqlalchemy import text

    from dao.prog.da_settlement import load_service

    service = load_service(local_config.path)
    service.consolidate(START_TS, START_TS + 900, publish=True, reason="mixed-savings")
    with local_config.db.engine.begin() as connection:
        connection.execute(text('CREATE TABLE prognoses AS SELECT * FROM "values" WHERE 0'))
        connection.execute(text("INSERT INTO variabel (id,code,name,dim,aggregate) VALUES (6,'bat_in','Charge','kWh','sum'),(7,'bat_out','Discharge','kWh','sum')"))
        connection.execute(text('INSERT INTO "values" (variabel,time,value) VALUES (5,:at,0.1)'), {"at": START_TS - 3600})
        connection.execute(text('INSERT INTO "values" (variabel,time,value) VALUES (:code,:at,:amount)'), [
            {"code": code, "at": at, "amount": amount}
            for at, quantities in ((START_TS - 3600, (2.0, 0.5, 1.0, 0.0)), (START_TS, (0.3, 0.3, 0.1, 0.0)))
            for code, amount in zip((1, 2, 6, 7), quantities, strict=True)])
    report = stored_savings_report(local_config.db, service, pricing_config())
    report.btw_l_def = report.btw_t_def = report.taxes_t_def = {"2026-01-01": 0.0}
    report.periodes = {"transition": {"vanaf": dt.datetime(2026, 12, 31, 23),
                                      "tot": dt.datetime(2027, 1, 1, 0, 15), "interval": "uur"}}
    frame = report.calc_saving_cost("transition")
    assert list(frame.netto_cost) == [Decimal("0.35"), Decimal("0.00")]
    assert list(frame.netto_cost_zonder) == [Decimal("0.10"), Decimal("-0.01")]
    assert list(frame.saving) == [Decimal("-0.25"), Decimal("-0.01")]


@pytest.fixture
def reference_month(system, db, terms, amsterdam_time):
    from datetime import date

    from sqlalchemy import text

    from dao.prog.settlement.types import MeterInterval, SpotInterval
    from dao.tests.settlement.conftest import MONTH_END_TS

    system.clock[0] = MONTH_END_TS + 86400
    system.service.snapshots = (terms(monthly_export_floor="spot_zero", supply_start=date(2027, 1, 1)),)
    system.meter.items, system.spots[:] = [], []
    energy = []
    for at in range(START_TS, MONTH_END_TS, 3600):
        production, discharge, spot = ("2", "2", "-0.2") if at == START_TS else (
            ("1", "0", "0.1") if at == MONTH_END_TS - 3600 else ("0", "0", "0"))
        system.meter.items.append(MeterInterval(at, at + 3600, "0", production, discharge,
                                               "unknown", "estimated", ("synthetic-month",)))
        system.spots.append(SpotInterval(at, at + 3600, spot, "synthetic-month-market", "measured"))
        energy.extend({"code": code, "at": at, "amount": amount}
                      for code, amount in ((1, 0.0), (2, float(production)), (6, 0.0), (7, float(discharge))))
    system.service.consolidate(START_TS, MONTH_END_TS, publish=True, reason="complete-reference-month")
    with db.engine.begin() as connection:
        connection.execute(text('CREATE TABLE prognoses AS SELECT * FROM "values" WHERE 0'))
        connection.execute(text("INSERT INTO variabel (id,code,name,dim,aggregate) VALUES (6,'bat_in','Charge','kWh','sum'),(7,'bat_out','Discharge','kWh','sum')"))
        connection.execute(text('INSERT INTO "values" (variabel,time,value) VALUES (:code,:at,:amount)'), energy)
    report = stored_savings_report(db, system.service, system.pricing)
    opening, closing = dt.datetime(2027, 1, 1), dt.datetime(2027, 2, 1)
    split = closing - dt.timedelta(hours=1)
    report.periodes = {key: {"vanaf": start, "tot": end, "interval": "dag"}
                      for key, start, end in (("month", opening, closing), ("before", opening, split), ("after", split, closing))}
    return report


def test_reference_month_matches_adjacent_reports_after_negative_battery_export(reference_month):
    month = reference_month.calc_saving_cost("month")
    before = reference_month.calc_saving_cost("before")
    after = reference_month.calc_saving_cost("after")
    assert month[["cons", "prod", "bat_in", "bat_out"]].iloc[[0, -1]].values.tolist() == [
        [0.0, 2.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0]]
    assert month.netto_cost_zonder.sum() == Decimal("-0.10")
    assert after.netto_cost_zonder.iloc[0] == Decimal("-0.10")
    assert before.saving.sum() == Decimal("-0.40") and after.saving.iloc[0] == Decimal("0.30")
    assert month.saving.sum() == before.saving.sum() + after.saving.sum() == Decimal("-0.10")


@pytest.mark.parametrize("component", ["cons", "prod", "bat_in", "bat_out"])
@pytest.mark.parametrize("missing", ["absent", "null"])
def test_missing_reference_prefix_is_unknown_not_actual_month_state(reference_month, component, missing):
    from sqlalchemy import text

    with reference_month.db_da.engine.begin() as connection:
        change = 'DELETE FROM "values"' if missing == "absent" else 'UPDATE "values" SET value=NULL'
        connection.execute(text(change + ' WHERE variabel=(SELECT id FROM variabel WHERE code=:component) AND time=:at'),
                           {"component": component, "at": START_TS})
    frames = {period: reference_month.calc_saving_cost(period) for period in ("month", "before", "after")}
    assert pd.isna(frames["after"].netto_cost_zonder.iloc[0]) and pd.isna(frames["after"].saving.iloc[0])
    reference_fields = ["cost_zonder", "profit_zonder", "netto_cost_zonder", "saving"]
    for period, net_cost in (("month", "0"), ("before", "0.40"), ("after", "-0.40")):
        frame = frames[period]
        # The closed actual book stays known, including its separate month credit.
        assert frame[["cost", "profit", "netto_cost"]].notna().all().all()
        assert frame.cost.sum() == Decimal("0")
        assert frame.netto_cost.sum() == Decimal(net_cost)
        assert frame[reference_fields].isna().all().all(), period
    fields = ["cost", "profit", "netto_cost", *reference_fields]
    pd.testing.assert_frame_equal(frames["month"][fields], pd.concat([frames["before"], frames["after"]])[fields])
