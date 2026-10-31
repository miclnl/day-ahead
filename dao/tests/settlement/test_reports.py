"""Stored amounts, not repriced history or energy times a mean tariff."""

import datetime as dt
import json
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


def test_saving_settles_reference_independently_not_at_mean_or_bonus_price(system, report):
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


def test_savings_reference_uses_registered_prices_after_source_or_config_edit(system, report):
    from dataclasses import replace
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


def test_active_savings_presentation_sums_decimal_amounts(system, report):
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
