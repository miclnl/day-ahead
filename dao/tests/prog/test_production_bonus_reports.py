"""The reports pay the feed-in price plus the production bonus, and still work without the column."""

import datetime

import pytest

pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

from dao.prog.da_report import Report  # noqa: E402

TIJD = datetime.datetime(2027, 6, 1, 12, 0)


def grid_frame(with_bonus: bool) -> pd.DataFrame:
    row = {
        "tijd": TIJD,
        "consumption": 1.0,
        "production": 2.0,
        "da_cons": 0.30,
        "da_prod": 0.10,
        "datasoort": "recorded",
    }
    if with_bonus:
        row["da_bonus"] = 0.03
    return pd.DataFrame([row])


def test_recalc_df_ha_pays_the_bonus_on_production():
    result = Report.__new__(Report).recalc_df_ha(grid_frame(with_bonus=True), "uur")
    assert result["profit"].iloc[0] == pytest.approx(2.0 * 0.13)
    assert result["cost"].iloc[0] == pytest.approx(1.0 * 0.30)


def test_recalc_df_ha_without_a_bonus_column_is_unchanged():
    result = Report.__new__(Report).recalc_df_ha(grid_frame(with_bonus=False), "uur")
    assert result["profit"].iloc[0] == pytest.approx(2.0 * 0.10)


def saving_frame(with_bonus: bool) -> pd.DataFrame:
    row = {
        "uur": "2027-06-01 12:00",
        "cons": 0.0,
        "prod": 2.0,
        "bat_in": 0.0,
        "bat_out": 0.0,
        "da_cons": 0.30,
        "da_prod": 0.10,
    }
    if with_bonus:
        row["da_bonus"] = 0.03
    return pd.DataFrame([row])


def make_saving_report(frame: pd.DataFrame) -> Report:
    report = Report.__new__(Report)
    report.saving_cost_dict = {"series": {}}
    report.periodes = {"vandaag": {"interval": "uur"}}
    report.get_columns = lambda calc_dict, period, _tot: frame.copy()
    report.clean_df = lambda calc_dict, df, columns, view, interval: df
    return report


def test_calc_saving_cost_pays_the_bonus_with_and_without_battery():
    result = make_saving_report(saving_frame(with_bonus=True)).calc_saving_cost("vandaag")
    assert result["profit"].iloc[0] == pytest.approx(2.0 * 0.13)
    assert result["profit_zonder"].iloc[0] == pytest.approx(2.0 * 0.13)


def test_calc_saving_cost_without_a_bonus_column_is_unchanged():
    result = make_saving_report(saving_frame(with_bonus=False)).calc_saving_cost("vandaag")
    assert result["profit"].iloc[0] == pytest.approx(2.0 * 0.10)
