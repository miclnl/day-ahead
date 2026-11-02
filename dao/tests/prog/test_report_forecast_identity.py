"""PV forecasts must retain their known instants through the report API."""

import datetime as dt
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from dao.forecast.pv.service import PVService
from dao.prog.config.models.devices.solar import SolarConfig
from dao.prog.da_report import Report

PERIOD = "vandaag en morgen"


class SyntheticWeather:
    def get_prognose_fields(self, codes, start_ts, end_ts, interval):
        if codes != ["gr", "dni", "dhi", "temp", "winds"] or interval != "1hour":
            raise AssertionError("Unexpected PV weather request")
        frame = pd.DataFrame({
            "time": [1792886400, 1792890000],
            "gr": [100.0, 200.0],
            "dni": [80.0, 160.0],
            "dhi": [20.0, 40.0],
            "temp": [15.0, 16.0],
            "winds": [3.0, 3.0],
        })
        return frame.loc[(frame.time >= start_ts) & (frame.time < end_ts)].copy()


@pytest.fixture
def make_report(monkeypatch, tmp_path):
    def simulate(params, weather, latitude, longitude, interval_s):
        # Deliberately synthetic kWh, not a second physical forecast model.
        return pd.Series([0.25, 0.5], index=weather.index) * params.total_kwp()

    monkeypatch.setattr("dao.forecast.pv.service.simulate", simulate)

    def make(field, *, source_zone="Europe/Amsterdam", report_zone="Europe/Amsterdam",
             history=(), forecast="available"):
        devices = [SolarConfig(name=f"Synthetic roof {number}", capacity=capacity,
                               tilt=30, orientation=0, calibration="off", model="physical")
                   for number, capacity in enumerate((1.0, 0.5))]
        if forecast == "no_pv":
            devices = []
        config = SimpleNamespace(
            solar=devices if field == "pv_ac" else [],
            battery=[SimpleNamespace(solar=[device]) for device in devices]
            if field == "pv_dc" else [],
        )
        service = PVService(config, SyntheticWeather(), None, 52.1, 5.2,
                            tmp_path / "pv", source_zone, "1hour")
        if forecast == "empty":
            monkeypatch.setattr(service, "forecast", lambda *args, **kwargs: pd.DataFrame(
                columns=["tijd", "prediction"]))

        report = Report.__new__(Report)
        report.config = config
        report.solar = config.solar
        report.time_zone = report_zone
        report.energy_balance_dict = {"pv_ac": {}}
        report.several_dict = {"pv_dc": {}}
        start = dt.datetime(2026, 10, 25, tzinfo=dt.UTC)
        report.periodes = {PERIOD: {"vanaf": start, "tot": start + dt.timedelta(days=2),
                                   "interval": "uur"}}
        monkeypatch.setattr(report, "pv_service", lambda: service)

        def recorded_balance(periode, *, field, _tot, col_dict):
            # Replace only the DB/HA reader, never the PV report/API methods.
            return pd.DataFrame({
                "tijd": [moment for moment, _ in history],
                field: [value for _, value in history],
                "datasoort": ["recorded"] * len(history),
            }), start

        monkeypatch.setattr(report, "get_energy_balance_data", recorded_balance)
        return report

    return make


@pytest.mark.parametrize("field", ["pv_ac", "pv_dc"])
@pytest.mark.parametrize("cumulate", [False, True])
@pytest.mark.parametrize(("source_zone", "report_zone", "labels"), [
    pytest.param("Europe/Amsterdam", "Europe/Amsterdam",
                 ["2026-10-25 02:00", "2026-10-25 02:00"], id="local-fold"),
    pytest.param("UTC", "Europe/Amsterdam",
                 ["2026-10-25 02:00", "2026-10-25 02:00"], id="utc-source"),
    pytest.param("America/New_York", "Europe/Amsterdam",
                 ["2026-10-25 02:00", "2026-10-25 02:00"], id="other-source"),
    pytest.param("Europe/Amsterdam", "UTC",
                 ["2026-10-25 00:00", "2026-10-25 01:00"], id="utc-report"),
    pytest.param("Europe/Amsterdam", "Asia/Kathmandu",
                 ["2026-10-25 05:45", "2026-10-25 06:45"], id="fractional-report"),
])
def test_pv_api_preserves_both_winter_hours(make_report, field, cumulate,
                                         source_zone, report_zone, labels):
    """Stripping known PV offsets must not collapse or shift the two instants."""
    report = make_report(field, source_zone=source_zone, report_zone=report_zone)

    payload = json.loads(report.get_api_data(field, PERIOD, cumulate=cumulate))

    assert payload["message"] == "Success"
    rows = payload["data"]
    assert [row["time_ts"] for row in rows] == [1792886400000, 1792890000000]
    assert [row["time"] for row in rows] == labels
    assert [row["value"] for row in rows] == ([0.375, 1.125] if cumulate else [0.375, 0.75])
    assert [row["datatype"] for row in rows] == ["expected", "expected"]
    assert all(set(row) == {"time_ts", "time", "value", "datatype"} for row in rows)


@pytest.mark.parametrize("field", ["pv_ac", "pv_dc"])
def test_pv_prognose_keeps_aware_time_and_index(make_report, field):
    report = make_report(field)
    start = dt.datetime(2026, 10, 25, tzinfo=dt.UTC)

    frame = report.get_pv_prognose(field, start, start + dt.timedelta(hours=2))

    assert frame.time.dt.tz is not None
    assert frame.index.tz is not None
    assert [int(moment.timestamp()) for moment in frame.time] == [1792886400, 1792890000]
    assert [int(moment.timestamp()) for moment in frame.index] == [1792886400, 1792890000]
    assert list(frame[field]) == [0.375, 0.75]


@pytest.mark.parametrize("field", ["pv_ac", "pv_dc"])
@pytest.mark.parametrize("cumulate", [False, True])
@pytest.mark.parametrize("legacy_time", [
    "2026-10-25 01:00",
    dt.datetime(2026, 10, 25, 1),
    "2026-10-24T23:00:00+00:00",
])
def test_pv_api_combines_legacy_labels_with_aware_forecast(make_report, field,
                                                        cumulate, legacy_time):
    report = make_report(field, history=[(legacy_time, 2.0)])

    rows = json.loads(report.get_api_data(field, PERIOD, cumulate=cumulate))["data"]

    assert [row["time_ts"] for row in rows] == [1792882800000, 1792886400000, 1792890000000]
    assert [row["time"] for row in rows] == ["2026-10-25 01:00", "2026-10-25 02:00",
                                            "2026-10-25 02:00"]
    assert [row["value"] for row in rows] == (
        [2.0, 2.375, 3.125] if cumulate else [2.0, 0.375, 0.75])
    assert [row["datatype"] for row in rows] == ["recorded", "expected", "expected"]


@pytest.mark.parametrize("field", ["pv_ac", "pv_dc"])
@pytest.mark.parametrize("cumulate", [False, True])
@pytest.mark.parametrize("forecast", ["empty", "no_pv"])
@pytest.mark.parametrize("history", [(), (("2026-10-25 01:00", 2.0),)])
def test_pv_api_handles_empty_forecast_or_no_pv(make_report, field, cumulate, forecast, history):
    report = make_report(field, forecast=forecast, history=history)

    payload = json.loads(report.get_api_data(field, PERIOD, cumulate=cumulate))

    assert payload["message"] == "Success"
    assert payload["data"] == ([{"time_ts": 1792882800000, "time": "2026-10-25 01:00",
                                "value": 2.0, "datatype": "recorded"}] if history else [])
