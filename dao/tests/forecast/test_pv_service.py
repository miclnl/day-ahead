"""Tests for PVService: forecasting and calibrating configured installations."""

from __future__ import annotations

import datetime as dt

import pytest

from dao.forecast.pv.calibrate import CalibrationResult
from dao.forecast.pv.physical import Plane, PVParams
from dao.forecast.pv.service import PVService
from dao.forecast.pv.store import calibration_path, save_calibration
from dao.lib.db_manager import DBmanagerObj
from dao.prog.config.models.devices.solar import SolarConfig

TZ = "Europe/Amsterdam"
LAT, LON = 52.1, 5.2
HOUR = 3600


def make_installation(**overrides) -> SolarConfig:
    defaults = dict(
        name="Roof South",
        tilt=35,
        orientation=0,
        capacity=3.6,
        calibration="off",
        entities_sensors=["sensor.test_pv_energy"],
    )
    defaults.update(overrides)
    return SolarConfig(**defaults)


def make_config(*, solar=None, battery=None):
    import types

    return types.SimpleNamespace(solar=solar or [], battery=battery or [])


@pytest.fixture
def da_db(tmp_path):
    from sqlalchemy import (
        BigInteger,
        Column,
        Float,
        ForeignKey,
        Integer,
        String,
        Table,
        UniqueConstraint,
        insert,
    )

    from dao.lib.db_manager import forecasts_table

    manager = DBmanagerObj(
        db_dialect="sqlite", db_name="day_ahead.db", db_path=str(tmp_path)
    )
    metadata = manager.metadata
    variabel = Table(
        "variabel",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("code", String(10), unique=True, nullable=False),
        Column("name", String(50), unique=True, nullable=False),
        Column("dim", String(10), nullable=False),
        Column("aggregate", String(3), nullable=False, default="avg"),
    )
    for name in ("values", "prognoses"):
        Table(
            name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column("variabel", Integer, ForeignKey("variabel.id"), nullable=False),
            Column("time", BigInteger, nullable=False),
            Column("value", Float),
            UniqueConstraint("variabel", "time"),
        )
    forecasts_table(metadata)
    metadata.create_all(manager.engine)
    with manager.engine.begin() as connection:
        connection.execute(
            insert(variabel),
            [
                {"id": 4, "code": "gr", "name": "Globale straling", "dim": "J/cm2"},
                {"id": 5, "code": "temp", "name": "Temperatuur", "dim": "C"},
                {"id": 23, "code": "winds", "name": "Windsnelheid", "dim": "m/s"},
                {"id": 28, "code": "dni", "name": "Directe straling", "dim": "J/cm2"},
                {"id": 29, "code": "dhi", "name": "Diffuse straling", "dim": "J/cm2"},
            ],
        )
    return manager


def put_prognose(db, code, rows):
    from sqlalchemy import Table, insert, select

    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    prognoses = Table("prognoses", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        ident = connection.execute(
            select(variabel.c.id).where(variabel.c.code == code)
        ).scalar_one()
        connection.execute(
            insert(prognoses),
            [{"variabel": ident, "time": t, "value": v} for t, v in rows],
        )


def put_value(db, code, rows):
    from sqlalchemy import Table, insert, select

    variabel = Table("variabel", db.metadata, autoload_with=db.engine)
    values = Table("values", db.metadata, autoload_with=db.engine)
    with db.engine.begin() as connection:
        ident = connection.execute(
            select(variabel.c.id).where(variabel.c.code == code)
        ).scalar_one()
        connection.execute(
            insert(values),
            [{"variabel": ident, "time": t, "value": v} for t, v in rows],
        )


@pytest.fixture
def pv_service_with_prognoses(da_db, tmp_path):
    start = dt.datetime(2026, 6, 21, 0, 0, tzinfo=dt.UTC)
    t0 = int(start.timestamp())
    for i in range(25):  # one extra point so 15-min interpolation has a next hour
        ts = t0 + i * HOUR
        hour = i % 24
        gr = max(0.0, 200 * (1 - abs(hour - 13) / 7)) if 6 <= hour <= 20 else 0.0
        put_prognose(da_db, "gr", [(ts, gr)])
        put_prognose(da_db, "temp", [(ts, 18.0)])
        put_prognose(da_db, "winds", [(ts, 3.0)])

    config = make_config(solar=[make_installation()])
    service = PVService(
        config, da_db, None, LAT, LON, tmp_path / "pv", TZ, "1hour",
        now=lambda: start,
    )
    return service, start


@pytest.fixture
def pv_service(tmp_path):
    config = make_config(solar=[make_installation()])
    return PVService(
        config, None, None, LAT, LON, tmp_path, TZ, "1hour",
        now=lambda: dt.datetime(2026, 6, 21, tzinfo=dt.UTC),
    )


@pytest.fixture
def pv_service_empty(da_db, tmp_path):
    config = make_config(solar=[make_installation()])
    return PVService(
        config, da_db, None, LAT, LON, tmp_path / "pv", TZ, "1hour",
        now=lambda: dt.datetime(2026, 6, 21, tzinfo=dt.UTC),
    )


def test_forecast_returns_one_row_per_interval(pv_service_with_prognoses):
    service, start = pv_service_with_prognoses
    installation = service.installations()[0]

    result = service.forecast(installation, start, start + dt.timedelta(hours=24), "1hour")

    assert len(result) == 24
    assert result["tijd"].dt.tz is not None
    assert (result["prediction"] >= 0).all()


def test_forecast_15min_uses_interpolated_weather(pv_service_with_prognoses):
    service, start = pv_service_with_prognoses
    installation = service.installations()[0]

    hourly = service.forecast(installation, start, start + dt.timedelta(hours=24), "1hour")
    quarter = service.forecast(installation, start, start + dt.timedelta(hours=24), "15min")

    assert len(quarter) == 96
    assert quarter["tijd"].dt.tz is not None
    hourly_sum = hourly["prediction"].sum()
    quarter_sum = quarter["prediction"].sum()
    assert quarter_sum == pytest.approx(hourly_sum, rel=0.10)


def test_params_for_prefers_calibration(tmp_path, pv_service):
    installation = make_installation(calibration="scale")
    pv_service.config.solar = [installation]

    config_params, source = pv_service.params_for(installation)
    assert source == "config"

    calibrated_params = PVParams(planes=[Plane(tilt=30, azimuth=180, pdc0_kw=2.5)])
    result = CalibrationResult(
        params=calibrated_params,
        mode="scale",
        window_start=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        window_end=dt.datetime(2026, 4, 1, tzinfo=dt.UTC),
        n_hours=1000,
        holdout_mae_fit=0.05,
        holdout_mae_config=0.08,
        ratio=1.0,
        created=pv_service._now(),
    )
    save_calibration(result, calibration_path(pv_service.data_dir, installation.name))

    params, source = pv_service.params_for(installation)
    assert source == "calibrated"
    assert params.planes[0].pdc0_kw == pytest.approx(2.5)


def test_params_for_warns_when_old(tmp_path, pv_service, caplog):
    installation = make_installation(calibration="scale")
    now = pv_service._now()
    old_result = CalibrationResult(
        params=PVParams(planes=[Plane(tilt=30, azimuth=180, pdc0_kw=2.5)]),
        mode="scale",
        window_start=now - dt.timedelta(days=460),
        window_end=now - dt.timedelta(days=100),
        n_hours=1000,
        holdout_mae_fit=0.05,
        holdout_mae_config=0.08,
        ratio=1.0,
        created=now - dt.timedelta(days=100),
    )
    save_calibration(old_result, calibration_path(pv_service.data_dir, installation.name))

    with caplog.at_level("WARNING"):
        pv_service.params_for(installation)

    assert any("dagen oud" in message for message in caplog.messages)


def test_forecast_without_weather_returns_zero_with_error(pv_service_empty, caplog):
    installation = pv_service_empty.installations()[0]
    start = dt.datetime(2026, 6, 21, 0, 0, tzinfo=dt.UTC)

    with caplog.at_level("ERROR"):
        result = pv_service_empty.forecast(
            installation, start, start + dt.timedelta(hours=6), "1hour"
        )

    assert len(result) == 6
    assert (result["prediction"] == 0.0).all()
    assert any("geen weerprognose" in message for message in caplog.messages)


@pytest.fixture
def pv_service_with_history(da_db, tmp_path):
    from dao.tests.forecast.conftest import RecorderHelper

    ha_dir = tmp_path / "ha"
    ha_dir.mkdir()
    manager = DBmanagerObj(
        db_dialect="sqlite", db_name="homeassistant.db", db_path=str(ha_dir)
    )
    from sqlalchemy import Column, Float, Integer, String, Table

    metadata = manager.metadata
    meta = Table(
        "statistics_meta",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("statistic_id", String(255)),
        Column("source", String(32)),
        Column("unit_of_measurement", String(255)),
        Column("has_mean", Integer, nullable=True),
        Column("has_sum", Integer),
        Column("name", String(255), nullable=True),
        Column("mean_type", Integer),
        Column("unit_class", String(255), nullable=True),
    )
    stats = Table(
        "statistics",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("created_ts", Float),
        Column("metadata_id", Integer),
        Column("start_ts", Float),
        Column("mean", Float, nullable=True),
        Column("min", Float, nullable=True),
        Column("max", Float, nullable=True),
        Column("last_reset_ts", Float, nullable=True),
        Column("state", Float, nullable=True),
        Column("sum", Float, nullable=True),
        Column("mean_weight", Float, nullable=True),
    )
    metadata.create_all(manager.engine, tables=[meta, stats])
    helper = RecorderHelper(manager, meta, stats)

    now = dt.datetime(2026, 6, 21, 6, 0, tzinfo=dt.UTC)
    start = now - dt.timedelta(days=62)
    n_hours = 62 * 24

    energy = {}
    total = 0.0
    for i in range(n_hours + 1):
        moment = start + dt.timedelta(hours=i)
        ts = int(moment.timestamp())
        energy[ts] = total
        if i < n_hours:
            hour = moment.hour
            total += 0.3 if 8 <= hour <= 18 else 0.0
    helper.add_energy("sensor.test_pv_energy", "kWh", energy)

    for i in range(n_hours):
        ts = int((start + dt.timedelta(hours=i)).timestamp())
        hour = (start + dt.timedelta(hours=i)).hour
        gr = 150.0 if 8 <= hour <= 18 else 0.0
        put_value(da_db, "gr", [(ts, gr)])
        put_value(da_db, "temp", [(ts, 18.0)])
        put_value(da_db, "winds", [(ts, 3.0)])

    config = make_config(solar=[make_installation(calibration="scale")])
    service = PVService(
        config, da_db, manager, LAT, LON, tmp_path / "pv", TZ, "1hour",
        now=lambda: now,
    )
    return service


@pytest.fixture
def pv_service_for_backtest(da_db, tmp_path):
    """Fourteen days of production and measured weather, inserted in bulk.

    pv_service_with_history writes one row per hour per code in its own
    transaction, which is fine for the two slow calibration tests but far
    too slow for a backtest that only needs a week.
    """
    from dao.tests.forecast.conftest import RecorderHelper

    ha_dir = tmp_path / "ha"
    ha_dir.mkdir()
    manager = DBmanagerObj(
        db_dialect="sqlite", db_name="homeassistant.db", db_path=str(ha_dir)
    )
    from sqlalchemy import Column, Float, Integer, String, Table

    metadata = manager.metadata
    meta = Table(
        "statistics_meta",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("statistic_id", String(255)),
        Column("source", String(32)),
        Column("unit_of_measurement", String(255)),
        Column("has_mean", Integer, nullable=True),
        Column("has_sum", Integer),
        Column("name", String(255), nullable=True),
        Column("mean_type", Integer),
        Column("unit_class", String(255), nullable=True),
    )
    stats = Table(
        "statistics",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("created_ts", Float),
        Column("metadata_id", Integer),
        Column("start_ts", Float),
        Column("mean", Float, nullable=True),
        Column("min", Float, nullable=True),
        Column("max", Float, nullable=True),
        Column("last_reset_ts", Float, nullable=True),
        Column("state", Float, nullable=True),
        Column("sum", Float, nullable=True),
        Column("mean_weight", Float, nullable=True),
    )
    metadata.create_all(manager.engine, tables=[meta, stats])
    helper = RecorderHelper(manager, meta, stats)

    now = dt.datetime(2026, 6, 21, 0, 0, tzinfo=dt.UTC)
    start = now - dt.timedelta(days=14)
    n_hours = 14 * 24

    energy: dict[int, float] = {}
    gr_rows: list[tuple[int, float]] = []
    temp_rows: list[tuple[int, float]] = []
    wind_rows: list[tuple[int, float]] = []
    total = 0.0
    for i in range(n_hours + 1):
        moment = start + dt.timedelta(hours=i)
        ts = int(moment.timestamp())
        energy[ts] = total
        if i < n_hours:
            hour = moment.hour
            gr = 150.0 if 8 <= hour <= 18 else 0.0
            total += 0.3 if 8 <= hour <= 18 else 0.0
            gr_rows.append((ts, gr))
            temp_rows.append((ts, 18.0))
            wind_rows.append((ts, 3.0))
    helper.add_energy("sensor.test_pv_energy", "kWh", energy)
    put_value(da_db, "gr", gr_rows)
    put_value(da_db, "temp", temp_rows)
    put_value(da_db, "winds", wind_rows)

    config = make_config(solar=[make_installation()])
    return PVService(
        config, da_db, manager, LAT, LON, tmp_path / "pv", TZ, "1hour",
        now=lambda: now,
    )


class _StubPredictor:
    """Stands in for a trained SolarPredictor: a fixed fraction of GHI."""

    prepared: list = []

    def __init__(self):
        self.installation = None

    def prepare_for(self, installation):
        self.installation = installation
        _StubPredictor.prepared.append(installation.name)

    def predict(self, weather):
        import pandas as pd

        return pd.DataFrame(
            {
                "date_time": weather.index,
                "prediction": weather["ghi"].to_numpy() * 0.002,
            }
        )


def test_auto_backtest_scores_two_different_models(
    pv_service_for_backtest, monkeypatch
):
    """The ml candidate used to be wired to forecast_from_weather, which is
    simulate() -- the physical model. The two scores then came out of one
    computation, min() always returned physical, and selection.json showed
    a comparison that never happened."""
    monkeypatch.setattr(
        "dao.prog.solar_predictor.SolarPredictor", _StubPredictor, raising=False
    )
    _StubPredictor.prepared = []
    service = pv_service_for_backtest
    installation = service.installations()[0]

    result = service._backtest_installation(installation, days=7)

    assert result is not None
    assert set(result.scores) == {"physical", "ml"}
    assert result.scores["ml"].n > 0
    assert result.scores["physical"].mae != result.scores["ml"].mae
    assert _StubPredictor.prepared == [installation.name]


def test_auto_backtest_scores_physical_alone_without_a_trained_model(
    pv_service_for_backtest, monkeypatch, caplog
):
    """No trained model means no ml score, not a second copy of the physical
    one, and the stored reason has to say which it was."""

    class _NoModel:
        def prepare_for(self, installation):
            raise FileNotFoundError("geen model")

    monkeypatch.setattr(
        "dao.prog.solar_predictor.SolarPredictor", _NoModel, raising=False
    )
    service = pv_service_for_backtest
    installation = service.installations()[0]

    with caplog.at_level("INFO"):
        result = service._backtest_installation(installation, days=7)

    assert set(result.scores) == {"physical"}
    assert result.winner == "physical"

    from dao.forecast.pv.select import select_pv_model

    selection = select_pv_model("auto", result)
    assert "geen getraind ML-model" in selection.reason


def test_calibration_weather_falls_back_to_archive_when_values_empty(da_db, caplog, tmp_path):
    """values has nothing for gr; the forecast archive at lead bucket 0/1 does."""
    now = dt.datetime(2026, 6, 21, tzinfo=dt.UTC)
    start = now - dt.timedelta(hours=3)
    config = make_config(solar=[make_installation()])
    service = PVService(
        config, da_db, None, LAT, LON, tmp_path / "pv", TZ, "1hour",
        now=lambda: now,
    )

    from sqlalchemy import Table

    forecasts = Table("forecasts", da_db.metadata, autoload_with=da_db.engine)
    with da_db.engine.begin() as connection:
        connection.execute(
            forecasts.insert(),
            [
                {
                    "variabel": 4,
                    "target_time": int((start + dt.timedelta(hours=i)).timestamp()),
                    "lead_bucket": 0,
                    "issued_time": int(start.timestamp()),
                    "value": 100.0,
                    "source": "meteoserver",
                }
                for i in range(3)
            ],
        )

    with caplog.at_level("WARNING"):
        weather = service._calibration_weather(start, now)

    assert not weather.empty
    assert any("prognose-archief" in message for message in caplog.messages)


@pytest.mark.slow
def test_calibrate_installation_uses_values_then_archive(pv_service_with_history, caplog):
    installation = pv_service_with_history.installations()[0]
    with caplog.at_level("WARNING"):
        pv_service_with_history.calibrate_installation(installation)

    assert not any("geen straling" in message for message in caplog.messages)


@pytest.mark.slow
def test_run_training_logs_per_installation(pv_service_with_history, caplog):
    with caplog.at_level("INFO"):
        pv_service_with_history.run_training()

    assert any("PV-kalibratie" in message for message in caplog.messages)


def test_one_failing_calibration_does_not_abort_the_whole_run(
    pv_service_with_prognoses, caplog
):
    """Calibration can raise for reasons the operator cannot see coming: a
    flat roof puts the initial guess outside least_squares' bounds, an
    all-NaN weather window makes the residuals non-finite. One installation
    going down must not take the rest of the task with it."""
    service, _start = pv_service_with_prognoses
    service.config.solar = [
        make_installation(name="Bad Roof"),
        make_installation(name="Good Roof"),
    ]

    def explode(installation):
        if installation.name == "Bad Roof":
            raise ValueError("Residuals are not finite in the initial point")
        return None

    service.calibrate_installation = explode

    with caplog.at_level("WARNING"):
        service.run_training()

    from dao.forecast.pv.store import selection_path

    assert selection_path(service.data_dir, "Good Roof").exists()
    assert selection_path(service.data_dir, "Bad Roof").exists()
    assert any("Bad Roof" in message for message in caplog.messages)


def test_service_forecast_ml_falls_back_to_physical_when_model_missing(
    pv_service_with_prognoses, caplog
):
    """The selection says ml, but no trained model exists: the physical
    model must answer anyway, with a warning rather than an exception."""
    from dao.forecast.baseload.store import write_json
    from dao.forecast.pv.store import selection_path

    service, start = pv_service_with_prognoses
    installation = service.installations()[0]
    write_json(
        selection_path(service.data_dir, installation.name),
        {
            "model": "ml",
            "scores": {},
            "decided_at": start.isoformat(),
            "reason": "geconfigureerd",
        },
    )

    with caplog.at_level("WARNING"):
        result = service.forecast(
            installation, start, start + dt.timedelta(hours=24), "1hour"
        )

    assert len(result) == 24
    assert (result["prediction"] >= 0).all()
    assert any("fysisch model" in message for message in caplog.messages)


def test_service_forecast_model_override(pv_service_with_prognoses):
    """An explicit model= beats the stored selection, which is how the
    solar report asks for the ML column next to the physical one."""
    service, start = pv_service_with_prognoses
    installation = service.installations()[0]

    physical = service.forecast(
        installation, start, start + dt.timedelta(hours=24), "1hour", model="physical"
    )
    # "ml" has no trained model here, so it falls back to the same physical
    # numbers -- the point is that the override is honoured without raising.
    overridden = service.forecast(
        installation, start, start + dt.timedelta(hours=24), "1hour", model="ml"
    )

    assert len(physical) == 24
    assert len(overridden) == 24


def test_selection_for_defaults_to_configured_without_a_file(pv_service_with_prognoses):
    service, _start = pv_service_with_prognoses
    installation = service.installations()[0]

    selection = service.selection_for(installation)

    assert selection.model == "physical"
    assert selection.reason == "geconfigureerd"
