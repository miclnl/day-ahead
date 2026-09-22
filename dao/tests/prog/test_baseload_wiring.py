"""Tests for the reporting side of the baseload: schema and sanity checks."""

import types

import pytest

pytest.importorskip("pandas")

from sqlalchemy import Table, inspect, select  # noqa: E402

from dao.prog.check_db import CheckDB  # noqa: E402
from dao.lib.db_manager import DBmanagerObj  # noqa: E402


@pytest.fixture
def checker(tmp_path):
    """A CheckDB bound to a throwaway database, without loading a config."""
    from sqlalchemy import Column, Integer, String, Table as T

    instance = CheckDB.__new__(CheckDB)
    instance.version = "2026.9.9"
    instance.last_version = None
    instance.db_da = DBmanagerObj(
        db_dialect="sqlite", db_name="day_ahead.db", db_path=str(tmp_path)
    )
    instance.engine = instance.db_da.engine
    # The forecast archive references variabel; give it something to point at.
    T(
        "variabel",
        instance.db_da.metadata,
        Column("id", Integer, primary_key=True),
        Column("code", String(10), unique=True, nullable=False),
        Column("name", String(50), unique=True, nullable=False),
        Column("dim", String(10), nullable=False),
        Column("aggregate", String(3), nullable=False, default="avg"),
    ).create(instance.engine)
    return instance


class TestSchema:
    def test_the_forecast_table_is_created(self, checker):
        checker.ensure_forecast_table()
        names = inspect(checker.engine).get_table_names()
        assert "forecasts" in names

    def test_creating_it_twice_is_harmless(self, checker):
        checker.ensure_forecast_table()
        checker.ensure_forecast_table()
        assert "forecasts" in inspect(checker.engine).get_table_names()

    def test_it_has_the_uniqueness_that_bounds_its_size(self, checker):
        checker.ensure_forecast_table()
        constraints = inspect(checker.engine).get_unique_constraints("forecasts")
        assert any(
            set(c["column_names"]) == {"variabel", "target_time", "lead_bucket"}
            for c in constraints
        )

    def test_the_uniqueness_is_actually_enforced(self, checker):
        """Belt and braces: the constraint is what caps the table size."""
        from sqlalchemy.exc import IntegrityError

        checker.ensure_forecast_table()
        table = Table("forecasts", checker.db_da.metadata, autoload_with=checker.engine)
        row = {
            "variabel": 1,
            "target_time": 1000,
            "lead_bucket": 4,
            "issued_time": 1,
            "value": 1.0,
        }
        with checker.engine.begin() as connection:
            connection.execute(table.insert().values(**row))
        with pytest.raises(IntegrityError):
            with checker.engine.begin() as connection:
                connection.execute(table.insert().values(**row))

    def test_the_target_index_exists(self, checker):
        checker.ensure_forecast_table()
        indexes = inspect(checker.engine).get_indexes("forecasts")
        assert any(i["column_names"] == ["target_time"] for i in indexes)


class TestAggregateColumn:
    def make_variabel(self, checker, with_aggregate: bool):
        from sqlalchemy import Column, Integer, MetaData, String, Table as T

        # Start from a clean slate: the fixture already made a table with the
        # aggregate column, and this class needs to test both shapes.
        checker.db_da.metadata.drop_all(checker.engine)
        metadata = MetaData()
        columns = [
            Column("id", Integer, primary_key=True),
            Column("code", String(10), unique=True, nullable=False),
            Column("name", String(50), unique=True, nullable=False),
            Column("dim", String(10), nullable=False),
        ]
        if with_aggregate:
            columns.append(
                Column("aggregate", String(3), nullable=False, default="avg")
            )
        table = T("variabel", metadata, *columns)
        metadata.create_all(checker.engine)
        checker.db_da.metadata = metadata
        return table

    def test_kwh_variables_get_the_sum_aggregate(self, checker):
        table = self.make_variabel(checker, with_aggregate=True)
        checker.upsert_variabel(table, [25, "m_house", "Gemeten huisvraag", "kWh"])
        with checker.engine.connect() as connection:
            row = connection.execute(
                select(table.c.code, table.c.aggregate)
            ).first()
        assert row.aggregate == "sum"

    def test_other_variables_get_avg(self, checker):
        table = self.make_variabel(checker, with_aggregate=True)
        checker.upsert_variabel(table, [99, "soctest", "SoC test", "%"])
        with checker.engine.connect() as connection:
            row = connection.execute(select(table.c.aggregate)).first()
        assert row.aggregate == "avg"

    def test_an_explicit_aggregate_wins(self, checker):
        table = self.make_variabel(checker, with_aggregate=True)
        checker.upsert_variabel(table, [98, "x", "X", "kWh", "avg"])
        with checker.engine.connect() as connection:
            assert connection.execute(select(table.c.aggregate)).scalar() == "avg"

    def test_an_old_database_without_the_column_still_works(self, checker):
        """Existing installations get the column added later in the same run."""
        table = self.make_variabel(checker, with_aggregate=False)
        checker.upsert_variabel(table, [25, "m_house", "Gemeten huisvraag", "kWh"])
        with checker.engine.connect() as connection:
            assert connection.execute(select(table.c.code)).scalar() == "m_house"

    def test_upserting_twice_updates_rather_than_duplicates(self, checker):
        table = self.make_variabel(checker, with_aggregate=True)
        checker.upsert_variabel(table, [25, "m_house", "Oud", "kWh"])
        checker.upsert_variabel(table, [25, "m_house", "Nieuw", "kWh"])
        with checker.engine.connect() as connection:
            rows = connection.execute(select(table.c.name)).all()
        assert len(rows) == 1 and rows[0].name == "Nieuw"

    def test_the_default_aggregate_rule(self):
        assert CheckDB.default_aggregate("kWh") == "sum"
        assert CheckDB.default_aggregate("euro") == "sum"
        assert CheckDB.default_aggregate("mm") == "sum"
        assert CheckDB.default_aggregate("%") == "avg"
        assert CheckDB.default_aggregate("°C") == "avg"


class TestDeEmbeddingCheck:
    """A modelled device without a meter is double counted, silently.

    The baseload is the measured total minus everything the optimizer
    schedules itself. If the heat pump is modelled but not metered, its
    consumption stays inside the baseload and the optimizer adds it again on
    top, every hour of every day, and nothing in the output shows it.
    """

    def report(self, **overrides):
        from dao.prog.da_report import Report

        instance = Report.__new__(Report)
        config = types.SimpleNamespace(
            boiler=types.SimpleNamespace(boiler_present=False),
            heating=types.SimpleNamespace(heater_present=False),
            electric_vehicle=[],
            machines=[],
            battery=[],
            solar=[],
        )
        instance.config = config
        instance.boiler_consumption_sensors = []
        instance.wp_consumption_sensors = []
        instance.ev_consumption_sensors = []
        instance.machine_consumption_sensors = []
        instance.battery_consumption_sensors = []
        instance.solar_production_ac_sensors = []
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(instance, key, value)
        return instance

    def test_a_consistent_setup_reports_nothing(self):
        report = self.report(
            heating=types.SimpleNamespace(heater_present=True),
            wp_consumption_sensors=["sensor.wp"],
        )
        assert report.check_baseload_sensors() == []

    def test_a_modelled_device_without_a_meter_is_flagged(self):
        report = self.report(heating=types.SimpleNamespace(heater_present=True))
        problems = report.check_baseload_sensors()
        assert len(problems) == 1
        assert "dubbel geteld" in problems[0]
        assert "warmtepomp" in problems[0]

    def test_a_meter_without_a_modelled_device_is_flagged(self):
        report = self.report(ev_consumption_sensors=["sensor.ev"])
        problems = report.check_baseload_sensors()
        assert len(problems) == 1
        assert "nergens weer opgeteld" in problems[0]

    def test_every_device_class_is_covered(self):
        report = self.report(
            boiler=types.SimpleNamespace(boiler_present=True),
            heating=types.SimpleNamespace(heater_present=True),
            electric_vehicle=[object()],
            machines=[object()],
            battery=[object()],
            solar=[object()],
        )
        problems = report.check_baseload_sensors()
        assert len(problems) == 6

    def test_an_empty_installation_is_consistent(self):
        assert self.report().check_baseload_sensors() == []
