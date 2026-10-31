"""Recorder bucket ends, explicit gaps and independent meter/price resolutions."""

import csv
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import Column, Float, Integer, String, Table, event, insert, select

from dao.lib.db_manager import DBmanagerObj
from dao.prog.settlement.metering import merge_inputs, read_ha_energy, read_meter_csv
from dao.prog.settlement.tariffs import quote
from dao.prog.settlement.types import MeterSelection, SpotInterval

ZONE = ZoneInfo("Europe/Amsterdam")
START = int(datetime(2027, 1, 1, tzinfo=ZONE).timestamp())
CSV_FIELDS = [
    "start_ts", "end_ts", "import_kwh", "export_kwh", "battery_discharge_kwh",
    "unit", "quality", "source", "origin",
]


@pytest.fixture
def recorder(tmp_path):
    """Real HA-shaped float statistics; duplicates intentionally remain possible."""
    db = DBmanagerObj("sqlite", "recorder.db", db_path=str(tmp_path))
    Table(
        "statistics_meta", db.metadata,
        Column("id", Integer, primary_key=True),
        Column("statistic_id", String),
        Column("unit_of_measurement", String),
        Column("has_sum", Integer),
    )
    for name in ("statistics", "statistics_short_term"):
        Table(
            name, db.metadata, Column("id", Integer, primary_key=True),
            Column("metadata_id", Integer), Column("start_ts", Float),
            Column("state", Float), Column("sum", Float),
            Column("last_reset_ts", Float),
        )
    db.metadata.create_all(db.engine)
    yield db
    db.engine.dispose()


def put_counter(db, sensor, points, *, unit="kWh", table="statistics_short_term", states=None):
    """Insert literal (bucket start, cumulative sum) recorder observations."""
    meta = db.metadata.tables["statistics_meta"]
    stats = db.metadata.tables[table]
    with db.engine.begin() as connection:
        meta_id = connection.execute(select(meta.c.id).where(meta.c.statistic_id == sensor)).scalar()
        if meta_id is None:
            meta_id = connection.execute(insert(meta).values(
                statistic_id=sensor, unit_of_measurement=unit, has_sum=1,
            )).inserted_primary_key[0]
        connection.execute(insert(stats), [
            {"metadata_id": meta_id, "start_ts": ts,
             "sum": None if total is None else float(total),
             "state": None if (states or {}).get(ts, total) is None else float((states or {}).get(ts, total))}
            for ts, total in points
        ])


def short_points(start, end, *, increment="0.1", initial="10"):
    """A bucket starting t ends at t+300, not at t (never exact boundary proof)."""
    return [(ts - 300, Decimal(initial) + i * Decimal(increment))
            for i, ts in enumerate(range(start, end + 1, 300))]


def selection(**overrides):
    return MeterSelection(**{
        "meter_id": "synthetic-grid", "import_sensors": ("sensor.import",),
        "export_sensors": ("sensor.export",), "battery_sources": (),
        "no_battery_evidence": "synthetic-installation-record", **overrides,
    })


def put_grid(db, start=START, end=START + 900, *, table="statistics_short_term"):
    if table == "statistics":
        points = [(start - 3600, "10"), (end - 3600, "12")]
    else:
        points = short_points(start, end)
    for sensor in ("sensor.import", "sensor.export"):
        put_counter(db, sensor, points, table=table)


def spot(start=START, end=START + 3600, price="0.10", source="synthetic-market", quality="measured"):
    return SpotInterval(start, end, None if price is None else Decimal(price), source, quality)


def write_csv(tmp_path, rows, *, fields=CSV_FIELDS):
    path = tmp_path / "normalized.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: value for key, value in {
                    "start_ts": START, "end_ts": START + 900,
                    "import_kwh": "0", "export_kwh": "1.2",
                    "battery_discharge_kwh": "0", "unit": "kWh",
                    "quality": "measured", "source": "synthetic-meter-file",
                    "origin": "verified_pv", **row,
                }.items() if key in fields
            })
    return path


def test_92_and_100_dst_quarters(recorder, snapshot, bonus_rules):
    """UTC quarter and quote alignment survives both real Amsterdam DST days."""
    days = []
    for month, day, next_day, count, bonus_hour in ((3, 28, 29, 92, 7), (10, 31, 1, 100, 8)):
        start = int(datetime(2027, month, day, tzinfo=ZONE).timestamp())
        end = int(datetime(2027, month if month == 3 else 11, next_day, tzinfo=ZONE).timestamp())
        put_grid(recorder, start, end)
        readings = read_ha_energy(recorder, selection(), start, end, 900)
        prices = [spot(ts, ts + 3600) for ts in range(start, end, 3600)]
        if month == 10:
            first = int(datetime(2027, 10, 31, 2, tzinfo=ZONE, fold=0).timestamp())
            second = int(datetime(2027, 10, 31, 2, tzinfo=ZONE, fold=1).timestamp())
            assert second - first == 3600
            assert datetime.fromtimestamp(first, ZoneInfo("UTC")).isoformat() == "2027-10-31T00:00:00+00:00"
            assert datetime.fromtimestamp(second, ZoneInfo("UTC")).isoformat() == "2027-10-31T01:00:00+00:00"
            prices[2] = spot(first, first + 3600, "0.20")
            prices[3] = spot(second, second + 3600, "0.40")
        snap = snapshot(bonus=bonus_rules)
        merged = merge_inputs(readings, prices, [snap], meter_id="synthetic-grid")
        quotes = [quote(snap, row.start_ts, row.end_ts, row.spot_ex_vat) for row in merged]
        assert len(readings) == len(merged) == len(quotes) == count
        assert all(row.end_ts - row.start_ts == 900 for row in readings)
        assert all(row.export_kwh == Decimal("0.3") for row in readings)
        assert all(row.quality == "estimated" for row in readings)
        assert quotes[0].export_rate == quotes[-1].export_rate == Decimal("0.10")
        at_window = int(datetime(2027, month, day, bonus_hour, tzinfo=ZONE).timestamp())
        by_start = {row.start_ts: row for row in quotes}
        assert by_start[at_window - 900].bonus_rate == Decimal("0")
        assert by_start[at_window].bonus_rate == Decimal("0.03")
        if month == 10:
            assert [q.export_rate for q in quotes[8:16]] == [Decimal("0.20")] * 4 + [Decimal("0.40")] * 4
        days.append(readings)
    spring_day, autumn_day = days
    assert len(spring_day) == 92 and len(autumn_day) == 100
    assert len({r.start_ts for r in autumn_day}) == 100


def test_hour_energy_not_fabricated_into_quarters(recorder, snapshot):
    """Retained hourly statistics yield one base estimate, not four fake meters."""
    put_grid(recorder, START, START + 3600, table="statistics")
    readings = read_ha_energy(recorder, selection(), START, START + 3600, 900)
    assert len(readings) == 1
    hour_only = readings[0]
    assert (hour_only.start_ts, hour_only.end_ts) == (START, START + 3600)
    assert hour_only.import_kwh == hour_only.export_kwh == Decimal("2")
    assert hour_only.quality == "estimated"
    assert any("3600" in source and "statistics" in source for source in hour_only.sources)
    native = merge_inputs(readings, [spot()], [snapshot()], meter_id="grid")[0]
    assert native.spot_ex_vat == Decimal("0.10")
    assert native.quality == "estimated"
    quarters = [spot(START + i * 900, START + (i + 1) * 900, price)
                for i, price in enumerate(("0.10", "0.40", "0.20", "0.30"))]
    unknown = merge_inputs(readings, quarters, [snapshot(price_interval_seconds=900)], meter_id="grid")[0]
    assert unknown.export_kwh == Decimal("2")
    assert unknown.spot_ex_vat is None and unknown.quality == "incomplete"


def test_missing_boundary_not_zero(recorder):
    put_counter(recorder, "sensor.import", short_points(START, START + 900))
    put_counter(recorder, "sensor.export", short_points(START, START + 600))
    missing = read_ha_energy(recorder, selection(), START, START + 900, 900)[0]
    assert missing.import_kwh == Decimal("0.3")
    assert missing.export_kwh is None
    assert missing.quality == "incomplete"


def test_bucket_end_semantics_and_internal_gap_are_not_endpoint_interpolation(recorder):
    put_counter(recorder, "sensor.import", [
        (START - 300, "10"), (START, "10.1"), (START + 300, "10.2"),
        (START + 600, "10.6"), (START + 900, "99"),
    ])
    put_counter(recorder, "sensor.export", [
        (START - 300, "20"), (START, "20.1"), (START + 600, "20.9"),
    ])
    row = read_ha_energy(recorder, selection(), START, START + 900, 900)[0]
    assert row.import_kwh == Decimal("0.6")  # end-bucket 10.6 minus preceding-bucket 10
    assert row.export_kwh is None  # endpoints alone do not conceal a missing bucket


def test_counter_reset_and_unit_conversion(recorder):
    put_counter(recorder, "sensor.import", [
        (START - 300, "1000"), (START, "1100"), (START + 300, "1200"), (START + 600, "1300"),
    ], unit="Wh")
    put_counter(recorder, "sensor.export", short_points(START, START + 900), states={START + 300: "0"})
    put_counter(recorder, "sensor.discharge", short_points(START, START + 900, increment="0.0001"), unit="MWh")
    row = read_ha_energy(recorder, selection(battery_sources=("sensor.discharge",), no_battery_evidence=None),
                         START, START + 900, 900)[0]
    assert row.import_kwh == Decimal("0.3")
    assert row.export_kwh is None  # reset-corrected sum must not hide a raw counter reset
    assert row.battery_discharge_kwh == Decimal("0.3")  # gross; never netted against grid import
    assert row.origin == "unknown" and row.quality == "incomplete"
    assert any("Wh" in source for source in row.sources)
    assert any("MWh" in source for source in row.sources)


@pytest.mark.parametrize("points", [
    [(START - 300, "10"), (START, "10.1"), (START + 300, "9"), (START + 600, "10.3")],
    [(START - 300, "10"), (START, None), (START + 300, "10.2"), (START + 600, "10.3")],
    [(START - 300, "10"), (START, "10.1"), (START, "10.9"), (START + 300, "10.2"), (START + 600, "10.3")],
])
def test_negative_missing_and_conflicting_statistics_stay_unknown(recorder, points):
    put_counter(recorder, "sensor.import", points)
    put_counter(recorder, "sensor.export", short_points(START, START + 900))
    row = read_ha_energy(recorder, selection(), START, START + 900, 900)[0]
    assert row.import_kwh is None and row.export_kwh == Decimal("0.3")
    assert row.quality == "incomplete"


def test_reset_inside_bucket_cannot_hide_behind_monotone_corrected_sum(recorder):
    points = [(START - 300, "10"), (START, "10.1"), (START + 300, "11.2"), (START + 600, "11.3")]
    states = {START + 300: "10.2", START + 600: "10.3"}
    put_counter(recorder, "sensor.import", points, states=states)
    put_counter(recorder, "sensor.export", short_points(START, START + 900))
    row = read_ha_energy(recorder, selection(), START, START + 900, 900)[0]
    assert row.import_kwh is None  # hidden reset/correction: sum delta 1.1, state delta 0.1
    assert row.export_kwh == Decimal("0.3") and row.quality == "incomplete"


def test_missing_battery_sensors_are_not_no_battery_evidence(recorder):
    put_grid(recorder)
    row = read_ha_energy(recorder, selection(no_battery_evidence=None), START, START + 900, 900)[0]
    assert row.battery_discharge_kwh is None
    assert row.origin == "unknown" and row.quality == "incomplete"
    proven = read_ha_energy(recorder, selection(), START, START + 900, 900)[0]
    assert proven.battery_discharge_kwh == Decimal("0")
    assert proven.origin == "unknown"  # no-battery proof is not proof that all export is PV


def test_sources_overlap_conflict(tmp_path, meter, snapshot):
    path = write_csv(tmp_path, [{}, {"start_ts": START + 600, "end_ts": START + 1500}])
    with pytest.raises(ValueError, match="overlap"):
        read_meter_csv(path, "grid")
    with pytest.raises(ValueError, match="overlap"):
        merge_inputs([meter(), meter()], [spot()], [snapshot()], meter_id="grid")
    with pytest.raises(ValueError, match="overlap"):
        merge_inputs([meter()], [spot(), spot(price="0.20", source="other")], [snapshot()], meter_id="grid")


def test_hour_price_rule_is_time_not_energy_weighted(meter, snapshot):
    readings = [meter(start_ts=START + i * 900, end_ts=START + (i + 1) * 900, export_kwh=energy)
                for i, energy in enumerate(("9", "1", "0", "0"))]
    prices = [spot(START + i * 900, START + (i + 1) * 900, price)
              for i, price in enumerate(("0.10", "0.40", "0.20", "0.30"))]
    result = merge_inputs(readings, prices, [snapshot(hour_price_rule="mean_quarters")], meter_id="grid")
    mean_of_four_spots = result[0].spot_ex_vat
    assert mean_of_four_spots == Decimal("0.25")  # not the export-weighted 0.13
    assert [row.spot_ex_vat for row in result] == [Decimal("0.25")] * 4
    assert all("mean_quarters" in row.spot_source for row in result)
    assert [row.export_kwh for row in result] == [Decimal("9"), Decimal("1"), Decimal("0"), Decimal("0")]


def test_merge_keeps_explicit_meter_identity_and_unregistered_revision(meter, snapshot):
    reading = meter(sources=("not-the-meter-id", "identical-proof"))
    first = merge_inputs([reading], [spot()], [snapshot()], meter_id="meter-one")[0]
    second = merge_inputs([reading], [spot()], [snapshot()], meter_id="meter-two")[0]
    assert first.meter_id == "meter-one" and second.meter_id == "meter-two"
    assert first.input_revision == second.input_revision == 0
    assert first.sources == second.sources == reading.sources
    assert first.input_hash != second.input_hash
    with pytest.raises(TypeError):
        merge_inputs([reading], [spot()], [snapshot()])


def test_csv_preserves_exact_zero_unknown_gross_discharge_and_source(tmp_path):
    path = write_csv(tmp_path, [{"import_kwh": "0.000000001", "export_kwh": "",
                                "battery_discharge_kwh": "0.3", "origin": "unknown"}])
    row = read_meter_csv(path, "grid")[0]
    assert row.import_kwh == Decimal("0.000000001")
    assert row.export_kwh is None and row.quality == "incomplete"
    assert row.battery_discharge_kwh == Decimal("0.3")
    assert any("synthetic-meter-file" in source for source in row.sources)
    assert any("kWh" in source and "900" in source for source in row.sources)
    zero = read_meter_csv(write_csv(tmp_path, [{"export_kwh": "0"}]), "grid")[0]
    assert zero.import_kwh == zero.export_kwh == zero.battery_discharge_kwh == Decimal("0")
    assert zero.quality == "measured" and zero.origin == "verified_pv"


def test_csv_optional_discharge_does_not_imply_no_battery(tmp_path):
    fields = [field for field in CSV_FIELDS if field not in ("battery_discharge_kwh", "origin")]
    row = read_meter_csv(write_csv(tmp_path, [{}], fields=fields), "grid")[0]
    assert row.battery_discharge_kwh is None
    assert row.origin == "unknown" and row.quality == "incomplete"


@pytest.mark.parametrize("change", [
    {"unit": "Wh"}, {"unit": "kW"}, {"unit": ""},
    {"start_ts": "2027-01-01T00:00:00"}, {"start_ts": "1.5"},
    {"end_ts": START}, {"export_kwh": "-1"}, {"import_kwh": "NaN"},
    {"export_kwh": "Infinity"}, {"export_kwh": "0.1234567891"},
    {"quality": "confirmed"}, {"source": ""}, {"origin": "assumed_pv"},
])
def test_csv_rejects_unsupported_units_bad_boundaries_and_untrustworthy_values(tmp_path, change):
    with pytest.raises((ValueError, TypeError)):
        read_meter_csv(write_csv(tmp_path, [change]), "grid")


def test_csv_rejects_unexplained_source_switch_and_reverse_order(tmp_path):
    for second in ({"start_ts": START + 900, "end_ts": START + 1800, "source": "other-meter"},
                   {"start_ts": START - 900, "end_ts": START}):
        with pytest.raises(ValueError):
            read_meter_csv(write_csv(tmp_path, [{}, second]), "grid")


def test_merge_inserts_explicit_internal_gap_not_zero(meter, snapshot):
    readings = [meter(), meter(start_ts=START + 1800, end_ts=START + 2700)]
    result = merge_inputs(readings, [spot()], [snapshot()], meter_id="grid")
    assert len(result) == 3
    gap = result[1]
    assert (gap.start_ts, gap.end_ts) == (START + 900, START + 1800)
    assert gap.import_kwh is gap.export_kwh is gap.battery_discharge_kwh is None
    assert gap.quality == "incomplete" and gap.origin == "unknown"
    assert gap.spot_ex_vat == Decimal("0.10")


@pytest.mark.parametrize("prices,rule", [
    ([], "native"), ([spot(price=None)], "native"),
    ([spot(START, START + 900)], "native"),
    ([spot(START + 300, START + 3900)], "native"),
    ([spot()], "unconfirmed"),
    ([spot(START + i * 900, START + (i + 1) * 900) for i in (0, 1, 3)], "mean_quarters"),
])
def test_missing_or_wrong_price_resolution_stays_unknown(meter, snapshot, prices, rule):
    row = merge_inputs([meter()], prices, [snapshot(hour_price_rule=rule)], meter_id="grid")[0]
    assert row.export_kwh == Decimal("2")
    assert row.spot_ex_vat is None and row.quality == "incomplete"


def test_quarter_prices_and_source_quality_are_not_optimizer_derived(meter, snapshot):
    row = merge_inputs([meter()], [spot(START, START + 900, "-0.20", quality="estimated")],
                       [snapshot(price_interval_seconds=900)], meter_id="grid")[0]
    assert row.spot_ex_vat == Decimal("-0.20") and row.quality == "estimated"
    missing = merge_inputs([meter()], [spot(quality="incomplete")], [snapshot()], meter_id="grid")[0]
    assert missing.spot_ex_vat is None and missing.quality == "incomplete"


def test_unknown_price_keeps_known_source_and_correction_identity(meter, snapshot):
    first = merge_inputs([meter()], [spot(price=None, source="market-revision-one")],
                         [snapshot()], meter_id="grid")[0]
    second = merge_inputs([meter()], [spot(price=None, source="market-revision-two")],
                          [snapshot()], meter_id="grid")[0]
    assert first.spot_ex_vat is second.spot_ex_vat is None
    assert first.spot_source is not None and "market-revision-one" in first.spot_source
    assert second.spot_source is not None and "market-revision-two" in second.spot_source
    assert first.input_hash != second.input_hash
    assert first.quality == second.quality == "incomplete"


def test_merge_hash_canonicalizes_decimals_and_tracks_corrections(meter, snapshot):
    first = merge_inputs([meter(export_kwh="2.00")], [spot(price="0.100")], [snapshot()], meter_id="grid")[0]
    same = merge_inputs([meter(export_kwh="2")], [spot(price="0.1")], [snapshot()], meter_id="grid")[0]
    assert len(first.input_hash) == 64 and first.input_hash == same.input_hash
    for reading, price in ((meter(export_kwh="3"), spot()),
                           (meter(quality="estimated"), spot()),
                           (meter(sources=("corrected-proof",)), spot()),
                           (meter(), spot(price="0.2"))):
        revised = merge_inputs([reading], [price], [snapshot()], meter_id="grid")[0]
        assert revised.input_hash != first.input_hash


def test_year_boundary_quarters_select_dated_quotes_without_splitting_energy(recorder, snapshot):
    start, end = START - 3600, START + 3600
    put_grid(recorder, start, end)
    december = snapshot(
        valid_from_ts=start, valid_to_ts=START, product="legacy", regime="legacy_interval",
        energy_taxes_consumption=((datetime(2026, 1, 1).date(), Decimal("0.1")),),
        energy_taxes_production=((datetime(2026, 1, 1).date(), Decimal("0.1")),),
        cost_supplier_consumption=((datetime(2026, 1, 1).date(), Decimal("0")),),
        cost_supplier_production=((datetime(2026, 1, 1).date(), Decimal("0")),),
        vat_consumption=((datetime(2026, 1, 1).date(), Decimal("0")),),
        vat_production=((datetime(2026, 1, 1).date(), Decimal("0")),),
        multiplier_consumption=((datetime(2026, 1, 1).date(), Decimal("1")),),
        multiplier_production=((datetime(2026, 1, 1).date(), Decimal("1")),),
    )
    january = snapshot()
    readings = read_ha_energy(recorder, selection(), start, end, 900)
    result = merge_inputs(readings, [spot(start, START), spot()], [december, january], meter_id="grid")
    assert len(result) == 8
    assert [quote(december if row.start_ts < START else january, row.start_ts, row.end_ts,
                  row.spot_ex_vat).export_rate for row in result] == [Decimal("0.20")] * 4 + [Decimal("0.10")] * 4
    coarse = replace(readings[0], start_ts=START - 1800, end_ts=START + 1800, export_kwh=Decimal("2"))
    unresolved = merge_inputs([coarse], [spot(start, START), spot()], [december, january], meter_id="grid")
    assert len(unresolved) == 1 and unresolved[0].export_kwh == Decimal("2")
    assert unresolved[0].spot_ex_vat is None and unresolved[0].quality == "incomplete"


def test_two_sensors_and_two_meters_do_not_mix_or_omit_missing_members(recorder):
    put_grid(recorder)
    put_counter(recorder, "sensor.second_import", short_points(START, START + 900, increment="0.2"))
    both = read_ha_energy(recorder, selection(import_sensors=("sensor.import", "sensor.second_import")),
                          START, START + 900, 900)[0]
    assert both.import_kwh == Decimal("0.9") and both.export_kwh == Decimal("0.3")
    second = read_ha_energy(recorder, selection(meter_id="other", import_sensors=("sensor.second_import",)),
                            START, START + 900, 900)[0]
    assert second.import_kwh == Decimal("0.6")
    absent = read_ha_energy(recorder, selection(import_sensors=("sensor.import", "sensor.purged")),
                            START, START + 900, 900)[0]
    assert absent.import_kwh is None and absent.quality == "incomplete"


def test_old_retention_gap_and_recent_short_term_keep_their_actual_resolutions(recorder):
    put_grid(recorder, START, START + 3600, table="statistics")
    put_grid(recorder, START + 3600, START + 7200)
    rows = read_ha_energy(recorder, selection(), START, START + 8100, 900)
    assert [row.end_ts - row.start_ts for row in rows] == [3600, 900, 900, 900, 900, 900]
    assert rows[0].export_kwh == Decimal("2") and rows[1].export_kwh == Decimal("0.3")
    assert rows[-1].export_kwh is None and rows[-1].quality == "incomplete"


def test_partial_short_term_does_not_fall_back_over_a_real_gap(recorder):
    put_grid(recorder, START, START + 3600, table="statistics")
    put_grid(recorder)
    rows = read_ha_energy(recorder, selection(), START, START + 3600, 900)
    assert len(rows) == 4
    assert rows[0].export_kwh == Decimal("0.3")
    assert rows[1].export_kwh is None and rows[-1].export_kwh is None


def test_zero_statistics_are_known_and_hour_request_sums_fine_buckets(recorder):
    for sensor in ("sensor.import", "sensor.export"):
        put_counter(recorder, sensor, short_points(START, START + 3600, increment="0"))
    rows = read_ha_energy(recorder, selection(), START, START + 3600, 3600)
    assert len(rows) == 1
    assert rows[0].import_kwh == rows[0].export_kwh == Decimal("0")
    assert rows[0].quality == "estimated"


def test_sensor_names_are_bound_and_reader_is_read_only(recorder):
    hostile = "sensor.grid' OR 1=1; DROP TABLE statistics; --"
    put_counter(recorder, hostile, short_points(START, START + 900))
    put_counter(recorder, "sensor.export", short_points(START, START + 900))
    queries = []

    def observe(connection, cursor, statement, parameters, context, executemany):
        queries.append((statement, parameters))

    event.listen(recorder.engine, "before_cursor_execute", observe)
    try:
        row = read_ha_energy(recorder, selection(import_sensors=(hostile,)), START, START + 900, 900)[0]
    finally:
        event.remove(recorder.engine, "before_cursor_execute", observe)
    assert row.import_kwh == Decimal("0.3")
    assert all(hostile not in statement for statement, _ in queries)
    assert any(hostile in str(parameters) for _, parameters in queries)
    assert all(not statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"))
               for statement, _ in queries)


@pytest.mark.parametrize("unit", ["W", "kW", "unknown"])
def test_recorder_rejects_non_energy_units(recorder, unit):
    put_counter(recorder, "sensor.import", short_points(START, START + 900), unit=unit)
    with pytest.raises(ValueError, match="unit"):
        read_ha_energy(recorder, selection(), START, START + 900, 900)


@pytest.mark.parametrize("start,end,resolution", [
    (START, START, 900), (START, START - 1, 900), (float(START), START + 900, 900),
    (START, START + 900, 60), (START + 1, START + 901, 900),
])
def test_reader_rejects_bad_boundaries_instead_of_rounding_them(recorder, start, end, resolution):
    with pytest.raises((TypeError, ValueError)):
        read_ha_energy(recorder, selection(), start, end, resolution)


def test_selection_overlap_is_not_double_counted(recorder):
    for selected in (selection(import_sensors=("sensor.import", "sensor.import")),
                     selection(export_sensors=("sensor.import",))):
        with pytest.raises(ValueError):
            read_ha_energy(recorder, selected, START, START + 900, 900)


def test_empty_sources_and_absent_recorder_tables_stay_unknown(recorder):
    recorder.metadata.tables["statistics_short_term"].drop(recorder.engine)
    recorder.metadata.tables["statistics"].drop(recorder.engine)
    row = read_ha_energy(recorder, selection(import_sensors=(), export_sensors=()),
                         START, START + 900, 900)[0]
    assert row.import_kwh is row.export_kwh is None
    assert row.quality == "incomplete"


def test_bounded_queries_handle_multiple_days_and_sensor_batches(recorder):
    sensors = tuple(f"sensor.import_{i}" for i in range(105))
    for sensor in sensors:
        put_counter(recorder, sensor, short_points(START, START + 900))
    put_counter(recorder, "sensor.export", short_points(START, START + 900))
    queries = []

    def observe(connection, cursor, statement, parameters, context, executemany):
        if context.compiled is not None:
            queries.extend(dict(parameters) for parameters in context.compiled_parameters)

    event.listen(recorder.engine, "before_cursor_execute", observe)
    try:
        rows = read_ha_energy(recorder, selection(import_sensors=sensors), START, START + 2 * 86400, 900)
    finally:
        event.remove(recorder.engine, "before_cursor_execute", observe)
    assert len(rows) == 192
    assert rows[0].import_kwh == Decimal("31.5")
    assert rows[-1].import_kwh is None
    time_queries = [query for query in queries if "start_ts_1" in query]
    assert len(time_queries) >= 4
    assert all(0 < query["start_ts_2"] - query["start_ts_1"] <= 86400 for query in time_queries)
    assert all(len(query) <= 102 for query in time_queries)  # <=100 sensors plus two epoch bounds


def test_overprecise_mean_is_rejected_not_silently_rounded(meter, snapshot):
    prices = [spot(START + i * 900, START + (i + 1) * 900, price)
              for i, price in enumerate(("0.000000000001", "0", "0", "0"))]
    with pytest.raises(ValueError, match="precision|decimal"):
        merge_inputs([meter()], prices, [snapshot(hour_price_rule="mean_quarters")], meter_id="grid")


@pytest.mark.parametrize("meter_id", ["", " ", None])
def test_meter_identity_must_be_explicit_nonempty(tmp_path, meter, snapshot, meter_id):
    with pytest.raises((TypeError, ValueError)):
        read_meter_csv(write_csv(tmp_path, [{}]), meter_id)
    with pytest.raises((TypeError, ValueError)):
        merge_inputs([meter()], [spot()], [snapshot()], meter_id=meter_id)
