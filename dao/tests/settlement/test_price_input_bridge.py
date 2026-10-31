"""Closed repricing must use archived kWh and cannot bypass registered raw evidence."""

from dataclasses import replace
from decimal import Decimal

import pytest

from dao.prog.settlement.repository import ConcurrentSettlementError
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_metering import (
    put_counter,
    selection,
    short_points,
)
from dao.tests.settlement.test_metering import (
    recorder as recorder,
)
from dao.tests.settlement.test_price_repository import frozen, price
from dao.tests.settlement.test_repository import (
    METER,
    publish,
    rows,
)
from dao.tests.settlement.test_repository import (
    calculate as calculate,
)
from dao.tests.settlement.test_repository import (
    inputs as inputs,
)
from dao.tests.settlement.test_repository import (
    repo as repo,
)
from dao.tests.settlement.test_repository import (
    terms as terms,
)


@pytest.fixture(autouse=True)
def closed_clock(monkeypatch):
    monkeypatch.setattr("dao.prog.settlement.repository.time", lambda: START_TS + 86400)


def test_historical_price_bridge_rejects_fresh_revision_with_stale_composite_and_recovers_archived_energy(
    repo, terms, inputs, calculate,
):
    from dao.prog.settlement.metering import reprice_inputs
    from dao.prog.settlement.repository import PriceInputMismatch

    snap = terms()
    original = tuple(replace(item, quality="estimated") for item in inputs([snap]))
    publish(repo, original, [snap], calculate)
    archived = repo.read_inputs(METER, START_TS, START_TS + 1800)
    assert repo.register_prices(METER, [price(value="0.2")], 1) == 2
    stale = repo.read_inputs(METER, START_TS, START_TS + 1800)
    assert stale.items == archived.items and stale.input_revision == 2
    assert repo.read(METER, START_TS, START_TS + 1800).freshness == "outdated"
    before = frozen(repo)
    with pytest.raises(PriceInputMismatch):
        repo.commit(METER, calculate(stale), stale, 1, "must-not-clean-stale-price")
    assert frozen(repo) == before
    prices = repo.read_prices(METER, START_TS, START_TS + 1800, expected_input_revision=2)
    candidates = reprice_inputs(stale, prices)  # Geen recorder-/CSV-bron meer nodig.
    assert [item.import_kwh for item in candidates] == [Decimal("1"), Decimal("1")]
    assert all(item.quality == "estimated" and item.input_revision == 0 for item in candidates)
    assert all(item.sources == original[i].sources and item.origin == original[i].origin
               and item.battery_discharge_kwh == original[i].battery_discharge_kwh
               for i, item in enumerate(candidates))
    assert all(item.spot_ex_vat == Decimal("0.2") for item in candidates)
    assert repo.register_inputs(METER, candidates, [snap], 2, price_batch=prices,
                                closed_before_ts=START_TS + 1800) == 3
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    assert repo.commit(METER, calculate(batch), batch, 1, "archived-reprice") == 2
    current = repo.read(METER, START_TS, START_TS + 1800)
    assert current.net_cost == Decimal("0.4") and current.dirty_from_ts is None
    assert current.freshness == "current" and current.source_watermark_ts == START_TS + 1800
    assert repo.read(METER, START_TS, START_TS + 1800, 1).net_cost == Decimal("0.2")


@pytest.mark.parametrize("kind", ["price", "contract"])
@pytest.mark.parametrize("partial", [False, True])
def test_raw_price_and_contract_aba_full_revalidation_is_no_version_but_partial_stays_dirty(
    repo, terms, inputs, calculate, kind, partial,
):
    from dao.prog.settlement.metering import reprice_inputs

    snap = terms()
    repo.register_inputs(METER, (), [snap], 0)
    repo.register_prices(METER, [price()], 1)
    prices = repo.read_prices(METER, START_TS, START_TS + 1800)
    repo.register_inputs(METER, inputs([snap]), [snap], 2, price_batch=prices,
                         closed_before_ts=START_TS + 1800)
    original = repo.read_inputs(METER, START_TS, START_TS + 1800)
    repo.commit(METER, calculate(original), original, 0, "raw-A")
    history = {name: rows(repo, name) for name in ("input", "run", "interval", "adjustment")}
    if kind == "price":
        repo.register_prices(METER, [price(value="0.2")], 3)
        repo.register_prices(METER, [price()], 4)
    else:
        other = terms(cost_supplier_consumption=((snap.cost_supplier_consumption[0][0], Decimal("0.2")),))
        repo.register_inputs(METER, (), [other], 3)
        repo.register_inputs(METER, (), [snap], 4)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    prices = repo.read_prices(METER, START_TS, START_TS + 1800)
    candidates = reprice_inputs(batch, prices)
    assert [item.input_hash for item in candidates] == [item.input_hash for item in original.items]
    assert repo.register_inputs(METER, candidates, [snap], 5, price_batch=prices,
                                closed_before_ts=START_TS + 1800) == 5
    if partial:
        prefix = repo.read_inputs(METER, START_TS, START_TS + 900)
        assert repo.commit(METER, calculate(prefix), prefix, 1, "partial-ABA") == 1
        assert rows(repo, "book")[0]["dirty_from_ts"] == START_TS
    assert repo.commit(METER, calculate(batch), batch, 1, "full-ABA") == 1
    current = repo.read(METER, START_TS, START_TS + 1800)
    assert current.input_revision == current.current_input_revision == 5
    assert current.dirty_from_ts is None and current.freshness == "current"
    assert {name: rows(repo, name) for name in history} == history
    assert [line.input_revision for line in current.lines] == [3, 3]
    with pytest.raises(ConcurrentSettlementError):
        repo.commit(METER, calculate(original), original, 1, "stale-A-token")
    repo.register_prices(METER, [price(value="0.3")], 5)
    newer = repo.read_inputs(METER, START_TS, START_TS + 1800)
    prices = repo.read_prices(METER, START_TS, START_TS + 1800)
    repo.register_inputs(METER, reprice_inputs(newer, prices), [snap], 6, price_batch=prices,
                         closed_before_ts=START_TS + 1800)
    newer = repo.read_inputs(METER, START_TS, START_TS + 1800)
    assert repo.commit(METER, calculate(newer), newer, 1, "real-money-change") == 2
    old = repo.read(METER, START_TS, START_TS + 1800, 1)
    assert old.input_revision == 3 and old.lines == current.lines
    assert rows(repo, "run")[0] == history["run"][0]


@pytest.mark.parametrize("mutation", ["value", "source", "quality", "missing-keywords", "one-keyword",
                                      "future", "unclosed", "meter", "row-revision", "hash", "omission",
                                      "selection", "range", "unregistered-value"])
def test_closed_registration_cannot_forge_price_evidence(repo, terms, inputs, mutation):
    from dao.prog.settlement.repository import PriceInputMismatch

    snap = terms()
    repo.register_inputs(METER, (), [snap], 0)
    repo.register_prices(METER, [price(quality="estimated")], 1)
    prices = repo.read_prices(METER, START_TS, START_TS + 900)
    candidate = inputs([snap], imports=("1",))[0]
    candidate = replace(candidate, quality="estimated", spot_source=candidate.spot_source.replace(
        '"measured"', '"estimated"'))
    kwargs = {"price_batch": prices, "closed_before_ts": START_TS + 900}
    if mutation == "value":
        candidate = replace(candidate, spot_ex_vat="0.2")
    elif mutation == "source":
        candidate = replace(candidate, spot_source="self-declared")
    elif mutation == "quality":
        candidate = replace(candidate, quality="measured")
    elif mutation == "missing-keywords":
        kwargs = {}
    elif mutation == "one-keyword":
        kwargs.pop("closed_before_ts")
    elif mutation == "future":
        kwargs["closed_before_ts"] = START_TS + 86401
    elif mutation == "unclosed":
        kwargs["closed_before_ts"] = START_TS + 899
    elif mutation == "meter":
        kwargs["price_batch"] = replace(prices, meter_id="other-grid")
    elif mutation == "row-revision":
        kwargs["price_batch"] = replace(prices, items=(replace(prices.items[0], input_revision=1),))
    elif mutation == "hash":
        kwargs["price_batch"] = replace(prices, items=(replace(prices.items[0], content_hash="b" * 64),))
    elif mutation == "omission":
        kwargs["price_batch"] = replace(prices, items=())
    elif mutation == "selection":
        kwargs["price_batch"] = replace(prices, snapshots=())
    elif mutation == "range":
        kwargs["price_batch"] = replace(prices, start_ts=START_TS + 3600, end_ts=START_TS + 4500)
    else:
        kwargs["price_batch"] = replace(prices, items=(replace(
            prices.items[0], spot=price(value="0.2", quality="estimated")),))
    before = frozen(repo)
    with pytest.raises(PriceInputMismatch):
        repo.register_inputs(METER, [candidate], [snap], 2, **kwargs)
    assert frozen(repo) == before


def test_empty_registered_batch_is_not_permission_to_smuggle_known_prices(repo, terms, inputs):
    from dao.prog.settlement.repository import PriceInputMismatch

    snap = terms()
    repo.register_inputs(METER, (), [snap], 0)
    prices = repo.read_prices(METER, START_TS, START_TS + 900)
    known = inputs([snap], imports=("1",))
    with pytest.raises(PriceInputMismatch):
        repo.register_inputs(METER, known, [snap], 1, price_batch=prices, closed_before_ts=START_TS + 900)
    unknown = (replace(known[0], spot_ex_vat=None, spot_source=None, quality="incomplete"),)
    assert repo.register_inputs(METER, unknown, [snap], 1, price_batch=prices,
                                closed_before_ts=START_TS + 900) == 2


def test_legacy_price_exception_is_same_boundaries_same_content_and_no_quality_upgrade(repo, terms, inputs):
    from dao.prog.settlement.repository import PriceInputMismatch

    snap = terms()
    original = tuple(replace(item, quality="estimated") for item in inputs([snap], imports=("1",)))
    repo.register_inputs(METER, original, [snap], 0)  # Bestaande offline positional compatibiliteit.
    prices = repo.read_prices(METER, START_TS, START_TS + 1800)
    for candidate in (replace(original[0], spot_ex_vat="0.2"),
                      replace(original[0], spot_source="new-source"),
                      replace(original[0], end_ts=START_TS + 450),
                      replace(original[0], start_ts=START_TS + 900, end_ts=START_TS + 1800),
                      replace(original[0], quality="measured")):
        with pytest.raises(PriceInputMismatch):
            repo.register_inputs(METER, [candidate], [snap], 1, price_batch=prices,
                                 closed_before_ts=START_TS + 1800)
    corrected = replace(original[0], import_kwh="2")
    assert repo.register_inputs(METER, [corrected], [snap], 1, price_batch=prices,
                                closed_before_ts=START_TS + 1800) == 2


def test_partial_required_window_and_withdrawal_cannot_fall_back_to_old_or_other_resolution(
    repo, terms, inputs, calculate,
):
    from dao.prog.settlement.metering import reprice_inputs
    from dao.prog.settlement.repository import PriceInputMismatch

    snap = terms(hour_price_rule="mean_quarters")
    old = inputs([terms()], imports=("1",))
    publish(repo, old, [snap], calculate)
    repo.register_prices(METER, [price(), price(step=900)], 1)
    batch = repo.read_inputs(METER, START_TS, START_TS + 900)
    prices = repo.read_prices(METER, START_TS, START_TS + 900)
    with pytest.raises(PriceInputMismatch):
        repo.commit(METER, calculate(batch), batch, 1, "partial-window")
    candidate = reprice_inputs(batch, prices)
    assert candidate[0].spot_ex_vat is None and candidate[0].quality == "incomplete"
    assert candidate[0].spot_source.startswith("unresolved:mean_quarters")
    repo.register_inputs(METER, candidate, [snap], 2, price_batch=prices, closed_before_ts=START_TS + 900)
    repo.register_prices(METER, [price(START_TS + i * 900, step=900) for i in range(4)], 3)
    batch = repo.read_inputs(METER, START_TS, START_TS + 900)
    candidate = reprice_inputs(batch, repo.read_prices(METER, START_TS, START_TS + 900))
    assert candidate[0].spot_ex_vat == Decimal("0.1") and candidate[0].quality == "incomplete"
    repo.register_prices(METER, [price(value=None, step=900, quality="incomplete")], 4)
    batch = repo.read_inputs(METER, START_TS, START_TS + 900)
    candidate = reprice_inputs(batch, repo.read_prices(METER, START_TS, START_TS + 900))
    assert candidate[0].spot_ex_vat is None and candidate[0].quality == "incomplete"


def test_unused_resolution_does_not_dirty_native_closed_inputs(repo, terms, inputs, calculate):
    snap = terms()
    publish(repo, inputs([snap], imports=("1",)), [snap], calculate)
    repo.register_prices(METER, [price(step=900)], 1)
    assert rows(repo, "book")[0]["dirty_from_ts"] is None
    assert repo.read(METER, START_TS, START_TS + 900).freshness == "current"
    repo.register_prices(METER, [price()], 2)  # Ook de eerste registratie vraagt hervalidatie.
    assert rows(repo, "book")[0]["dirty_from_ts"] == START_TS


@pytest.mark.parametrize("correction", ["source", "quality"])
def test_raw_source_and_quality_corrections_dirty_and_require_new_closed_price_evidence(
    repo, terms, inputs, calculate, correction,
):
    from dao.prog.settlement.metering import reprice_inputs
    from dao.prog.settlement.repository import PriceInputMismatch

    snap = terms()
    publish(repo, inputs([snap], imports=("1",)), [snap], calculate)
    corrected = price(source="corrected-market") if correction == "source" else price(quality="estimated")
    repo.register_prices(METER, [corrected], 1)
    assert rows(repo, "book")[0]["dirty_from_ts"] == START_TS
    batch = repo.read_inputs(METER, START_TS, START_TS + 900)
    with pytest.raises(PriceInputMismatch):
        repo.commit(METER, calculate(batch), batch, 1, "old-price-provenance")
    prices = repo.read_prices(METER, START_TS, START_TS + 900)
    candidate = reprice_inputs(batch, prices)
    assert candidate[0].spot_ex_vat == Decimal("0.1")
    assert corrected.source in candidate[0].spot_source
    assert candidate[0].quality == ("estimated" if correction == "quality" else "measured")
    repo.register_inputs(METER, candidate, [snap], 2, price_batch=prices, closed_before_ts=START_TS + 900)
    batch = repo.read_inputs(METER, START_TS, START_TS + 900)
    repo.commit(METER, calculate(batch), batch, 1, "corrected-price-provenance")
    assert rows(repo, "book")[0]["dirty_from_ts"] is None


def test_first_mean_quarter_marks_the_existing_later_closed_quarter_not_a_fictional_prefix(
    repo, terms, inputs, calculate,
):
    from dao.prog.settlement.engine import settle_series
    from dao.prog.settlement.types import SettlementState

    snap = terms(hour_price_rule="mean_quarters")
    repo.register_inputs(METER, inputs([snap], start=START_TS + 900, imports=("1",), spot=None), [snap], 0)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    result = settle_series(batch.items, batch.snapshots, SettlementState(START_TS, None, None))
    repo.commit(METER, result, batch, 0, "revalidate-from-contract-dependency")
    assert rows(repo, "book")[0]["dirty_from_ts"] is None
    repo.register_prices(METER, [price(step=900)], 1)
    assert rows(repo, "book")[0]["dirty_from_ts"] == START_TS + 900
    assert len(rows(repo, "input")) == 1 and rows(repo, "input")[0]["start_ts"] == START_TS + 900


def test_reprice_requires_same_revision_selection_meter_and_price_envelope(repo, terms, inputs):
    from dao.prog.settlement.metering import reprice_inputs

    snap = terms()
    repo.register_inputs(METER, inputs([snap]), [snap], 0)
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    prices = repo.read_prices(METER, START_TS, START_TS + 1800)
    for invalid in (replace(prices, input_revision=2), replace(prices, snapshots=()),
                    replace(prices, meter_id="other-grid"),
                    replace(prices, start_ts=START_TS + 3600, end_ts=START_TS + 4500)):
        with pytest.raises(ValueError):
            reprice_inputs(batch, invalid)
    assert reprice_inputs(replace(batch, items=()), prices) == ()
    unchanged = reprice_inputs(batch, prices)
    assert [item.input_hash for item in unchanged] == [item.input_hash for item in batch.items]
    assert all(item.input_revision == 0 for item in unchanged)


@pytest.mark.parametrize("quality", ["estimated", "incomplete"])
def test_reprice_after_recorder_retention_preserves_real_archived_kwh_and_quality(
    repo, terms, recorder, calculate, quality,
):
    from dao.prog.settlement.metering import merge_inputs, read_ha_energy, reprice_inputs

    put_counter(recorder, "sensor.import", short_points(START_TS, START_TS + 900))
    put_counter(recorder, "sensor.export", short_points(START_TS, START_TS + 900, increment="0"))
    snap = terms()
    measured = read_ha_energy(recorder, selection(), START_TS, START_TS + 900, 900)
    original = tuple(replace(item, quality=quality) for item in merge_inputs(measured, [price()], [snap], meter_id=METER))
    publish(repo, original, [snap], calculate)
    for name in ("statistics_short_term", "statistics", "statistics_meta"):
        recorder.metadata.tables[name].drop(recorder.engine)
    assert read_ha_energy(recorder, selection(), START_TS, START_TS + 900, 900)[0].import_kwh is None
    repo.register_prices(METER, [price(value="0.2")], 1)
    archived = repo.read_inputs(METER, START_TS, START_TS + 900)
    prices = repo.read_prices(METER, START_TS, START_TS + 900)
    candidate = reprice_inputs(archived, prices)
    assert candidate[0].import_kwh == Decimal("0.3") and candidate[0].export_kwh == Decimal("0")
    assert candidate[0].battery_discharge_kwh == Decimal("0") and candidate[0].quality == quality
    assert candidate[0].sources == original[0].sources and candidate[0].origin == "unknown"
    repo.register_inputs(METER, candidate, [snap], 2, price_batch=prices, closed_before_ts=START_TS + 900)
    batch = repo.read_inputs(METER, START_TS, START_TS + 900)
    repo.commit(METER, calculate(batch), batch, 1, "retained-archive-reprice")
    current = repo.read(METER, START_TS, START_TS + 900)
    assert current.known_subtotal == Decimal("0.06") and current.freshness == "current"
    assert current.net_cost == (Decimal("0.06") if quality == "estimated" else None)


def test_legacy_closed_hour_cannot_claim_measured_quality_without_raw_evidence(repo, terms, inputs):
    from dao.prog.settlement.repository import PriceInputMismatch

    snap = terms()
    hour = replace(inputs([snap], imports=("1",))[0], end_ts=START_TS + 3600)
    repo.register_inputs(METER, [hour], [snap], 0)
    prices = repo.read_prices(METER, START_TS, START_TS + 3600)
    with pytest.raises(PriceInputMismatch):
        repo.register_inputs(METER, [replace(hour, import_kwh="2")], [snap], 1,
                             price_batch=prices, closed_before_ts=START_TS + 3600)
