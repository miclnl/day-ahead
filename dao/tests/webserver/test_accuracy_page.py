"""The /v2/accuracy page and its JSON endpoint.

The page has to render on a fresh install, where none of the forecast
artefacts exist yet: the endpoint then returns nulls rather than a 500, and
the page shows its empty state.
"""

import json
import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_wtf")

from .conftest import INGRESS, SUPERVISOR  # noqa: E402


def test_accuracy_page_renders(client):
    response = client.get("/v2/accuracy", headers=INGRESS, environ_base=SUPERVISOR)

    assert response.status_code == 200
    assert b"Accuracy" in response.data
    assert b"Nog geen prognoses gearchiveerd" in response.data


def test_api_without_archive_returns_json_nulls(client):
    response = client.get(
        "/v2/api/accuracy/", headers=INGRESS, environ_base=SUPERVISOR
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["accuracy"] is None
    assert payload["weather"] is None
    assert payload["baseload"]["selection"] is None
    assert payload["baseload"]["profile_created"] is None
    assert payload["pv"] == {}
    assert payload["days"] == 28


def test_api_reads_written_files(client, site):
    forecast = site / "data" / "forecast"
    (forecast / "baseload").mkdir(parents=True, exist_ok=True)
    (forecast / "pv").mkdir(parents=True, exist_ok=True)

    (forecast / "accuracy.json").write_text(
        json.dumps(
            {
                "created": "2026-06-21T12:00:00+00:00",
                "days": [7, 28],
                "components": {
                    "temp": {
                        "component": "temp",
                        "unit": "C",
                        "windows": {
                            "28": {
                                "by_lead": {"0": {"mae": 0.2, "rmse": 0.3,
                                                  "bias": 0.1, "n": 24}},
                                "by_hour": {},
                                "by_weekday": {},
                                "by_regime": {},
                                "by_source": {},
                                "pairs": 24,
                                "missing": 0,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (forecast / "baseload" / "selection.json").write_text(
        json.dumps(
            {
                "model": "ml",
                "scores": {},
                "decided_at": "2026-06-21T12:00:00+00:00",
                "reason": "backtest over 28 dagen",
            }
        ),
        encoding="utf-8",
    )
    (forecast / "pv" / "Roof_South.selection.json").write_text(
        json.dumps(
            {
                "model": "physical",
                "scores": {},
                "decided_at": "2026-06-21T12:00:00+00:00",
                "reason": "geen archief",
            }
        ),
        encoding="utf-8",
    )

    try:
        response = client.get(
            "/v2/api/accuracy/?days=28", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["accuracy"]["components"]["temp"]["unit"] == "C"
        assert payload["baseload"]["selection"]["model"] == "ml"
        assert payload["pv"]["Roof_South"]["selection"]["model"] == "physical"
    finally:
        # "site" is module-scoped and shared with the other tests in this
        # file; leave it as it was found.
        for path in (
            forecast / "accuracy.json",
            forecast / "baseload" / "selection.json",
            forecast / "pv" / "Roof_South.selection.json",
        ):
            path.unlink(missing_ok=True)


def test_api_serves_strict_json_for_a_candidate_that_scored_nothing(client, site):
    """A candidate that scored no days gets mae/rmse/bias NaN. Flask's
    serialiser writes a bare NaN for that, which is not JSON: a strict
    parser rejects the whole document, the page's .catch() fires, and the
    user sees the same "no archive yet" message a fresh install shows."""
    forecast = site / "data" / "forecast"
    (forecast / "baseload").mkdir(parents=True, exist_ok=True)
    selection = forecast / "baseload" / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "model": "profile",
                "scores": {
                    "profile": {"mae": 0.21, "rmse": 0.3, "bias": 0.0, "n": 672},
                    "ml": {"mae": float("nan"), "rmse": float("nan"),
                           "bias": float("nan"), "n": 0},
                },
                "decided_at": "2026-06-21T12:00:00+00:00",
                "reason": "backtest over 28 dagen",
            }
        ),
        encoding="utf-8",
    )

    try:
        response = client.get(
            "/v2/api/accuracy/", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        # json.loads without parse_constant is what a browser's JSON.parse
        # does: it rejects NaN, Infinity and -Infinity outright.
        payload = json.loads(
            response.get_data(as_text=True),
            parse_constant=_reject_json_constant,
        )
        scores = payload["baseload"]["selection"]["scores"]
        assert scores["ml"]["mae"] is None
        assert scores["profile"]["mae"] == 0.21
    finally:
        selection.unlink(missing_ok=True)


def _reject_json_constant(name):
    raise AssertionError(f"niet-JSON constante in de response: {name}")


def test_written_artefacts_never_contain_nan():
    """Same rule one step earlier: nothing non-finite reaches disk either,
    or every reader of the artefact has the same problem."""
    import math
    import tempfile
    from pathlib import Path

    from dao.forecast.baseload.store import write_json

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "selection.json"
        write_json(
            path,
            {"scores": {"ml": {"mae": float("nan"), "rmse": math.inf, "n": 0}}},
        )
        text = path.read_text(encoding="utf-8")

    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text, parse_constant=_reject_json_constant)["scores"]["ml"][
        "mae"
    ] is None


def test_api_days_parameter_falls_back_on_garbage(client):
    response = client.get(
        "/v2/api/accuracy/?days=nonsense", headers=INGRESS, environ_base=SUPERVISOR
    )

    assert response.status_code == 200
    assert response.get_json()["days"] == 28


def test_api_resolves_a_window_the_report_never_produced(client, site):
    """The report decides which windows exist, not the page.

    The report writes the window it was run with -- 30 days by default --
    while the page asked for "28". The lookup found nothing, so the table
    stayed empty and the page announced that there was no archive at all,
    while the archive held thousands of scored pairs.
    """
    forecast = site / "data" / "forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    (forecast / "accuracy.json").write_text(
        json.dumps(
            {
                "created": "2026-11-09T15:09:42+00:00",
                "days": [7, 30],
                "components": {
                    "base": {
                        "component": "base",
                        "unit": "kWh",
                        "windows": {
                            "7": {"pairs": 24},
                            "30": {
                                "by_lead": {"0": {"mae": 0.1, "rmse": 0.2,
                                                  "bias": 0.0, "n": 24}},
                                "by_hour": {}, "by_weekday": {}, "by_regime": {},
                                "by_source": {}, "pairs": 24, "missing": 0,
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        response = client.get(
            "/v2/api/accuracy/?days=28", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        payload = response.get_json()
        # Which windows the report produced, and which one is being shown.
        assert payload["windows"] == [7, 30]
        assert payload["days"] == 30
    finally:
        (forecast / "accuracy.json").unlink(missing_ok=True)


def test_api_prefers_the_component_windows_over_the_metadata(client, site):
    """A window the components do not have is not offered.

    The page looks its data up by the window keys of the components, so a
    window that only the metadata mentions would show an empty table and
    the claim that there is no archive.
    """
    forecast = site / "data" / "forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    (forecast / "accuracy.json").write_text(
        json.dumps(
            {
                "created": "2026-11-09T15:09:42+00:00",
                "days": [7, 365],
                "components": {
                    "base": {
                        "component": "base",
                        "unit": "kWh",
                        "windows": {"7": {}, "30": {}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        response = client.get(
            "/v2/api/accuracy/?days=28", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["windows"] == [7, 30]
        assert payload["days"] == 30
    finally:
        (forecast / "accuracy.json").unlink(missing_ok=True)


@pytest.mark.parametrize("broken", [30, "30", [7, 30], None, 3.5])
def test_api_survives_a_report_with_a_broken_component_windows(client, site, broken):
    """A component whose windows are not a mapping must not take the page down.

    The windows of the other components are used, and the metadata of the
    report when there are none.
    """
    forecast = site / "data" / "forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    (forecast / "accuracy.json").write_text(
        json.dumps(
            {
                "created": "2026-11-09T15:09:42+00:00",
                "days": [7, 30],
                "components": {
                    "base": {"component": "base", "unit": "kWh", "windows": broken},
                    "temp": {"component": "temp", "unit": "C", "windows": {"30": {}}},
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        response = client.get(
            "/v2/api/accuracy/?days=28", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["windows"] == [30]
        assert payload["days"] == 30
    finally:
        (forecast / "accuracy.json").unlink(missing_ok=True)


def test_api_ignores_a_window_key_the_page_cannot_look_up(client, site):
    """"030" is not the window 30: the page looks its data up by key.

    Offering it would show a window that holds nothing.
    """
    forecast = site / "data" / "forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    (forecast / "accuracy.json").write_text(
        json.dumps(
            {
                "created": "2026-11-09T15:09:42+00:00",
                "components": {
                    "base": {
                        "component": "base",
                        "unit": "kWh",
                        "windows": {"030": {}, "7": {}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        response = client.get(
            "/v2/api/accuracy/?days=28", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["windows"] == [7]
        assert payload["days"] == 7
    finally:
        (forecast / "accuracy.json").unlink(missing_ok=True)


def test_api_keeps_a_window_the_report_did_produce(client, site):
    """A window the report produced is used as asked, not replaced."""
    forecast = site / "data" / "forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    (forecast / "accuracy.json").write_text(
        json.dumps({"created": "2026-11-09T15:09:42+00:00", "days": [7, 30],
                    "components": {}}),
        encoding="utf-8",
    )

    try:
        response = client.get(
            "/v2/api/accuracy/?days=7", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["windows"] == [7, 30]
        assert payload["days"] == 7
    finally:
        (forecast / "accuracy.json").unlink(missing_ok=True)


@pytest.mark.parametrize(
    "days", [None, "30", 30, [], [7, "x"], [True], [0], [-7]]
)
def test_api_survives_a_report_with_a_broken_days_field(client, site, days):
    """A report with unusable metadata must not take the page down.

    The windows are then read from the components themselves, which carry
    the same information: without them the page would claim again that
    there is no archive.
    """
    forecast = site / "data" / "forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    (forecast / "accuracy.json").write_text(
        json.dumps(
            {
                "created": "2026-11-09T15:09:42+00:00",
                "days": days,
                "components": {
                    "base": {
                        "component": "base",
                        "unit": "kWh",
                        "windows": {"7": {}, "30": {}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        response = client.get(
            "/v2/api/accuracy/?days=28", headers=INGRESS, environ_base=SUPERVISOR
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["windows"] == [7, 30]
        assert payload["days"] == 30
    finally:
        (forecast / "accuracy.json").unlink(missing_ok=True)


def test_page_takes_its_windows_from_the_report(client):
    """One source of truth for the window list.

    The template used to hardcode "7 dagen" and "28 dagen". The report
    writes the window it was run with, so the two could disagree -- and
    did: the page then looked for a window that was never written.
    """
    response = client.get("/v2/accuracy", headers=INGRESS, environ_base=SUPERVISOR)

    assert response.status_code == 200
    assert b'<option value="7"' not in response.data
    assert b'<option value="28"' not in response.data


def test_page_waits_for_the_chart_bundle_before_it_renders(client):
    """window.Chart only exists once the deferred bundle has run.

    The page's inline script fired its first request while the document
    was still being parsed, so the response could arrive before the
    deferred module bundle had executed. The charts were then silently
    skipped and both canvases stayed empty, although the data was there.
    """
    response = client.get("/v2/accuracy", headers=INGRESS, environ_base=SUPERVISOR)

    assert response.status_code == 200
    assert b'addEventListener("DOMContentLoaded", load)' in response.data


def test_page_draws_the_series_in_colours_of_their_own(client):
    """Chart.js v4 has no palette: a dataset without a colour is grey.

    Forecast and measurement were both drawn in that grey, so the chart
    could not show which line was which.
    """
    response = client.get("/v2/accuracy", headers=INGRESS, environ_base=SUPERVISOR)

    assert response.status_code == 200
    assert b"FORECAST_COLOR = " in response.data
    assert b"MEASURED_COLOR = " in response.data


def test_page_gives_the_charts_room_to_draw_in(client):
    """A chart of 120 pixels has no room for a plot area.

    The legend and the hour labels took 92 of those pixels, which left 28
    for the lines themselves. The height belongs on the container, because
    a responsive chart takes the height of its box.
    """
    response = client.get("/v2/accuracy", headers=INGRESS, environ_base=SUPERVISOR)

    assert response.status_code == 200
    for canvas in (b"recent-chart", b"bias-chart"):
        match = re.search(
            rb'<div style="height: (\d+)px;">\s*<canvas id="' + canvas + rb'"',
            response.data,
        )
        assert match, f"canvas {canvas.decode()} has no box of its own"
        assert int(match.group(1)) >= 200
    # Without this the chart keeps the ratio of the canvas and grows with
    # the width of the page instead of taking the height of its box.
    assert response.data.count(b"maintainAspectRatio: false") == 2
