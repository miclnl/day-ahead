"""The /v2/accuracy page and its JSON endpoint.

The page has to render on a fresh install, where none of the forecast
artefacts exist yet: the endpoint then returns nulls rather than a 500, and
the page shows its empty state.
"""

import json

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
