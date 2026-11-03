"""_requested_timezone(): the v2 API's 'timezone' query parameter.

`request.args.get('timezone') if None else "Europe/Amsterdam"` is always the
default branch (None is falsy), so a request's own timezone parameter was
silently ignored no matter what was sent.

Uses a throwaway Flask app instead of the real one: the real app package
loads the add-on configuration at import time, which this fix does not
depend on.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("flask")

from flask import Flask
from werkzeug.exceptions import BadRequest

from dao.webserver.app.v2.api import routes
from dao.webserver.app.v2.api.routes import _requested_timezone


@pytest.fixture
def app():
    application = Flask(__name__)
    application.config.update(TESTING=True, PROPAGATE_EXCEPTIONS=False)
    application.register_blueprint(routes.api)
    return application


def test_the_default_is_used_when_no_parameter_is_given(app):
    with app.test_request_context("/data/"):
        assert _requested_timezone() == "Europe/Amsterdam"


def test_an_explicit_timezone_is_honoured(app):
    with app.test_request_context("/data/?timezone=America/New_York"):
        assert _requested_timezone() == "America/New_York"


def test_utc_is_accepted(app):
    with app.test_request_context("/data/?timezone=UTC"):
        assert _requested_timezone() == "UTC"


def test_an_unknown_timezone_is_a_bad_request_not_a_crash(app):
    with app.test_request_context("/data/?timezone=Not/AZone"):
        with pytest.raises(BadRequest):
            _requested_timezone()


def test_an_empty_parameter_falls_back_to_the_default(app):
    with app.test_request_context("/data/?timezone="):
        assert _requested_timezone() == "Europe/Amsterdam"


@pytest.fixture(params=["/data/", "/data-sql-ha/"])
def endpoint(request):
    return request.param


@pytest.fixture
def report_spy(monkeypatch):
    report = SimpleNamespace(
        get_data=Mock(return_value=[]), get_ha_data_query=Mock(return_value="SELECT synthetic")
    )
    constructor = Mock(return_value=report)
    monkeypatch.setattr(routes, "Report", constructor)
    return constructor, report


def backend_kwargs(endpoint, report_spy):
    report = report_spy[1]
    method = report.get_data if endpoint == "/data/" else report.get_ha_data_query
    return method.call_args.kwargs


@pytest.mark.parametrize("zone", ["Europe/Amsterdam", "America/New_York", "UTC"])
def test_aware_query_preserves_utc_instant(app, endpoint, report_spy, zone):
    response = app.test_client().get(endpoint, query_string={
        "start": "2026-01-01T10:00:00+03:00",
        "end": "2026-01-01T12:00:00+03:00",
        "aggregate": "hour", "timezone": zone,
    })

    assert response.status_code == 200
    kwargs = backend_kwargs(endpoint, report_spy)
    assert kwargs["start"].astimezone(UTC) == datetime(2026, 1, 1, 7, tzinfo=UTC)
    assert kwargs["end"].astimezone(UTC) == datetime(2026, 1, 1, 9, tzinfo=UTC)
    assert kwargs["start"].tzinfo.key == zone
    assert kwargs["end"].tzinfo.key == zone


@pytest.mark.parametrize("boundary", ["start", "end"])
@pytest.mark.parametrize("zone,ambiguous", [
    ("Europe/Amsterdam", "2026-10-25T02:30:00"),
    ("America/New_York", "2026-11-01T01:30:00"),
    ("Australia/Lord_Howe", "2026-04-05T01:45:00"),
])
def test_ambiguous_naive_query_is_rejected(
    app, endpoint, report_spy, boundary, zone, ambiguous
):
    query = {
        "start": "2026-01-01T00:00:00", "end": "2026-12-01T00:00:00",
        "aggregate": "hour", "timezone": zone, boundary: ambiguous,
    }
    response = app.test_client().get(endpoint, query_string=query)

    assert response.status_code == 400
    report_spy[0].assert_not_called()


@pytest.mark.parametrize("boundary", ["start", "end"])
@pytest.mark.parametrize("zone,nonexistent", [
    ("Europe/Amsterdam", "2026-03-29T02:30:00"),
    ("America/New_York", "2026-03-08T02:30:00"),
    ("Australia/Lord_Howe", "2026-10-04T02:15:00"),
])
def test_nonexistent_naive_query_is_rejected(
    app, endpoint, report_spy, boundary, zone, nonexistent
):
    query = {
        "start": "2026-01-01T00:00:00", "end": "2026-12-01T00:00:00",
        "aggregate": "hour", "timezone": zone, boundary: nonexistent,
    }
    response = app.test_client().get(endpoint, query_string=query)

    assert response.status_code == 400
    report_spy[0].assert_not_called()


def test_aware_fold_range_is_compared_in_utc_not_wall_time(app, endpoint, report_spy):
    response = app.test_client().get(endpoint, query_string={
        "start": "2026-10-25T02:45:00+02:00", "end": "2026-10-25T02:15:00+01:00",
        "aggregate": "hour", "timezone": "Europe/Amsterdam",
    })

    assert response.status_code == 200
    kwargs = backend_kwargs(endpoint, report_spy)
    assert kwargs["start"].astimezone(UTC) == datetime(2026, 10, 25, 0, 45, tzinfo=UTC)
    assert kwargs["end"].astimezone(UTC) == datetime(2026, 10, 25, 1, 15, tzinfo=UTC)
    assert kwargs["start"].fold == 0
    assert kwargs["end"].fold == 1


@pytest.mark.parametrize("start,end", [
    ("2026-01-01T10:00:00", "2026-01-01T12:00:00+01:00"),
    ("2026-01-01T10:00:00+01:00", "2026-01-01T12:00:00"),
])
def test_mixed_aware_and_naive_query_is_allowed(app, endpoint, report_spy, start, end):
    response = app.test_client().get(endpoint, query_string={
        "start": start, "end": end, "aggregate": "hour", "timezone": "Europe/Amsterdam",
    })

    assert response.status_code == 200
    kwargs = backend_kwargs(endpoint, report_spy)
    assert kwargs["start"].astimezone(UTC) == datetime(2026, 1, 1, 9, tzinfo=UTC)
    assert kwargs["end"].astimezone(UTC) == datetime(2026, 1, 1, 11, tzinfo=UTC)
