"""_requested_timezone(): the v2 API's 'timezone' query parameter.

`request.args.get('timezone') if None else "Europe/Amsterdam"` is always the
default branch (None is falsy), so a request's own timezone parameter was
silently ignored no matter what was sent.

Uses a throwaway Flask app instead of the real one: the real app package
loads the add-on configuration at import time, which this fix does not
depend on.
"""

import pytest

pytest.importorskip("flask")

from flask import Flask
from werkzeug.exceptions import BadRequest

from dao.webserver.app.v2.api.routes import _requested_timezone


@pytest.fixture
def app():
    application = Flask(__name__)
    application.config["TESTING"] = True
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
