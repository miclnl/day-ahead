"""Both data endpoints reject invalid input before touching the report backend."""

import logging
import unicodedata
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flask import Flask

from dao.webserver.app.v2.api import routes

SYNTHETIC_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijklmnop"


@pytest.fixture(params=["/data/", "/data-sql-ha/"])
def endpoint(request):
    return request.param


@pytest.fixture
def api_client():
    app = Flask(__name__)
    # An uncaught exception is an HTTP 500, not a fixture/test setup error.
    app.config.update(TESTING=True, PROPAGATE_EXCEPTIONS=False)
    app.register_blueprint(routes.api)
    return app.test_client()


@pytest.fixture
def report_spy(monkeypatch):
    report = SimpleNamespace(
        get_data=Mock(return_value=[]),
        get_ha_data_query=Mock(return_value="SELECT synthetic_query"),
    )
    constructor = Mock(return_value=report)
    monkeypatch.setattr(routes, "Report", constructor)
    return constructor, report


@pytest.fixture
def valid_query():
    return {
        "start": "2026-01-01T10:00:00",
        "end": "2026-01-01T12:00:00",
        "timezone": "Europe/Amsterdam",
        "aggregate": "hour",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"start": None},
        {"end": None},
        {"start": ""},
        {"end": ""},
        {"start": "not-an-iso-date"},
        {"end": "2026-02-30T12:00:00"},
        {"aggregate": "invalid"},
        {"aggregate": None},
        {"aggregate": ""},
        {"timezone": "Not/AZone"},
        {"end": "2026-01-01T10:00:00"},
        {"end": "2026-01-01T09:59:59"},
        # Equal/reversed UTC instants despite increasing local clock labels.
        {"start": "2026-10-25T02:15:00+01:00", "end": "2026-10-25T02:45:00+02:00"},
        {"start": "2026-10-25T02:00:00+02:00", "end": "2026-10-25T02:00:00+02:00"},
    ],
)
def test_invalid_query_returns_400_without_report(
    api_client, endpoint, report_spy, valid_query, changes
):
    query = {**valid_query, **changes}
    query = {key: value for key, value in query.items() if value is not None}

    response = api_client.get(endpoint, query_string=query)

    assert response.status_code == 400
    report_spy[0].assert_not_called()


@pytest.mark.parametrize("aggregate", ["15min", "hour", "day", "week", "month"])
def test_valid_query_retains_response_shape(
    api_client, endpoint, report_spy, valid_query, aggregate
):
    constructor, report = report_spy
    moment = datetime(2026, 1, 1, 10, 15)
    report.get_data.return_value = [{"ts": moment, "cons": {"v": 1.25, "f": None}}]
    query = {**valid_query, "fields": "cons,prod", "aggregate": aggregate}

    response = api_client.get(endpoint, query_string=query)

    assert response.status_code == 200
    constructor.assert_called_once_with()
    if endpoint == "/data/":
        expected_ts = (
            "2026-01-01 10:15" if aggregate == "15min"
            else "2026-01-01 10:00" if aggregate == "hour"
            else "2026-01-01"
        )
        assert response.json == [{"ts": expected_ts, "cons": {"v": 1.25, "f": None}}]
        kwargs = report.get_data.call_args.kwargs
        assert kwargs["aggregate"] == aggregate
    else:
        assert response.get_data(as_text=True) == "SELECT synthetic_query"
        kwargs = report.get_ha_data_query.call_args.kwargs
        assert kwargs["step"].total_seconds() == 86400
    assert kwargs["var_codes"] == ["cons", "prod"]
    assert kwargs["start"].isoformat() == "2026-01-01T10:00:00+01:00"
    assert kwargs["end"].isoformat() == "2026-01-01T12:00:00+01:00"


@pytest.mark.parametrize("failure_at", ["constructor", "backend", "result"])
@pytest.mark.parametrize("diagnostic,secrets", [
    ("\r\t\x1b\x85\u2028\u202eafter_controls", ()),
    (
        "api_token=B11_TEST_TOKEN client_secret='B11 SECRET WITH SPACES' "
        'DB_PASSWORD="B11 QUOTED PASSWORD" access-token=B11_ACCESS_TOKEN',
        ("B11_TEST_TOKEN", "B11 SECRET WITH SPACES", "B11 QUOTED PASSWORD", "B11_ACCESS_TOKEN"),
    ),
    (
        "pass=B11_SHORT_PASSWORD passphrase='B11 PASSPHRASE' "
        "private_key=B11_PRIVATE_KEY authorization=B11_AUTHORIZATION",
        ("B11_SHORT_PASSWORD", "B11 PASSPHRASE", "B11_PRIVATE_KEY", "B11_AUTHORIZATION"),
    ),
    (
        "https://B11_URL_USER:B11_URL_PASS@example.invalid/path "
        "postgresql+psycopg://B11_DB_USER:B11_DB_PASS@db.invalid/catalog "
        "//B11_PROXY_USER:B11_PROXY_PASS@proxy.invalid/path",
        ("B11_URL_USER", "B11_URL_PASS", "B11_DB_USER", "B11_DB_PASS",
         "B11_PROXY_USER", "B11_PROXY_PASS"),
    ),
    ("password=" + "B11_LONG_SECRET" * 70 + " detail=" + "x" * 1000, ("B11_LONG_SECRET",)),
    pytest.param("authorization=Bearer B11_BEARER_TOKEN", ("B11_BEARER_TOKEN",),
                 id="equals-bearer-token"),
    pytest.param("authorization=Basic dXNlcjpwYXNzd29yZA==", ("dXNlcjpwYXNzd29yZA==",),
                 id="equals-basic-payload"),
    pytest.param("password: hunter2", ("hunter2",), id="colon-password"),
    pytest.param("token: abc123def456", ("abc123def456",), id="colon-token"),
    pytest.param("Authorization: Bearer B11_BEARER_TOKEN", ("B11_BEARER_TOKEN",),
                 id="colon-bearer-token"),
    pytest.param("Bearer " + SYNTHETIC_JWT, (SYNTHETIC_JWT,), id="bearer-jwt"),
    pytest.param("Basic dXNlcjpwYXNzd29yZA==", ("dXNlcjpwYXNzd29yZA==",),
                 id="standalone-basic-payload"),
    pytest.param("Bearer B11_OPAQUE_TOKEN", ("B11_OPAQUE_TOKEN",),
                 id="standalone-opaque-bearer"),
    pytest.param(SYNTHETIC_JWT, (SYNTHETIC_JWT,), id="standalone-jwt"),
    pytest.param("Authorization: Basic dXNlcjpwYXNzd29yZA==", ("dXNlcjpwYXNzd29yZA==",),
                 id="colon-basic-payload"),
    pytest.param("pAsSwOrD : 'B11 COLON PASSWORD' api_token:\tB11_COLON_TOKEN\nFORGED_LOG",
                 ("B11 COLON PASSWORD", "B11_COLON_TOKEN"), id="quoted-colon-controls"),
    pytest.param("dbPassword=B11_CAMEL_PASSWORD accessToken: B11_CAMEL_TOKEN",
                 ("B11_CAMEL_PASSWORD", "B11_CAMEL_TOKEN"), id="camel-case-keys"),
    *[
        pytest.param(f"{key}{separator}B11_SECRET_VALUE", ("B11_SECRET_VALUE",),
                     id=f"credential-key-{key}-{separator.strip()}")
        for key in (
            "password", "passwd", "passphrase", "pwd", "pass", "token", "secret",
            "secretkey", "secret_key", "authorization", "credential", "wachtwoord",
            "apikey", "api_key", "private_key", "access_key", "x-api-key",
            "refresh_token", "db_password", "dbpassword", "DBPASSWORD", "clientsecret",
            "authtoken", "mytoken", "password2", "db2password3",
            "config.password.value", "config_token_value",
        )
        for separator in ("=", ": ")
    ],
    *[
        pytest.param(f"{key}{separator}B11_VISIBLE_VALUE", (),
                     id=f"innocent-key-{key}-{separator.strip()}")
        for key in (
            "compass", "compassion", "tokenizer", "tokenize", "secretive",
            "passwordless", "credentialing", "author", "pass2", "dbpass", "db.pass",
            "pass_value", "mytokenizer", "dbpasswordless",
        )
        for separator in ("=", ": ")
    ],
    *[
        pytest.param(f"{key}{separator}{scheme} B11_SCHEME_CREDENTIAL",
                     ("B11_SCHEME_CREDENTIAL",), id=f"compound-{key}-{scheme}-{separator.strip()}")
        for key in ("dbpassword", "secretkey", "password2")
        for separator in ("=", ": ")
        for scheme in ("Bearer", "Basic")
    ],
])
def test_backend_exception_does_not_leak_sentinel(
    api_client, endpoint, report_spy, valid_query, failure_at, diagnostic, secrets, caplog
):
    constructor, report = report_spy
    innocent = (
        "sqlite3.OperationalError: no such table: readings; "
        "postgresql+psycopg://db.internal.example/report; "
        "sqlite3.OperationalError.message; schema.readings.daily; basic; bearer; "
        "compass: calibration; tokenizer: unexpected; "
    )
    error = RuntimeError(
        "BACKEND_SENTINEL password=B11_TEST_SECRET\nFORGED_LOG "
        "missing table synthetic_values " + innocent + diagnostic
    )

    def broken_rows():
        raise error
        yield  # The backend can fail while its lazy result is consumed.

    class BrokenQuery:
        def __str__(self):
            raise error

    if failure_at == "constructor":
        constructor.side_effect = error
    elif failure_at == "backend":
        report.get_data.side_effect = error
        report.get_ha_data_query.side_effect = error
    else:
        report.get_data.return_value = broken_rows()
        report.get_ha_data_query.return_value = BrokenQuery()

    with caplog.at_level(logging.ERROR):
        response = api_client.get(endpoint, query_string=valid_query)

    assert response.status_code == 500
    assert response.json == {"error": "Internal server error"}
    assert b"BACKEND_SENTINEL" not in response.data
    constructor.assert_called_once_with()
    assert "RuntimeError" in caplog.text
    assert "missing table synthetic_values" in caplog.text
    assert "missing table synthetic_values" not in response.get_data(as_text=True)
    assert "BACKEND_SENTINEL" in caplog.text
    assert "B11_TEST_SECRET" not in caplog.text
    assert "<redacted>" in caplog.text
    assert "\nFORGED_LOG" not in caplog.text
    assert len(caplog.records) == 1
    record = caplog.records[0]
    message = record.getMessage()
    assert innocent in message
    # A printable diagnostic with no declared credentials must survive verbatim.
    if not secrets and diagnostic.isprintable():
        assert diagnostic in message
    # Keep the injection witness, but only as inert text on the existing line.
    assert "FORGED_LOG" in message
    assert len(message.splitlines()) == 1
    assert not any(unicodedata.category(char).startswith("C") for char in message)
    assert len(message) <= 550
    assert record.exc_info is None
    for secret in secrets:
        assert secret not in caplog.text
