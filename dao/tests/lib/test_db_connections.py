"""Connection caching and error handling in db_connections.py.

_build_db_da had no error handling at all (unlike its _build_db_ha twin), so
an unreachable non-sqlite server raised straight out of make_db_da instead
of returning None as the docstring promises. The cache also used to pin
whichever config was seen first forever; a changed database host, user or
password after a reload of options.json had no effect.
"""

import types

import pytest

pytest.importorskip("sqlalchemy")

from dao.lib import db_connections


@pytest.fixture(autouse=True)
def _reset():
    db_connections.reset_connections()
    yield
    db_connections.reset_connections()


def _db_config(path, name="day_ahead.db", engine="sqlite", **overrides):
    fields = dict(
        engine=engine,
        database=name,
        server=None,
        username=None,
        password=None,
        port=None,
        db_path=str(path),
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _config(path, **overrides):
    return types.SimpleNamespace(
        database_da=_db_config(path, **overrides),
        database_ha=_db_config(path, **overrides),
        time_zone="Europe/Amsterdam",
    )


def test_make_db_da_returns_the_same_instance_for_an_unchanged_config(tmp_path):
    config = _config(tmp_path)

    first = db_connections.make_db_da(config, {}, check_create=True)
    second = db_connections.make_db_da(config, {}, check_create=True)

    assert first is not None
    assert first is second


def test_make_db_da_reconnects_when_the_database_changes(tmp_path):
    config = _config(tmp_path, name="a.db")
    first = db_connections.make_db_da(config, {}, check_create=True)

    changed = _config(tmp_path, name="b.db")
    second = db_connections.make_db_da(changed, {}, check_create=True)

    assert second is not None
    assert second is not first
    # The old engine was disposed, not leaked.
    assert first.engine.pool.checkedout() == 0


def test_make_db_da_returns_none_on_a_connection_error_instead_of_raising(tmp_path):
    """An unreachable non-sqlite server must not propagate an exception:
    make_db_da's docstring promises None."""
    config = _config(
        tmp_path,
        engine="postgresql",
        server="host.invalid",
        username="u",
        port=5432,
    )

    result = db_connections.make_db_da(config, {})

    assert result is None


def test_make_db_ha_returns_none_on_a_connection_error_instead_of_raising(tmp_path):
    config = _config(
        tmp_path,
        engine="postgresql",
        server="host.invalid",
        username="u",
        port=5432,
    )

    result = db_connections.make_db_ha(config, {})

    assert result is None


def test_make_db_da_reports_a_missing_sqlite_file_without_check_create(tmp_path):
    config = _config(tmp_path, name="missing.db")

    result = db_connections.make_db_da(config, {}, check_create=False)

    assert result is None


def test_da_and_ha_caches_are_independent(tmp_path):
    config = _config(tmp_path, name="shared.db")

    da = db_connections.make_db_da(config, {}, check_create=True)
    ha = db_connections.make_db_ha(config, {})

    assert da is not None and ha is not None
    assert da is not ha
