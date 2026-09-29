"""
DB connection factories with module-level singletons.

``make_db_da`` and ``make_db_ha`` each return the same ``DBmanagerObj``
instance for every call within a process, as long as the configuration
keeps resolving to the same database. The underlying SQLAlchemy engine
connects lazily on the first query, so construction is cheap and does not
require the database server to be reachable up front.

The cache is keyed on the resolved connection URL rather than pinned
forever to whichever config was seen first: the scheduler reloads its
configuration after every edit to options.json (see da_scheduler.py /
watchdog.sh), and a changed database host, user or password must actually
take effect instead of silently continuing to use the old connection.

Pass ``check_create=True`` to ``make_db_da`` to have the database created
automatically if it does not exist yet.  The resulting instance is still
cached as the singleton.  This is intended for the one-off ``check_db``
setup tool only.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

from dao.lib.db_manager import DBmanagerObj

if TYPE_CHECKING:
    from dao.prog.config.versions.v0 import ConfigurationV0

logger = logging.getLogger(__name__)

# Module-level singletons — one DBmanagerObj per database per process, keyed
# on the connection URL it was built from (see the cache helpers below).
_db_da: Optional[DBmanagerObj] = None
_db_da_url: Optional[str] = None
_db_ha: Optional[DBmanagerObj] = None
_db_ha_url: Optional[str] = None
# Lock protecting singleton initialisation; held only for fast local work.
_db_lock = threading.Lock()


def _connection_kwargs(db, secrets: dict) -> dict:
    return dict(
        db_dialect=db.engine,
        db_name=db.database,
        db_server=db.server,
        db_user=db.username,
        db_password=db.password.resolve(secrets) if db.password is not None else None,
        db_port=db.port,
        db_path=db.db_path,
    )


def reset_connections() -> None:
    """Drop both cached connections. Mainly for tests."""
    global _db_da, _db_da_url, _db_ha, _db_ha_url
    with _db_lock:
        for instance in (_db_da, _db_ha):
            if instance is not None:
                instance.engine.dispose()
        _db_da = _db_da_url = None
        _db_ha = _db_ha_url = None


def _build_db_da(
    config: ConfigurationV0,
    secrets: dict,
    check_create: bool = False,
) -> Optional[DBmanagerObj]:
    """Construct a DBmanagerObj for the Day Ahead database."""
    db = config.database_da
    kwargs = _connection_kwargs(db, secrets)

    try:
        import sqlalchemy_utils

        db_url = DBmanagerObj.db_url(**kwargs)
        if not sqlalchemy_utils.database_exists(db_url):
            if check_create:
                sqlalchemy_utils.create_database(db_url)
            else:
                if db.engine == "sqlite":
                    logger.error(
                        f"day_ahead database not found: {db.db_path}/{db.database}"
                    )
                else:
                    logger.error(
                        f"day_ahead database does not exist ({db.engine} / {db.server})"
                    )
                return None
        return DBmanagerObj(**kwargs, db_time_zone=config.time_zone)
    except Exception as ex:  # noqa: BLE001 - any driver/network failure
        # The docstrings on make_db_da/make_db_ha promise None on a
        # connection error; this branch used to be missing here, so an
        # unreachable non-sqlite server raised straight out of make_db_da
        # instead.
        if db.engine == "sqlite":
            logger.error(f"day_ahead database not found: {db.db_path}/{db.database}")
        else:
            logger.error(
                f"Cannot connect to day_ahead database ({db.engine} / {db.server}): {ex}"
            )
        return None


def _build_db_ha(
    config: ConfigurationV0,
    secrets: dict,
) -> Optional[DBmanagerObj]:
    """Construct a DBmanagerObj for the Home Assistant database."""
    db = config.database_ha
    kwargs = _connection_kwargs(db, secrets)
    try:
        import sqlalchemy_utils

        db_url = DBmanagerObj.db_url(**kwargs)
        if not sqlalchemy_utils.database_exists(db_url):
            if db.engine == "sqlite":
                logger.error(
                    f"Home Assistant database not found: {db.db_path}/{db.database}"
                )
            else:
                logger.error(
                    f"Home Assistant database does not exist ({db.engine} / {db.server})"
                )
            return None
        return DBmanagerObj(**kwargs, db_time_zone=config.time_zone)
    except Exception as ex:
        if db.engine == "sqlite":
            logger.error(
                f"Home Assistant database not found: {db.db_path}/{db.database}"
            )
        else:
            logger.error(
                f"Cannot connect to Home Assistant database ({db.engine} / {db.server}): {ex}"
            )
        return None


def make_db_da(
    config: ConfigurationV0,
    secrets: dict,
    check_create: bool = False,
) -> Optional[DBmanagerObj]:
    """
    Return the singleton DBmanagerObj for the Day Ahead database.

    The instance is created on first call and reused on subsequent calls.
    Pass ``check_create=True`` on the first call to have the database created
    automatically when it does not yet exist.  This is intended for the
    one-off ``check_db`` setup tool only.

    Returns None on connection error.
    """
    global _db_da, _db_da_url
    with _db_lock:
        wanted_url = DBmanagerObj.db_url(**_connection_kwargs(config.database_da, secrets))
        if _db_da is not None and wanted_url == _db_da_url:
            return _db_da
        result = _build_db_da(config, secrets, check_create=check_create)
        if result is not None:
            if _db_da is not None:
                logger.info("day_ahead database configuration changed, reconnecting")
                _db_da.engine.dispose()
            _db_da = result
            _db_da_url = wanted_url
        return _db_da


def make_db_ha(
    config: ConfigurationV0,
    secrets: dict,
) -> Optional[DBmanagerObj]:
    """
    Return the singleton DBmanagerObj for the Home Assistant database.

    The instance is created on first call and reused on subsequent calls.
    Returns None on connection error.
    """
    global _db_ha, _db_ha_url
    with _db_lock:
        wanted_url = DBmanagerObj.db_url(**_connection_kwargs(config.database_ha, secrets))
        if _db_ha is not None and wanted_url == _db_ha_url:
            return _db_ha
        result = _build_db_ha(config, secrets)
        if result is not None:
            if _db_ha is not None:
                logger.info("Home Assistant database configuration changed, reconnecting")
                _db_ha.engine.dispose()
            _db_ha = result
            _db_ha_url = wanted_url
        return _db_ha
