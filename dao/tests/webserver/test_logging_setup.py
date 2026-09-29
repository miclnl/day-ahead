"""The dashboard logs to stdout, which is what the add-on log shows.

It used to configure a TimedRotatingFileHandler on ../data/log/dashboard.log
at import time in app/routes.py, and pass it to basicConfig as the *only*
handler. Two problems:

* gunicorn runs two worker processes (gunicorn_config.py), so two processes
  held a rotating handler on one file. At midnight both try to rename it;
  one wins and the other keeps writing to a file renamed out from under it.
* replacing the default handler meant nothing the dashboard logged reached
  stdout, so none of it appeared in the Home Assistant add-on log, while
  the scheduler and the task subprocesses did show up there. Half the
  system was quietly missing from the log the user actually reads.

The setup also lived in the v1 module, so v2 and the API silently depended
on v1 being imported; it is in app/__init__.py now.
"""

import logging
from logging.handlers import TimedRotatingFileHandler

import pytest


def test_the_dashboard_installs_a_stdout_handler(client):
    import sys

    root = logging.getLogger()
    stream_handlers = [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.StreamHandler)
        and getattr(handler, "stream", None) is sys.stdout
    ]

    assert stream_handlers, "dashboard logging does not reach stdout"


def test_no_rotating_file_handler_is_installed(client):
    """Two worker processes rotating one file is the bug; there is nothing
    to rotate now that the log goes to stdout."""
    root = logging.getLogger()

    assert not [
        handler
        for handler in root.handlers
        if isinstance(handler, TimedRotatingFileHandler)
    ]


def test_configuring_twice_does_not_duplicate_the_handler(client):
    """gunicorn imports the app once per worker, but a reload or a repeated
    import inside one process must not stack handlers and log every line
    twice."""
    import importlib

    app_module = importlib.import_module("app")
    root = logging.getLogger()
    marked_before = [
        h for h in root.handlers if getattr(h, "_dao_dashboard", False)
    ]

    app_module._configure_logging()

    marked_after = [h for h in root.handlers if getattr(h, "_dao_dashboard", False)]
    assert len(marked_after) == len(marked_before) == 1


def test_logging_does_not_depend_on_the_retired_v1_module(client):
    """v2 and the API had no logging setup of their own and relied on the v1
    module configuring the root logger as an import side effect. That module
    is gone now, so the setup in app/__init__.py is the only thing keeping
    the dashboard's log alive -- and the stdout handler asserted above is
    the proof that it does."""
    import importlib
    import sys

    assert "app.routes" not in sys.modules
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.routes")


def test_a_report_in_a_request_does_not_change_the_dashboard_level(client):
    """DaBase configured the root logger unconditionally in __init__, and ~8
    routes construct a Report (a DaBase subclass) to render a page. So
    opening one report page reset the dashboard's log level to whatever
    logging_level the DAO configuration carried -- with debug that meant
    every later request logged SQLAlchemy, urllib3 and matplotlib output
    into the add-on log, as a side effect of rendering a page.

    Driven through the dashboard's real setup so the two stay consistent if
    either side changes.
    """
    from dao.prog.da_base import DaBase

    root = logging.getLogger()
    level_before = root.level
    handlers_before = root.handlers[:]

    instance = DaBase.__new__(DaBase)
    instance.log_level = logging.DEBUG

    owns = instance._configure_root_logging()

    assert owns is False, "DaBase took over the dashboard's root logger"
    assert root.level == level_before
    assert root.handlers == handlers_before
