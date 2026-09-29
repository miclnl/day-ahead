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


def test_the_v1_module_no_longer_configures_logging(client):
    """v2 and the API had no logging setup of their own and relied on v1's
    import side effect, so retiring the v1 UI would have left the web server
    with no logging configuration at all."""
    import importlib

    routes = importlib.import_module("app.routes")

    assert not hasattr(routes, "logname")
    assert not hasattr(routes, "handler")
