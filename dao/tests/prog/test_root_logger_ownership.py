"""DaBase only configures the root logger when nothing else has.

It used to call logging.getLogger().setLevel() unconditionally in
__init__, twice: once with the default and once with the level from the
DAO configuration. That is right for the command line processes, which
have no logging set up when they build a DaBase.

Inside the web server it was not. The dashboard configures the root logger
in app/__init__.py, and ~8 routes construct a Report (a DaBase subclass) to
render a page. So opening one report page reset the whole dashboard's log
level to whatever logging_level the DAO configuration carried -- with
logging_level: debug that meant every later request logged debug output
from SQLAlchemy, urllib3 and matplotlib into the add-on log, as a side
effect of rendering a page.
"""

import logging

import pytest

from dao.prog.da_base import DaBase


@pytest.fixture
def bare_instance():
    """A DaBase with only what _configure_root_logging touches.

    __init__ needs a live configuration, database and Home Assistant, so
    the method under test is exercised on its own.
    """
    instance = DaBase.__new__(DaBase)
    instance.log_level = logging.DEBUG
    return instance


@pytest.fixture
def fresh_logger():
    """A logger with no handlers, standing in for the root logger of a
    freshly started command line process.

    A throwaway logger rather than the real root one: pytest's own logging
    plugin attaches a capture handler to the root logger for the duration of
    every test, which is itself the "somebody already configured this" case
    and would make the unconfigured scenario impossible to reach in-process.

    Named loggers live in the logging module's registry for the whole
    session, so it is emptied on the way in as well as on the way out.
    """
    logger = logging.getLogger("dao.tests.root_logger_ownership")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    yield logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)


class TestCommandLineProcess:
    def test_it_takes_ownership_when_nothing_is_configured(
        self, bare_instance, fresh_logger
    ):
        owns = bare_instance._configure_root_logging(fresh_logger)

        assert owns is True
        assert fresh_logger.handlers, "no handler installed"
        assert fresh_logger.level == logging.DEBUG

    def test_the_configured_level_is_applied(self, bare_instance, fresh_logger):
        bare_instance.log_level = logging.WARNING

        bare_instance._configure_root_logging(fresh_logger)

        assert fresh_logger.level == logging.WARNING


class TestInsideAHostApplication:
    def test_it_leaves_an_already_configured_root_logger_alone(
        self, bare_instance, fresh_logger
    ):
        """This is the web server case: app/__init__.py installed a handler
        at import time, long before any request constructs a Report."""
        host_handler = logging.StreamHandler()
        fresh_logger.addHandler(host_handler)
        fresh_logger.setLevel(logging.INFO)

        owns = bare_instance._configure_root_logging(fresh_logger)

        assert owns is False
        assert fresh_logger.level == logging.INFO
        assert fresh_logger.handlers == [host_handler]

    def test_repeated_construction_does_not_stack_handlers(
        self, bare_instance, fresh_logger
    ):
        """Eight routes construct a Report; a handler per construction would
        multiply every log line by the number of pages visited."""
        host_handler = logging.StreamHandler()
        fresh_logger.addHandler(host_handler)

        for _ in range(5):
            bare_instance._configure_root_logging(fresh_logger)

        assert fresh_logger.handlers == [host_handler]

    def test_the_second_setlevel_in_init_is_gated_on_ownership(self):
        """__init__ calls setLevel again once the configuration has been
        read, because only then is the real level known. That call is gated
        on the same ownership flag; without the gate the first fix would be
        undone a few lines later."""
        import inspect

        source = inspect.getsource(DaBase.__init__)
        for line_number, line in enumerate(source.splitlines()):
            if "setLevel(self.log_level)" in line:
                preceding = source.splitlines()[max(0, line_number - 1)]
                assert "_owns_root_logger" in preceding, (
                    f"unguarded setLevel in __init__: {line.strip()!r}"
                )
