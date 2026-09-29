"""An expired plan keeps saying so.

Reported from a live add-on running in shadow mode: the fast control event
log held exactly one entry, two days old,

    setpoint_change  plan_expired  0.0 W  0.0 W

and nothing since. The layer was running fine. The event log only records a
setpoint *change*, so a plan that stays expired produces one entry when
everything first goes to 0 W and then nothing, and the max_plan_age warning
fires once and is only re-armed when a new plan loads. The result was a
layer parked at 0 W for days with no trace anywhere of why.

Parking at 0 W is the right call -- without a plan there is no reference to
correct against -- but it has to stay visible.
"""

import logging
import time
from types import SimpleNamespace

import pytest

from dao.prog.fastctrl import runner as runner_module
from dao.prog.fastctrl.runner import EXPIRED_WARN_EVERY_S, FastControlRunner


@pytest.fixture
def instance():
    """A runner with only what _warn_if_plan_expired touches.

    __init__ needs a live Home Assistant and a configuration, so the method
    under test is exercised on its own.
    """
    obj = FastControlRunner.__new__(FastControlRunner)
    obj._expired_warned_at = 0.0
    obj.plan_path = "../data/fast_plan.json"
    return obj


def a_plan(created_ts):
    return SimpleNamespace(
        created_ts=created_ts,
        age=lambda now: max(0.0, now - created_ts),
    )


def expired():
    return SimpleNamespace(reason="plan_expired")


def normal():
    return SimpleNamespace(reason="follow_plan")


class TestItWarns:
    def test_an_expired_plan_warns(self, instance, caplog):
        now = time.time()
        plan = a_plan(now - 2 * 24 * 3600)

        with caplog.at_level(logging.WARNING):
            instance._warn_if_plan_expired(expired(), plan, now)

        assert "dekt" in caplog.text
        assert "0 W" in caplog.text

    def test_the_message_names_the_age_and_where_to_look(self, instance, caplog):
        now = time.time()
        plan = a_plan(now - 48 * 3600)

        with caplog.at_level(logging.WARNING):
            instance._warn_if_plan_expired(expired(), plan, now)

        assert "48.0 uur oud" in caplog.text
        assert "fast_plan.json" in caplog.text
        # Points at the log line the optimiser writes, so the operator can
        # find out why the plan is not being refreshed.
        assert "Plan voor de snelle regellaag" in caplog.text


class TestItRepeats:
    def test_it_does_not_warn_on_every_tick(self, instance, caplog):
        """The loop ticks every 10-20 seconds; warning each time would bury
        the add-on log."""
        now = time.time()
        plan = a_plan(now - 3600)

        with caplog.at_level(logging.WARNING):
            for offset in range(0, 60, 10):
                instance._warn_if_plan_expired(expired(), plan, now + offset)

        assert caplog.text.count("dekt") == 1

    def test_it_warns_again_after_the_interval(self, instance, caplog):
        """This is the actual fix: the old code warned once and then stayed
        silent for as long as the condition lasted."""
        now = time.time()
        plan = a_plan(now - 3600)

        with caplog.at_level(logging.WARNING):
            instance._warn_if_plan_expired(expired(), plan, now)
            instance._warn_if_plan_expired(
                expired(), plan, now + EXPIRED_WARN_EVERY_S + 1
            )

        assert caplog.text.count("dekt") == 2

    def test_the_throttle_resets_once_the_plan_is_usable_again(
        self, instance, caplog
    ):
        """So that a plan expiring a second time warns straight away rather
        than waiting out the remainder of the interval."""
        now = time.time()
        plan = a_plan(now - 3600)

        with caplog.at_level(logging.WARNING):
            instance._warn_if_plan_expired(expired(), plan, now)
            instance._warn_if_plan_expired(normal(), plan, now + 10)
            instance._warn_if_plan_expired(expired(), plan, now + 20)

        assert caplog.text.count("dekt") == 2


class TestItStaysQuietOtherwise:
    def test_a_usable_plan_does_not_warn(self, instance, caplog):
        now = time.time()

        with caplog.at_level(logging.WARNING):
            instance._warn_if_plan_expired(normal(), a_plan(now), now)

        assert "dekt" not in caplog.text

    def test_other_reasons_do_not_warn(self, instance, caplog):
        now = time.time()

        with caplog.at_level(logging.WARNING):
            for reason in ("sensor_stale", "plan_stale", "deadband", "override"):
                instance._warn_if_plan_expired(
                    SimpleNamespace(reason=reason), a_plan(now), now
                )

        assert "dekt" not in caplog.text


def test_the_interval_is_long_enough_to_be_readable():
    """Half an hour: often enough that the condition cannot be missed, rare
    enough that a week of it does not drown the log."""
    assert 600 <= EXPIRED_WARN_EVERY_S <= 3600
