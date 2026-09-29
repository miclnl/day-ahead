"""The task registry replaced four drifted-apart task lists with one.

These tests pin down the two things callers depend on: that every historical
key from the v1/v2 dashboards still resolves (so bookmarks and
``/api/run/<key>`` URLs keep working), and that the scheduler's allowed
actions stay in step with the registry instead of drifting again.
"""

import pytest

from dao.prog import tasks


class TestAliases:
    @pytest.mark.parametrize(
        "old_key, expected",
        [
            # v1 dashboard (bewerkingen)
            ("calc_met_debug", "calc_optimum_met_debug"),
            ("calc_zonder_debug", "calc_optimum"),
            ("get_tibber", "tibber"),
            ("get_meteo", "meteo"),
            ("get_prices", "prices"),
            ("fast_simulate", "fast_control_simulate"),
            # v2 dashboard (inline match)
            ("optimize_debug", "calc_optimum_met_debug"),
            ("optimize_regular", "calc_optimum"),
            ("update_tibber", "tibber"),
            ("update_meteo", "meteo"),
            ("update_prices", "prices"),
            ("train_ml", "train_ml_predictions"),
            # keys that were already identical everywhere
            ("calc_baseloads", "calc_baseloads"),
            ("fast_once", "fast_once"),
        ],
    )
    def test_every_historical_key_still_resolves(self, old_key, expected):
        assert tasks.resolve(old_key) == expected

    def test_an_unknown_key_resolves_to_none_so_callers_can_404(self):
        assert tasks.resolve("no_such_task") is None
        assert tasks.get("no_such_task") is None

    def test_canonical_keys_resolve_to_themselves(self):
        for key in tasks.TASKS:
            assert tasks.resolve(key) == key

    def test_no_alias_shadows_a_canonical_key(self):
        """An alias equal to some other task's canonical key would make
        /api/run/<key> ambiguous; _build_alias_map refuses it."""
        assert not set(tasks.ALIASES) & set(tasks.TASKS)


class TestSchedulerActionsStayInStep:
    def test_the_literal_matches_the_registry(self):
        """config/models/scheduler.py keeps its own Literal so the JSON
        schema carries an enum for the settings UI. That is a second list,
        so it can drift -- it already had: consolidate_data was a valid
        registry entry the scheduler configuration rejected. This test is
        what keeps the two identical from now on."""
        from typing import get_args

        from dao.prog.config.models.scheduler import SchedulerAction

        assert set(get_args(SchedulerAction)) == tasks.schedulable_functions()

    def test_consolidate_is_schedulable_now(self):
        """It has a function and is a periodic maintenance job, but was
        missing from the scheduler literal, so it could only ever be run by
        hand from the command line."""
        assert "consolidate_data" in tasks.schedulable_functions()

    def test_diagnostics_are_not_schedulable(self):
        """fast_once fights the fast control thread the scheduler already
        hosts; fast_control_simulate is a backtest nothing acts on."""
        assert tasks.TASKS["fast_once"]["schedulable"] is False
        assert tasks.TASKS["fast_control_simulate"]["schedulable"] is False

    def test_every_schedulable_function_exists_on_dabase(self):
        """The scheduler maps a configured action onto a task and then calls
        that method; a typo here would only surface at the scheduled time."""
        from dao.prog.da_base import DaBase

        for function in tasks.schedulable_functions():
            assert hasattr(DaBase, function), function


class TestBuildCmd:
    def test_parameters_are_appended_in_declared_order(self):
        cmd = tasks.build_cmd(
            "prices", {"prijzen_tot": "2026-01-03", "prijzen_start": "2026-01-01"}
        )
        assert cmd == [
            "python3", "../prog/day_ahead.py", "prices", "2026-01-01", "2026-01-03",
        ]

    def test_missing_and_empty_parameters_are_skipped(self):
        """The v1 dashboard behaved this way: 'prices' without dates fetches
        the default range instead of being handed an empty string to parse."""
        assert tasks.build_cmd("prices", {}) == [
            "python3", "../prog/day_ahead.py", "prices",
        ]
        assert tasks.build_cmd("prices", {"prijzen_start": "  "}) == [
            "python3", "../prog/day_ahead.py", "prices",
        ]

    def test_an_alias_builds_the_same_command_as_its_canonical_key(self):
        assert tasks.build_cmd("fast_simulate", {"days": "14"}) == tasks.build_cmd(
            "fast_control_simulate", {"days": "14"}
        )

    def test_days_lands_after_the_flag_it_belongs_to(self):
        assert tasks.build_cmd("fast_simulate", {"days": "14"}) == [
            "python3", "../prog/da_fast.py", "simulate", "--days", "14",
        ]

    def test_an_unknown_key_yields_none(self):
        assert tasks.build_cmd("no_such_task") is None

    def test_the_returned_list_is_a_copy(self):
        """Callers append to it (extra parameters, debug flags); mutating the
        registry's own list would corrupt every later run in the process."""
        first = tasks.build_cmd("meteo")
        first.append("--nonsense")
        assert tasks.build_cmd("meteo") == [
            "python3", "../prog/day_ahead.py", "meteo",
        ]


class TestApiTimeout:
    def test_defaults_to_the_cap_below_gunicorns_own_timeout(self):
        assert tasks.api_timeout_s("meteo") == tasks.API_RUN_TIMEOUT_S
        assert tasks.API_RUN_TIMEOUT_S < 120  # gunicorn_config.py timeout

    def test_a_task_cannot_raise_its_own_cap_above_the_gunicorn_limit(self):
        """The v1 dashboard gave fast_simulate timeout_s=600 with a comment
        claiming it matched the gunicorn timeout. It did not: gunicorn kills
        the worker at 120 s, so the 600 s cap could never be reached and the
        run ended as a killed worker plus an orphaned child instead of a
        timeout message."""
        assert tasks.api_timeout_s("fast_control_simulate") <= tasks.API_RUN_TIMEOUT_S

    def test_an_unknown_key_still_gets_a_cap(self):
        assert tasks.api_timeout_s("no_such_task") == tasks.API_RUN_TIMEOUT_S


class TestRegistryShape:
    def test_every_task_has_the_fields_callers_read(self):
        for key, task in tasks.TASKS.items():
            assert task["name"], key
            assert task["cmd"], key
            assert task["file_name"], key
            assert "function" in task, key
            assert isinstance(task["schedulable"], bool), key
            assert isinstance(task["aliases"], tuple), key

    def test_log_file_prefixes_are_unique(self):
        """Two tasks sharing a prefix would make the dashboard's "newest
        matching log file" lookup pick up the other task's output."""
        prefixes = [task["file_name"] for task in tasks.TASKS.values()]
        assert len(prefixes) == len(set(prefixes))
