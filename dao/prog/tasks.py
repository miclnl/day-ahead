"""The task registry: one list of everything that can be run as a task.

Before this module the same set of tasks was declared in four places, each
with its own naming scheme, and they had drifted apart:

* ``DaBase.generate_tasks()`` (11 tasks, keys like ``calc_optimum``)
* ``bewerkingen`` in the v1 dashboard (9 tasks, keys like
  ``calc_zonder_debug``)
* an inline ``match`` in the v2 dashboard (9 tasks, keys like
  ``optimize_regular``)
* the ``SchedulerAction`` literal in ``config/models/scheduler.py`` (9
  function names)

The drift was not cosmetic: ``clean``, ``consolidate`` and
``forecast_accuracy`` existed only in the first list, so neither dashboard
could start them; ``fast_once`` existed only in the two dashboards, so it
could not be scheduled; and ``consolidate_data`` was missing from the
scheduler literal, so a perfectly valid registry entry was rejected by
configuration validation.

Every task is declared once here. The old keys live on as ``aliases`` so
existing bookmarks, form values and ``/api/run/<key>`` URLs keep working.

Deliberately free of heavy imports (no pandas, no Home Assistant client, no
database): the web server imports this to render its task list and must not
pay for the whole optimiser to do so.
"""

import logging
import os
import signal
from typing import Any, Iterable, Optional

_PY = "python3"
_DAY_AHEAD = "../prog/day_ahead.py"
_DA_FAST = "../prog/da_fast.py"

#: Wall-clock cap for the synchronous ``/api/run`` endpoints. Those run the
#: task inside the request, and gunicorn kills a worker that does not answer
#: within its own ``timeout`` (120 s, see webserver/gunicorn_config.py).
#: Staying under that turns "worker killed, child orphaned, no output" into a
#: normal timeout message with the output collected so far. Tasks that
#: legitimately take longer belong on the background path (the dashboard's
#: task page), not on this one.
API_RUN_TIMEOUT_S = 110

#: name        human readable label, shown in both dashboards
#: cmd         argv of the subprocess that runs the task
#: function    the DaBase method, used by the scheduler to map a configured
#:             action onto a task and by run_task_function to call it
#:             in-process. None for tasks that only exist as a subprocess.
#: file_name   prefix of the task's log file in ../data/log
#: parameters  form fields appended to cmd, in this order, when present
#: timeout_s   optional override of API_RUN_TIMEOUT_S, never above it
#: schedulable whether the task may appear in the scheduler configuration
#: aliases     historical keys from the v1/v2 dashboards
TASKS: dict[str, dict[str, Any]] = {
    "calc_optimum_met_debug": {
        "name": "Optimaliseringsberekening met debug",
        "cmd": [_PY, _DAY_AHEAD, "debug", "calc"],
        "function": "calc_optimum_met_debug",
        "file_name": "calc_debug",
        "schedulable": True,
        "aliases": ("calc_met_debug", "optimize_debug"),
    },
    "calc_optimum": {
        "name": "Optimaliseringsberekening zonder debug",
        "cmd": [_PY, _DAY_AHEAD, "calc"],
        "function": "calc_optimum",
        "file_name": "calc",
        "schedulable": True,
        "aliases": ("calc_zonder_debug", "optimize_regular"),
    },
    "tibber": {
        "name": "Verbruiksgegevens bij Tibber ophalen",
        "cmd": [_PY, _DAY_AHEAD, "tibber"],
        "function": "get_tibber_data",
        "file_name": "tibber",
        "schedulable": True,
        "aliases": ("get_tibber", "update_tibber"),
    },
    "meteo": {
        "name": "Meteoprognoses ophalen",
        "cmd": [_PY, _DAY_AHEAD, "meteo"],
        "function": "get_meteo_data",
        "file_name": "meteo",
        "schedulable": True,
        "aliases": ("get_meteo", "update_meteo"),
    },
    "prices": {
        "name": "Day ahead prijzen ophalen",
        "cmd": [_PY, _DAY_AHEAD, "prices"],
        "function": "get_day_ahead_prices",
        "file_name": "prices",
        "parameters": ("prijzen_start", "prijzen_tot"),
        "schedulable": True,
        "aliases": ("get_prices", "update_prices"),
    },
    "calc_baseloads": {
        "name": "Bereken de baseloads",
        "cmd": [_PY, _DAY_AHEAD, "calc_baseloads"],
        "function": "calc_baseloads",
        "file_name": "baseloads",
        "schedulable": True,
        "aliases": (),
    },
    "clean": {
        "name": "Bestanden opschonen",
        "cmd": [_PY, _DAY_AHEAD, "clean_data"],
        "function": "clean_data",
        "file_name": "clean",
        "schedulable": True,
        "aliases": (),
    },
    "train_ml_predictions": {
        "name": "ML modellen trainen",
        "cmd": [_PY, _DAY_AHEAD, "train"],
        "function": "train_ml_predictions",
        "file_name": "train",
        "schedulable": True,
        "aliases": ("train_ml",),
    },
    "consolidate": {
        "name": "Verbruik/productie consolideren",
        "cmd": [_PY, _DAY_AHEAD, "consolidate"],
        "function": "consolidate_data",
        "file_name": "consolidate",
        "schedulable": True,
        "aliases": (),
    },
    "forecast_accuracy": {
        "name": "Prognosefout rapporteren",
        "cmd": [_PY, _DAY_AHEAD, "accuracy"],
        "function": "forecast_accuracy",
        "file_name": "accuracy",
        "schedulable": True,
        "aliases": (),
    },
    "fast_once": {
        "name": "Snelle regellaag: één regelcyclus",
        "cmd": [_PY, _DA_FAST, "once"],
        # No DaBase method: this one only ever ran as a subprocess from the
        # dashboard. Scheduling it would fight the fast control thread that
        # the scheduler process already hosts, so it stays on-demand only.
        "function": None,
        "file_name": "fast_once",
        "schedulable": False,
        "aliases": (),
    },
    "fast_control_simulate": {
        "name": "Snelle regellaag: terugrekenen op historie",
        "cmd": [_PY, _DA_FAST, "simulate", "--days"],
        "function": "fast_control_simulate",
        "file_name": "fast_simulate",
        "parameters": ("days",),
        # A backtest over a long history, started by hand when you want to
        # look at it. Nothing acts on the result, so there is no point in
        # having cron run it.
        "schedulable": False,
        "aliases": ("fast_simulate",),
    },
}


def _build_alias_map() -> dict[str, str]:
    """Every historical key mapped onto its canonical key.

    Raises on a duplicate rather than silently letting one entry win: two
    tasks claiming the same alias would make ``/api/run/<alias>`` ambiguous,
    and which one you got would depend on dict order.
    """
    aliases: dict[str, str] = {}
    for key, task in TASKS.items():
        for alias in task.get("aliases", ()):
            if alias in TASKS:
                raise ValueError(
                    f"Alias {alias!r} of task {key!r} collides with a canonical task key"
                )
            if alias in aliases:
                raise ValueError(
                    f"Alias {alias!r} is claimed by both {aliases[alias]!r} and {key!r}"
                )
            aliases[alias] = key
    return aliases


ALIASES: dict[str, str] = _build_alias_map()


def resolve(key: str) -> Optional[str]:
    """The canonical task key for *key*, which may be an alias.

    Returns None for an unknown key, so callers can answer 404 rather than
    raise.
    """
    if key in TASKS:
        return key
    return ALIASES.get(key)


def get(key: str) -> Optional[dict[str, Any]]:
    """The task definition for *key* (canonical or alias), or None."""
    canonical = resolve(key)
    return TASKS[canonical] if canonical is not None else None


def schedulable_functions() -> frozenset[str]:
    """The DaBase method names that may appear as a scheduler action.

    ``config/models/scheduler.py`` keeps its own ``Literal`` so the JSON
    schema still carries an enum for the settings UI; a test asserts that
    literal and this set stay identical.
    """
    return frozenset(
        task["function"]
        for task in TASKS.values()
        if task.get("schedulable") and task.get("function")
    )


def task_key_for_function(action: str) -> Optional[str]:
    """The canonical task key whose ``function`` is *action*."""
    for key, task in TASKS.items():
        if task.get("function") == action:
            return key
    return None


def build_cmd(key: str, values: Optional[dict[str, Any]] = None) -> Optional[list[str]]:
    """The argv for *key*, with its declared parameters appended in order.

    Empty and missing values are skipped, matching what the v1 dashboard did
    with its form fields: ``prices`` without dates fetches the default range
    rather than being handed an empty string to parse.
    """
    task = get(key)
    if task is None:
        return None
    cmd = list(task["cmd"])
    for parameter in task.get("parameters", ()):
        value = (values or {}).get(parameter)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            cmd.append(text)
    return cmd


def api_timeout_s(key: str) -> int:
    """Wall-clock cap for running *key* synchronously in a web request."""
    task = get(key)
    if task is None:
        return API_RUN_TIMEOUT_S
    return min(int(task.get("timeout_s", API_RUN_TIMEOUT_S)), API_RUN_TIMEOUT_S)


def kill_process_group(proc) -> None:
    """Kill a task subprocess and everything it spawned.

    Tasks are started with ``start_new_session=True``, which gives them
    their own process group. A bare ``proc.kill()`` only reaches that one
    process; if the script it runs forks or execs a helper of its own, that
    helper would be left running as an orphan after a cancel.

    Lives here rather than in either dashboard because both of them cancel
    tasks and the two copies had already started to differ.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass  # already exited
    except OSError as exception:
        logging.warning(f"Kon procesgroep van pid {proc.pid} niet stoppen: {exception}")


def menu_entries(keys: Optional[Iterable[str]] = None) -> dict[str, dict[str, Any]]:
    """Task definitions keyed by canonical key, for rendering a task list.

    Without *keys* every task is returned; pass a subset to keep a dashboard
    showing only the tasks it wants, in the order given.
    """
    if keys is None:
        return dict(TASKS)
    selected = {}
    for key in keys:
        canonical = resolve(key)
        if canonical is not None:
            selected[canonical] = TASKS[canonical]
    return selected
