"""The documented HTTP API: /api/run and /api/report.

These two endpoints are documented in DOCS.md with worked Home Assistant
examples -- a ``rest_command`` that starts a calculation, and REST sensors
that read report figures -- so they are wired into people's own
configurations. They lived in the v1 interface module, but they are the
public data API rather than part of that interface, and the security
problems the review found in v1 (path traversal and reflected XSS in the
settings editor) were in its pages, not here. Removing the interface
therefore should not take them with it.

They are reachable only with ``allow_direct_access`` enabled, since
everything else is refused by the ingress guard in ``app/__init__.py``;
that is what DOCS.md already tells people.
"""

import logging

from flask import Blueprint, abort, request
from markupsafe import escape

from dao.prog import task_state
from dao.prog import tasks as task_registry
from dao.prog.da_report import Report

public_api = Blueprint("public_api", __name__)

#: Where options.json and the database live, relative to the web server's
#: working directory. Matches the other modules.
app_datapath = "../data/"


@public_api.route("/api/run/<string:bewerking>", methods=["GET", "POST"])
def run_api(bewerking: str):
    """Ask the scheduler to run a task.

    This used to run the task inside the request, which meant it had to
    finish within gunicorn's per-request timeout or the worker was killed
    and the child orphaned. Tasks are executed by the scheduler process now
    (see dao/prog/task_state.py), so this records a request and answers
    straight away: an optimisation that takes minutes now actually
    completes, and the ``rest_command`` in DOCS.md no longer blocks while it
    runs.

    The response is plain text rather than the HTML page it used to render.
    The documented use is a fire-and-forget ``rest_command`` which does not
    read the body, and the task's output belongs in its log file, which the
    dashboard shows.
    """
    task = task_registry.get(bewerking)
    if task is None:
        # Never echo the path segment back: it is attacker controlled.
        abort(404)
    canonical = task_registry.resolve(bewerking)

    if not task_state.request(canonical, source="api"):
        holder = task_state.running_tasks().get(canonical, {})
        return (
            f"Taak {task['name']} draait al "
            f"(gestart door {holder.get('source', 'onbekend')}).\n",
            409,
            {"Content-Type": "text/plain; charset=utf-8"},
        )

    if task_state.scheduler_alive() is False:
        logging.warning(
            f"Taak {canonical} aangevraagd via de api terwijl de planner niet "
            f"lijkt te draaien; de aanvraag verloopt als hij niet wordt opgepakt."
        )

    return (
        f"Taak {task['name']} aangevraagd.\n",
        202,
        {"Content-Type": "text/plain; charset=utf-8"},
    )


@public_api.route("/api/report/<string:fld>/<string:periode>", methods=["GET"])
def api_report(fld: str, periode: str):
    """Report figures as JSON, for use as a Home Assistant REST sensor."""
    cumulate_raw = request.args.get("cumulate")
    cumulate = False
    if cumulate_raw is not None:
        try:
            cumulate = int(cumulate_raw) == 1
        except ValueError:
            cumulate = False

    report = Report(app_datapath + "/options.json")
    result = report.get_api_data(
        str(escape(fld)), str(escape(periode)), cumulate=cumulate
    )
    return result, {"Content-Type": "application/json"}
