from flask import Blueprint, abort, render_template, request, redirect, url_for
from markupsafe import escape
from dao.prog.da_report import Report
from subprocess import TimeoutExpired, run as subprocess_run
from dao.prog import task_state
from dao.prog import tasks as task_registry
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, available_timezones

api = Blueprint("api", __name__)

_DEFAULT_TIMEZONE = "Europe/Amsterdam"


def _requested_timezone() -> str:
    """The 'timezone' query parameter, or the default when it is absent.

    ``x if None else y`` is always y (None is falsy), so this used to return
    the default unconditionally and every request's own timezone parameter
    was silently ignored.
    """
    raw = request.args.get("timezone") or _DEFAULT_TIMEZONE
    if raw not in available_timezones():
        abort(400, description=f"Unknown timezone: {raw!r}")
    return raw

@api.route("/data/")
def data():
    """
    Retourneert in json de data
    :return: de gevraagde data in json formaat
    """
    data_report = Report()
    start = request.args.get('start')
    end = request.args.get('end')
    aggregate = request.args.get('aggregate')
    fields = request.args.get('fields')

    if fields:
        fields = fields.split(",")

    timezone_raw = _requested_timezone()

    try:
        data = data_report.get_data(
            start=datetime.fromisoformat(start).replace(tzinfo=ZoneInfo(timezone_raw)),
            end=datetime.fromisoformat(end).replace(tzinfo=ZoneInfo(timezone_raw)),
            aggregate=aggregate,
            var_codes=fields,
        )

    except Exception as e:
        return {"error": str(e)}, 500

    def format_ts(dt, aggregate: str) -> str:
        if aggregate == "15min":
            return dt.strftime("%Y-%m-%d %H:%M")
        elif aggregate == "hour":
            return dt.strftime("%Y-%m-%d %H:00")
        else:
            return dt.strftime("%Y-%m-%d")

    data = [
        {**row, "ts": format_ts(row["ts"], aggregate)}
        for row in data
    ]

    return data

@api.route("/run/<string:task>")
def run(task: str):
    definition = task_registry.get(task)
    if definition is None:
        return f"Unknown task: {escape(task)}", 404
    canonical = task_registry.resolve(task)

    # This endpoint runs the task inside the request, so it needs the same
    # claim as every other starter (otherwise it happily runs a second
    # optimisation alongside the one cron just started) and a cap below
    # gunicorn's own per-request timeout. It had neither: without a timeout
    # gunicorn killed the worker at 120 s and the child was orphaned, with
    # no output returned at all.
    if not task_state.claim(canonical, source="api"):
        holder = task_state.running_tasks().get(canonical, {})
        return (
            f"Task already running (started by "
            f"{holder.get('source', 'unknown')}): {canonical}",
            409,
            {"Content-Type": "text/plain"},
        )

    timeout_s = task_registry.api_timeout_s(canonical)
    status = "error"
    returncode = None
    try:
        proc = subprocess_run(
            definition["cmd"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        log_content = proc.stdout + proc.stderr
        returncode = proc.returncode
        status = "done" if returncode == 0 else "error"
    except TimeoutExpired as exception:
        log_content = (
            f"Task {canonical} aborted after {timeout_s}s timeout.\n"
            f"Use the task page for tasks that take longer: those run in "
            f"the background without this limit.\n"
            f"stdout so far:\n{exception.stdout or ''}\n"
            f"stderr so far:\n{exception.stderr or ''}\n"
        )
    finally:
        task_state.release(canonical, status, returncode=returncode)

    return log_content, {"Content-Type": "text/plain"}


@api.route("/data-sql-ha/")
def data_sql_ha():
    """
    Retourneert in json de data
    :return: de gevraagde data in json formaat
    """
    data_report = Report()
    start = request.args.get('start')
    end = request.args.get('end')
    aggregate = request.args.get('aggregate')
    fields = request.args.get('fields')

    if fields:
        fields = fields.split(",")

    timezone_raw = _requested_timezone()

    query = data_report.get_ha_data_query(
            start=datetime.fromisoformat(start).replace(tzinfo=ZoneInfo(timezone_raw)),
            end=datetime.fromisoformat(end).replace(tzinfo=ZoneInfo(timezone_raw)),
            var_codes=fields,
            step=timedelta(days=1)
        )

    return str(query)