from flask import Blueprint, abort, render_template, request, redirect, url_for
from markupsafe import escape
from dao.prog.da_report import AGGREGATE_DELTAS, Report
from dao.prog import task_state
from dao.prog import tasks as task_registry
from datetime import UTC, datetime, timedelta
from pathlib import Path
import json
import base64
import logging
import re
import unicodedata
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


def _parse_query_timestamp(raw: str, zone: ZoneInfo) -> datetime:
    """Preserve aware instants; accept only unique, existing naive local times."""
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is not None:
        return parsed.astimezone(zone)

    # A gap has no UTC round-trip; a fold has two distinct UTC instants.
    # Outside transitions both fold values identify the same instant.
    instants = set()
    for fold in (0, 1):
        instant = parsed.replace(tzinfo=zone, fold=fold).astimezone(UTC)
        if instant.astimezone(zone).replace(tzinfo=None) == parsed:
            instants.add(instant)
    if len(instants) != 1:
        raise ValueError("Ambiguous or nonexistent local timestamp")
    return instants.pop().astimezone(zone)


def _parse_data_query(args, timezone_raw: str):
    """Pure shared query validation, without constructing a backend or doing I/O."""
    if not args.get("start") or not args.get("end"):
        raise ValueError("Start and end are required")
    aggregate = args.get("aggregate")
    if aggregate not in AGGREGATE_DELTAS:
        raise ValueError("Invalid aggregate interval")
    zone = ZoneInfo(timezone_raw)
    start = _parse_query_timestamp(args["start"], zone)
    end = _parse_query_timestamp(args["end"], zone)
    # Same-zone datetime comparisons ignore fold and compare wall-clock labels.
    if end.astimezone(UTC) <= start.astimezone(UTC):
        raise ValueError("End must be later than start")
    fields = args.get("fields")
    if fields:
        fields = fields.split(",")
    return start, end, aggregate, fields


def _validated_data_query():
    try:
        return _parse_data_query(request.args, _requested_timezone())
    except (TypeError, ValueError, OverflowError):
        abort(400, description="Invalid data query")


def _sanitized_backend_error(error: Exception) -> str:
    """Bound diagnostic details to one line, without credential values/userinfo."""
    detail = f"{type(error).__name__}: {error}"
    detail = " ".join("".join(
        " " if unicodedata.category(char).startswith("C") else char for char in detail
    ).split())
    detail = re.sub(
        r"((?:[a-z][a-z0-9+.-]*:)?//)[^\s/?#]*@",
        r"\1<redacted>@", detail, flags=re.IGNORECASE,
    )
    credential_core = (
        r"(?:password|passwd|passphrase|pwd|token|secret|"
        r"(?:api|private|access|secret)[_-]?key|authorization|credential|wachtwoord)"
    )
    # A core ends the key (digits allowed), or occupies a complete segment.
    # The short, ambiguous name 'pass' is sensitive only as the entire key.
    credential_key = (
        rf"(?:pass|[\w.-]*{credential_core}[0-9]*|"
        rf"(?:[\w.-]*[_.-])?{credential_core}[0-9]*[_.-][\w.-]*)"
    )
    detail = re.sub(
        rf"((?<![\w.-]){credential_key}\s*(?:=|:(?!//))\s*)"
        r'''(?:(?:Bearer|Basic)\s+)?(?:"(?:\\.|[^"\\])*(?:"|$)|'(?:\\.|[^'\\])*(?:'|$)|[^\s,;]+)''',
        r"\1<redacted>", detail, flags=re.IGNORECASE,
    )
    detail = re.sub(
        r"(?<![\w.-])((?:Bearer|Basic)\s+)[A-Za-z0-9._~+/-]+=*(?![\w./~+=-])",
        r"\1<redacted>", detail, flags=re.IGNORECASE,
    )

    def redact_jwt(match):
        # Dotted identifiers/DSN hosts are not JWTs: require a JSON object header.
        try:
            header = match[1] + "=" * (-len(match[1]) % 4)
            header = json.loads(base64.urlsafe_b64decode(header))
        except (ValueError, UnicodeError):
            return match[0]
        return "<redacted>" if isinstance(header, dict) else match[0]

    detail = re.sub(
        r"(?<![\w./+-])([A-Za-z0-9_-]+)\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?![\w./+-])",
        redact_jwt, detail,
    )
    return detail if len(detail) <= 512 else detail[:509] + "..."


@api.route("/data/")
def data():
    """
    Retourneert in json de data
    :return: de gevraagde data in json formaat
    """
    start, end, aggregate, fields = _validated_data_query()

    def format_ts(dt, aggregate: str) -> str:
        if aggregate == "15min":
            return dt.strftime("%Y-%m-%d %H:%M")
        elif aggregate == "hour":
            return dt.strftime("%Y-%m-%d %H:00")
        else:
            return dt.strftime("%Y-%m-%d")

    try:
        data_report = Report()
        data = data_report.get_data(
            start=start, end=end, aggregate=aggregate, var_codes=fields
        )
        return [{**row, "ts": format_ts(row["ts"], aggregate)} for row in data]
    except Exception as error:
        logging.error("Data backend failed (%s)", _sanitized_backend_error(error))
        return {"error": "Internal server error"}, 500

@api.route("/run/<string:task>")
def run(task: str):
    """Queue a task for the scheduler, returning 202 without executing it.

    The plain-text response acknowledges the request, not its result. Task
    output is written by the scheduler's task and shown on the task page.
    Pending and running claims both return 409, including through aliases.
    """
    definition = task_registry.get(task)
    if definition is None:
        return f"Unknown task: {escape(task)}", 404
    canonical = task_registry.resolve(task)

    if not task_state.request(canonical, source="api"):
        holder = task_state.running_tasks().get(canonical, {})
        return (
            f"Task already running (started by "
            f"{holder.get('source', 'unknown')}): {canonical}",
            409,
            {"Content-Type": "text/plain"},
        )

    return f"Task requested: {canonical}\n", 202, {"Content-Type": "text/plain"}


@api.route("/data-sql-ha/")
def data_sql_ha():
    """
    Retourneert in json de data
    :return: de gevraagde data in json formaat
    """
    start, end, _aggregate, fields = _validated_data_query()

    try:
        data_report = Report()
        query = data_report.get_ha_data_query(
            start=start,
            end=end,
            var_codes=fields,
            step=timedelta(days=1)
        )
        return str(query)
    except Exception as error:
        logging.error("HA data query backend failed (%s)", _sanitized_backend_error(error))
        return {"error": "Internal server error"}, 500


_DATA_PATH = Path("../data")


def _read_json_or_none(path: Path):
    """A JSON artefact, or ``None`` when it is absent or unreadable.

    The accuracy page must render on a fresh install, where none of these
    files exist yet; every one of them is therefore optional and a missing
    one is an empty block on the page, not a 500.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _pv_artefacts():
    """Calibration and selection per PV installation, keyed by artefact key."""
    pv_dir = _DATA_PATH / "forecast" / "pv"
    result: dict = {}
    if not pv_dir.is_dir():
        return result
    for path in sorted(pv_dir.glob("*.json")):
        if path.name.endswith(".selection.json"):
            key = path.name[: -len(".selection.json")]
            result.setdefault(key, {})["selection"] = _read_json_or_none(path)
        else:
            key = path.stem
            result.setdefault(key, {})["calibration"] = _read_json_or_none(path)
    return result


def _recent_pairs(days: int = 2):
    """Forecast against measured for the last 48 hours, per component.

    Best-effort: this needs a live database and Home Assistant history, and
    the page is still useful without it, so any failure yields an empty
    mapping rather than an error.
    """
    try:
        from dao.forecast.evaluate import COMPONENTS, measured_series
        from dao.forecast.history import HistoryReader
        from dao.prog.da_base import DaBase
    except ImportError:
        return {}

    try:
        base = DaBase()
        if base.config is None:
            return {}
        now = datetime.now(ZoneInfo(base.time_zone))
        start = now - timedelta(days=days)
        reader = HistoryReader(base.db_ha, base.time_zone) if base.db_ha else None
        recent: dict = {}
        for component in COMPONENTS:
            rows = base.db_da.forecast_rows(
                [component], [0], int(start.timestamp()), int(now.timestamp())
            )
            if rows is None or len(rows) == 0:
                continue
            measured = measured_series(
                component, reader, base.config, base.db_da, start, now
            )
            lookup = (
                {int(moment.timestamp()): float(value) for moment, value in measured.items()}
                if measured is not None
                else {}
            )
            recent[component] = [
                {
                    "tijd": datetime.fromtimestamp(
                        int(row.target_time), tz=ZoneInfo(base.time_zone)
                    ).isoformat(),
                    "forecast": float(row.value),
                    "measured": lookup.get(int(row.target_time)),
                }
                for row in rows.itertuples()
            ]
        return recent
    except Exception:  # noqa: BLE001 - the page renders without this block
        return {}


@api.route("/accuracy/")
def accuracy():
    """Everything the accuracy page shows, in one call.

    Every block is independently optional: a fresh install has no archive,
    no calibration and no selection, and must still get valid JSON.

    The whole payload goes through json_safe: a score for a candidate that
    could not be evaluated is NaN, and Flask would serialise that as a bare
    NaN, which JSON.parse rejects -- the page would then show its "no
    archive yet" message over a perfectly good archive.
    """
    from dao.prog.config.loader import json_safe

    try:
        days = int(request.args.get("days", 28))
    except ValueError:
        days = 28

    baseload_dir = _DATA_PATH / "forecast" / "baseload"
    profile = _read_json_or_none(baseload_dir / "profile.json")

    return json_safe(
        {
            "days": days,
            "accuracy": _read_json_or_none(_DATA_PATH / "forecast" / "accuracy.json"),
            "baseload": {
                "selection": _read_json_or_none(baseload_dir / "selection.json"),
                "status": _read_json_or_none(baseload_dir / "status.json"),
                "profile_created": (profile or {}).get("created"),
            },
            "pv": _pv_artefacts(),
            "weather": _read_json_or_none(
                _DATA_PATH / "forecast" / "weather" / "status.json"
            ),
            "recent": _recent_pairs(),
        }
    )
