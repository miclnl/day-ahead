import time, os, fnmatch, re, datetime, json, logging
from flask import Blueprint, abort, render_template, request, redirect, url_for

from dao.prog.version import __version__
from pathlib import Path
from dao.prog.da_report import Report
from dao.prog import task_state
from dao.prog import tasks as task_registry
from dao.prog.config.loader import (
    ConfigurationLoader,
    atomic_write_text,
    set_fast_control_mode,
    validate_config_data,
)

v2 = Blueprint("v2", __name__)

@v2.context_processor
def inject_data():
    return {
        "version": __version__,
        "vite_tags": vite_tags("assets/main.js")
    }

# globals
# The data directory lives outside Flask's static folder on purpose: it holds
# secrets.json and the database. Graphs are served through the /images route.
app_datapath = "../data/"

VITE_DEV_SERVER = "http://localhost:5173"
VITE_MANIFEST = Path("app/static/build/.vite/manifest.json")

def vite_tags(entry: str) -> str:
    if os.getenv("VITE_DEV") == "1":
        return f'<script type="module" src="{VITE_DEV_SERVER}/@vite/client"></script>' \
               f'<script type="module" src="{VITE_DEV_SERVER}/{entry}"></script>'

    if not VITE_MANIFEST.exists():
        raise RuntimeError("Vite manifest not found. Run 'npm run build' in the Vite server directory.")

    with VITE_MANIFEST.open() as f:
        manifest = json.load(f)

    asset = manifest[entry]

    tags = []

    for css in asset.get("css", []):
        href = url_for("static", filename=f"build/{css}")
        tags.append(f'<link rel="stylesheet" href="{href}">')

    src = url_for("static", filename=f'build/{asset["file"]}')
    tags.append(f'<script type="module" src="{src}"></script>')

    return "\n".join(tags)


def get_file_list_with_ts(path: str, pattern: str) -> list:
    """
    get a time-ordered file list with name and timestamp from filename
    :parameter path: folder
    :parameter pattern: wildcards to search for
    """
    flist = []
    for f in os.listdir(path):
        if fnmatch.fnmatch(f, pattern):
            # Extract timestamp from filename (e.g. calc_2026-02-17__08-45.png) because datetime picker works with
            # absolute timestamps and the file modification date might differ from the timestamp in the filename, which is the intended reference time for the user
            m = re.search(r"(\d{4}-\d{2}-\d{2})__(\d{2})[:-](\d{2})(?:[:-](\d{2}))?", f)
            if m:
                try:
                    seconds = m.group(4) or "00"
                    dt_str = f"{m.group(1)} {m.group(2)}:{m.group(3)}:{seconds}"
                    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    timestamp = dt.timestamp()  # Local time as epoch
                    flist.append({
                        "name": f,
                        "time": timestamp,
                    })
                except (ValueError, OSError):
                    # Fallback to mtime if filename parsing fails
                    fullname = os.path.join(path, f)
                    flist.append({
                        "name": f,
                        "time": int(os.path.getmtime(fullname)),
                    })

    flist.sort(key=lambda x: (x["time"], x["name"].lower()))
    return flist


def get_closest_index_from_list(flist: list, ts: float) -> int:
    return min(
        range(len(flist)),
        key=lambda i: abs(flist[i].get("time", 0) - ts)
    )


#: How often a running task refreshes its claim and checks for a cancel.
#: Well under task_state.STALE_AFTER_S.
HEARTBEAT_S = 1


def get_task_state() -> dict:
    """The task this dashboard's task page reports on.

    Claims are per task (see dao/prog/task_state.py) but this page shows one
    task at a time: the running one, or the most recently finished when
    nothing runs. Returns the flat shape the template and the poll endpoint
    already expected.
    """
    state = task_state.read()
    running = state["running"]
    if running:
        key = max(running, key=lambda k: running[k].get("started") or 0)
        entry = running[key]
        return {
            "status": "cancelled" if entry.get("cancel") else "running",
            "task": key,
            "logfile": entry.get("logfile"),
            "started": entry.get("started"),
            "returncode": None,
        }
    finished = state["last"]
    if finished:
        key = max(finished, key=lambda k: finished[k].get("finished") or 0)
        entry = finished[key]
        return {
            "status": entry.get("status", "idle"),
            "task": key,
            "logfile": entry.get("logfile"),
            "started": entry.get("started"),
            "returncode": entry.get("returncode"),
        }
    return {"status": "idle", "task": None, "logfile": None, "started": None}


def log_chart(datapath: str, pattern: str):
    #  By design; the get_file_list() is called over and over again to ensure an accurate reflection of the files
    flist = get_file_list_with_ts(app_datapath + datapath, pattern)
    last_index = len(flist) - 1
    if len(flist) == 0:
        return None

    show_index = request.args.get("i")
    if show_index is None:
        show_index = last_index
    else:
        show_index = int(show_index)

    show_index = max(0, min(show_index, last_index))

    rq_ts = request.args.get("ts")
    if rq_ts is not None:
        show_index = get_closest_index_from_list(flist, datetime.datetime.fromisoformat(rq_ts).timestamp())

    first_index = 0
    prev_index = max(0, show_index - 1)
    next_index = min(last_index, show_index + 1)
    ffprev_index = max(0,
                       get_closest_index_from_list(flist, flist[show_index]["time"] - (6 * 3600)))  # Subtract 6 hours
    ffnext_index = min(last_index,
                       get_closest_index_from_list(flist, flist[show_index]["time"] + (6 * 3600)))  # Add 6 hours
    show_ts = datetime.datetime.fromtimestamp(flist[show_index]["time"]).isoformat()

    return {
        "filename": flist[show_index]["name"],
        "first_index": first_index,
        "ffprev_index": ffprev_index,
        "prev_index": prev_index,
        "show_index": show_index,
        "next_index": next_index,
        "ffnext_index": ffnext_index,
        "last_index": last_index,
        "show_ts": show_ts,
    }


def get_solar_items_with_ml():
    loader = ConfigurationLoader(Path(app_datapath + "options.json"))
    config = loader.load_and_validate()
    if config is None:
        return {}

    solar_options = [
        *config.solar,
        *(
            solar_option
            for battery_option in config.battery
            for solar_option in battery_option.solar
        ),
    ]

    return {
        solar_option.name or "default": solar_option
        for solar_option in solar_options
        if solar_option.ml_prediction
    }


@v2.route("/")
@v2.route("/chart")
def chart():
    kwargs = log_chart("images/", "*.png")
    if kwargs is None:
        return render_template("v2/no-task.html", )

    kwargs["image"] = url_for("image", name=kwargs["filename"])
    return render_template(
        "v2/chart.html",
        **kwargs
    )


@v2.route("/log")
def log():
    kwargs = log_chart("log/", "*.log")
    if kwargs is None:
        return render_template("v2/no-task.html", )

    log_file = app_datapath + "log/" + kwargs["filename"]
    with open(log_file, "r") as f:
        kwargs["logdata"] = f.read()

    return render_template(
        "v2/log.html",
        **kwargs
    )


@v2.route("/delete-file", methods=["POST"])
def delete_file():
    post_data = request.form.to_dict(flat=True)

    action = post_data.get("action", "")
    if action not in ("chart", "log"):
        abort(400)
    target = post_data.get("file", "")
    if post_data.get("confirm") == "1" and re.match(
        r"^(images|log)/[\w.\-]+\.(log|png)$", target
    ):
        try:
            os.remove(app_datapath + target)
        except FileNotFoundError:
            pass

    return redirect(url_for("v2." + action, i=post_data.get("show_index", 0)))


@v2.route("/tasks")
def tasks():
    return render_template("v2/tasks.html")

@v2.route("/task-cancel")
def task_cancel():
    try:
        for key in task_state.running_tasks():
            task_state.request_cancel(key)
        return render_template("v2/tasks.html")
    except Exception:
        logging.exception("Afbreken van de taak is mislukt")
        return "Error cancelling task", 500


@v2.route("/task-exec", methods=["POST"])
def task_exec():
    requested = request.form.to_dict().get("task", "")
    canonical = task_registry.resolve(requested)
    if canonical is None:
        return "Invalid action", 400

    values = {"days": request.form.get("days", "14")}

    # Hand the work to the scheduler process instead of running it here.
    # This request is served by a gunicorn worker that gets recycled on
    # every configuration change (the watchdog sends gunicorn a HUP), which
    # used to kill the thread doing the work while its subprocess carried
    # on, leaving nothing to record the result. The request is a claim in
    # the "pending" state, so it excludes cron and the other dashboard from
    # the same task straight away.
    if not task_state.request(canonical, source="dashboard", parameters=values):
        holder = task_state.running_tasks().get(canonical, {})
        return (
            f"Taak draait al (gestart door {holder.get('source', 'onbekend')}): "
            f"{task_registry.get(canonical)['name']}",
            409,
        )

    if task_state.scheduler_alive() is False:
        logging.warning(
            f"Taak {canonical} aangevraagd terwijl de planner niet lijkt te "
            f"draaien; de aanvraag verloopt als hij niet wordt opgepakt."
        )

    return redirect(url_for('v2.task_state'))


# endpoint="task_state" keeps url_for('v2.task_state') and the two
# templates using it working; the function itself is renamed because
# task_state is also the module holding the shared claims (imported above)
# and a route function of that name would shadow it for the whole module.
@v2.route("/task-state", methods=["GET"], endpoint="task_state")
def task_state_page():
    current_state = get_task_state()

    content = "No logfile available"
    started = None
    seconds_running = None
    status = current_state["status"]

    if status != "idle":
        started_timestamp = current_state.get("started")
        started = (
            datetime.datetime.fromtimestamp(started_timestamp)
            if started_timestamp is not None
            else None
        )

        last_update = (
            time.time()
            if status == "running"
            else current_state.get("last_update")
        )

        seconds_running = None
        if last_update is not None and started is not None:
            last_update = datetime.datetime.fromtimestamp(last_update)
            seconds_running = int((last_update - started).total_seconds())

        if current_state.get("logfile") is None:
            content = "No log data available yet"
        else:
            try:
                with open(current_state["logfile"], "r") as f:
                    content = f.read()
            except:
                content = "Could not read logfile"

    headers = {
        "HX-Push-Url": "false",
    }

    return render_template(
        "v2/task-status.html",
        task_state=current_state,
        started=started,
        seconds_running=seconds_running,
        content=content,
    ), headers


def reports_gen(subject: str, view: str, period: str, solar_item=None, date: datetime.datetime = None):
    report = Report(app_datapath + "/options.json")
    prognose = period in ["vandaag en morgen", "morgen", "today_with_forecast"]
    if period == "today_with_forecast":
        period = "vandaag"

    if date is None:
        date = datetime.date.today()
    tot = None

    if not prognose:
        now = datetime.datetime.now()
        tot = report.periodes[period]["tot"]
        tot = min(tot, datetime.datetime(now.year, now.month, now.day, now.hour))

    interval = report.periodes[period]["interval"]

    if subject == "grid":
        report_df = report.get_grid_data(period, _tot=tot)
        report_df = report.calc_grid_columns(
            report_df, interval, view
        )
    elif subject == "balans":
        report_df, lastmoment = report.get_energy_balance_data(
            period, _tot=tot
        )
        report_df = report.calc_balance_columns(
            report_df, interval, view
        )
    # else:  # co2
    #     report_df = report.calc_co2_emission(
    #         period,
    #         _tot=tot,
    #         active_interval="uur",
    #         active_view=view,
    #     )
    elif subject == "save_cons":
        report_df = report.calc_saving_consumption(
            active_period=period,
            _tot=tot,
            active_interval=interval,
            active_view=view,
        )
    elif subject == "save_cost":
        report_df = report.calc_saving_cost(
            active_period=period,
            _tot=tot,
            active_interval=interval,
            active_view=view,
        )
    elif subject == "solar":
        report_df = report.calc_solar_data(
            device=solar_item,
            day=date,
            active_view=view,
        )
    else:
        raise Exception("Invalid subject")

    report_df.round(3)

    if view == "tabel":
        report_data = [
            report_df.to_html(
                index=False,
                justify="right",
                decimal=",",
                classes="data",
                border=0,
                float_format="{:.3f}".format,
            )
        ]
    else:
        if subject == "grid":
            report_data = report.make_graph(report_df, period)
        elif subject == "balans":
            report_data = report.make_graph(
                report_df, period, report.balance_graph_options
            )
        # else:  # co2
        #     report_data = report.make_graph(
        #         report_df, period, report.co2_graph_options
        #     )
        elif subject == "save_cons":
            report_data = report.make_graph(
                report_df, period, report.saving_cons_graph_options
            )
        elif subject == "save_cost":
            report_data = report.make_graph(
                report_df, period, report.saving_cost_graph_options
            )
        elif subject == "solar":
            report_data = report.make_graph(
                report_df,
                "vandaag",
                _options=report.solar_graph_options,
                _title=f"Solar production {date.strftime('%Y-%m-%d')}"
            )
        else:
            raise Exception("Invalid subject")

    return report_data


@v2.route("/reports", methods=["GET"])
def reports():
    subject = request.args.get("subject", default="grid")
    view = request.args.get("view", default="tabel")
    period = request.args.get("period", default="vandaag")
    report_data = reports_gen(subject, view, period)
    return render_template(
        "v2/report.html",
        title="Reports",
        period=period,
        subject=subject,
        view=view,
        report_data=report_data,
        subject_options=[{"label": "Grid", "value": "grid"},
                         {"label": "Balance", "value": "balans"}]
    )


@v2.route("/savings", methods=["GET"])
def savings():
    subject = request.args.get("subject", default="save_cons")
    view = request.args.get("view", default="tabel")
    period = request.args.get("period", default="vandaag")
    report_data = reports_gen(subject, view, period)
    return render_template(
        "v2/report.html",
        title="Savings",
        period=period,
        subject=subject,
        view=view,
        report_data=report_data,
        subject_options=[{"label": "Consumption", "value": "save_cons"},
                         {"label": "Cost", "value": "save_cost"}]
    )


@v2.route("/solar")
def solar():
    solar_items = get_solar_items_with_ml()

    if len(solar_items) == 0:
        return render_template("v2/solar-not-found.html")

    subject = request.args.get("subject", default=next(iter(solar_items.keys())))
    view = request.args.get("view", default="grafiek")
    period = request.args.get("period", default="vandaag")
    date_str = request.args.get("date")

    if date_str:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    else:
        date = datetime.datetime.today()

    date_str = date.strftime("%Y-%m-%d")

    report_data = reports_gen("solar", view, period, solar_item=solar_items[subject], date=date)
    return render_template(
        "v2/report.html",
        title="Solar",
        period=period,
        subject=subject,
        view=view,
        report_data=report_data,
        hide_period=True,
        show_datepicker=True,
        date=date_str,
        subject_options=[
            {"label": key, "value": key}
            for key in solar_items.keys()
        ]
    )


@v2.route("/reports-v2", methods=["GET"])
def reportsv2():
    today = datetime.datetime.combine(
        datetime.date.today(),
        datetime.time.min
    )

    tomorrow = today + datetime.timedelta(days=1)
    start = request.args.get("start", default=today.isoformat())
    end = request.args.get("end", default=tomorrow.isoformat())
    fields = request.args.get("fields", default="prod,cons,cost,profit")
    aggregate = request.args.get("aggregate", default="hour")

    fields = fields.split(",")

    report = Report(app_datapath + "/options.json")
    vars = report.get_vars()

    return render_template(
        "v2/reports-v2.html",
        start=start,
        end=end,
        aggregate=aggregate,
        vars=vars,
        fields=fields,
    )


@v2.route("/config", methods=["GET", "POST"])
def config():
    path = app_datapath + "options.json"
    error = None
    success = None

    if request.method == "POST" and "config" in request.form:
        newconfig = request.form["config"]
        try:
            # Syntax and schema: a config that does not validate would put
            # the scheduler in a restart loop.
            validate_config_data(json.loads(newconfig))
            atomic_write_text(Path(path), newconfig)
            success = "Config updated successfully"
        except (ValueError, OSError) as err:
            error = "Error: " + str(err)

    with open(path, "r") as file:
        content = file.read()

    return render_template(
        "v2/config.html",
        content=content,
        success=success,
        error=error,
    )


@v2.route("/secrets", methods=["GET", "POST"])
def secrets():
    path = app_datapath + "secrets.json"
    error = None
    success = None

    if request.method == "POST" and "secrets" in request.form:
        newsecrets = request.form["secrets"]
        try:
            if not isinstance(json.loads(newsecrets), dict):
                raise ValueError("secrets.json moet een JSON-object met sleutel/waarde zijn")
            atomic_write_text(Path(path), newsecrets)
            success = "Secrets updated successfully"
        except (ValueError, OSError) as err:
            error = "Error: " + str(err)

    with open(path, "r") as file:
        content = file.read()

    return render_template(
        "v2/secrets.html",
        content=content,
        success=success,
        error=error,
    )


FAST_STATE_PATH = "../data/fast_state.json"


def _load_fast_state():
    try:
        with open(FAST_STATE_PATH, "r") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _load_config():
    from dao.prog.config.loader import ConfigurationLoader
    from pathlib import Path
    loader = ConfigurationLoader(Path(app_datapath + "options.json"))
    try:
        return loader.load_and_validate()
    except Exception:
        return None


def _resolved_mode(config):
    """Return (display_string, is_entity_backed)."""
    if config is None:
        return "off", False
    fast = getattr(config, "fast_control", None)
    if fast is None:
        return "off", False
    mode_field = fast.mode
    raw_value = getattr(mode_field, "value", mode_field)
    is_entity = hasattr(mode_field, "is_entity_id") and mode_field.is_entity_id(raw_value)
    if is_entity:
        return "off", True
    raw = str(raw_value).strip().lower()
    return (raw if raw in ("off", "shadow", "active") else "off"), False


@v2.route("/fast-control")
def fast_control():
    state = _load_fast_state()
    config = _load_config()
    mode, mode_is_entity = _resolved_mode(config)
    last_decision = state.get("last_decision")
    events = state.get("events", [])
    return render_template(
        "v2/fast-control.html",
        state=state,
        mode=mode,
        mode_is_entity=mode_is_entity,
        last_decision=last_decision,
        events=events[-50:][::-1],   # last 50, newest first
    )


@v2.route("/fast-control/state")
def fast_control_state():
    state = _load_fast_state()
    last_decision = state.get("last_decision")
    return render_template(
        "v2/fast-control-state.html",
        state=state,
        last_decision=last_decision,
    )


@v2.route("/fast-control/mode", methods=["POST"])
def fast_control_mode():
    new_mode = request.form.get("mode", "").strip()
    try:
        # Edits only the mode key in the raw document; dumping the whole model
        # would rewrite the user's file with every default pinned.
        set_fast_control_mode(Path(app_datapath + "options.json"), new_mode)
    except (ValueError, OSError) as ex:
        return str(ex), 400
    return redirect(url_for("v2.fast_control"))
