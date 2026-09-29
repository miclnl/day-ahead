import collections
import datetime
import re
import time

# from sqlalchemy.sql.coercions import expect_col_expression_collection

from . import app, csrf
from flask import (
    abort,
    render_template,
    request,
    jsonify,
    session as flask_session,
    url_for,
)
from markupsafe import escape
import fnmatch
import os
import threading
from subprocess import Popen, PIPE, run, STDOUT, TimeoutExpired
import logging
from pathlib import Path
from dao.prog.config.loader import (
    ConfigurationLoader,
    atomic_write_text,
    set_fast_control_mode,
    validate_config_data,
)
from dao.prog.da_report import Report
from dao.prog.version import __version__
from dao.prog import task_state
from dao.prog import tasks as task_registry
import json

# globals
# The data directory lives outside Flask's static folder on purpose: it holds
# secrets.json and the database. Graphs are served through the /images route.
app_datapath = "../data/"
config = None

# Introduced previous_time and active_view as global variables
# This is used to enable switching between "grafiek" and "tabel" and retaining the (closest) timestamp
previous_time = None
active_view = "grafiek"


def create_config():
    global config
    try:
        loader = ConfigurationLoader(Path(app_datapath + "options.json"))
        config = loader.load_and_validate()
    except (ValueError, RuntimeError, OSError) as ex:
        logging.error(app_datapath)
        logging.error(ex)
        config = None


def validate_settings_document(setting: str, text: str) -> None:
    """Check an options.json or secrets.json document before it is written.

    Raises ValueError with a message that can be shown to the user. A config
    that does not validate would otherwise put the scheduler into a restart
    loop until the user notices the log.
    """
    data = json.loads(text)
    if setting == "options":
        validate_config_data(data)
    elif not isinstance(data, dict):
        raise ValueError("secrets.json moet een JSON-object met sleutel/waarde zijn")



browse = {}

views = {
    "tabel": {"name": "Tabel", "icon": "tabel.png"},
    "grafiek": {"name": "Grafiek", "icon": "grafiek.png"},
}

actions = {
    "first": {"icon": "first.png"},
    "prev": {"icon": "prev.png"},
    "next": {"icon": "next.png"},
    "last": {"ison": "last.png"},
}

periods = {
    "list": [
        "vandaag",
        "morgen",
        "vandaag en morgen",
        "gisteren",
        "deze week",
        "vorige week",
        "deze maand",
        "vorige maand",
        "dit jaar",
        "vorig jaar",
        "dit contractjaar",
        "365 dagen",
    ],
    "prognose": ["vandaag", "deze week", "deze maand", "dit jaar", "dit contractjaar"],
}

web_menu = {
    "home": {
        "name": "Home",
        "submenu": {},
        "views": views,
        "actions": actions,
        "function": "home",
    },
    "run": {
        "name": "Run",
    },
    "fast_control": {
        "name": "Fast control",
        "submenu": {},
    },
    "reports": {
        "name": "Reports",
        "submenu": {
            "grid": {
                "name": "Grid",
                "views": views,
                "periods": periods,
                "calculate": "calc_grid",
            },
            "balans": {"name": "Balans", "views": views, "periods": periods},
            "co2": {"name": "CO2", "views": views, "periods": periods.copy()},
        },
    },
    "savings": {
        "name": "Savings",
        "submenu": {
            "consumption": {
                "name": "Verbruik",
                "views": views,
                "periods": periods,
                "calculate": "calc_saving_consumption",
                "graph_options": "saving_cons_graph_options",
            },
            "cost": {
                "name": "Kosten",
                "views": views,
                "periods": periods,
                "calculate": "calc_saving_cost",
                "graph_options": "saving_cost_graph_options",
            },
            "co2": {
                "name": "CO2-emissie",
                "views": views,
                "periods": periods.copy(),
                "calculate": "calc_saving_co2",
                "graph_options": "saving_co2_graph_options",
            },
        },
    },
    "solar": {
        "name": "Solar",
        "submenu": {
            "items": {},
            "views": views,
            "actions": actions,
        },
    },
    "settings": {
        "name": "Config",
        "submenu": {
            "options": {"name": "Options", "views": "json-editor"},
            "secrets": {"name": "Secrets", "views": "json-editor"},
        },
    },
}

solar_web_menu = {
    "solar": {
        "name": "Solar",
        "submenu": {
            "items": {},
            "views": views,
            "actions": actions,
        },
    },
}


def generate_solar_items():
    global web_menu
    solar_options = config.solar if config else []
    battery_options = config.battery if config else []
    for battery_option in battery_options:
        for sol_opt in battery_option.solar:
            solar_options.append(sol_opt)
    result = {}
    for solar_option in solar_options:
        if solar_option.ml_prediction:
            key = solar_option.name or "default"
            result[key] = solar_option
    if len(result) == 0:
        if "solar" in web_menu.keys():
            del web_menu["solar"]
    else:
        if not "solar" in web_menu.keys():
            web_menu.update(solar_web_menu)
            key_order = ("home", "run", "reports", "savings", "solar", "settings")
            web_menu = collections.OrderedDict((k, web_menu[k]) for k in key_order)
    return result


def get_web_menu_items():
    items = {}
    for key, value in web_menu.items():
        items[key] = value["name"]
    return items


web_menu_items = {}
solar_items = {}


def check_web_menu_items():
    global solar_items, web_menu_items
    create_config()
    solar_items = generate_solar_items()
    if len(solar_items) > 0 and "solar" in web_menu.keys():
        web_menu["solar"]["submenu"]["items"] = solar_items
    web_menu_items = get_web_menu_items()


check_web_menu_items()

if config is not None:
    sensor_co2_intensity = (
        config.report.co2_intensity_sensor if config and config.report else None
    )
else:
    sensor_co2_intensity = None

if sensor_co2_intensity is None:
    del web_menu["reports"]["submenu"]["co2"]
    del web_menu["savings"]["submenu"]["co2"]
else:
    web_menu["reports"]["submenu"]["co2"]["periods"]["prognose"] = []
    web_menu["reports"]["submenu"]["co2"]["periods"]["list"] = periods["list"].copy()
    web_menu["reports"]["submenu"]["co2"]["periods"]["list"].remove("vandaag en morgen")
    web_menu["reports"]["submenu"]["co2"]["periods"]["list"].remove("morgen")
    web_menu["savings"]["submenu"]["co2"]["periods"]["prognose"] = []
    web_menu["savings"]["submenu"]["co2"]["periods"]["list"] = periods["list"].copy()
    web_menu["savings"]["submenu"]["co2"]["periods"]["list"].remove("vandaag en morgen")
    web_menu["savings"]["submenu"]["co2"]["periods"]["list"].remove("morgen")

# The tasks this dashboard offers, derived from the one registry in
# dao/prog/tasks.py. Keyed by the historical v1 keys (which are aliases in
# the registry) so existing form values, bookmarks and /api/run/<key> URLs
# keep working. clean, consolidate and forecast_accuracy are new here: they
# had a complete registry entry all along but were listed in neither
# dashboard, so they could only be run from the command line.
_V1_TASK_KEYS = (
    "calc_met_debug",
    "calc_zonder_debug",
    "get_prices",
    "get_meteo",
    "get_tibber",
    "calc_baseloads",
    "consolidate",
    "forecast_accuracy",
    "train_ml_predictions",
    "clean",
    "fast_once",
    "fast_simulate",
)


def _build_bewerkingen() -> dict:
    """Registry entries under their v1 keys, with the fields run.html reads.

    run.html iterates value["parameters"] and renders value["wait"] into a
    setTimeout call. Jinja renders a missing key as an empty string, so the
    entries that had no "wait" (all of them) produced
    ``setTimeout(callback, )`` -- a JavaScript syntax error, which is why
    the "bewerking wordt uitgevoerd" page never resubmitted itself. Both
    keys are always present now.
    """
    entries = {}
    for key in _V1_TASK_KEYS:
        task = task_registry.get(key)
        if task is None:  # pragma: no cover - guards a typo in _V1_TASK_KEYS
            logging.error(f"Onbekende taak {key!r} in de v1-takenlijst, overgeslagen")
            continue
        entries[key] = {
            "name": task["name"],
            "cmd": list(task["cmd"]),
            "file_name": task["file_name"],
            "parameters": list(task.get("parameters", ())),
            "wait": 1000,
        }
    return entries


bewerkingen = _build_bewerkingen()


def get_file_list(path: str, pattern: str) -> list:
    """
    get a time-ordered file list with name and timestamp from filename
    :parameter path: folder
    :parameter pattern: wildcards to search for
    """
    flist = []
    for f in os.listdir(path):
        if fnmatch.fnmatch(f, pattern):
            # Extract timestamp from filename (e.g. calc_2026-02-17__08-45.png) because datetime picker works with
            # absolut timestamps and then file modification date might differ from the timestamp in the filename, which is the intended reference time for the user
            m = re.search(r"(\d{4}-\d{2}-\d{2})__(\d{2})[:-](\d{2})(?:[:-](\d{2}))?", f)
            if m:
                try:
                    seconds = m.group(4) or "00"
                    dt_str = f"{m.group(1)} {m.group(2)}:{m.group(3)}:{seconds}"
                    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    timestamp = dt.timestamp()  # Local time as epoch
                    flist.append({"name": f, "time": timestamp})
                except (ValueError, OSError):
                    # Fallback to mtime if filename parsing fails
                    fullname = os.path.join(path, f)
                    flist.append({"name": f, "time": os.path.getmtime(fullname)})
            else:
                # Fallback to mtime if no timestamp in filename
                fullname = os.path.join(path, f)
                flist.append({"name": f, "time": os.path.getmtime(fullname)})
    flist.sort(key=lambda x: x.get("time"), reverse=True)
    return flist


@app.route("/", methods=["POST", "GET"])
def menu():
    # check_web_menu_items()
    lst = request.form.to_dict(flat=False)
    if "current_menu" in lst:
        current_menu = lst["current_menu"][0]
        if current_menu == "home":
            return home()
        elif current_menu == "run":
            return run_process()
        elif current_menu == "fast_control":
            return fast_control()
        elif current_menu == "reports" or current_menu == "savings":
            return reports(current_menu)
        elif current_menu == "solar" and "solar" in web_menu_items.keys():
            return solar()
        elif current_menu == "settings":
            return settings()
        else:
            return home()
    else:
        if "menu_home" in lst:
            return home()
        elif "menu_run" in lst:
            return run_process()
        elif "menu_fast_control" in lst:
            return fast_control()
        elif "menu_reports" in lst:
            return reports("reports")
        elif "menu_savings" in lst:
            return reports("savings")
        elif "menu_solar" in lst:
            return solar()
        elif "menu_settings" in lst:
            return settings()
        else:
            return home()


FAST_STATE_PATH_V1 = "../data/fast_state.json"


def _load_fast_state_v1():
    try:
        with open(FAST_STATE_PATH_V1, "r") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _config_v1():
    """Return a freshly-loaded config object, mirroring v2's _load_config()."""
    try:
        loader = ConfigurationLoader(Path(app_datapath + "options.json"))
        return loader.load_and_validate()
    except Exception:
        return None


def _cached_config_v1():
    """The module-level config already loaded at import time, for read-only
    display paths that are hit on every poll.

    _config_v1() re-parses and re-validates options.json from scratch and
    takes the same fcntl.flock() the migration path uses; a GET request
    polled every 5 seconds (the fast-control status widget) doing that on
    every tick is needless load, and it can even *write* options.json (the
    migration branch) from what looks like a read-only GET. The watchdog
    already restarts this worker on every options.json change (see
    watchdog.sh), which re-runs create_config() at import and keeps this
    cache current; a fresh load is only necessary right after this process
    itself just wrote the file (see the POST handler below).
    """
    return config if config is not None else _config_v1()


def _resolved_mode_v1(config):
    """Return (display_string, is_entity_backed) for v1 page rendering.

    When mode is bound to an HA entity, the static config cannot tell us
    the current value. Read the live state from fast_state.json instead
    so the UI displays what the runner is actually doing rather than
    always reporting "off".
    """
    if config is None:
        return "off", False
    fast = getattr(config, "fast_control", None)
    if fast is None:
        return "off", False
    mode_field = fast.mode
    raw_value = getattr(mode_field, "value", mode_field)
    is_entity = hasattr(mode_field, "is_entity_id") and mode_field.is_entity_id(raw_value)
    if is_entity:
        # Prefer the runtime value the runner last published; fall back
        # to "off" only if no state file exists yet.
        state = _load_fast_state_v1()
        runtime_mode = state.get("last_decision", {}).get("mode")
        if runtime_mode in ("off", "shadow", "active"):
            return runtime_mode, True
        return "off", True
    raw = str(raw_value).strip().lower()
    return (raw if raw in ("off", "shadow", "active") else "off"), False


@app.route("/fast_control", methods=["GET", "POST"])
def fast_control():
    state = _load_fast_state_v1()
    last_decision = state.get("last_decision")
    events = list(reversed(state.get("events", [])[-50:]))
    success = None
    error = None
    config = _cached_config_v1()

    if request.method == "POST":
        new_mode = request.form.get("mode", "").strip()
        try:
            # Edits only the mode key in the raw document; dumping the whole
            # model would rewrite the user's file with every default pinned.
            set_fast_control_mode(Path(app_datapath + "options.json"), new_mode)
            success = f"Modus gezet op {new_mode}"
            # A fresh load here (not the cache): the mode was just written to
            # disk by this same request and must be reflected immediately,
            # not after the next worker restart.
            config = _config_v1()
        except (ValueError, OSError) as ex:
            error = str(ex)

    mode, mode_is_entity = _resolved_mode_v1(config)

    return render_template(
        "fast_control.html",
        title="Fast control",
        active_menu_list=web_menu_items,
        active_menu="fast_control",
        state=state,
        last_decision=last_decision,
        events=events,
        mode=mode,
        mode_is_entity=mode_is_entity,
        success=success,
        error=error,
        version=__version__,
    )


@app.route("/fast_control/state.json")
def fast_control_state_json():
    state = _load_fast_state_v1()
    last_decision = state.get("last_decision") or {}
    # Polled every 5 seconds by the page's own JS: the cached config, not a
    # fresh parse+validate+possible-migration-write of options.json.
    mode, mode_is_entity = _resolved_mode_v1(_cached_config_v1())
    return jsonify({
        "mode": mode,
        "mode_is_entity": mode_is_entity,
        "override": last_decision.get("override"),
        "reason": last_decision.get("reason"),
        "house_w": last_decision.get("house_w"),
        "benefit_eur_h": last_decision.get("benefit_eur_h"),
        "pv_w": last_decision.get("pv_w"),
        "saved_today_eur": state.get("saved_today_eur", 0),
        "saved_today_is_estimate": state.get("saved_today_is_estimate", False),
        "daily_extra_throughput_used": state.get("daily_extra_throughput_used", 0),
        "energy_budget_used": state.get("energy_budget_used", 0),
        "ts": last_decision.get("ts"),
    })


@app.route("/", methods=["POST", "GET"])
def home():
    subjects = ["balans"]
    views = ["grafiek", "tabel"]
    cur_subject = "grid"
    active_subject = "grid"
    cur_view = "grafiek"
    # global active_view
    global previous_time
    active_time = None
    action = None
    confirm_delete = False
    # get active_view from session if available, otherwise use the default value;
    # this is to enable switching between grafiek and tabel while retaining the active time
    active_view = flask_session.get("active_view", "grafiek")

    if config is not None:
        battery_options = config.battery
        for b in range(len(battery_options)):
            subjects.append(battery_options[b].name)
    if request.method == "POST":
        # ImmutableMultiDict([('cur_subject', 'Accu2'), ('subject', 'Accu1')])
        lst = request.form.to_dict(flat=False)
        # print('Home:', lst)
        if "cur_subject" in lst:
            #   active_subject = lst["cur_subject"][0]
            cur_subject = lst["cur_subject"][0]
        if "cur_view" in lst:
            #   active_view = lst["cur_view"][0]
            cur_view = lst["cur_view"][0]
        if "subject" in lst:
            active_subject = lst["subject"][0]
        if "view" in lst:
            active_view = lst["view"][0]
            flask_session["active_view"] = (
                active_view  # update session with the new active_view
            )
        if "active_time" in lst:
            # Ignore active_time from POST if switching between grafiek & table; keep the active_time from the previous call from session.
            if cur_view != active_view:
                active_time = flask_session.get("active_time")
            else:
                active_time = float(lst["active_time"][0])
        if "action" in lst:
            action = lst["action"][0]
        if "file_delete" in lst:
            confirm_delete = lst["file_delete"][0] == "delete"

    #  Every mouse click on Home page calls the home() function
    #  By design; the get_file_list() is called over and over again to ensure an accurate reflection of the files
    #  Files might have been added/removed since last call

    if active_view == "grafiek":
        active_map = "/images/"
        active_filter = "*.png"
    else:
        active_map = "/log/"
        active_filter = "*.log"

    flist = get_file_list(app_datapath + active_map, active_filter)
    index = 0

    if active_time:
        # Find index in the current flist with timestamp closest to active_time (possibly from other flist)
        # The timestamp between e.g. calc_2026-02-17__08-45.png & calc_2026-02-17__08-45.log are NOT identical
        # The intent is to be able to switch between grafiek and table while keeping the active time
        active_time = float(active_time)
        diff_time = active_time  # high intialization value
        for i in range(len(flist)):
            if abs(flist[i]["time"] - active_time) < diff_time:
                diff_time = abs(flist[i]["time"] - active_time)
                index = i
    # Ensure index is within valid range
    index = max(0, min(index, len(flist) - 1))

    if action == "first":
        index = 0
    if action == "previous":
        index = max(0, index - 1)
    if action == "next":
        index = min(len(flist) - 1, index + 1)
    if action == "last":
        index = len(flist) - 1

    if action in ["fast_forward", "fast_reverse"]:
        if type(active_time) != float:
            active_time = float(active_time)
        if action == "fast_forward":
            target_time = active_time - (6 * 3600)  # Add 6 hours
        if action == "fast_reverse":
            target_time = active_time + (6 * 3600)  # Subtract 6 hours
        diff_time = active_time  # high intialization value
        for i in range(len(flist)):
            if abs(flist[i]["time"] - target_time) < diff_time:
                diff_time = abs(flist[i]["time"] - target_time)
                index = i

    if action == "delete" and confirm_delete:
        os.remove(app_datapath + active_map + flist[index]["name"])
        flist = get_file_list(app_datapath + active_map, active_filter)
        index = min(len(flist) - 1, index)

    if len(flist) > 0:
        # print('Active index:', index )
        # print(flist[index]["name"], datetime.datetime.fromtimestamp(flist[index]["time"]))
        active_time = str(flist[index]["time"])
        if active_view == "grafiek":
            image = url_for("image", name=flist[index]["name"])
            tabel = None
        else:
            image = None
            with open(app_datapath + active_map + flist[index]["name"], "r") as f:
                tabel = f.read()
    else:
        active_time = None
        image = None
        tabel = None

    # Remember this active time in global variable
    # previous_time = active_time
    flask_session["active_time"] = (
        active_time  # Store active_time in session to enable switching between grafiek and tabel while retaining the active time
    )

    flatpickr_times = [
        datetime.datetime.fromtimestamp(f["time"]).strftime("%Y-%m-%d %H:%M:%S")
        for f in flist
    ]
    flatpickr_default_ts = float(active_time) if active_time else None
    flatpickr_default = (
        datetime.datetime.fromtimestamp(float(active_time)).strftime("%Y-%m-%d %H:%M::%S")
        if active_time
        else ""
    )

    return render_template(
        "home.html",
        title="Optimization",
        active_menu_list=web_menu_items,
        active_menu="home",
        subjects=subjects,
        views=views,
        active_subject=active_subject,
        active_view=active_view,
        image=image,
        tabel=tabel,
        active_time=active_time,
        flatpickr_times=flatpickr_times,
        flatpickr_default_ts=flatpickr_default_ts,
        flatpickr_default=flatpickr_default,
        version=__version__,
    )


# logfile = "../data/log/run.log"
"""
task_state = {
    "status": "idle",    # idle | running | done | error
    "task" : "",
    "msg" : "",
    "returncode": None
}
"""
lock = threading.Lock()

#: How often a running task refreshes its claim, and how often it checks
#: whether a cancel was requested. Well under task_state.STALE_AFTER_S.
HEARTBEAT_S = 5


def _tracked_task() -> tuple:
    """The task this dashboard's status and log polling should report on.

    Claims are per task now (see dao/prog/task_state.py), but this UI shows
    one task at a time: the running one, or the most recently finished when
    nothing runs. Returns (task_key, flat_record) where the record has the
    shape the /status and /log routes already expected.
    """
    state = task_state.read()
    running = state["running"]
    if running:
        key = max(running, key=lambda k: running[k].get("started") or 0)
        return key, {"status": "running", **running[key]}
    finished = state["last"]
    if finished:
        key = max(finished, key=lambda k: finished[k].get("finished") or 0)
        return key, finished[key]
    return None, {"status": "idle", "logfile": None}


def run_and_log(cmd, task_key, logfile):
    """Run a claimed task, streaming its output into *logfile*.

    The caller must already hold the claim on *task_key* (see run_process);
    taking it here would reopen the check-then-act race this is meant to
    close. Releasing it is this function's job, including when the task
    fails, so a crashed run does not block the next one for the full
    staleness window.
    """
    status = "error"
    returncode = None
    try:
        with open(logfile, "w") as handle:
            # start_new_session=True: the task gets its own process group, so
            # a cancel reaches everything it spawned and a signal aimed at
            # the web server does not kill it halfway through.
            proc = Popen(
                cmd, stdout=PIPE, stderr=STDOUT, text=True, start_new_session=True
            )
            task_state.heartbeat(task_key, logfile=logfile)
            cancelled = False
            last_beat = time.time()
            for line in proc.stdout:
                handle.write(line)
                handle.flush()
                now = time.time()
                if now - last_beat >= HEARTBEAT_S:
                    last_beat = now
                    entry = task_state.heartbeat(task_key)
                    if entry.get("cancel"):
                        cancelled = True
                        handle.write("\nOpdracht afgebroken op verzoek.\n")
                        handle.flush()
                        task_registry.kill_process_group(proc)
                        break
            proc.wait()
            returncode = proc.returncode
            if cancelled:
                status = "cancelled"
            else:
                status = "done" if returncode == 0 else "error"
    except Exception:
        logging.exception(f"Taak {task_key} is mislukt")
        raise
    finally:
        task_state.release(task_key, status, returncode=returncode, logfile=logfile)


@app.route("/run", methods=["POST", "GET"])
def run_process():
    bewerking = ""
    current_bewerking = ""
    log_content = ""
    parameters = {}

    if request.method in ["POST", "GET"]:
        if task_state.running_tasks():
            log_content = "Er draait al een opdracht."
            state = "running"
        else:
            dct = request.form.to_dict(flat=False)
            if "current_bewerking" in dct:
                current_bewerking = dct["current_bewerking"][0]
                run_bewerking = bewerkingen.get(current_bewerking)
                if run_bewerking is None:
                    abort(404)
                canonical = task_registry.resolve(current_bewerking)
                values = {
                    parameter: dct[parameter][0]
                    for parameter in run_bewerking["parameters"]
                    if parameter in dct
                }
                cmd = task_registry.build_cmd(canonical, values)
                logfile = (
                    "../data/log/"
                    + run_bewerking["file_name"]
                    + "_tmp_"
                    + datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
                    + ".log"
                )
                # Claim before starting the thread, not inside it. The old
                # code read the state here and only wrote "running" once the
                # worker thread got scheduled, so two near-simultaneous
                # requests both passed the check above and started the task
                # twice. claim() does the check and the write atomically.
                if not task_state.claim(
                    canonical, source="dashboard", logfile=logfile
                ):
                    log_content = "Er draait al een opdracht."
                    state = "running"
                else:
                    bewerking = ""
                    threading.Thread(
                        target=run_and_log,
                        args=(cmd, canonical, logfile),
                        daemon=True,
                    ).start()
                    log_content = "Opdracht is gestart"
                    state = "running"
            else:
                for i in range(len(dct.keys())):
                    bew = list(dct.keys())[i]
                    if bew in bewerkingen:
                        bewerking = bew
                        if "parameters" in bewerkingen[bewerking]:
                            for j in range(len(bewerkingen[bewerking]["parameters"])):
                                if bewerkingen[bewerking]["parameters"][j] in dct:
                                    param_str = bewerkingen[bewerking]["parameters"][j]
                                    param_value = dct[
                                        bewerkingen[bewerking]["parameters"][j]
                                    ][0]
                                    parameters[param_str] = param_value
                        break
                state = "idle"

    return render_template(
        "run.html",
        title="Run",
        active_menu_list=web_menu_items,
        active_menu="run",
        bewerkingen=bewerkingen,
        bewerking=bewerking,
        current_bewerking=current_bewerking,
        parameters=parameters,
        status=state,
        log_content=log_content,
        version=__version__,
    )


@app.route("/status")
def status():
    task, record = _tracked_task()
    definition = task_registry.get(task) if task else None
    task_name = definition["name"] if definition else task
    state = record.get("status")
    if state == "running":
        msg = f"Opdracht '{task_name}' wordt uitgevoerd"
    elif state == "done":
        msg = f"✅ Opdracht '{task_name}' succesvol afgerond"
    elif state == "error":
        msg = f"❌ Opdracht '{task_name}' geëindigd met fout"
    elif state == "cancelled":
        msg = f"Opdracht '{task_name}' is afgebroken"
    else:
        msg = f"Opdracht '{task}' : {state}"
    return jsonify({"status": state, "msg": msg})


@app.route("/log")
def show_log():
    _task, record = _tracked_task()
    logfile = record.get("logfile")
    if logfile is None or not os.path.exists(logfile):
        return "Nog geen log beschikbaar"
    with open(logfile, "r") as f:
        lines = f.readlines()
    if record.get("status") == "running":
        text = "".join(lines[-20:])  # laatste 20 regels
    else:
        text = "".join(lines)
    return text


@app.route("/reports", methods=["POST", "GET"])
def reports(active_menu: str):
    report = Report(app_datapath + "/options.json")
    menu_dict = web_menu[active_menu]
    title = menu_dict["name"]
    subjects_lst = list(menu_dict["submenu"].keys())
    active_subject = subjects_lst[0]
    views_lst = list(menu_dict["submenu"][active_subject]["views"].keys())
    active_view = views_lst[0]
    period_lst = menu_dict["submenu"][active_subject]["periods"]["list"]
    active_period = period_lst[0]
    show_prognose = False
    met_prognose = False
    if request.method in ["POST", "GET"]:
        # ImmutableMultiDict([('cur_subject', 'Accu2'), ('subject', 'Accu1')])
        lst = request.form.to_dict(flat=False)
        if "cur_subject" in lst:
            active_subject = lst["cur_subject"][0]
            if active_subject not in subjects_lst:
                active_subject = subjects_lst[0]
        if "cur_view" in lst:
            active_view = lst["cur_view"][0]
        if "cur_periode" in lst:
            active_period = lst["cur_periode"]
        if "subject" in lst:
            active_subject = lst["subject"][0]
            period_lst = menu_dict["submenu"][active_subject]["periods"]["list"]
        if "view" in lst:
            active_view = lst["view"][0]
        if "periode-select" in lst:
            active_period = lst["periode-select"][0]
        if not (active_period in period_lst):
            active_period = period_lst[0]
        if "met_prognose" in lst:
            met_prognose = lst["met_prognose"][0]
    tot = None
    if active_period in menu_dict["submenu"][active_subject]["periods"]["prognose"]:
        show_prognose = True
    else:
        show_prognose = False
        met_prognose = False
    if not met_prognose:
        now = datetime.datetime.now()
        tot = report.periodes[active_period]["tot"]
        if (
            active_period in menu_dict["submenu"][active_subject]["periods"]["prognose"]
            or menu_dict["submenu"][active_subject]["periods"]["prognose"] == []
        ):
            tot = min(tot, datetime.datetime(now.year, now.month, now.day, now.hour))
    views_lst = list(menu_dict["submenu"][active_subject]["views"].keys())
    period_lst = menu_dict["submenu"][active_subject]["periods"]["list"]
    active_interval = report.periodes[active_period]["interval"]
    if active_menu == "reports":
        if active_subject == "grid":
            report_df = report.get_grid_data(active_period, _tot=tot)
            report_df = report.calc_grid_columns(
                report_df, active_interval, active_view
            )
        elif active_subject == "balans":
            report_df, lastmoment = report.get_energy_balance_data(
                active_period, _tot=tot
            )
            report_df = report.calc_balance_columns(
                report_df, active_interval, active_view
            )
        else:  # co2
            report_df = report.calc_co2_emission(
                active_period,
                _tot=tot,
                active_interval=active_interval,
                active_view=active_view,
            )
        report_df.round(3)
    else:  # savings
        calc_function = getattr(
            report, menu_dict["submenu"][active_subject]["calculate"]
        )
        report_df = calc_function(
            active_period,
            _tot=tot,
            active_interval=active_interval,
            active_view=active_view,
        )
    if active_view == "tabel":
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
        if active_menu == "reports":
            if active_subject == "grid":
                report_data = report.make_graph(report_df, active_period)
            elif active_subject == "balans":
                report_data = report.make_graph(
                    report_df, active_period, report.balance_graph_options
                )
            else:  # co2
                report_data = report.make_graph(
                    report_df, active_period, report.co2_graph_options
                )
        else:  # "savings"
            graph_options = getattr(
                report, menu_dict["submenu"][active_subject]["graph_options"]
            )
            report_data = report.make_graph(report_df, active_period, graph_options)
    return render_template(
        "report.html",
        title=title,
        active_menu_list=web_menu_items,
        active_menu=active_menu,
        subjects=subjects_lst,
        views=views_lst,
        periode_options=period_lst,
        active_period=active_period,
        show_prognose=show_prognose,
        met_prognose=met_prognose,
        active_subject=active_subject,
        active_view=active_view,
        report_data=report_data,
        version=__version__,
    )


@app.route("/solar", methods=["POST", "GET"])
def solar():
    report = Report(app_datapath + "/options.json")
    menu_dict = web_menu["solar"]
    title = menu_dict["name"]
    subjects_lst = list(menu_dict["submenu"]["items"].keys())
    active_subject = subjects_lst[0]
    views_lst = list(menu_dict["submenu"]["views"].keys())
    active_view = views_lst[0]
    active_date = datetime.date.today()

    if request.method in ["POST", "GET"]:
        lst = request.form.to_dict(flat=False)
        if "cur_subject" in lst:
            active_subject = lst["cur_subject"][0]
            if active_subject not in subjects_lst:
                active_subject = subjects_lst[0]
        if "cur_view" in lst:
            active_view = lst["cur_view"][0]
        if "subject" in lst:
            active_subject = lst["subject"][0]
        if "view" in lst:
            active_view = lst["view"][0]
        if "active_date" in lst:
            active_date = datetime.datetime.strptime(
                lst["active_date"][0], "%Y-%m-%d"
            ).date()
        if "action" in lst:
            action = lst["action"][0]
            if action == "previous":
                active_date -= datetime.timedelta(days=1)
            else:
                active_date += datetime.timedelta(days=1)
    report_df = report.calc_solar_data(
        solar_items[active_subject], active_date, active_view
    )
    report_df.round(3)
    if active_view == "tabel":
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
        report_data = report.make_graph(
            report_df,
            "vandaag",
            _options=report.solar_graph_options,
            _title=f"Solar production {active_date.strftime('%Y-%m-%d')}",
        )
    return render_template(
        "solar.html",
        title=title,
        active_menu_list=web_menu_items,
        active_menu="solar",
        subjects=subjects_lst,
        views=views_lst,
        active_subject=active_subject,
        active_view=active_view,
        active_date=active_date,
        report_data=report_data,
        version=__version__,
    )


@app.route("/settings", methods=["POST", "GET"])
@app.route("/settings/<filename>", methods=["POST", "GET"])
def settings(filename: str | None = None):
    def get_file(fname):
        with open(fname, "r") as file:
            return file.read()

    settngs = ["options", "secrets"]
    active_setting = filename or "options"
    cur_setting = ""
    lst = request.form.to_dict(flat=False)
    if request.method in ["POST", "GET"]:
        if "cur_setting" in lst:
            active_setting = lst["cur_setting"][0]
            cur_setting = active_setting
        if "setting" in lst:
            active_setting = lst["setting"][0]
    # The form value names the file that is read and written. Anything other
    # than the two known files is a traversal attempt.
    if active_setting not in settngs or cur_setting not in ("", *settngs):
        abort(400)
    message = None
    filename_ext = app_datapath + active_setting + ".json"

    options = None
    if (cur_setting != active_setting) or ("setting" in lst):
        options = get_file(filename_ext)
    else:
        lst = request.form.to_dict(flat=False)
        if "codeinput" in lst:
            updated_data = request.form["codeinput"]
            if "action" in lst:
                action = request.form["action"]
                if action == "update":
                    try:
                        validate_settings_document(active_setting, updated_data)
                        atomic_write_text(Path(filename_ext), updated_data)
                        message = "JSON data updated successfully"
                        check_web_menu_items()
                    except ValueError as err:
                        message = "Error: " + str(err)
                    except OSError as err:
                        message = "Error: " + str(err)
                    options = updated_data
                if action == "cancel":
                    options = get_file(filename_ext)
        else:
            # Load initial JSON data from a file
            options = get_file(filename_ext)
    return render_template(
        "settings.html",
        title="Instellingen",
        active_menu_list=web_menu_items,
        active_menu="settings",
        settings=settngs,
        active_setting=active_setting,
        options_data=options,
        message=message,
        version=__version__,
    )


'''
@app.route('/api/prognose/<string:fld>', methods=['GET'])
def api_prognose(fld: str):
    """
    retourneert in json de data van
    :param fld: de code van de gevraagde data
    :return: de gevraagde data in json formaat
    """
    report = dao.prog.da_report.Report()
    start = request.args.get('start')
    end = request.args.get('end')
    data = report.get_api_data(fld, prognose=True, start=start, end=end)
    return jsonify({'data': data})
'''


@app.route("/api/report/<string:fld>/<string:periode>", methods=["GET"])
@csrf.exempt
def api_report(fld: str, periode: str):
    """
    Retourneert in json de data van
    :param fld: de code van de gevraagde data
    :param periode: de periode van de gevraagde data
    :return: de gevraagde data in json formaat
    """
    cumulate = request.args.get("cumulate")
    cumulate = escape(cumulate)
    report = Report(app_datapath + "/options.json")
    # start = request.args.get('start')
    # end = request.args.get('end')
    if cumulate is None:
        cumulate = False
    else:
        try:
            cumulate = int(cumulate)
            cumulate = cumulate == 1
        except ValueError:
            cumulate = False
    fld = str(escape(fld))
    periode = str(escape(periode))
    result = report.get_api_data(fld, periode, cumulate=cumulate)

    headers = {
        "Content-Type": "application/json",
    }
    return result, headers


@app.route("/api/run/<string:bewerking>", methods=["GET", "POST"])
@csrf.exempt
def run_api(bewerking: str):
    task = task_registry.get(bewerking)
    if task is None:
        # Never echo the path segment back: it is attacker controlled and
        # the response is HTML.
        abort(404)
    canonical = task_registry.resolve(bewerking)

    # This endpoint runs the task inside the request. Claim it like any
    # other starter so it cannot run alongside the same task started by
    # cron or from the task page.
    if not task_state.claim(canonical, source="api"):
        return (
            render_template(
                "api_run.html",
                log_content=f"Taak {task['name']} draait al.",
                version=__version__,
                active_menu_list=web_menu_items,
            ),
            409,
        )

    # The cap has to stay under gunicorn's own per-request timeout (120 s,
    # see gunicorn_config.py). It used to be 300 s with a comment claiming
    # that matched gunicorn; it did not, so the worker was killed at 120 s
    # and the run ended as a dead worker plus an orphaned child instead of
    # a timeout message with the output collected so far.
    timeout_s = task_registry.api_timeout_s(canonical)
    status = "error"
    returncode = None
    try:
        proc = run(
            task["cmd"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        log_content = proc.stdout + proc.stderr
        returncode = proc.returncode
        status = "done" if returncode == 0 else "error"
    except TimeoutExpired as exception:
        log_content = (
            f"Taak {bewerking} afgebroken na {timeout_s}s timeout.\n"
            f"Gebruik de takenpagina voor taken die langer duren: die "
            f"draaien in de achtergrond zonder deze limiet.\n"
            f"stdout tot timeout:\n{exception.stdout or ''}\n"
            f"stderr tot timeout:\n{exception.stderr or ''}\n"
        )
    filename = (
        "../data/log/"
        + task["file_name"]
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
        + ".log"
    )
    try:
        with open(filename, "w") as f:
            f.write(log_content)
    finally:
        task_state.release(
            canonical, status, returncode=returncode, logfile=filename
        )
    return render_template(
        "api_run.html",
        log_content=log_content,
        version=__version__,
        active_menu_list=web_menu_items,
    )
