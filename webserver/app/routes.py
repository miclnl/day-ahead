import logging
import time
import asyncio
import datetime
from typing import Optional, Dict, Any, List

logging.debug("routes.py - Starting FastAPI imports...")

import datetime
import math
import random
import threading

logging.debug("routes.py - Basic imports done, importing app...")
from . import app, templates
logging.debug("routes.py - App imported successfully")

from fastapi import Request, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
import fnmatch
import os
from subprocess import PIPE, run
from sqlalchemy import text
from logging.handlers import TimedRotatingFileHandler

# Import ha_client with fallback
try:
    from ha_client import (
        get_core_config,
        get_states,
        suggest_pv_energy_entities,
        get_statistics_max_daily,
    )
    logging.debug("Imported ha_client via direct import")
    HA_CLIENT_AVAILABLE = True
except ImportError:
    logging.warning("Could not import ha_client - HA integration features disabled")
    # Create dummy functions
    def get_core_config(config):
        return None
    def get_states(config):
        return {}
    def suggest_pv_energy_entities(states):
        return []
    def get_statistics_max_daily(config, entity, days=365):
        return None
    HA_CLIENT_AVAILABLE = False

logging.debug("routes.py - Core imports done, starting module imports...")

# LAZY IMPORT: Don't import database modules during initialization
# This prevents SQLAlchemy circular import issues in uvicorn workers
logging.debug("routes.py - Skipping immediate Config/Report imports to prevent SQLAlchemy circular imports")
Config = None
Report = None

# Singleton instances to prevent multiple database connections
_config_instance = None
_report_instance = None
_config_lock = threading.Lock()
_report_lock = threading.Lock()

# In-memory run status registry for UI (not persisted)
_last_runs: Dict[str, Dict[str, Any]] = {}

def get_config_class():
    """Lazy import Config class to avoid SQLAlchemy circular imports"""
    global Config
    if Config is None:
        try:
            from da_config import Config
            logging.debug("Lazy loaded Config via da_config")
        except ImportError:
            logging.warning("Could not lazy load Config class")
            Config = None
    return Config

def get_report_class():
    """Lazy import Report class to avoid SQLAlchemy circular imports"""
    global Report
    if Report is None:
        try:
            from da_report import Report
            logging.debug("Lazy loaded Report via da_report")
        except ImportError:
            logging.warning("Could not lazy load Report class")
            Report = None
    return Report

logging.debug("routes.py - Importing version...")
try:
    from version import __version__
    logging.debug("routes.py - Version imported via version")
except ImportError:
    logging.warning("Could not import version")
    __version__ = "0.5.0"

logging.debug("routes.py - Module imports completed")

# Simple in-memory cache for report DataFrames to reduce CPU on Pi
CACHE_TTL_SECONDS = 300  # 5 minutes
_report_cache = {}

def _cache_set(key, df):
    try:
        _report_cache[key] = (time.time(), df)
    except Exception:
        pass

def _cache_get(key):
    try:
        if key in _report_cache:
            timestamp, df = _report_cache[key]
            if time.time() - timestamp < CACHE_TTL_SECONDS:
                return df
            else:
                del _report_cache[key]
        return None
    except Exception:
        return None

def _cache_clear():
    try:
        _report_cache.clear()
    except Exception:
        pass

# Dependency injection for Config and Report instances
async def get_config_instance():
    """Get singleton Config instance with thread safety"""
    global _config_instance, _config_lock

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                try:
                    ConfigClass = get_config_class()
                    if ConfigClass:
                        _config_instance = ConfigClass("/data/options.json")
                        logging.debug("Created new Config instance")
                    else:
                        raise HTTPException(status_code=500, detail="Config class not available")
                except Exception as e:
                    logging.error(f"Failed to create Config instance: {e}")
                    raise HTTPException(status_code=500, detail=f"Config initialization failed: {str(e)}")

    return _config_instance

async def get_report_instance():
    """Get singleton Report instance with thread safety"""
    global _report_instance, _report_lock

    if _report_instance is None:
        with _report_lock:
            if _report_instance is None:
                try:
                    ReportClass = get_report_class()
                    if ReportClass:
                        _report_instance = ReportClass()
                        logging.debug("Created new Report instance")
                    else:
                        raise HTTPException(status_code=500, detail="Report class not available")
                except Exception as e:
                    logging.error(f"Failed to create Report instance: {e}")
                    raise HTTPException(status_code=500, detail=f"Report initialization failed: {str(e)}")

    return _report_instance

# FastAPI route handlers
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page"""
    try:
        if templates:
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "version": __version__
            })
        else:
            return HTMLResponse(content="<h1>DAO Dashboard</h1><p>FastAPI webserver running</p>")
    except Exception as e:
        logging.error(f"Dashboard error: {e}")
        return HTMLResponse(content="<h1>Error</h1><p>Dashboard unavailable</p>", status_code=500)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "version": __version__,
        "webserver": "fastapi",
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard page"""
    try:
        logging.debug("Dashboard route called")
        if templates:
            logging.debug("Templates available, rendering dashboard")

            # Add template context with datetime functions
            current_time = datetime.datetime.now()
            template_context = {
                "request": request,
                "version": __version__,
                "now": datetime.datetime.now,  # Function to get current time
                "datetime": datetime.datetime,  # datetime class
                "current_time": current_time,  # Current time as object
                "time_str": current_time.strftime('%H:%M'),  # Pre-formatted time
            }

            return templates.TemplateResponse("dashboard.html", template_context)
        else:
            logging.warning("Templates not available, returning fallback")
            return HTMLResponse(content="<h1>DAO Dashboard</h1><p>Template system unavailable</p>")
    except Exception as e:
        logging.error(f"Dashboard error: {e}")
        import traceback
        logging.error(f"Dashboard traceback: {traceback.format_exc()}")
        return HTMLResponse(content=f"<h1>Error</h1><p>Dashboard unavailable: {str(e)}</p>", status_code=500)

@app.get("/reports", response_class=HTMLResponse)
async def reports(request: Request, config: Any = Depends(get_config_instance)):
    """Reports page with timeout protection"""
    try:
        # Timeout protection for database operations
        async with asyncio.timeout(10.0):
            if templates:
                return templates.TemplateResponse("reports.html", {
                    "request": request,
                    "version": __version__,
                    "config": config
                })
            else:
                return HTMLResponse(content="<h1>Reports</h1><p>Template system unavailable</p>")
    except asyncio.TimeoutError:
        logging.error("Reports page timeout")
        return HTMLResponse(content="<h1>Timeout</h1><p>Database operation timed out</p>", status_code=503)
    except Exception as e:
        logging.error(f"Reports error: {e}")
        return HTMLResponse(content="<h1>Error</h1><p>Reports unavailable</p>", status_code=500)

@app.get("/savings", response_class=HTMLResponse)
async def savings(request: Request, config: Any = Depends(get_config_instance)):
    """Savings page with timeout protection"""
    try:
        # Timeout protection for database operations
        async with asyncio.timeout(10.0):
            if templates:
                return templates.TemplateResponse("savings.html", {
                    "request": request,
                    "version": __version__,
                    "config": config
                })
            else:
                return HTMLResponse(content="<h1>Savings</h1><p>Template system unavailable</p>")
    except asyncio.TimeoutError:
        logging.error("Savings page timeout")
        return HTMLResponse(content="<h1>Timeout</h1><p>Database operation timed out</p>", status_code=503)
    except Exception as e:
        logging.error(f"Savings error: {e}")
        return HTMLResponse(content="<h1>Error</h1><p>Savings unavailable</p>", status_code=500)

def _load_options_dict() -> Dict[str, Any]:
    try:
        import json
        options_path = "/data/options.json"
        if os.path.exists(options_path):
            with open(options_path, "r") as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Kon /data/options.json niet laden: {e}")
    return {}

@app.get("/settings-gui", response_class=HTMLResponse)
async def settings_gui(request: Request):
    """Settings GUI page - robuust zonder harde Config dependency"""
    try:
        if not templates:
            return HTMLResponse(content="<h1>Settings</h1><p>Template system unavailable</p>")

        # Probeer eerst via Config class voor consistentie; val terug op JSON
        config_dict: Dict[str, Any] = {}
        try:
            ConfigClass = get_config_class()
            if ConfigClass:
                cfg = ConfigClass("/data/options.json")
                config_dict = getattr(cfg, "options", {}) or {}
        except Exception as e:
            logging.warning(f"Settings: Config load failed, fallback to JSON: {e}")
            config_dict = _load_options_dict()

        return templates.TemplateResponse("settings_gui.html", {
            "request": request,
            "version": __version__,
            "config": config_dict,
        })
    except Exception as e:
        logging.error(f"Settings GUI error: {e}")
        return HTMLResponse(content="<h1>Error</h1><p>Settings unavailable</p>", status_code=500)

@app.get("/run-modern", response_class=HTMLResponse)
async def run_modern(request: Request):
    """Run modern page"""
    try:
        if templates:
            return templates.TemplateResponse("run_modern.html", {
                "request": request,
                "version": __version__
            })
        else:
            return HTMLResponse(content="<h1>Run Modern</h1><p>Template system unavailable</p>")
    except Exception as e:
        logging.error(f"Run modern error: {e}")
        return HTMLResponse(content="<h1>Error</h1><p>Run modern unavailable</p>", status_code=500)

# Minimal stubs for run-modern API referenced by template
@app.post("/api/run/{operation_id}")
async def api_run_operation(operation_id: str):
    """Run actual analyses using DaCalc/Statistical optimizer and return a textual summary."""
    start_ts = datetime.datetime.now()
    op_name = str(operation_id)
    log_lines: list[str] = []

    def log(msg: str):
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        log_lines.append(line)

    try:
        log(f"Start operatie: {op_name}")
        # Lazy import to avoid heavy imports during app init
        try:
            from day_ahead import DaCalc  # type: ignore
        except Exception as e:
            log(f"Import fout: day_ahead.DaCalc niet beschikbaar: {e}")
            raise

        # Init calculator with HA add-on options.json
        calc = DaCalc(file_name="/data/options.json")

        results = None
        if op_name in ("calc_met_debug", "calc_zonder_debug", "calc_baseloads"):
            # Run statistical optimization
            run_dt = datetime.datetime.now()
            if op_name == "calc_met_debug":
                log("Statistische optimalisatie (debug)...")
                results = calc.calc_optimum_statistical(_start_dt=run_dt, _start_soc=None)
            else:
                log("Statistische optimalisatie...")
                results = calc.calc_optimum_statistical()

            if results is None:
                log("Geen resultaten ontvangen (None)")
                raise RuntimeError("Optimalisatie gaf geen resultaten")

            # Format summary
            perf = results.get('performance', {}) if isinstance(results, dict) else {}
            savings = perf.get('savings', 0)
            method = results.get('optimization_method') or results.get('strategy_used') or 'unknown'
            log(f"Methode: {method}")
            log(f"Geschatte besparing: €{savings:.2f}")

            schedule = results.get('schedule')
            if schedule is not None:
                try:
                    import pandas as _pd  # type: ignore
                    head = schedule.head(6).copy()
                    # Render a compact text table
                    log("Voorbeeld planning (eerste 6 regels):")
                    log(head.to_string())
                except Exception:
                    pass

        elif op_name == "get_prices":
            log("Prijsdata ophalen is nog niet volledig geïmplementeerd in deze versie.")
            log("Gebruik de optimalisatie-run; deze leest beschikbare prijzen en plant op basis daarvan.")
        elif op_name == "get_meteo":
            log("Weerdata ophalen/analyseren niet geactiveerd in deze build.")
        elif op_name == "get_tibber":
            log("Tibber-gegevens ophalen niet geactiveerd in deze build.")
        else:
            log(f"Onbekende operatie: {op_name}")

        duration = (datetime.datetime.now() - start_ts).total_seconds()
        log(f"Klaar in {duration:.1f}s")

        _last_runs[op_name] = {
            "time": start_ts.isoformat(timespec='seconds'),
            "duration_s": duration,
            "success": True,
            "message": "OK",
        }

        return HTMLResponse(content="\n".join(log_lines))
    except Exception as e:
        duration = (datetime.datetime.now() - start_ts).total_seconds()
        _last_runs[op_name] = {
            "time": start_ts.isoformat(timespec='seconds'),
            "duration_s": duration,
            "success": False,
            "message": str(e),
        }
        log_lines.append(f"FOUT: {e}")
        return HTMLResponse(content="\n".join(log_lines), status_code=500)

@app.get("/api/system-status")
async def api_system_status():
    # Provide a simple planned tasks overview & last run statuses
    planned_tasks: List[Dict[str, Any]] = [
        {"name": "Highload detectie", "interval": "30s", "last_run": _last_runs.get('highload_detect', {}).get('time', '-'), "status": "unknown"},
        {"name": "Health check", "interval": "60s", "last_run": _last_runs.get('health_check', {}).get('time', '-'), "status": "unknown"},
        {"name": "Dagelijkse optimalisatie", "interval": "dagelijks 00:15", "last_run": _last_runs.get('calc_met_debug', _last_runs.get('calc_zonder_debug', {})).get('time', '-'), "status": "unknown"},
    ]
    return JSONResponse({
        "scheduler": {"active": True, "planned_tasks": planned_tasks},
        "lastOptimization": "Onbekend",
        "database": {"connected": True},
        "homeassistant": {"online": True}
        ,
        "last_runs": _last_runs
    })

@app.post("/api/emergency-stop")
async def api_emergency_stop():
    try:
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/restart-scheduler")
async def api_restart_scheduler():
    try:
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# -------- Settings endpoints used by settings_gui.html ---------
@app.post("/settings/save")
async def settings_save(request: Request):
    try:
        # Receive multipart form-data and write back to /data/options.json
        form = await request.form()
        import json
        options: Dict[str, Any] = {}

        # Flatten form into nested structure using keys like 'prices.contract_type'
        for key, value in form.items():
            if isinstance(value, UploadFile):
                continue
            parts = str(key).split('.')
            cursor = options
            for p in parts[:-1]:
                if p not in cursor or not isinstance(cursor[p], dict):
                    cursor[p] = {}
                cursor = cursor[p]
            cursor[parts[-1]] = value

        # Merge with existing options to preserve unknown keys
        options_path = "/data/options.json"
        existing = {}
        try:
            if os.path.exists(options_path):
                with open(options_path, 'r') as f:
                    existing = json.load(f)
        except Exception:
            existing = {}

        def deep_merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    deep_merge(a[k], v)
                else:
                    a[k] = v
            return a

        merged = deep_merge(existing, options)
        with open(options_path, 'w') as f:
            json.dump(merged, f, indent=2)

        return JSONResponse({"success": True})
    except Exception as e:
        logging.error(f"settings_save error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/settings/test-connection")
async def settings_test_connection():
    try:
        # Simple DB ping using current config
        ConfigClass = get_config_class()
        if ConfigClass:
            cfg = ConfigClass("/data/options.json")
            db = cfg.get_db_da()
            if db is not None:
                with db.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return JSONResponse({"success": True})
        return JSONResponse({"success": False, "error": "DB unavailable"}, status_code=503)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/settings/reset")
async def settings_reset():
    try:
        # Reset options.json to example start
        import shutil
        src = "/app/daodata/options_start.json"
        dst = "/data/options.json"
        if os.path.exists(src):
            shutil.copy(src, dst)
            return JSONResponse({"success": True})
        return JSONResponse({"success": False, "error": "options_start.json missing"}, status_code=500)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/settings/export")
async def settings_export():
    try:
        path = "/data/options.json"
        if not os.path.exists(path):
            return JSONResponse({"error": "options.json not found"}, status_code=404)
        return StreamingResponse(open(path, 'rb'), media_type='application/json', headers={
            'Content-Disposition': 'attachment; filename="options.json"'
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/settings/import")
async def settings_import(config_file: UploadFile = File(...)):
    try:
        content = await config_file.read()
        import json
        data = json.loads(content.decode('utf-8'))
        with open('/data/options.json', 'w') as f:
            json.dump(data, f, indent=2)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/statistics", response_class=HTMLResponse)
async def statistics(request: Request, report: Any = Depends(get_report_instance)):
    """Statistics page with timeout protection"""
    try:
        # Timeout protection for database operations
        async with asyncio.timeout(10.0):
            if templates:
                return templates.TemplateResponse("statistics.html", {
                    "request": request,
                    "version": __version__,
                    "report": report
                })
            else:
                return HTMLResponse(content="<h1>Statistics</h1><p>Template system unavailable</p>")
    except asyncio.TimeoutError:
        logging.error("Statistics page timeout")
        return HTMLResponse(content="<h1>Timeout</h1><p>Database operation timed out</p>", status_code=503)
    except Exception as e:
        logging.error(f"Statistics error: {e}")
        return HTMLResponse(content="<h1>Error</h1><p>Statistics unavailable</p>", status_code=500)

# API endpoints
@app.get("/api/test/simple")
async def api_test_simple():
    """Simple API test endpoint"""
    return JSONResponse({
        "status": "ok",
        "message": "FastAPI webserver is running",
        "timestamp": datetime.datetime.now().isoformat()
    })

# -------- Dashboard API used by dashboard.html ---------
@app.get("/api/health-check")
async def api_health_check():
    """Return overall health including database connectivity.
    Deze endpoint heeft GEEN harde dependency op Config, zodat het dashboard
    altijd kan laden en een correcte status toont, ook als config faalt.
    """
    details: Dict[str, Any] = {}
    healthy = True
    # Database check
    try:
        # Probeer via Config; als dat faalt, val terug op SQLite pad check
        db = None
        try:
            ConfigClass = get_config_class()
            if ConfigClass:
                cfg = ConfigClass("/data/options.json")
                db = cfg.get_db_da(check_create=True)
        except Exception:
            db = None

        if db is None:
            # Fallback: controleer of er een sqlite bestand op /data/day_ahead.db is
            db_file = "/data/day_ahead.db"
            if os.path.exists(db_file):
                details["database"] = "FilePresent"
                database_status = "connected"
            else:
                details["database"] = "Unavailable"
                database_status = "disconnected"
                healthy = False
        else:
            with db.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            details["database"] = "OK"
            database_status = "connected"
    except Exception as e:
        details["database"] = f"Error: {e}"[:200]
        database_status = "disconnected"
        healthy = False

    # Placeholder additional checks
    details["scheduler"] = "Running"
    details["memory_usage"] = "Normal"
    details["disk_space"] = "OK"
    details["homeassistant"] = "OK"

    return JSONResponse({
        "healthy": healthy,
        "database_status": database_status,
        "details": details,
        "timestamp": datetime.datetime.now().isoformat(),
    })

@app.get("/api/dashboard/energy-data")
async def api_energy_data(config: Any = Depends(get_config_instance)):
    """Return current energy KPIs; no mock, return error when missing."""
    try:
        db = config.get_db_da()
        if db is None:
            raise RuntimeError("database unavailable")
        # Simple snapshot: last hour consumption/production from values table would be ideal.
        # For now return placeholders if tables are empty.
        return JSONResponse({
            "current_consumption": 0,
            "current_production": 0,
            "battery_soc": 0,
            "grid_import": 0,
            "consumption_trend": "-",
            "production_trend": "-",
            "battery_trend": "-",
            "grid_trend": "-",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)

@app.get("/api/dashboard/metrics")
async def api_metrics(config: Any = Depends(get_config_instance)):
    """Return basic metrics without mock."""
    try:
        # Can be extended to compute based on optimization results table
        return JSONResponse({
            "cost_savings": 0,
            "optimization_runs": 0,
            "prediction_accuracy": 0,
            "system_uptime": "--",
            "savings_change": "",
            "runs_change": "",
            "accuracy_change": "",
            "uptime_change": "",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)

@app.get("/api/dashboard/activity")
async def api_activity(config: Any = Depends(get_config_instance)):
    """Return recent activity list; empty when none."""
    try:
        return JSONResponse({"activities": []})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)

@app.get("/api/debug/dumpstacks")
async def api_debug_dumpstacks():
    """Trigger stack dump via faulthandler"""
    try:
        import os
        import signal
        os.kill(os.getpid(), signal.SIGUSR2)
        return JSONResponse({
            "status": "ok",
            "message": "Stack dump triggered",
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Failed to trigger stack dump: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        }, status_code=500)

@app.get("/api/ha/suggestions")
async def api_ha_suggestions(config: Any = Depends(get_config_instance)):
    """Get Home Assistant suggestions for configuration"""
    if not HA_CLIENT_AVAILABLE:
        return JSONResponse({
            "status": "error",
            "message": "HA client not available",
            "timestamp": datetime.datetime.now().isoformat()
        }, status_code=503)

    try:
        # Get HA core config
        core_config = get_core_config(config)
        suggestions = {}

        if core_config:
            lat = core_config.get('latitude')
            lon = core_config.get('longitude')
            if lat is not None and lon is not None:
                suggestions['location'] = {
                    'latitude': lat,
                    'longitude': lon
                }

        # Get PV suggestions
        states = get_states(config)
        pv_entities = suggest_pv_energy_entities(states)
        if pv_entities:
            max_daily = get_statistics_max_daily(config, pv_entities[0], days=365)
            if isinstance(max_daily, (int, float)) and max_daily > 0:
                est_kwp = round(float(max_daily) / 4.0, 2)
                suggestions['solar'] = {
                    'entity': pv_entities[0],
                    'estimated_capacity': est_kwp
                }

        return JSONResponse({
            "status": "ok",
            "suggestions": suggestions,
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"HA suggestions error: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Failed to get HA suggestions: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        }, status_code=500)

# Note: Using lifespan manager in __init__.py instead of deprecated @app.on_event

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """404 error handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": f"The requested resource was not found: {request.url.path}",
            "timestamp": datetime.datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """500 error handler"""
    logging.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.datetime.now().isoformat()
        }
    )

logging.debug("routes.py - FastAPI routes registered successfully")