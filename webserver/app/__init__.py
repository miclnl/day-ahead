#!/usr/bin/env python3
"""
DAO FastAPI application with async support and better performance
"""
import sys
import os
import traceback
import logging
import time
import pathlib
import faulthandler
from contextlib import asynccontextmanager

# Laad add-on configuratie VOOR logging setup
try:
    from .addon_config import addon_config
    log_level = addon_config.get_log_level()
    logger = logging.getLogger(__name__)
    logger.debug(f"Add-on configuratie geladen - loglevel: {log_level}")
except ImportError as e:
    log_level = "info"
    logger = logging.getLogger(__name__)
    logger.warning(f"Kon addon_config niet importeren: {e}")

# Configure logging based on addon config
level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR
}

logging_level = level_map.get(log_level, logging.INFO)

# Configure logging early
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s',
    handlers=[logging.StreamHandler()],
    force=True  # Force override any existing config
)

# Set specific loggers
logging.getLogger('fastapi').setLevel(logging_level)
logging.getLogger('uvicorn').setLevel(logging_level)
logging.getLogger('uvicorn.access').setLevel(logging_level)
logging.getLogger('uvicorn.error').setLevel(logging_level)

logger = logging.getLogger(__name__)

# Debug information
logger.debug(f"Initializing FastAPI app from {__file__}")
logger.debug(f"Working directory: {os.getcwd()}")
logger.debug(f"Python version: {sys.version}")
logger.debug(f"Python executable: {sys.executable}")
logger.debug(f"Python path: {sys.path[:5]}...")
logger.debug(f"Logging level set to: {log_level.upper()}")

# Ensure log directory exists
try:
    log_dir = pathlib.Path('/data/log')
    log_dir.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.warning(f"Kon log directory niet aanmaken: {e}")

# Enable faulthandler
try:
    _fh_path = log_dir / f"faulthandler_worker_{os.getpid()}.log"
    _fh_file = open(_fh_path, 'a', buffering=1)
    faulthandler.enable(_fh_file, all_threads=True)
    import signal as _sig
    faulthandler.register(_sig.SIGUSR2, file=_fh_file, all_threads=True)
    logger.debug(f"Faulthandler geactiveerd: {_fh_path}")
except Exception as e:
    logger.warning(f"Faulthandler niet geactiveerd: {e}")

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
prog_dir = os.path.join(parent_dir, 'prog')

logger.debug(f"Current dir: {current_dir}")
logger.debug(f"Parent dir: {parent_dir}")
logger.debug(f"Prog dir: {prog_dir}")

# Add multiple paths to ensure imports work
paths_to_add = [
    prog_dir,  # /app/dao/prog
    parent_dir,  # /app/dao
    os.path.join(parent_dir, '..'),  # /app
]

for path in paths_to_add:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        logger.debug(f"Added {path} to Python path")
    else:
        logger.debug(f"Path {path} already in sys.path or doesn't exist")

logger.debug(f"Final Python path: {sys.path[:5]}...")

# Import FastAPI
try:
    logger.debug("Attempting to import FastAPI...")
    start_time = time.time()
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    import_time = time.time() - start_time
    logger.debug(f"FastAPI import successful in {import_time:.3f}s")
except ImportError as e:
    logger.critical(f"FastAPI import failed: {e}")
    traceback.print_exc()
    raise

# Global state for singleton instances
_config_instance = None
_report_instance = None
_config_lock = None
_report_lock = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 FastAPI application starting up...")
    try:
        # Initialize global state
        global _config_instance, _report_instance, _config_lock, _report_lock
        import threading
        _config_lock = threading.Lock()
        _report_lock = threading.Lock()
        logger.info("✅ Global state initialized")

        # Clear any existing cache
        from .routes import _cache_clear
        _cache_clear()
        logger.info("✅ Cache cleared on startup")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 FastAPI application shutting down...")
    try:
        # Cleanup global state
        _config_instance = None
        _report_instance = None

        # Clear cache on shutdown
        from .routes import _cache_clear
        _cache_clear()
        logger.info("✅ Global state and cache cleaned up")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Create FastAPI app
try:
    logger.debug("Creating FastAPI app...")
    start_time = time.time()

    from dao.prog.version import __version__ as _DAO_VERSION
    app = FastAPI(
        title="DAO Modern Enhanced",
        description="Day Ahead Optimizer - Statistical Intelligence",
        version=_DAO_VERSION,
        lifespan=lifespan,
        docs_url=None,  # Disable auto-generated docs
        redoc_url=None,  # Disable auto-generated docs
    )

    creation_time = time.time() - start_time
    logger.debug(f"FastAPI app created successfully in {creation_time:.3f}s")
except Exception as e:
    logger.critical(f"FastAPI app creation failed: {e}")
    traceback.print_exc()
    raise

# Setup static files and templates
try:
    static_dir = os.path.join(current_dir, 'static')
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.debug(f"Static files mounted from: {static_dir}")

    templates_dir = os.path.join(current_dir, 'templates')
    if os.path.exists(templates_dir):
        templates = Jinja2Templates(directory=templates_dir)

        # Add useful modules to template context
        import datetime
        import math
        import random

        templates.env.globals.update({
            'now': datetime.datetime.now,  # Function to get current time
            'datetime': datetime.datetime,  # datetime class for other operations
            'timedelta': datetime.timedelta,
            'math': math,
            'random': random,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'len': len,
            'round': round,
            'min': min,
            'max': max,
        })

        logger.debug(f"Templates loaded from: {templates_dir}")
    else:
        templates = None
        logger.warning(f"Templates directory not found: {templates_dir}")
except Exception as e:
    logger.warning(f"Static/templates setup failed: {e}")
    templates = None

# Import routes with timeout protection
logger.debug("Attempting to import routes...")
import signal

def timeout_handler(signum, frame):
    logger.critical("Routes import timeout - deadlock detected!")
    raise TimeoutError("Routes import timed out")

try:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(15)

    logger.debug("Starting routes import with timeout protection...")
    start_time = time.time()

    from . import routes
    routes_time = time.time() - start_time

    signal.alarm(0)
    logger.debug(f"Routes import successful in {routes_time:.3f}s")

except TimeoutError as e:
    signal.alarm(0)
    logger.critical(f"Routes import timed out: {e}")
    traceback.print_exc()
except Exception as e:
    signal.alarm(0)
    logger.critical(f"Routes import failed: {e}")
    traceback.print_exc()

    # Add minimal fallback routes
    logger.debug("Adding minimal fallback routes...")

    @app.get("/")
    async def index():
        return RedirectResponse(url="/health")

    @app.get("/health")
    async def health():
        return JSONResponse({
            'status': 'healthy',
            'version': _DAO_VERSION,
            'webserver': 'fastapi',
            'error': 'Routes import failed - minimal mode active'
        })

    @app.exception_handler(404)
    async def not_found(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=404,
            content={'error': 'Not found', 'message': str(exc)}
        )

    @app.exception_handler(500)
    async def server_error(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=500,
            content={'error': 'Server error', 'message': str(exc)}
        )

    logger.debug("Fallback routes added")

logger.debug("FastAPI app initialization complete")
logger.info(f"DAO webserver ready - FastAPI app created successfully")

# Expose both FastAPI app and ASGI app for compatibility
asgi_app = app

# Export templates for routes
__all__ = ['app', 'asgi_app', 'templates']