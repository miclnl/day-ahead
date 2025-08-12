#!/usr/bin/env python3
"""
Uvicorn configuration for DAO FastAPI webserver
Pure Uvicorn without Gunicorn for better performance
"""
import uvicorn
import os
import logging
import json

# Load addon config for log level
def get_addon_log_level():
    """Get log level from addon configuration"""
    try:
        config_file = "/data/options.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            log_level = config.get("log_level", "info")
            # Validate log level
            valid_levels = ["debug", "info", "warning", "error"]
            if log_level.lower() in valid_levels:
                return log_level.lower()
    except Exception as e:
        print(f"Warning: Could not load addon config: {e}")
    return "info"

# Server configuration
HOST = "0.0.0.0"
PORT = 5001
WORKERS = 2

# Performance settings
WORKER_CONNECTIONS = 1000
BACKLOG = 2048
LIMIT_MAX_REQUESTS = 1000
LIMIT_CONCURRENCY = 1000

# Timeout settings
TIMEOUT_KEEP_ALIVE = 5
TIMEOUT_GRACEFUL_SHUTDOWN = 30

# Logging configuration
LOG_LEVEL = get_addon_log_level()
ACCESS_LOG = True
USE_COLORS = False

# Development settings
RELOAD = False
RELOAD_DIRS = []

def get_uvicorn_config():
    """Get Uvicorn configuration dictionary"""
    return {
        "app": "app:asgi_app",
        "host": HOST,
        "port": PORT,
        "workers": WORKERS,
        "worker_connections": WORKER_CONNECTIONS,
        "backlog": BACKLOG,
        "limit_max_requests": LIMIT_MAX_REQUESTS,
        "limit_concurrency": LIMIT_CONCURRENCY,
        "timeout_keep_alive": TIMEOUT_KEEP_ALIVE,
        "timeout_graceful_shutdown": TIMEOUT_GRACEFUL_SHUTDOWN,
        "log_level": LOG_LEVEL,
        "access_log": ACCESS_LOG,
        "use_colors": USE_COLORS,
        "reload": RELOAD,
        "reload_dirs": RELOAD_DIRS,
        "loop": "asyncio",
        "http": "httptools",
        "ws": "websockets",
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
        "server_header": False,
        "date_header": False,
    }

if __name__ == "__main__":
    # Run Uvicorn directly
    config = get_uvicorn_config()
    print(f"Starting Uvicorn with log level: {LOG_LEVEL}")
    uvicorn.run(**config)
