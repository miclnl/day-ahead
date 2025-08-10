# Use ingress port 8099 for Home Assistant integration
port = 8099
workers = 2
bind = f"0.0.0.0:{port}"
forwarded_allow_ips = "*"
secure_scheme_headers = {"X-Forwarded-Proto": "https"}
timeout = 120

# Force fresh module imports to prevent cache issues
preload_app = True
max_worker_connections = 1000
worker_class = "sync"

# Enable reload for development/debugging
reload = True
reload_extra_files = ["app/__init__.py", "app/routes.py"]

# Enhanced debug logging - responds to Home Assistant DEBUG log level
import os
log_level = os.getenv('LOG_LEVEL', 'info').lower()
if log_level == 'debug':
    loglevel = 'debug'
    access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
    accesslog = '-'  # Log to stdout
    errorlog = '-'   # Log to stderr
else:
    loglevel = 'info'

# Additional debug settings
capture_output = True
enable_stdio_inheritance = True
