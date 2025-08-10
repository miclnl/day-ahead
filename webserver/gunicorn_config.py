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
