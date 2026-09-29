import sys
from pathlib import Path

sys.path.append("../../")
from dao.prog.config.loader import ConfigurationLoader
from dao.prog.config.models.dashboard import DashboardConfig

# Working directory is dao/webserver, so ../data is the add-on data directory
# (linked to /config/dao_data by run.sh). It must not live under app/static.
app_datapath = "../data/"
try:
    _loader = ConfigurationLoader(Path(app_datapath + "options.json"))
    _config = _loader.load_and_validate()
    port = _config.dashboard.port
except Exception:
    port = DashboardConfig().port
workers = 2
bind = f"0.0.0.0:{port}"
# Only Home Assistant's supervisor sits in front of this server; nobody else
# may rewrite the scheme or the script name through X-Forwarded-* headers.
forwarded_allow_ips = "172.30.32.2"
secure_scheme_headers = {"X-Forwarded-Proto": "https"}
timeout = 120
