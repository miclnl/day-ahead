import sys
import os

sys.path.append("../../")
from dao.prog.da_config import get_config

app_datapath = "app/static/data/"
try:
    port = get_config(app_datapath + "options.json", ["dashboard", "port"], 5001)
except:
    # Fallback to 5001 if config not available
    port = 5001
workers = 2
bind = f"0.0.0.0:{port}"
forwarded_allow_ips = "*"
secure_scheme_headers = {"X-Forwarded-Proto": "https"}
timeout = 120
