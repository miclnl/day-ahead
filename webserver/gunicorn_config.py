# Hardcoded port for DAO Modern to ensure consistency  
port = 5001
workers = 2
bind = f"0.0.0.0:{port}"
forwarded_allow_ips = "*"
secure_scheme_headers = {"X-Forwarded-Proto": "https"}
timeout = 120
