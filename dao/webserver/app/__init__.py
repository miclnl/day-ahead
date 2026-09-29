import datetime
import ipaddress
import logging
import os
import re
import secrets
from pathlib import Path

from flask import Flask, abort, request, send_from_directory
from flask_wtf.csrf import CSRFProtect

# The add-on data directory. The web server runs with dao/webserver as its
# working directory, so this resolves to /root/dao/data in the container,
# which run.sh links to /config/dao_data. It must never live inside Flask's
# static folder: everything under static/ is served to anyone who can reach
# the port, and this directory holds secrets.json and the database.
DATA_DIR = Path("../data")
IMAGES_DIR = (DATA_DIR / "images").resolve()

# Home Assistant's supervisor proxies ingress traffic from this fixed address
# (see the add-on ingress documentation) and always adds X-Ingress-Path.
# Requests that do not come through it are only accepted when the operator
# opts in: the add-on option "allow_direct_access" (run.sh turns it into
# DAO_ALLOW_DIRECT=1) or the same variable set by hand during development.
SUPERVISOR_NETWORK = ipaddress.ip_network("172.30.32.2/32")
ALLOW_DIRECT = os.environ.get("DAO_ALLOW_DIRECT") == "1"

_IMAGE_NAME = re.compile(r"^[\w.\-]+\.png$")


class IngressMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        ingress_path = environ.get("HTTP_X_INGRESS_PATH", "").rstrip("/")

        if ingress_path:
            environ["SCRIPT_NAME"] = ingress_path

        return self.app(environ, start_response)


def _load_secret_key() -> str:
    """Persist one random secret key per installation.

    A hard-coded key makes every session cookie forgeable. The key is stored
    next to the other runtime files so it survives a restart, and generated
    once when it does not exist yet.
    """
    key_file = DATA_DIR / ".flask_secret"
    try:
        return key_file.read_text(encoding="utf-8").strip() or secrets.token_hex(32)
    except FileNotFoundError:
        pass
    except OSError as ex:
        logging.warning(f"Kon {key_file} niet lezen ({ex}), tijdelijke sleutel")
        return secrets.token_hex(32)
    key = secrets.token_hex(32)
    try:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(key, encoding="utf-8")
        os.chmod(key_file, 0o600)
    except OSError as ex:
        logging.warning(f"Kon {key_file} niet schrijven ({ex}), tijdelijke sleutel")
    return key


app = Flask(__name__)
app.secret_key = _load_secret_key()
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"
app.config["SESSION_COOKIE_HTTPONLY"] = True
# Every state-changing form carries a CSRF token (see the templates); HTMX
# sends it in the X-CSRFToken header from the body tag of the v2 base template.
# The token does not expire with time, only with the session, so a dashboard
# tab that stays open overnight keeps working. The REST endpoints under /api
# are exempted explicitly where they are defined: they are called from Home
# Assistant automations that cannot obtain a token.
app.config["WTF_CSRF_TIME_LIMIT"] = None
csrf = CSRFProtect(app)
app.wsgi_app = IngressMiddleware(app.wsgi_app)


def _from_supervisor() -> bool:
    try:
        return ipaddress.ip_address(request.remote_addr or "") in SUPERVISOR_NETWORK
    except ValueError:
        return False


@app.before_request
def ingress_only():
    """Refuse requests that bypass Home Assistant's ingress.

    The UI has no login of its own; HA's ingress is the authentication. A
    direct port mapping would expose the configuration editor, the secrets
    editor and the task runner to anyone on the network.
    """
    if ALLOW_DIRECT:
        return None
    if _from_supervisor() and "X-Ingress-Path" in request.headers:
        return None
    abort(401)


@app.route("/images/<name>")
def image(name: str):
    """Serve one generated graph. Only plain PNG names, only from images/."""
    if not _IMAGE_NAME.match(name):
        abort(404)
    return send_from_directory(IMAGES_DIR, name, max_age=3600)


@app.template_filter("human_ts")
def _human_ts(value):
    """Render a Unix epoch as a human-readable local-time string.

    Used by templates that need to show timestamps from fast-control
    events and task metadata. Returns '—' for None or non-numeric
    values so missing data shows as a placeholder rather than an
    epoch dump.
    """
    if value is None:
        return "—"
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return str(value)
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


from . import routes  # noqa: E402
from .v2.routes import v2  # noqa: E402
from .v2.api.routes import api  # noqa: E402

app.register_blueprint(v2, name="v2", url_prefix="/v2")
app.register_blueprint(api, name="api", url_prefix="/v2/api")
csrf.exempt(api)
