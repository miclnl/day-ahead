from flask import Flask
import datetime


class IngressMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        ingress_path = environ.get("HTTP_X_INGRESS_PATH", "").rstrip("/")

        if ingress_path:
            environ["SCRIPT_NAME"] = ingress_path

        return self.app(environ, start_response)


# sys.path.append("../")

app = Flask(__name__)
app.secret_key = "secret_cookie_key"
app.wsgi_app = IngressMiddleware(app.wsgi_app)


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


from . import routes
from .v2.routes import v2
from .v2.api.routes import api

app.register_blueprint(v2, name="v2", url_prefix="/v2")
app.register_blueprint(api, name="api", url_prefix="/v2/api")

#  if __name__ == '__main__':
#      app.run()
#  app.run(port=5000, host='0.0.0.0')
#  if __name__ == '__main__':
#      app.run(port=5000, host='0.0.0.0')
