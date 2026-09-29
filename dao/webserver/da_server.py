"""Development entry point: run the dashboard with Flask's built-in server.

The data directory is ../data relative to dao/webserver, the same layout the
add-on uses. Set DAO_ALLOW_DIRECT=1 when running outside Home Assistant,
otherwise every request is refused because it does not come through ingress.
"""

import argparse
import os

from app import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Start Flask in debug mode")
    args = parser.parse_args()

    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(port=port, host="0.0.0.0", debug=args.debug, use_reloader=args.debug)
