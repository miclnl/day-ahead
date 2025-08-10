import os

if not os.path.lexists("app/static/data"):
    os.symlink("../data", "app/static/data")

from app import app

if __name__ == "__main__":
    # Use ingress port 8099 for Home Assistant integration
    port = int(os.getenv('PORT', 8099))
    # Only allow connections from ingress proxy for security
    host = "172.30.32.2" if os.getenv('INGRESS', 'false').lower() == 'true' else "0.0.0.0"
    app.run(port=port, host=host)
