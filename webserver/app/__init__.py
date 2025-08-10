from flask import Flask
import sys
import os

# Add parent directory to Python path to access prog modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
prog_dir = os.path.join(parent_dir, 'prog')
if prog_dir not in sys.path:
    sys.path.insert(0, prog_dir)

app = Flask(__name__)

# Import routes after setting up path
try:
    from . import routes
except ImportError as e:
    print(f"Failed to import routes: {e}")
    # Try alternative import method
    from dao.webserver.app import routes


#  if __name__ == '__main__':
#      app.run()
#  app.run(port=5000, host='0.0.0.0')
#  if __name__ == '__main__':
#      app.run(port=5000, host='0.0.0.0')
