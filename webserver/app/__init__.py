#!/usr/bin/env python3
"""
DAO Flask application with debug WSGI loading
"""
import sys
import os
import traceback

# Debug information
print(f"DEBUG: Initializing Flask app from {__file__}")
print(f"DEBUG: Working directory: {os.getcwd()}")
print(f"DEBUG: Python version: {sys.version}")

# Add parent directory to Python path to access prog modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
prog_dir = os.path.join(parent_dir, 'prog')

print(f"DEBUG: Current dir: {current_dir}")
print(f"DEBUG: Parent dir: {parent_dir}")  
print(f"DEBUG: Prog dir: {prog_dir}")

if os.path.exists(prog_dir) and prog_dir not in sys.path:
    sys.path.insert(0, prog_dir)
    print(f"DEBUG: Added {prog_dir} to Python path")

# Import Flask
try:
    from flask import Flask
    print("DEBUG: Flask import successful")
except ImportError as e:
    print(f"CRITICAL: Flask import failed: {e}")
    traceback.print_exc()
    raise

# Create Flask app
try:
    app = Flask(__name__)
    print("DEBUG: Flask app created successfully")
except Exception as e:
    print(f"CRITICAL: Flask app creation failed: {e}")
    traceback.print_exc()
    raise

# Import routes AFTER app is created to avoid circular import
print("DEBUG: Attempting to import routes...")
import sys
import time
import signal

def timeout_handler(signum, frame):
    print("CRITICAL: Routes import timeout - deadlock detected!")
    raise TimeoutError("Routes import timed out")

try:
    # Set a timeout for routes import to detect deadlocks
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout
    
    print("DEBUG: Starting routes import with timeout protection...")
    
    # Import the routes module which will register routes with app
    from . import routes
    
    # Cancel the timeout
    signal.alarm(0)
    
    print("DEBUG: Routes import successful")
except TimeoutError as e:
    signal.alarm(0)
    print(f"CRITICAL: Routes import timed out: {e}")
    traceback.print_exc()
except Exception as e:
    signal.alarm(0)
    print(f"CRITICAL: Routes import failed: {e}")
    traceback.print_exc()
    
    # Try fallback minimal routes
    print("DEBUG: Adding minimal fallback routes...")
    from flask import jsonify, redirect
    
    @app.route('/')
    def index():
        return redirect('/health')
    
    @app.route('/health')  
    def health():
        return jsonify({
            'status': 'healthy',
            'version': '1.3.12',
            'webserver': 'running',
            'error': 'Routes import failed - minimal mode active'
        })
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found', 'message': str(e)}), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'error': 'Server error', 'message': str(e)}), 500
    
    print("DEBUG: Fallback routes added")

print("DEBUG: Flask app initialization complete")