#!/usr/bin/env python3
"""
Debug script to check if gunicorn imports the same app as da_server.py
"""
import sys
import os

print("=== DEBUG GUNICORN IMPORT ===")
print(f"Python path: {sys.path[:3]}")
print(f"Working directory: {os.getcwd()}")

# Change to webserver directory like gunicorn does
os.chdir('/app/dao/webserver')
print(f"Changed working directory to: {os.getcwd()}")

# Add current directory to Python path
if '.' not in sys.path:
    sys.path.insert(0, '.')

print("=== IMPORTING APP MODULE ===")
try:
    # Import exactly what gunicorn imports: app:app
    import app
    print(f"App module imported from: {app.__file__}")
    print(f"App object: {app.app}")
    
    # Check if it has the debug print statements
    import inspect
    source = inspect.getsource(app)
    if "DEBUG: Initializing Flask app" in source:
        print("✅ App module contains DEBUG statements - using new version")
    else:
        print("❌ App module does NOT contain DEBUG statements - using old version")
        
except Exception as e:
    print(f"❌ Failed to import app: {e}")
    import traceback
    traceback.print_exc()

print("=== CHECKING ROUTES MODULE ===")
try:
    from app import routes
    import inspect
    source = inspect.getsource(routes)
    if "Lazy loaded" in source:
        print("✅ Routes module contains lazy loading - using new version")
    else:
        print("❌ Routes module does NOT contain lazy loading - using old version")
except Exception as e:
    print(f"❌ Failed to import routes: {e}")
    import traceback
    traceback.print_exc()

print("=== VERSION CHECK ===")
try:
    print(f"App version in fallback: {getattr(app, '__version__', 'not found')}")
except Exception as e:
    print(f"Version check failed: {e}")