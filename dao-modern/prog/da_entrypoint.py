#!/usr/bin/env python3
"""
Secure entrypoint for DAO Modern without shell dependencies.
Replaces run.sh with pure Python implementation.
"""

import os
import sys
import json
import shutil
import subprocess
import logging
import signal
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class DaoEntrypoint:
    def __init__(self):
        self.config_dir = Path("/config/dao_modern_data")
        self.app_dir = Path("/app/dao")
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def setup_directories(self):
        """Setup data directories and symlinks"""
        logger.info("Setting up directories and configuration")
        
        # Create config directory if it doesn't exist
        if not self.config_dir.exists():
            logger.info("Creating dao_modern_data directory and copying default files")
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy default data
            default_data = Path("/app/daodata")
            if default_data.exists():
                shutil.copytree(default_data, self.config_dir, dirs_exist_ok=True)
            
            # Setup default config files
            options_file = self.config_dir / "options.json"
            if not options_file.exists():
                shutil.copy(self.config_dir / "options_start.json", options_file)
            
            secrets_file = self.config_dir / "secrets.json" 
            if not secrets_file.exists():
                shutil.copy(self.config_dir / "secrets_vb.json", secrets_file)
        else:
            logger.info("dao_modern_data directory already exists")
        
        # Create symlink for data access
        data_link = self.app_dir / "data"
        if not data_link.exists():
            logger.info("Creating data symlink")
            data_link.symlink_to(self.config_dir)
        
        # Create symlink for webserver static data
        static_data = self.app_dir / "webserver" / "app" / "static" / "data"
        if not static_data.exists():
            logger.info("Creating webserver static data symlink")
            static_data.symlink_to(self.config_dir)
    
    def setup_environment(self):
        """Setup environment variables"""
        # Set MIP library path if exists
        mip_dir = self.app_dir / "prog" / "miplib"
        if mip_dir.exists():
            os.environ["PMIP_CBC_LIBRARY"] = str(mip_dir / "lib" / "libCbc.so")
            os.environ["LD_LIBRARY_PATH"] = str(mip_dir / "lib")
        
        # Set Python path
        os.environ["PYTHONPATH"] = f"/app:/app/dao:/app/dao/prog"
    
    def check_database(self):
        """Check and setup database"""
        logger.info("Checking database configuration")
        
        try:
            # Change to prog directory for database check
            os.chdir(self.app_dir / "prog")
            
            # Set data symlink for options.json access
            data_symlink = Path("../data")
            if data_symlink.exists() and data_symlink.is_symlink():
                data_symlink.unlink()
            elif data_symlink.exists():
                shutil.rmtree(data_symlink)
            
            # Create symlink to config directory
            data_symlink.symlink_to(self.config_dir)
            
            result = subprocess.run([sys.executable, "check_db.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Database check completed successfully")
            else:
                logger.error(f"Database check failed: {result.stderr}")
                if result.stdout:
                    logger.error(f"Database check stdout: {result.stdout}")
                
        except Exception as e:
            logger.error(f"Error during database check: {e}")
    
    def start_webserver(self):
        """Start the web server in background"""
        logger.info("Starting web server")
        
        try:
            os.chdir(self.app_dir / "webserver")
            
            # Use virtual environment python for gunicorn
            venv_python = Path("/app/dao/venv/day_ahead/bin/python3")
            if venv_python.exists():
                # Start gunicorn within virtual environment
                cmd = [str(venv_python), "-m", "gunicorn", "--config", "gunicorn_config.py", "app:app"]
            else:
                # Fallback to system gunicorn
                cmd = ["gunicorn", "--config", "gunicorn_config.py", "app:app"]
            
            process = subprocess.Popen(cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
            
            logger.info(f"Web server started with PID: {process.pid}")
            
            # Check if process started successfully
            time.sleep(1)
            if process.poll() is not None:
                output = process.stdout.read()
                logger.error(f"Web server failed to start: {output}")
                return None
            
            return process
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            return None
    
    def start_smart_services(self):
        """Start smart services if available"""
        smart_services = self.app_dir / "prog" / "start_smart_services.py"
        
        if smart_services.exists():
            logger.info("Starting smart services (WebSocket + Smart Optimization)")
            
            try:
                os.chdir(self.app_dir / "prog")
                
                # Use virtual environment python if available
                venv_python = Path("/app/dao/venv/day_ahead/bin/python3")
                python_cmd = str(venv_python) if venv_python.exists() else sys.executable
                
                process = subprocess.Popen([python_cmd, "start_smart_services.py"],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         universal_newlines=True)
                
                logger.info(f"Smart services started with PID: {process.pid}")
                
                # Check if process started successfully
                time.sleep(1)
                if process.poll() is not None:
                    output = process.stdout.read()
                    logger.error(f"Smart services failed to start: {output}")
                    return None
                
                return process
                
            except Exception as e:
                logger.error(f"Failed to start smart services: {e}")
                return None
        else:
            logger.info("Smart services not available")
            return None
    
    def start_scheduler(self):
        """Start the main scheduler"""
        logger.info("Starting DAO scheduler")
        
        try:
            os.chdir(self.app_dir / "prog")
            
            # Try modern scheduler first, fallback to simple
            scheduler_file = "da_modern_scheduler.py"
            if not Path(scheduler_file).exists():
                scheduler_file = "da_simple_scheduler.py"
            
            if not Path(scheduler_file).exists():
                scheduler_file = "da_scheduler.py"
            
            logger.info(f"Using scheduler: {scheduler_file}")
            
            # Use subprocess for all schedulers to avoid import issues
            logger.info(f"Starting scheduler: {scheduler_file}")
            
            # Use virtual environment python if available
            venv_python = Path("/app/dao/venv/day_ahead/bin/python3")
            python_cmd = str(venv_python) if venv_python.exists() else sys.executable
            
            process = subprocess.Popen([python_cmd, scheduler_file],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
            
            # Stream output
            while self.running and process.poll() is None:
                line = process.stdout.readline()
                if line:
                    print(line.strip(), flush=True)
                time.sleep(0.1)
            
            if process.poll() is not None:
                logger.error(f"Scheduler exited with code: {process.returncode}")
                # Read remaining output
                remaining_output = process.stdout.read()
                if remaining_output:
                    print(remaining_output.strip(), flush=True)
                    
                # Keep container alive by waiting
                logger.info("Scheduler stopped, but keeping container alive for web services")
                while self.running:
                    time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            raise
    
    def run(self):
        """Main entry point"""
        logger.info("Starting DAO Modern Enhanced")
        
        try:
            # Setup
            self.setup_directories()
            self.setup_environment()
            self.check_database()
            
            # Start background services
            webserver = self.start_webserver()
            smart_services = self.start_smart_services()
            
            # Give services time to start
            time.sleep(2)
            
            # Start main scheduler (blocking)
            self.start_scheduler()
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1)
        
        logger.info("DAO Modern Enhanced stopped")


if __name__ == "__main__":
    entrypoint = DaoEntrypoint()
    entrypoint.run()