#!/usr/bin/env python3
"""
Secure entrypoint for DAO Light without shell dependencies.
Replaces run_light.sh with pure Python implementation.
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

class DaoLightEntrypoint:
    def __init__(self):
        self.config_dir = Path("/config/dao_light_data")
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
            logger.info("Creating dao_light_data directory and copying default files")
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
            logger.info("dao_light_data directory already exists")
        
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
            result = subprocess.run([sys.executable, "check_db.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Database check completed successfully")
            else:
                logger.error(f"Database check failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error during database check: {e}")
    
    def start_webserver(self):
        """Start the web server in background"""
        logger.info("Starting web server")
        
        try:
            os.chdir(self.app_dir / "webserver")
            
            # Start gunicorn in background
            cmd = ["gunicorn", "--config", "gunicorn_config.py", "app:app"]
            process = subprocess.Popen(cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
            
            logger.info(f"Web server started with PID: {process.pid}")
            return process
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            return None
    
    def start_scheduler(self):
        """Start the simple scheduler (no smart services for Light version)"""
        logger.info("Starting DAO Light scheduler")
        
        try:
            os.chdir(self.app_dir / "prog")
            
            # Use simple scheduler for Light version
            scheduler_file = "da_simple_scheduler.py"
            if not Path(scheduler_file).exists():
                scheduler_file = "da_scheduler.py"
            
            logger.info(f"Using scheduler: {scheduler_file}")
            
            # Import and run scheduler
            if scheduler_file == "da_simple_scheduler.py":
                from da_simple_scheduler import main
                main()
            else:
                # Fallback to subprocess for old scheduler
                subprocess.call([sys.executable, scheduler_file])
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            raise
    
    def run(self):
        """Main entry point"""
        logger.info("Starting DAO Light (Minimal & Stable)")
        
        try:
            # Setup
            self.setup_directories()
            self.setup_environment()
            self.check_database()
            
            # Start background services
            webserver = self.start_webserver()
            
            # Give services time to start
            time.sleep(2)
            
            # Start main scheduler (blocking)
            self.start_scheduler()
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1)
        
        logger.info("DAO Light stopped")


if __name__ == "__main__":
    entrypoint = DaoLightEntrypoint()
    entrypoint.run()