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
            
            # Try schedulers in order of preference with fallback
            scheduler_files = [
                ("da_simple_scheduler.py", "Simple scheduler (reliable)"),
                ("da_scheduler.py", "Basic scheduler (fallback)"),
                ("da_modern_scheduler.py", "Modern scheduler (advanced)")
            ]
            
            scheduler_file = None
            scheduler_desc = None
            
            for sched_file, sched_desc in scheduler_files:
                if Path(sched_file).exists():
                    scheduler_file = sched_file
                    scheduler_desc = sched_desc
                    break
            
            if not scheduler_file:
                logger.error("No scheduler found!")
                return
            
            logger.info(f"Using scheduler: {scheduler_file} ({scheduler_desc})")
            
            # Try multiple schedulers with automatic fallback on failure
            for attempt, (sched_file, sched_desc) in enumerate(scheduler_files):
                if not Path(sched_file).exists():
                    continue
                    
                logger.info(f"Attempt {attempt + 1}: Starting {sched_file} ({sched_desc})")
                
                try:
                    # Use system Python for better compatibility
                    python_cmd = sys.executable
                    
                    process = subprocess.Popen([python_cmd, sched_file],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,
                                             universal_newlines=True,
                                             cwd=str(self.app_dir / "prog"))
                    
                    # Stream output with timeout
                    start_time = time.time()
                    stable_time = 10  # Consider stable after 10 seconds
                    
                    while self.running and process.poll() is None:
                        line = process.stdout.readline()
                        if line:
                            print(line.strip(), flush=True)
                        time.sleep(0.1)
                        
                        # If running stable for enough time, consider it successful
                        if time.time() - start_time > stable_time:
                            logger.info(f"Scheduler {sched_file} running stable")
                            break
                    
                    # If we get here and process is still running, it's successful
                    if process.poll() is None:
                        # Continue streaming until process ends
                        while self.running and process.poll() is None:
                            line = process.stdout.readline()
                            if line:
                                print(line.strip(), flush=True)
                            time.sleep(0.1)
                    
                    exit_code = process.returncode
                    logger.error(f"Scheduler {sched_file} exited with code: {exit_code}")
                    
                    # Read remaining output
                    remaining_output = process.stdout.read()
                    if remaining_output:
                        print(remaining_output.strip(), flush=True)
                    
                    # If this was not the last scheduler, try the next one
                    if attempt < len(scheduler_files) - 1:
                        logger.info(f"Trying next scheduler due to exit code {exit_code}")
                        time.sleep(2)  # Brief delay before retry
                        continue
                    else:
                        logger.error("All schedulers failed, keeping container alive for web services")
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to start {sched_file}: {e}")
                    if attempt < len(scheduler_files) - 1:
                        logger.info("Trying next scheduler due to startup error")
                        continue
                    else:
                        logger.error("All schedulers failed to start")
                        break
            
            # Keep container alive for web services
            logger.info("Scheduler management complete, keeping container alive for web services")
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