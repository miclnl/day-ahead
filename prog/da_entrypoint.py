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
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Verhoog naar DEBUG voor meer details
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class DaoEntrypoint:
    def __init__(self):
        self.config_dir = Path("/config/dao_modern_data")
        self.app_dir = Path("/app/dao")
        self.running = True
        self.processes = {}  # Track alle processen
        self.monitor_thread = None
        self.watchdog_thread = None
        self.last_heartbeat = time.time()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("DaoEntrypoint geïnitialiseerd")

    def _start_watchdog(self):
        """Start watchdog timer om freezes te detecteren"""
        def watchdog_loop():
            logger.info("🕐 Watchdog timer gestart - controleert elke 60 seconden")
            try:
                while self.running:
                    try:
                        current_time = time.time()
                        time_since_heartbeat = current_time - self.last_heartbeat

                        if time_since_heartbeat > 300:  # 5 minuten geen heartbeat
                            logger.critical(f"🚨 WATCHDOG ALARM: Geen heartbeat voor {time_since_heartbeat:.0f}s! Container bevroren!")

                            # Force restart van kritieke processen
                            for name, process in self.processes.items():
                                if name in ['webserver', 'scheduler'] and process and process.poll() is None:
                                    logger.critical(f"🚨 Force restart van {name}")
                                    try:
                                        process.terminate()
                                        time.sleep(2)
                                        if process.poll() is None:
                                            process.kill()
                                    except Exception as kill_error:
                                        logger.error(f"🚨 Error killing process {name}: {kill_error}")

                            # Reset heartbeat
                            self.last_heartbeat = current_time

                        time.sleep(60)  # Check elke minuut

                    except Exception as e:
                        logger.error(f"🚨 Watchdog iteration error: {e}")
                        import traceback
                        logger.error(f"🚨 Watchdog traceback: {traceback.format_exc()}")
                        logger.info("🕐 Continuing watchdog despite error...")
                        time.sleep(60)

            except Exception as e:
                logger.critical(f"🚨 FATAL WATCHDOG ERROR: {e}")
                import traceback
                logger.critical(f"🚨 FATAL WATCHDOG TRACEBACK: {traceback.format_exc()}")
                logger.critical("🕐 Watchdog crashed - attempting recovery...")

                # Probeer watchdog te herstarten
                try:
                    logger.info("🕐 Attempting watchdog recovery...")
                    while self.running:
                        try:
                            time.sleep(60)
                        except Exception as recovery_error:
                            logger.error(f"🚨 Watchdog recovery error: {recovery_error}")
                            time.sleep(60)
                except Exception as recovery_fatal:
                    logger.critical(f"🚨 Watchdog recovery failed: {recovery_fatal}")

        try:
            self.watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
            self.watchdog_thread.start()
            logger.info("🕐 Watchdog thread gestart")
        except Exception as e:
            logger.error(f"🚨 Failed to start watchdog: {e}")
            # Watchdog failure is niet kritiek - main loop kan nog steeds draaien

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self._cleanup_processes()

    def _cleanup_processes(self):
        """Cleanup alle processen bij shutdown"""
        logger.info("Cleaning up processes...")
        for name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info(f"Terminating {name} process (PID: {process.pid})")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name} process")
                    process.kill()
                except Exception as e:
                    logger.error(f"Error terminating {name}: {e}")

    def _handle_debug_commands(self):
        """Handle debug commands via stdin"""
        try:
            import select
            import sys

            # Check of er input beschikbaar is
            if select.select([sys.stdin], [], [], 0.1)[0]:
                command = sys.stdin.readline().strip().lower()

                if command == 'ps' or command == 'status':
                    self._show_process_status()
                elif command == 'memory':
                    try:
                        import psutil
                        memory = psutil.virtual_memory()
                        logger.info(f"💾 Memory: {memory.percent}% gebruikt ({memory.available // (1024**3)} GB beschikbaar)")
                    except:
                        logger.warning("⚠️ Kon memory status niet ophalen")
                elif command == 'help':
                    logger.info("🔧 Debug commands: ps/status, memory, help")
                elif command:
                    logger.info(f"🔧 Onbekend command: {command}")

        except Exception as e:
            # Ignore errors in debug command handling
            pass

    def _show_process_status(self):
        """Toon huidige process status voor debugging"""
        try:
            logger.info("🔍 === PROCESS STATUS DEBUG ===")

            # Toon Python processen
            try:
                import subprocess
                ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
                if ps_output.returncode == 0:
                    # Filter Python processen
                    lines = ps_output.stdout.split('\n')
                    python_processes = [line for line in lines if 'python' in line.lower()]
                    if python_processes:
                        logger.info("🐍 Python processen:")
                        for proc in python_processes[:10]:  # Max 10 regels
                            logger.info(f"  {proc}")
                    else:
                        logger.warning("⚠️ Geen Python processen gevonden")
                else:
                    logger.warning("⚠️ Kon ps output niet ophalen")
            except Exception as e:
                logger.error(f"❌ Fout bij ophalen process status: {e}")

            # Toon onze tracked processen
            logger.info("📋 Onze tracked processen:")
            for name, process in self.processes.items():
                if process:
                    status = "Running" if process.poll() is None else f"Crashed (exit: {process.returncode})"
                    logger.info(f"  {name}: {status}")
                else:
                    logger.info(f"  {name}: Not started")

            # Memory status
            try:
                import psutil
                memory = psutil.virtual_memory()
                logger.info(f"💾 Memory: {memory.percent}% gebruikt ({memory.available // (1024**3)} GB beschikbaar)")
            except:
                logger.warning("⚠️ Kon memory status niet ophalen")

            logger.info("🔍 === END PROCESS STATUS ===")

        except Exception as e:
            logger.error(f"❌ Fout in process status debug: {e}")

    def _monitor_processes(self):
        """Monitor alle subprocesses en log crashes"""
        while self.running:
            try:
                time.sleep(30)  # Check elke 30 seconden

                for name, process in self.processes.items():
                    if process and process.poll() is not None:
                        exit_code = process.returncode
                        logger.error(f"🚨 Process {name} crashed with exit code {exit_code}")

                        # Haal process info op met ps
                        try:
                            import subprocess
                            ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
                            if ps_output.returncode == 0:
                                logger.debug(f"📊 Process status:\n{ps_output.stdout}")
                        except Exception as e:
                            logger.debug(f"Kon ps output niet ophalen: {e}")

                        # Restart kritieke processen
                        if name in ['webserver', 'scheduler']:
                            logger.critical(f"🚨 KRITIEK: {name} is gecrashed! Container wordt onstabiel!")
                            # Probeer te herstarten
                            try:
                                if name == 'webserver':
                                    self.start_webserver()
                                elif name == 'scheduler':
                                    self.start_scheduler()
                                logger.info(f"🔄 {name} herstart poging voltooid")
                            except Exception as e:
                                logger.error(f"❌ Herstart van {name} gefaald: {e}")

                # Check memory usage elke 30 seconden
                try:
                    import psutil
                    memory = psutil.virtual_memory()
                    if memory.percent > 90:
                        logger.critical(f"🚨 MEMORY CRISIS: {memory.percent}% gebruikt! Container kan crashen!")
                    elif memory.percent > 80:
                        logger.warning(f"⚠️ Hoog geheugengebruik: {memory.percent}%")
                except:
                    pass

            except Exception as e:
                logger.error(f"🚨 Error in process monitoring: {e}")
                time.sleep(10)

    def setup_directories(self):
        """Setup data directories and symlinks"""
        logger.info("Setting up directories and configuration")

        try:
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

            logger.info("Directory setup completed successfully")

        except Exception as e:
            logger.error(f"Directory setup failed: {e}")
            raise

    def _first_start_enrichment(self):
        """Bij eerste start: vul HA-gegevens (locatie en PV) automatisch in."""
        try:
            marker = Path('/data/.dao_first_start_done')
            options_path = self.config_dir / 'options.json'
            if marker.exists() or not options_path.exists():
                return
            # Backup
            try:
                backup = Path(f"/data/options.backup-{int(time.time())}.json")
                backup.write_bytes(options_path.read_bytes())
                logger.info(f"Backup gemaakt: {backup}")
            except Exception as e:
                logger.warning(f"Backup mislukt: {e}")

            try:
                opts = json.loads(options_path.read_text())
            except Exception as e:
                logger.error(f"Kan options.json niet lezen: {e}")
                return

            class _Cfg:
                def __init__(self, data):
                    self._d = data
                def get(self, path: list, default_parent=None, default_value=None):
                    ref = self._d
                    for i, key in enumerate(path):
                        if not isinstance(ref, dict):
                            return default_value
                        if key not in ref:
                            if i == len(path) - 1:
                                return default_value
                            if default_parent is not None:
                                ref[key] = {}
                        ref = ref.get(key, {})
                    return ref

            try:
                from dao.prog.ha_client import (
                    get_core_config,
                    get_states,
                    suggest_pv_energy_entities,
                    get_statistics_max_daily,
                )
            except Exception as e:
                logger.warning(f"HA client niet beschikbaar: {e}")
                return

            cfg = _Cfg(opts)
            # Locatie uit HA
            core = get_core_config(cfg)
            if core:
                lat = core.get('latitude')
                lon = core.get('longitude')
                if lat is not None and lon is not None:
                    opts.setdefault('location', {})
                    opts['location'].setdefault('latitude', lat)
                    opts['location'].setdefault('longitude', lon)
                    logger.info(f"HA locatie ingevuld: lat={lat}, lon={lon}")

            # PV schatting
            states = get_states(cfg)
            pv_entities = suggest_pv_energy_entities(states)
            if pv_entities:
                max_daily = get_statistics_max_daily(cfg, pv_entities[0], days=365)
                if isinstance(max_daily, (int, float)) and max_daily > 0:
                    est_kwp = round(float(max_daily) / 4.0, 2)
                    solar = opts.setdefault('solar', {})
                    solar.setdefault('capacity', est_kwp)
                    solar.setdefault('entity', pv_entities[0])
                    logger.info(f"PV schatting: capacity~{est_kwp} kWp, entity={pv_entities[0]}")

            try:
                options_path.write_text(json.dumps(opts, indent=2))
                marker.touch()
                logger.info("Eerste-start verrijking toegepast")
            except Exception as e:
                logger.error(f"Kon options.json niet bijwerken: {e}")
        except Exception as e:
            logger.error(f"Fout in eerste-start verrijking: {e}")

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
        """Start the web server in background using pure Uvicorn"""
        logger.info("Starting FastAPI webserver with Uvicorn on port 5001")

        try:
            os.chdir(self.app_dir / "webserver")
            logger.debug(f"Changed to webserver directory: {os.getcwd()}")

            # Check if uvicorn config exists
            config_file = Path("uvicorn_config.py")
            if not config_file.exists():
                logger.error(f"Uvicorn config file not found: {config_file}")
                return None

            logger.debug(f"Uvicorn config file found: {config_file}")

            # Get log level from addon config
            try:
                from webserver.app.addon_config import addon_config
                log_level = addon_config.get_log_level()
                logger.info(f"Using log level from addon config: {log_level}")
            except Exception as e:
                log_level = "info"
                logger.warning(f"Could not get log level from addon config: {e}")

            # Use virtual environment python for uvicorn
            venv_python = Path("/app/dao/venv/day_ahead/bin/python3")
            if venv_python.exists():
                # Start uvicorn directly (no Gunicorn wrapper)
                cmd = [
                    str(venv_python), "-m", "uvicorn",
                    "app:asgi_app",
                    "--host", "0.0.0.0",
                    "--port", "5001",
                    "--workers", "2",
                    "--log-level", log_level
                ]
                logger.info("Using virtual environment uvicorn")
                logger.debug(f"Virtual environment path: {venv_python}")
            else:
                # Fallback to system uvicorn
                cmd = [
                    "uvicorn",
                    "app:asgi_app",
                    "--host", "0.0.0.0",
                    "--port", "5001",
                    "--workers", "2",
                    "--log-level", log_level
                ]
                logger.info("Using system uvicorn")
                logger.warning("Virtual environment not found, using system uvicorn")

            logger.info(f"Starting command: {' '.join(cmd)}")

            # Start process with better error handling
            process = subprocess.Popen(cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True,
                                     bufsize=1)  # Line buffered

            logger.info(f"FastAPI webserver started with PID: {process.pid}")

            # Store process for monitoring
            self.processes['webserver'] = process

            # Give it more time to start and capture initial output
            logger.debug("Waiting for webserver to start...")
            time.sleep(5)  # Verhoog naar 5 seconden

            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else "No output"
                logger.error(f"FastAPI webserver failed to start: {output}")
                logger.error(f"Exit code: {process.returncode}")
                return None
            else:
                logger.info("FastAPI webserver is running successfully")

                # Read any initial output
                try:
                    if process.stdout:
                        # Non-blocking read
                        import select
                        if select.select([process.stdout], [], [], 0.1)[0]:
                            output = process.stdout.readline()
                            if output:
                                logger.info(f"FastAPI webserver initial output: {output.strip()}")
                except Exception as e:
                    logger.debug(f"Could not read initial output: {e}")

                return process

        except Exception as e:
            logger.error(f"Failed to start FastAPI webserver: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def start_smart_services(self):
        """Start smart services if available"""
        smart_services = self.app_dir / "prog" / "start_smart_services.py"

        if smart_services.exists():
            logger.info("Starting smart services (WebSocket + Smart Optimization)")

            try:
                os.chdir(self.app_dir / "prog")
                logger.debug(f"Changed to prog directory: {os.getcwd()}")

                # Use virtual environment python if available
                venv_python = Path("/app/dao/venv/day_ahead/bin/python3")
                python_cmd = str(venv_python) if venv_python.exists() else sys.executable

                logger.debug(f"Using Python: {python_cmd}")

                process = subprocess.Popen([python_cmd, "start_smart_services.py"],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         universal_newlines=True,
                                         bufsize=1)

                logger.info(f"Smart services started with PID: {process.pid}")

                # Store process for monitoring
                self.processes['smart_services'] = process

                # Check if process started successfully
                logger.debug("Waiting for smart services to start...")
                time.sleep(2)
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout else "No output"
                    logger.error(f"Smart services failed to start: {output}")
                    logger.error(f"Exit code: {process.returncode}")
                    return None
                else:
                    logger.info("Smart services started successfully")

                return process

            except Exception as e:
                logger.error(f"Failed to start smart services: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None
        else:
            logger.info("Smart services not available")
            return None

    def start_scheduler(self):
        """Start the main scheduler"""
        logger.info("Starting DAO scheduler")

        try:
            os.chdir(self.app_dir / "prog")
            logger.debug(f"Changed to prog directory: {os.getcwd()}")

            # Try schedulers in order of preference with fallback
            scheduler_files = [
                ("da_simple_scheduler.py", "Simple scheduler (reliable)"),
                ("da_scheduler.py", "Basic scheduler (fallback)"),
                ("da_modern_scheduler.py", "Modern scheduler (advanced)")
            ]

            for scheduler_file, description in scheduler_files:
                scheduler_path = Path(scheduler_file)
                if scheduler_path.exists():
                    logger.info(f"Using scheduler: {scheduler_file} ({description})")
                    logger.debug(f"Scheduler path: {scheduler_path.absolute()}")

                    try:
                        # Use virtual environment python if available
                        venv_python = Path("/app/dao/venv/day_ahead/bin/python3")
                        python_cmd = str(venv_python) if venv_python.exists() else sys.executable

                        logger.debug(f"Using Python: {python_cmd}")

                        cmd = [python_cmd, scheduler_file]
                        logger.debug(f"Scheduler command: {' '.join(cmd)}")

                        process = subprocess.Popen(cmd,
                                                 stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT,
                                                 universal_newlines=True,
                                                 bufsize=1)

                        logger.info(f"Attempt 1: Starting {scheduler_file} ({description})")
                        logger.info(f"Scheduler started with PID: {process.pid}")

                        # Store process for monitoring
                        self.processes['scheduler'] = process

                        # Give it time to start
                        logger.debug("Waiting for scheduler to start...")
                        time.sleep(3)

                        if process.poll() is not None:
                            output = process.stdout.read() if process.stdout else "No output"
                            logger.error(f"Scheduler {scheduler_file} failed to start: {output}")
                            logger.error(f"Exit code: {process.returncode}")
                            continue  # Try next scheduler
                        else:
                            logger.info(f"Scheduler {scheduler_file} started successfully")
                            return process

                    except Exception as e:
                        logger.error(f"Failed to start scheduler {scheduler_file}: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue  # Try next scheduler
                else:
                    logger.debug(f"Scheduler file not found: {scheduler_path}")

            # If we get here, no scheduler started successfully
            logger.error("All schedulers failed to start")
            return None

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def run(self):
        """Main entry point"""
        logger.info("Starting DAO Modern Enhanced")

        try:
            # Setup environment
            self.setup_environment()
            logger.debug("Environment setup completed")

            # Setup directories
            self.setup_directories()
            logger.debug("Directory setup completed")

            # First-start enrichment (HA lat/lon en PV)
            try:
                self._first_start_enrichment()
            except Exception as e:
                logger.warning(f"Eerste-start verrijking mislukt: {e}")

            # Check database
            self.check_database()
            logger.debug("Database check completed")

            # Start webserver
            webserver_process = self.start_webserver()
            if webserver_process:
                logger.info("Webserver started successfully")
            else:
                logger.error("Failed to start webserver")

            # Start smart services
            smart_services_process = self.start_smart_services()
            if smart_services_process:
                logger.info("Smart services started successfully")
            else:
                logger.warning("Smart services failed to start")

            # Start scheduler
            scheduler_process = self.start_scheduler()
            if scheduler_process:
                logger.info("Scheduler started successfully")
            else:
                logger.error("Failed to start scheduler")

            # Log all running processes
            logger.info("=== Process Status ===")
            for name, process in self.processes.items():
                if process and process.poll() is None:
                    logger.info(f"✅ {name}: Running (PID: {process.pid})")
                else:
                    status = "Stopped" if process else "Not started"
                    logger.warning(f"❌ {name}: {status}")

            # Keep container alive and monitor processes
            logger.info("All services started, monitoring processes...")
            logger.info("💓 Main monitoring loop started - container should stay alive")
            try:
                while self.running:
                    try:
                        self.last_heartbeat = time.time() # Update heartbeat

                        # Log process status elke 5 minuten
                        if int(time.time()) % 300 == 0:  # Elke 5 minuten
                            self._show_process_status()

                        # Check alle processen
                        all_running = True
                        for name, process in self.processes.items():
                            if process and process.poll() is not None:
                                logger.error(f"🚨 Process {name} is gestopt!")
                                all_running = False

                        if not all_running:
                            logger.warning("⚠️ Niet alle processen draaien - container kan onstabiel zijn")

                        # Log heartbeat elke minuut
                        if int(time.time()) % 60 == 0:
                            logger.info("💓 Container heartbeat - alle services actief")

                        # Handle debug commands
                        self._handle_debug_commands()

                        time.sleep(30)  # Check elke 30 seconden

                    except Exception as e:
                        logger.error(f"🚨 Error in main loop iteration: {e}")
                        import traceback
                        logger.error(f"🚨 Traceback: {traceback.format_exc()}")
                        logger.info("💓 Continuing main loop despite error...")
                        time.sleep(30)

            except Exception as e:
                logger.critical(f"🚨 FATAL ERROR in main loop: {e}")
                import traceback
                logger.critical(f"🚨 FATAL Traceback: {traceback.format_exc()}")
                logger.critical("🚨 Main loop crashed - attempting recovery...")

                # Probeer de main loop te herstarten
                try:
                    logger.info("💓 Attempting main loop recovery...")
                    while self.running:
                        try:
                            self.last_heartbeat = time.time()
                            time.sleep(30)
                        except Exception as recovery_error:
                            logger.error(f"🚨 Recovery loop error: {recovery_error}")
                            time.sleep(30)
                except Exception as recovery_fatal:
                    logger.critical(f"🚨 Recovery failed: {recovery_fatal}")
                    raise

        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        logger.info("Shutdown sequence completed")


if __name__ == "__main__":
    entrypoint = DaoEntrypoint()
    entrypoint.run()