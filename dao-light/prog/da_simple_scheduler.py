"""
Simple scheduler implementation without external dependencies.
Replaces APScheduler with built-in threading and time-based scheduling.
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Callable, Optional
from da_base import DaBase


class SimpleScheduler(DaBase):
    """Simple thread-based scheduler without external dependencies"""
    
    def __init__(self, config_file: str):
        super().__init__(config_file)
        self.running = False
        self.threads = []
        self.jobs = {}
        
        logging.info("Simple Scheduler initialized")
    
    def start(self):
        """Start the scheduler"""
        self.running = True
        self.load_schedule()
        
        # Start main scheduler thread
        scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        scheduler_thread.start()
        self.threads.append(scheduler_thread)
        
        logging.info("Simple Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        logging.info("Simple Scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                current_time = datetime.now()
                current_time_str = current_time.strftime("%H%M")
                
                # Check scheduled tasks
                for time_pattern, task_info in self.tasks.items():
                    if self._should_run_task(time_pattern, current_time_str, current_time):
                        self._run_task(task_info)
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logging.error(f"Scheduler loop error: {e}")
                time.sleep(60)
    
    def _should_run_task(self, pattern: str, current_time: str, dt: datetime) -> bool:
        """Check if task should run based on time pattern"""
        try:
            if pattern.startswith('xx'):
                # Every hour at specific minute (xx15 = every hour at :15)
                return current_time.endswith(pattern[2:])
            elif pattern.endswith('xx'):
                # Every minute in specific hour (02xx = every minute in 02:00 hour)
                return current_time.startswith(pattern[:2])
            else:
                # Specific time (0215 = 02:15)
                return current_time == pattern
        except:
            return False
    
    def _run_task(self, task_info: Dict):
        """Run a scheduled task in a separate thread"""
        def task_runner():
            try:
                task_name = task_info.get('description', 'Unknown')
                function_name = task_info.get('function')
                
                logging.info(f"Running scheduled task: {task_name}")
                
                if function_name == 'get_day_ahead_prices':
                    self._fetch_prices()
                elif function_name == 'get_meteo_data':
                    self._fetch_weather()
                elif function_name == 'calc_optimum':
                    self._run_optimization()
                elif function_name == 'consolidate':
                    self._consolidate_data()
                else:
                    logging.warning(f"Unknown function: {function_name}")
                
                logging.info(f"Task completed: {task_name}")
                
            except Exception as e:
                logging.error(f"Task execution error: {e}")
        
        # Start task in separate thread
        task_thread = threading.Thread(target=task_runner, daemon=True)
        task_thread.start()
    
    def _fetch_prices(self):
        """Fetch energy prices"""
        try:
            from da_prices import DaPrices
            da_prices = DaPrices(self.file_name)
            da_prices.get_day_ahead_prices()
        except Exception as e:
            logging.error(f"Price fetching error: {e}")
    
    def _fetch_weather(self):
        """Fetch weather data"""
        try:
            from da_meteo import DaMeteo
            da_meteo = DaMeteo(self.file_name)
            da_meteo.get_meteo_data()
        except Exception as e:
            logging.error(f"Weather fetching error: {e}")
    
    def _run_optimization(self):
        """Run optimization"""
        try:
            from day_ahead import DaCalc
            da_calc = DaCalc(self.file_name)
            da_calc.calc_optimum()
        except Exception as e:
            logging.error(f"Optimization error: {e}")
    
    def _consolidate_data(self):
        """Consolidate historical data"""
        try:
            # Run data consolidation
            self.run_task_function("consolidate", False)
        except Exception as e:
            logging.error(f"Data consolidation error: {e}")


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s : %(message)s'
    )
    
    scheduler = SimpleScheduler("../data/options.json")
    
    try:
        scheduler.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user")
        scheduler.stop()
    except Exception as e:
        logging.error(f"Scheduler error: {e}")


if __name__ == "__main__":
    main()