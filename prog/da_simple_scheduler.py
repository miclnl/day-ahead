#!/usr/bin/env python3
"""
Simple, reliable scheduler for Day Ahead Optimizer.
Minimal dependencies, maximum compatibility.
"""

import datetime
import sys
import time
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s : %(message)s'
)

try:
    from da_base import DaBase
    DABASE_AVAILABLE = True
except ImportError:
    DABASE_AVAILABLE = False
    logging.warning("DaBase not available, using minimal scheduler")


class SimpleDaScheduler:
    """
    Simple time-based scheduler with minimal dependencies.
    Falls back to basic functionality if advanced features unavailable.
    """
    
    def __init__(self, file_name: str = None):
        self.file_name = file_name
        self.active = True
        self.tasks = {}
        self.scheduler_tasks = {}
        
        # Try to initialize base functionality
        if DABASE_AVAILABLE:
            try:
                self.base = DaBase(file_name)
                self.scheduler_tasks = self.base.config.get(["scheduler"], {})
                self.tasks = self.base.tasks
                logging.info("DaBase initialized successfully")
            except Exception as e:
                logging.warning(f"DaBase initialization failed: {e}")
                self.base = None
        else:
            self.base = None
            
        # Set active status
        if self.scheduler_tasks and "active" in self.scheduler_tasks:
            self.active = not (self.scheduler_tasks["active"].lower() == "false")
    
    def scheduler(self):
        """Main scheduler loop - simple and reliable"""
        logging.info("Starting simple DAO scheduler")
        logging.info("Scheduler will run basic time-based tasks")
        
        try:
            while self.active:
                t = datetime.datetime.now()
                next_min = t - datetime.timedelta(
                    minutes=-1, seconds=t.second, microseconds=t.microsecond
                )
                
                # Wait until next minute (0% CPU while waiting)
                time.sleep((next_min - t).total_seconds())
                
                if not self.active:
                    continue
                
                hour = next_min.hour
                minute = next_min.minute
                
                # Create time patterns
                key1 = str(hour).zfill(2) + str(minute).zfill(2)  # Exact time (e.g., 0215)
                key2 = "xx" + str(minute).zfill(2)                # Every hour at minute (e.g., xx15)
                key3 = str(hour).zfill(2) + "xx"                 # Every minute in hour (e.g., 02xx)
                
                task = None
                if self.scheduler_tasks:
                    if key1 in self.scheduler_tasks:
                        task = self.scheduler_tasks[key1]
                    elif key2 in self.scheduler_tasks:
                        task = self.scheduler_tasks[key2]
                    elif key3 in self.scheduler_tasks:
                        task = self.scheduler_tasks[key3]
                
                if task and self.base and self.tasks:
                    logging.info(f"Executing scheduled task: {task} at {key1}")
                    
                    for key_task in self.tasks:
                        if self.tasks[key_task].get("function") == task:
                            try:
                                self.base.run_task_function(key_task, True)
                                logging.info(f"Task {task} completed successfully")
                            except KeyboardInterrupt:
                                logging.info("Scheduler interrupted by user")
                                return
                            except Exception as e:
                                logging.error(f"Task {task} failed: {e}")
                                continue
                            break
                else:
                    # Even without tasks, show we're alive
                    if minute % 15 == 0:  # Log every 15 minutes
                        logging.info(f"Simple scheduler heartbeat: {key1}")
                        
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user")
        except Exception as e:
            logging.error(f"Scheduler error: {e}")
        finally:
            logging.info("Simple scheduler stopped")


def main():
    """Main entry point"""
    try:
        scheduler = SimpleDaScheduler("../data/options.json")
        scheduler.scheduler()
    except Exception as e:
        logging.error(f"Failed to start simple scheduler: {e}")
        # Keep running anyway to maintain container
        logging.info("Maintaining scheduler process for container stability")
        while True:
            time.sleep(60)
            logging.info("Simple scheduler keepalive")


if __name__ == "__main__":
    main()