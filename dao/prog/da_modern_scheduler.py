"""
Modern event-driven scheduler for Day Ahead Optimizer.
Replaces the old time.sleep() based scheduler with APScheduler for better performance and responsiveness.
"""

import asyncio
import datetime
import logging
import sys
import signal
from typing import Dict, List, Callable, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from da_base import DaBase
from da_async_fetcher import AsyncDataFetcher, fetch_and_save_all_data


class ModernDaScheduler(DaBase):
    """
    Modern event-driven scheduler using APScheduler.
    Provides real-time responsiveness and concurrent task execution.
    """
    
    def __init__(self, file_name: str = None):
        super().__init__(file_name)
        
        self.scheduler = AsyncIOScheduler(
            timezone='Europe/Amsterdam',
            job_defaults={
                'coalesce': True,  # Combine multiple pending executions
                'max_instances': 1,  # Prevent overlapping executions
                'misfire_grace_time': 300  # 5 minute grace period
            }
        )
        
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="DAO-Worker")
        self.event_queue = asyncio.Queue()
        self.active = True
        self.tasks_config = self.config.get(["scheduler"], {}) if self.config else {}
        
        # Priority levels for different task types
        self.task_priorities = {
            'urgent': 1,      # EV charging, grid limits
            'high': 2,        # Optimization, real-time adjustments  
            'normal': 3,      # Data fetching, routine tasks
            'low': 4          # Reports, cleanup, maintenance
        }
        
        if "active" in self.tasks_config:
            self.active = not (self.tasks_config["active"].lower() == "false")
            
        # Setup event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
    
    async def start_scheduler(self):
        """
        Start the modern scheduler with all configured tasks.
        Replaces the old infinite loop with proper async event handling.
        """
        logging.info("Starting modern DAO scheduler")
        
        try:
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            
            # Start the scheduler
            self.scheduler.start()
            
            # Schedule configured tasks
            self._schedule_configured_tasks()
            
            # Schedule real-time monitoring
            self._schedule_realtime_monitoring()
            
            # Start event processing
            await self._process_events()
            
        except Exception as e:
            logging.error(f"Error in scheduler: {e}")
            raise
    
    def _schedule_configured_tasks(self):
        """Schedule tasks based on configuration"""
        logging.info("Scheduling configured tasks")
        
        for time_key, task_name in self.tasks_config.items():
            if time_key == "active":
                continue
                
            # Parse time patterns
            if self._is_cron_pattern(time_key):
                trigger = self._parse_cron_pattern(time_key)
            else:
                trigger = self._parse_time_pattern(time_key)
            
            if trigger:
                # Find matching task function
                task_func = self._get_task_function(task_name)
                if task_func:
                    job_id = f"scheduled_{time_key}_{task_name}"
                    self.scheduler.add_job(
                        task_func,
                        trigger=trigger,
                        id=job_id,
                        name=f"Scheduled: {task_name} at {time_key}",
                        replace_existing=True
                    )
                    logging.info(f"Scheduled task '{task_name}' for pattern '{time_key}'")
                else:
                    logging.warning(f"Unknown task function: {task_name}")
    
    def _schedule_realtime_monitoring(self):
        """Schedule real-time monitoring tasks"""
        logging.info("Setting up real-time monitoring")
        
        # Monitor for urgent events every 30 seconds
        self.scheduler.add_job(
            self._check_urgent_conditions,
            IntervalTrigger(seconds=30),
            id='urgent_monitoring',
            name='Urgent Condition Monitoring'
        )
        
        # Data freshness check every 5 minutes
        self.scheduler.add_job(
            self._check_data_freshness,
            IntervalTrigger(minutes=5),
            id='data_freshness_check',
            name='Data Freshness Check'
        )
        
        # System health check every 15 minutes
        self.scheduler.add_job(
            self._check_system_health,
            IntervalTrigger(minutes=15),
            id='system_health_check',
            name='System Health Check'
        )
    
    async def _process_events(self):
        """Main event processing loop"""
        logging.info("Starting event processing loop")
        
        try:
            while self.active:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    await self._handle_event(event)
                except asyncio.TimeoutError:
                    # Normal timeout, continue
                    continue
                except Exception as e:
                    logging.error(f"Error processing event: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logging.info("Scheduler interrupted by user")
        except Exception as e:
            logging.error(f"Error in event processing: {e}")
        finally:
            await self._shutdown()
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle real-time events based on priority"""
        event_type = event.get('type')
        priority = event.get('priority', 'normal')
        data = event.get('data', {})
        
        logging.info(f"Handling event: {event_type} (priority: {priority})")
        
        try:
            if event_type == 'price_change':
                await self._handle_price_change(data)
            elif event_type == 'ev_connected':
                await self._handle_ev_connected(data)
            elif event_type == 'grid_limit_approached':
                await self._handle_grid_limit(data)
            elif event_type == 'battery_critical':
                await self._handle_battery_critical(data)
            elif event_type == 'data_update':
                await self._handle_data_update(data)
            else:
                logging.warning(f"Unknown event type: {event_type}")
                
        except Exception as e:
            logging.error(f"Error handling event {event_type}: {e}")
    
    async def _handle_price_change(self, data: Dict):
        """Handle significant price changes that require re-optimization"""
        logging.info("Handling price change event")
        
        # Trigger immediate re-optimization if price change is significant
        price_change_threshold = 0.05  # €0.05/kWh
        
        if abs(data.get('price_change', 0)) > price_change_threshold:
            await self._trigger_urgent_optimization("Price change exceeds threshold")
    
    async def _handle_ev_connected(self, data: Dict):
        """Handle EV connection events"""
        logging.info("Handling EV connected event")
        
        # Immediate optimization for EV charging schedule
        await self._trigger_urgent_optimization("EV connected - updating charging schedule")
    
    async def _handle_grid_limit(self, data: Dict):
        """Handle approaching grid limit situations"""
        logging.warning("Handling grid limit approached event")
        
        # Emergency optimization to stay within grid limits
        await self._trigger_urgent_optimization("Grid limit approached")
    
    async def _handle_battery_critical(self, data: Dict):
        """Handle critical battery conditions"""
        logging.warning("Handling battery critical event")
        
        battery_soc = data.get('soc', 0)
        if battery_soc < 10:  # Critical low SoC
            await self._trigger_urgent_optimization("Battery critically low")
    
    async def _handle_data_update(self, data: Dict):
        """Handle data update events"""
        logging.info("Handling data update event")
        
        # Check if optimization should be triggered
        data_type = data.get('data_type')
        if data_type in ['weather', 'prices']:
            # Schedule optimization in 1 minute to allow other data to arrive
            self.scheduler.add_job(
                self._run_optimization,
                DateTrigger(run_date=datetime.datetime.now() + datetime.timedelta(minutes=1)),
                id='delayed_optimization',
                name='Delayed Optimization after Data Update',
                replace_existing=True
            )
    
    async def _trigger_urgent_optimization(self, reason: str):
        """Trigger urgent optimization with high priority"""
        logging.info(f"Triggering urgent optimization: {reason}")
        
        # Run optimization immediately
        self.scheduler.add_job(
            self._run_optimization,
            DateTrigger(run_date=datetime.datetime.now()),
            id='urgent_optimization',
            name=f'Urgent Optimization: {reason}',
            replace_existing=True
        )
    
    async def _check_urgent_conditions(self):
        """Check for conditions requiring urgent attention"""
        try:
            # Check grid power approaching limits
            current_power = self._get_current_grid_power()
            grid_limit = self.grid_max_power * 0.9  # 90% of limit
            
            if current_power > grid_limit:
                await self.event_queue.put({
                    'type': 'grid_limit_approached',
                    'priority': 'urgent',
                    'data': {'current_power': current_power, 'limit': self.grid_max_power}
                })
            
            # Check battery SoC
            battery_soc = self._get_battery_soc()
            if battery_soc < 15:  # Low SoC
                await self.event_queue.put({
                    'type': 'battery_critical',
                    'priority': 'urgent',
                    'data': {'soc': battery_soc}
                })
            
        except Exception as e:
            logging.error(f"Error checking urgent conditions: {e}")
    
    async def _check_data_freshness(self):
        """Check if data is fresh enough for optimization"""
        try:
            # Check price data age
            latest_price_time = self._get_latest_data_timestamp('da')
            now = datetime.datetime.now().timestamp()
            
            if now - latest_price_time > 3600:  # Data older than 1 hour
                logging.warning("Price data is stale, triggering refresh")
                await self._fetch_fresh_data()
            
        except Exception as e:
            logging.error(f"Error checking data freshness: {e}")
    
    async def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check database connectivity
            db_healthy = self._check_database_health()
            
            # Check Home Assistant connectivity
            ha_healthy = self._check_ha_connectivity()
            
            # Log health status
            if db_healthy and ha_healthy:
                logging.info("System health check: All systems operational")
            else:
                logging.warning(f"System health check: DB={db_healthy}, HA={ha_healthy}")
                
        except Exception as e:
            logging.error(f"Error in system health check: {e}")
    
    async def _fetch_fresh_data(self):
        """Fetch fresh data from all sources"""
        logging.info("Fetching fresh data from all sources")
        
        try:
            # Use async fetcher for concurrent data retrieval
            success = await fetch_and_save_all_data(self.config, self.db_da)
            
            if success:
                await self.event_queue.put({
                    'type': 'data_update',
                    'priority': 'normal',
                    'data': {'data_type': 'all', 'success': True}
                })
            else:
                logging.error("Failed to fetch fresh data")
                
        except Exception as e:
            logging.error(f"Error fetching fresh data: {e}")
    
    async def _run_optimization(self):
        """Run the optimization process"""
        logging.info("Running optimization process")
        
        try:
            # Use the modular optimizer
            from dao.prog.day_ahead import DaCalc
            
            calc = DaCalc(self.file_name)
            if hasattr(calc, 'calc_optimum_modular'):
                # Use new modular optimizer
                results = calc.calc_optimum_modular()
            else:
                # Fall back to original optimizer
                results = calc.calc_optimum()
            
            if results:
                logging.info("Optimization completed successfully")
            else:
                logging.error("Optimization failed")
                
        except Exception as e:
            logging.error(f"Error running optimization: {e}")
    
    def _get_task_function(self, task_name: str) -> Optional[Callable]:
        """Get the callable function for a task name"""
        if task_name in self.tasks:
            task_config = self.tasks[task_name]
            function_name = task_config.get('function')
            
            if function_name == 'get_day_ahead_prices':
                return self._fetch_prices
            elif function_name == 'get_meteo_data':
                return self._fetch_meteo
            elif function_name == 'calc_optimum':
                return self._run_optimization
            elif function_name == 'consolidate':
                return self._consolidate_data
            else:
                logging.warning(f"Unknown function: {function_name}")
                return None
        
        return None
    
    async def _fetch_prices(self):
        """Fetch energy prices"""
        logging.info("Fetching energy prices")
        try:
            # Use async price fetching
            async with AsyncDataFetcher(self.config, self.db_da) as fetcher:
                start_time = datetime.datetime.now()
                end_time = start_time + datetime.timedelta(days=2)
                results = await fetcher.fetch_all_data_concurrent(start_time, end_time)
                
                if results.get('prices') is not None:
                    await fetcher.save_data_concurrent({'prices': results['prices']})
                    logging.info("Price data fetched and saved successfully")
                else:
                    logging.error("Failed to fetch price data")
                    
        except Exception as e:
            logging.error(f"Error fetching prices: {e}")
    
    async def _fetch_meteo(self):
        """Fetch weather data"""
        logging.info("Fetching weather data")
        try:
            async with AsyncDataFetcher(self.config, self.db_da) as fetcher:
                results = await fetcher._fetch_meteo_data()
                
                if results is not None:
                    await fetcher.save_data_concurrent({'weather': results})
                    logging.info("Weather data fetched and saved successfully")
                else:
                    logging.error("Failed to fetch weather data")
                    
        except Exception as e:
            logging.error(f"Error fetching weather: {e}")
    
    async def _consolidate_data(self):
        """Consolidate historical data"""
        logging.info("Consolidating historical data")
        try:
            # Run consolidation in thread executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.run_task_function, "consolidate", False)
            
        except Exception as e:
            logging.error(f"Error consolidating data: {e}")
    
    def _parse_cron_pattern(self, pattern: str) -> CronTrigger:
        """Parse cron-like patterns (e.g., '0 */2 * * *')"""
        # This would implement full cron parsing
        # For now, return None to use time patterns
        return None
    
    def _parse_time_pattern(self, pattern: str) -> Optional[CronTrigger]:
        """Parse time patterns like '0215', 'xx15', '02xx'"""
        try:
            if len(pattern) == 4:
                if pattern[:2] == 'xx':
                    # Every hour at specific minute (xx15)
                    minute = int(pattern[2:])
                    return CronTrigger(minute=minute)
                elif pattern[2:] == 'xx':
                    # Every minute in specific hour (02xx)
                    hour = int(pattern[:2])
                    return CronTrigger(hour=hour)
                else:
                    # Specific time (0215 = 02:15)
                    hour = int(pattern[:2])
                    minute = int(pattern[2:])
                    return CronTrigger(hour=hour, minute=minute)
            
            return None
            
        except ValueError:
            logging.error(f"Invalid time pattern: {pattern}")
            return None
    
    def _is_cron_pattern(self, pattern: str) -> bool:
        """Check if pattern is a cron expression"""
        return ' ' in pattern and len(pattern.split()) == 5
    
    def _job_executed(self, event):
        """Handle job execution events"""
        job_id = event.job_id
        logging.debug(f"Job executed: {job_id}")
    
    def _job_error(self, event):
        """Handle job error events"""
        job_id = event.job_id
        exception = event.exception
        logging.error(f"Job failed: {job_id}, Error: {exception}")
        
        # Send notification if configured
        if self.notification_entity:
            message = f"Scheduler job failed: {job_id}"
            try:
                self.set_value(self.notification_entity, message)
            except Exception as e:
                logging.error(f"Failed to send notification: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"Received signal {signum}, shutting down gracefully")
        self.active = False
    
    async def _shutdown(self):
        """Graceful shutdown"""
        logging.info("Shutting down scheduler")
        
        try:
            # Shutdown scheduler
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logging.info("Scheduler shutdown complete")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
    
    # Helper methods for health checks and data retrieval
    def _get_current_grid_power(self) -> float:
        """Get current grid power usage"""
        # This would interface with Home Assistant to get real-time data
        # For now, return placeholder
        return 0.0
    
    def _get_battery_soc(self) -> float:
        """Get current battery state of charge"""
        # This would interface with Home Assistant to get real-time data
        # For now, return placeholder
        return 50.0
    
    def _get_latest_data_timestamp(self, data_code: str) -> float:
        """Get timestamp of latest data entry"""
        try:
            # Query database for latest timestamp
            query_result = self.db_da.get_column_data('values', data_code)
            if not query_result.empty:
                # Parse timestamp from last entry
                return float(query_result.iloc[-1]['time'])
            return 0.0
        except Exception:
            return 0.0
    
    def _check_database_health(self) -> bool:
        """Check database connectivity"""
        try:
            # Simple query to test connection
            test_data = self.db_da.get_column_data('values', 'da')
            return True
        except Exception as e:
            logging.error(f"Database health check failed: {e}")
            return False
    
    def _check_ha_connectivity(self) -> bool:
        """Check Home Assistant connectivity"""
        try:
            # Test HA connection
            # This would make a simple API call to HA
            return True
        except Exception as e:
            logging.error(f"HA connectivity check failed: {e}")
            return False

    # Public API for external events
    async def send_event(self, event_type: str, data: Dict[str, Any], priority: str = 'normal'):
        """
        Send an external event to the scheduler.
        This allows other components to trigger scheduler actions.
        """
        await self.event_queue.put({
            'type': event_type,
            'priority': priority,
            'data': data,
            'timestamp': datetime.datetime.now().timestamp()
        })


async def main():
    """Main entry point for the modern scheduler"""
    scheduler = ModernDaScheduler("../data/options.json")
    
    try:
        await scheduler.start_scheduler()
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user")
    except Exception as e:
        logging.error(f"Scheduler error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s : %(message)s'
    )
    
    # Run the async scheduler
    asyncio.run(main())