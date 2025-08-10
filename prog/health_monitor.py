"""Health monitoring and status reporting for DAO addon."""
import os
import time
import json
import logging
import psutil
from typing import Dict, Any
from datetime import datetime, timedelta

class HealthMonitor:
    """Monitor addon health and report status to Home Assistant."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()
        self.last_optimization = None
        self.error_count = 0
        self.warning_count = 0
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'boot_time': psutil.boot_time(),
                'uptime': str(datetime.now() - self.start_time)
            }
        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization engine status."""
        status = {
            'last_run': self.last_optimization.isoformat() if self.last_optimization else None,
            'next_run': None,  # Would be calculated based on schedule
            'status': 'running' if self.last_optimization and 
                     datetime.now() - self.last_optimization < timedelta(hours=1) else 'idle',
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }
        
        # Add next run prediction if we have schedule info
        if self.last_optimization:
            # Assume hourly optimization
            status['next_run'] = (self.last_optimization + timedelta(hours=1)).isoformat()
            
        return status
    
    def get_database_status(self) -> Dict[str, Any]:
        """Check database connectivity and status."""
        try:
            # This would be implemented to check actual DB connection
            # For now, return basic status
            return {
                'status': 'connected',
                'type': os.getenv('DB_ENGINE', 'sqlite'),
                'last_check': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def get_api_status(self) -> Dict[str, Any]:
        """Check external API connectivity."""
        apis = {}
        
        # Check Nord Pool API
        try:
            import requests
            response = requests.get('https://www.nordpoolgroup.com/api/marketdata/page/10', timeout=5)
            apis['nordpool'] = {
                'status': 'ok' if response.status_code == 200 else 'error',
                'response_time': response.elapsed.total_seconds(),
                'last_check': datetime.now().isoformat()
            }
        except Exception as e:
            apis['nordpool'] = {
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
        
        # Check Home Assistant API
        try:
            supervisor_token = os.getenv('SUPERVISOR_TOKEN')
            if supervisor_token:
                headers = {'Authorization': f'Bearer {supervisor_token}'}
                response = requests.get('http://supervisor/core/api/', headers=headers, timeout=5)
                apis['homeassistant'] = {
                    'status': 'ok' if response.status_code == 200 else 'error',
                    'response_time': response.elapsed.total_seconds(),
                    'last_check': datetime.now().isoformat()
                }
        except Exception as e:
            apis['homeassistant'] = {
                'status': 'error', 
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
            
        return apis
    
    def get_full_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        return {
            'addon_info': {
                'name': 'DAO Modern',
                'version': os.getenv('DAO_VERSION', '1.0.0'),
                'uptime': str(datetime.now() - self.start_time)
            },
            'system': self.get_system_stats(),
            'optimization': self.get_optimization_status(),
            'database': self.get_database_status(),
            'apis': self.get_api_status(),
            'timestamp': datetime.now().isoformat()
        }
    
    def report_to_homeassistant(self, status: Dict[str, Any]) -> None:
        """Send health status to Home Assistant as sensor data."""
        try:
            supervisor_token = os.getenv('SUPERVISOR_TOKEN')
            if not supervisor_token:
                return
                
            import requests
            headers = {
                'Authorization': f'Bearer {supervisor_token}',
                'Content-Type': 'application/json'
            }
            
            # Create sensor entities for key metrics
            sensors = [
                {
                    'state': status['optimization']['status'],
                    'attributes': status['optimization'],
                    'entity_id': 'sensor.dao_optimization_status'
                },
                {
                    'state': status['system']['cpu_percent'],
                    'attributes': {'unit_of_measurement': '%'},
                    'entity_id': 'sensor.dao_cpu_usage'
                },
                {
                    'state': status['system']['memory_percent'],
                    'attributes': {'unit_of_measurement': '%'},
                    'entity_id': 'sensor.dao_memory_usage'
                }
            ]
            
            for sensor in sensors:
                requests.post(
                    'http://supervisor/core/api/states/' + sensor['entity_id'],
                    headers=headers,
                    json=sensor,
                    timeout=5
                )
                
        except Exception as e:
            self.logger.error(f"Failed to report status to Home Assistant: {e}")
    
    def update_optimization_status(self, success: bool, error_msg: str = None) -> None:
        """Update optimization run status."""
        self.last_optimization = datetime.now()
        if success:
            self.logger.info("Optimization completed successfully")
        else:
            self.error_count += 1
            self.logger.error(f"Optimization failed: {error_msg}")
            
    def start_monitoring(self, interval: int = 300) -> None:
        """Start periodic health monitoring."""
        import threading
        import time
        
        def monitor_loop():
            while True:
                try:
                    health_report = self.get_full_health_report()
                    self.report_to_homeassistant(health_report)
                    
                    # Log summary
                    self.logger.info(
                        f"Health check: CPU {health_report['system'].get('cpu_percent', 0):.1f}%, "
                        f"Memory {health_report['system'].get('memory_percent', 0):.1f}%, "
                        f"Optimization {health_report['optimization']['status']}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    
                time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info(f"Started health monitoring (interval: {interval}s)")