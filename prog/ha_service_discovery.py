"""Home Assistant Service Discovery integration for DAO."""
import os
import json
import logging
from typing import Optional, Dict, Any

class HAServiceDiscovery:
    """Discover and configure services via Home Assistant Supervisor."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supervisor_token = os.getenv('SUPERVISOR_TOKEN')
        
    def get_mysql_config(self) -> Optional[Dict[str, Any]]:
        """Auto-discover MySQL/MariaDB addon configuration."""
        try:
            import requests
            headers = {'Authorization': f'Bearer {self.supervisor_token}'}
            
            # Try MySQL service first
            response = requests.get(
                'http://supervisor/services/mysql',
                headers=headers
            )
            if response.status_code == 200:
                mysql_config = response.json()['data']
                return {
                    'host': mysql_config.get('host', 'core-mariadb'),
                    'port': mysql_config.get('port', 3306),
                    'username': mysql_config.get('username'),
                    'password': mysql_config.get('password'),
                    'database': 'dao_energy'
                }
                
            # Try MariaDB service
            response = requests.get(
                'http://supervisor/services/mariadb',
                headers=headers
            )
            if response.status_code == 200:
                mariadb_config = response.json()['data']
                return {
                    'host': mariadb_config.get('host', 'core-mariadb'),
                    'port': mariadb_config.get('port', 3306),
                    'username': mariadb_config.get('username'),
                    'password': mariadb_config.get('password'),
                    'database': 'dao_energy'
                }
                
        except Exception as e:
            self.logger.debug(f"MySQL/MariaDB service discovery failed: {e}")
        return None
    
    def get_mqtt_config(self) -> Optional[Dict[str, Any]]:
        """Auto-discover MQTT broker configuration."""
        try:
            import requests
            headers = {'Authorization': f'Bearer {self.supervisor_token}'}
            response = requests.get(
                'http://supervisor/services/mqtt',
                headers=headers
            )
            if response.status_code == 200:
                mqtt_config = response.json()['data']
                return {
                    'host': mqtt_config.get('host', 'core-mosquitto'),
                    'port': mqtt_config.get('port', 1883),
                    'username': mqtt_config.get('username'),
                    'password': mqtt_config.get('password'),
                    'ssl': mqtt_config.get('ssl', False)
                }
        except Exception as e:
            self.logger.debug(f"MQTT service discovery failed: {e}")
        return None
            
    def get_homeassistant_config(self) -> Dict[str, Any]:
        """Get Home Assistant configuration from supervisor."""
        return {
            'url': os.getenv('HASSIO_HA_URL', 'http://supervisor/core'),
            'token': self.supervisor_token,
            'websocket_url': os.getenv('HASSIO_HA_WS_URL', 'ws://supervisor/core/websocket')
        }
    
    def auto_configure_database(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically configure database based on available services."""
        # Try MySQL/MariaDB first
        mysql_config = self.get_mysql_config()
        if mysql_config:
            self.logger.info("Auto-configured MySQL/MariaDB database from service discovery")
            config['database da'] = {
                'engine': 'mysql',
                'host': mysql_config['host'],
                'port': mysql_config['port'],
                'database': mysql_config['database'],
                'username': mysql_config['username'],
                'password': mysql_config['password']
            }
            return config
            
        # Fall back to SQLite
        self.logger.info("Using SQLite database (no MySQL service found)")
        config['database da'] = {
            'engine': 'sqlite',
            'db_path': '/data'
        }
        return config