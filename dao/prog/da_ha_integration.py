"""
Enhanced Home Assistant Integration Service
Provides deeper integration with HA for real-time monitoring and control
"""

import logging
import asyncio
import json
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import websockets
from dao.prog.da_config import Config


class HomeAssistantIntegration:
    """Enhanced Home Assistant integration with real-time capabilities"""
    
    def __init__(self, config: Config):
        self.config = config
        self.ha_url = config.get(['homeassistant', 'url'])
        self.ha_token = config.get(['homeassistant', 'token'])
        self.ws_connection = None
        self.entity_cache = {}
        self.event_handlers = {}
        
        if not self.ha_url or not self.ha_token:
            logging.warning("Home Assistant URL or token not configured")
            return
            
        # Clean URL
        if not self.ha_url.startswith(('http://', 'https://')):
            self.ha_url = 'http://' + self.ha_url
        if self.ha_url.endswith('/'):
            self.ha_url = self.ha_url[:-1]
            
        self.headers = {
            'Authorization': f'Bearer {self.ha_token}',
            'Content-Type': 'application/json'
        }
        
        logging.info(f"HA Integration initialized for {self.ha_url}")

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Home Assistant"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ha_url}/api/", headers=self.headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "version": data.get('version', 'Unknown'),
                            "message": f"Connected to HA {data.get('version', 'Unknown')}"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {await response.text()}"
                        }
        except asyncio.TimeoutError:
            return {"success": False, "error": "Connection timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_entities(self, domains: List[str] = None) -> Dict[str, List[Dict]]:
        """Get entities from Home Assistant, optionally filtered by domain"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ha_url}/api/states", headers=self.headers) as response:
                    if response.status == 200:
                        entities = await response.json()
                        
                        categorized = {
                            'sensors': [],
                            'switches': [],
                            'binary_sensors': [],
                            'input_numbers': [],
                            'input_booleans': [],
                            'automations': [],
                            'scripts': [],
                            'scenes': [],
                            'climate': [],
                            'lights': [],
                            'covers': [],
                            'fans': [],
                            'waters': [],
                            'device_trackers': []
                        }
                        
                        for entity in entities:
                            entity_id = entity['entity_id']
                            domain = entity_id.split('.')[0]
                            
                            if domains and domain not in domains:
                                continue
                                
                            entity_info = {
                                'entity_id': entity_id,
                                'name': entity.get('attributes', {}).get('friendly_name', entity_id),
                                'state': entity.get('state'),
                                'unit': entity.get('attributes', {}).get('unit_of_measurement', ''),
                                'device_class': entity.get('attributes', {}).get('device_class', ''),
                                'last_changed': entity.get('last_changed'),
                                'attributes': entity.get('attributes', {})
                            }
                            
                            # Categorize by domain
                            if domain in categorized:
                                categorized[domain].append(entity_info)
                            elif domain == 'sensor':
                                categorized['sensors'].append(entity_info)
                            elif domain == 'switch':
                                categorized['switches'].append(entity_info)
                            elif domain == 'binary_sensor':
                                categorized['binary_sensors'].append(entity_info)
                        
                        # Cache entities
                        self.entity_cache = categorized
                        return {"success": True, "entities": categorized}
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            logging.error(f"Error getting HA entities: {e}")
            return {"success": False, "error": str(e)}

    async def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        """Get state of a specific entity"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ha_url}/api/states/{entity_id}", headers=self.headers) as response:
                    if response.status == 200:
                        entity = await response.json()
                        return {
                            "success": True,
                            "entity_id": entity_id,
                            "state": entity.get('state'),
                            "attributes": entity.get('attributes', {}),
                            "last_changed": entity.get('last_changed'),
                            "last_updated": entity.get('last_updated')
                        }
                    else:
                        return {"success": False, "error": f"Entity not found: {entity_id}"}
                        
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def call_service(self, domain: str, service: str, entity_id: str = None, data: Dict = None) -> Dict[str, Any]:
        """Call a Home Assistant service"""
        try:
            service_data = data or {}
            if entity_id:
                service_data['entity_id'] = entity_id
                
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ha_url}/api/services/{domain}/{service}",
                    headers=self.headers,
                    json=service_data
                ) as response:
                    if response.status in [200, 201]:
                        return {"success": True, "message": f"Service {domain}.{service} called successfully"}
                    else:
                        return {"success": False, "error": f"HTTP {response.status}: {await response.text()}"}
                        
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def set_entity_state(self, entity_id: str, state: Any, attributes: Dict = None) -> Dict[str, Any]:
        """Set state of an entity (for input helpers, etc.)"""
        try:
            domain = entity_id.split('.')[0]
            
            if domain == 'input_number':
                return await self.call_service('input_number', 'set_value', entity_id, {'value': state})
            elif domain == 'input_boolean':
                service = 'turn_on' if state else 'turn_off'
                return await self.call_service('input_boolean', service, entity_id)
            elif domain == 'input_text':
                return await self.call_service('input_text', 'set_value', entity_id, {'value': state})
            elif domain == 'input_select':
                return await self.call_service('input_select', 'select_option', entity_id, {'option': state})
            elif domain == 'switch':
                service = 'turn_on' if state else 'turn_off'
                return await self.call_service('switch', service, entity_id)
            else:
                return {"success": False, "error": f"Cannot set state for domain: {domain}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_history(self, entity_ids: List[str], start_time: datetime, end_time: datetime = None) -> Dict[str, Any]:
        """Get historical data for entities"""
        try:
            if not end_time:
                end_time = datetime.now()
                
            params = {
                'filter_entity_id': ','.join(entity_ids),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ha_url}/api/history/period", headers=self.headers, params=params) as response:
                    if response.status == 200:
                        history = await response.json()
                        return {"success": True, "history": history}
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_energy_data(self, start_date: datetime, end_date: datetime = None) -> Dict[str, Any]:
        """Get energy dashboard data"""
        try:
            if not end_date:
                end_date = datetime.now()
                
            # Get energy entities based on configuration
            energy_entities = self.get_energy_entities()
            
            if not energy_entities:
                return {"success": False, "error": "No energy entities configured"}
            
            # Get history for energy entities
            history_result = await self.get_history(energy_entities, start_date, end_date)
            
            if not history_result["success"]:
                return history_result
            
            # Process energy data
            processed_data = self.process_energy_history(history_result["history"])
            
            return {
                "success": True,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "data": processed_data
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_energy_entities(self) -> List[str]:
        """Get configured energy entities from DAO config"""
        entities = []
        
        # Add battery entities
        batteries = self.config.get(['battery'], [])
        for battery in batteries:
            if 'entity' in battery:
                entities.append(battery['entity'])
        
        # Add EV entities  
        evs = self.config.get(['electric_vehicle'], [])
        for ev in evs:
            if 'entity' in ev:
                entities.append(ev['entity'])
                
        # Add solar entities
        solar = self.config.get(['solar'], {})
        if 'entity' in solar:
            entities.append(solar['entity'])
            
        # Add grid/consumption entities
        grid_entity = self.config.get(['grid', 'entity'])
        if grid_entity:
            entities.append(grid_entity)
            
        consumption_entity = self.config.get(['consumption', 'entity'])
        if consumption_entity:
            entities.append(consumption_entity)
            
        return entities

    def process_energy_history(self, history_data: List) -> Dict[str, Any]:
        """Process raw history data into usable energy metrics"""
        processed = {
            'consumption': [],
            'production': [],
            'battery_soc': [],
            'grid_power': [],
            'timestamps': []
        }
        
        for entity_history in history_data:
            if not entity_history:
                continue
                
            entity_id = entity_history[0]['entity_id']
            
            for state in entity_history:
                try:
                    timestamp = datetime.fromisoformat(state['last_changed'].replace('Z', '+00:00'))
                    value = float(state['state'])
                    
                    # Categorize based on entity type/name
                    if 'consumption' in entity_id.lower() or 'verbruik' in entity_id.lower():
                        processed['consumption'].append({'time': timestamp, 'value': value})
                    elif 'production' in entity_id.lower() or 'solar' in entity_id.lower() or 'pv' in entity_id.lower():
                        processed['production'].append({'time': timestamp, 'value': value})
                    elif 'battery' in entity_id.lower() and ('soc' in entity_id.lower() or 'level' in entity_id.lower()):
                        processed['battery_soc'].append({'time': timestamp, 'value': value})
                    elif 'grid' in entity_id.lower() or 'net' in entity_id.lower():
                        processed['grid_power'].append({'time': timestamp, 'value': value})
                        
                except (ValueError, KeyError):
                    continue
                    
        return processed

    async def start_websocket_listener(self):
        """Start WebSocket connection for real-time events"""
        try:
            ws_url = self.ha_url.replace('http', 'ws') + '/api/websocket'
            
            async with websockets.connect(ws_url) as websocket:
                self.ws_connection = websocket
                
                # Authentication
                auth_message = await websocket.recv()
                auth_data = json.loads(auth_message)
                
                if auth_data.get('type') == 'auth_required':
                    await websocket.send(json.dumps({
                        'type': 'auth',
                        'access_token': self.ha_token
                    }))
                    
                    auth_result = await websocket.recv()
                    auth_result_data = json.loads(auth_result)
                    
                    if auth_result_data.get('type') == 'auth_ok':
                        logging.info("WebSocket authenticated successfully")
                        
                        # Subscribe to events
                        await self.subscribe_to_events()
                        
                        # Listen for events
                        await self.listen_for_events()
                    else:
                        logging.error("WebSocket authentication failed")
                        
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
            self.ws_connection = None

    async def subscribe_to_events(self):
        """Subscribe to relevant Home Assistant events"""
        if not self.ws_connection:
            return
            
        # Subscribe to state changes for energy entities
        energy_entities = self.get_energy_entities()
        
        await self.ws_connection.send(json.dumps({
            'id': 1,
            'type': 'subscribe_events',
            'event_type': 'state_changed'
        }))

    async def listen_for_events(self):
        """Listen for WebSocket events"""
        if not self.ws_connection:
            return
            
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                
                if data.get('type') == 'event':
                    event = data.get('event', {})
                    if event.get('event_type') == 'state_changed':
                        await self.handle_state_change(event.get('data', {}))
                        
        except Exception as e:
            logging.error(f"WebSocket listener error: {e}")

    async def handle_state_change(self, event_data: Dict):
        """Handle state change events"""
        entity_id = event_data.get('entity_id')
        new_state = event_data.get('new_state', {})
        old_state = event_data.get('old_state', {})
        
        if not entity_id:
            return
            
        # Check if this is an energy-related entity
        energy_entities = self.get_energy_entities()
        if entity_id not in energy_entities:
            return
            
        # Log significant changes
        new_value = new_state.get('state')
        old_value = old_state.get('state', '') if old_state else ''
        
        logging.info(f"Energy entity changed: {entity_id} {old_value} -> {new_value}")
        
        # Trigger event handlers
        for handler_name, handler_func in self.event_handlers.items():
            try:
                await handler_func(entity_id, old_state, new_state)
            except Exception as e:
                logging.error(f"Event handler {handler_name} error: {e}")

    def register_event_handler(self, name: str, handler_func):
        """Register an event handler for state changes"""
        self.event_handlers[name] = handler_func
        logging.info(f"Registered event handler: {name}")

    async def create_dao_entities(self) -> Dict[str, Any]:
        """Create DAO-specific entities in Home Assistant"""
        entities_to_create = [
            {
                'entity_id': 'input_boolean.dao_optimization_enabled',
                'name': 'DAO Optimalisatie Ingeschakeld',
                'icon': 'mdi:lightning-bolt'
            },
            {
                'entity_id': 'input_number.dao_battery_target_soc',
                'name': 'DAO Batterij Doel SOC',
                'min': 0,
                'max': 100,
                'step': 5,
                'unit_of_measurement': '%',
                'icon': 'mdi:battery'
            },
            {
                'entity_id': 'input_number.dao_ev_departure_time',
                'name': 'DAO EV Vertrektijd',
                'min': 0,
                'max': 23,
                'step': 1,
                'unit_of_measurement': 'hour',
                'icon': 'mdi:car-electric'
            },
            {
                'entity_id': 'sensor.dao_optimization_status',
                'name': 'DAO Optimalisatie Status',
                'icon': 'mdi:cog'
            },
            {
                'entity_id': 'sensor.dao_next_optimization',
                'name': 'DAO Volgende Optimalisatie',
                'device_class': 'timestamp',
                'icon': 'mdi:clock'
            },
            {
                'entity_id': 'sensor.dao_daily_savings',
                'name': 'DAO Dagelijkse Besparing',
                'unit_of_measurement': '€',
                'icon': 'mdi:currency-eur'
            }
        ]
        
        created_count = 0
        errors = []
        
        for entity_config in entities_to_create:
            try:
                entity_id = entity_config['entity_id']
                domain = entity_id.split('.')[0]
                
                # Check if entity already exists
                state_result = await self.get_entity_state(entity_id)
                if state_result['success']:
                    continue  # Entity already exists
                
                # Create the entity using appropriate service
                if domain == 'input_boolean':
                    result = await self.call_service('input_boolean', 'create', data={
                        'name': entity_config['name'],
                        'icon': entity_config.get('icon')
                    })
                elif domain == 'input_number':
                    result = await self.call_service('input_number', 'create', data={
                        'name': entity_config['name'],
                        'min': entity_config.get('min', 0),
                        'max': entity_config.get('max', 100),
                        'step': entity_config.get('step', 1),
                        'unit_of_measurement': entity_config.get('unit_of_measurement'),
                        'icon': entity_config.get('icon')
                    })
                else:
                    # For sensors, we need to use a different approach (MQTT or template)
                    continue
                    
                if result['success']:
                    created_count += 1
                    logging.info(f"Created DAO entity: {entity_id}")
                else:
                    errors.append(f"{entity_id}: {result['error']}")
                    
            except Exception as e:
                errors.append(f"{entity_config['entity_id']}: {str(e)}")
        
        return {
            'success': created_count > 0 or len(errors) == 0,
            'created_count': created_count,
            'total_entities': len(entities_to_create),
            'errors': errors
        }

    async def update_dao_sensors(self, optimization_data: Dict) -> Dict[str, Any]:
        """Update DAO-specific sensor states"""
        updates = []
        errors = []
        
        sensor_updates = [
            ('sensor.dao_optimization_status', optimization_data.get('status', 'Unknown')),
            ('sensor.dao_next_optimization', optimization_data.get('next_run', datetime.now().isoformat())),
            ('sensor.dao_daily_savings', optimization_data.get('daily_savings', 0))
        ]
        
        for entity_id, value in sensor_updates:
            try:
                # Use the state setting API (requires MQTT or custom integration)
                result = await self.call_service('homeassistant', 'update_entity', data={
                    'entity_id': entity_id,
                    'state': value
                })
                
                if result['success']:
                    updates.append(entity_id)
                else:
                    errors.append(f"{entity_id}: {result['error']}")
                    
            except Exception as e:
                errors.append(f"{entity_id}: {str(e)}")
        
        return {
            'success': len(updates) > 0,
            'updated': updates,
            'errors': errors
        }

    async def get_dao_settings(self) -> Dict[str, Any]:
        """Get DAO settings from Home Assistant input helpers"""
        settings = {}
        
        dao_entities = [
            'input_boolean.dao_optimization_enabled',
            'input_number.dao_battery_target_soc',
            'input_number.dao_ev_departure_time'
        ]
        
        for entity_id in dao_entities:
            result = await self.get_entity_state(entity_id)
            if result['success']:
                setting_name = entity_id.replace('input_boolean.dao_', '').replace('input_number.dao_', '')
                settings[setting_name] = result['state']
        
        return {'success': True, 'settings': settings}


# Convenience function for integration
def create_ha_integration(config: Config) -> HomeAssistantIntegration:
    """Create and return a HA integration instance"""
    return HomeAssistantIntegration(config)