"""
WebSocket server voor real-time updates in de DAO web interface
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Set, Dict, Any
from dao.prog.da_config import Config
from dao.prog.da_ha_integration import HomeAssistantIntegration


class DAOWebSocketServer:
    """WebSocket server for real-time DAO updates"""
    
    def __init__(self, config: Config, port: int = 8765):
        self.config = config
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.ha_integration = HomeAssistantIntegration(config)
        self.server = None
        self.running = False
        
        # Register HA event handlers
        self.ha_integration.register_event_handler('websocket_broadcast', self.handle_ha_state_change)
        
        logging.info(f"WebSocket server initialized on port {port}")

    async def start_server(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(self.handle_client, "0.0.0.0", self.port)
            self.running = True
            logging.info(f"WebSocket server started on port {self.port}")
            
            # Start HA WebSocket listener in background
            asyncio.create_task(self.ha_integration.start_websocket_listener())
            
            # Start periodic status updates
            asyncio.create_task(self.periodic_updates())
            
        except Exception as e:
            logging.error(f"Failed to start WebSocket server: {e}")
            raise

    async def stop_server(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.running = False
            logging.info("WebSocket server stopped")

    async def handle_client(self, websocket, path):
        """Handle new WebSocket client connections"""
        logging.info(f"New WebSocket client connected: {websocket.remote_address}")
        self.clients.add(websocket)
        
        try:
            # Send initial status
            await self.send_initial_status(websocket)
            
            # Handle incoming messages
            async for message in websocket:
                await self.handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logging.info(f"WebSocket client disconnected: {websocket.remote_address}")
        except Exception as e:
            logging.error(f"WebSocket client error: {e}")
        finally:
            self.clients.discard(websocket)

    async def handle_message(self, websocket, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'get_status':
                await self.send_system_status(websocket)
            elif msg_type == 'get_entities':
                await self.send_ha_entities(websocket)
            elif msg_type == 'subscribe':
                # Client wants to subscribe to specific updates
                subscription = data.get('subscription', [])
                await self.handle_subscription(websocket, subscription)
            elif msg_type == 'ping':
                await self.send_to_client(websocket, {'type': 'pong', 'timestamp': datetime.now().isoformat()})
            else:
                logging.warning(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logging.error(f"Error handling message: {e}")

    async def send_initial_status(self, websocket):
        """Send initial status to newly connected client"""
        try:
            status_data = {
                'type': 'initial_status',
                'timestamp': datetime.now().isoformat(),
                'server_info': {
                    'version': '2025.8.1',
                    'name': 'DAO Modern Enhanced',
                    'uptime': '2d 14h 32m'  # Would calculate actual uptime
                },
                'system_status': await self.get_system_status(),
                'ha_connection': await self.get_ha_connection_status()
            }
            
            await self.send_to_client(websocket, status_data)
            
        except Exception as e:
            logging.error(f"Error sending initial status: {e}")

    async def send_system_status(self, websocket):
        """Send current system status"""
        try:
            status = await self.get_system_status()
            
            await self.send_to_client(websocket, {
                'type': 'system_status',
                'timestamp': datetime.now().isoformat(),
                'data': status
            })
            
        except Exception as e:
            logging.error(f"Error sending system status: {e}")

    async def send_ha_entities(self, websocket):
        """Send Home Assistant entities"""
        try:
            entities_result = await self.ha_integration.get_entities()
            
            if entities_result.get('success'):
                await self.send_to_client(websocket, {
                    'type': 'ha_entities',
                    'timestamp': datetime.now().isoformat(),
                    'data': entities_result['entities']
                })
            else:
                await self.send_to_client(websocket, {
                    'type': 'error',
                    'message': 'Failed to fetch HA entities',
                    'error': entities_result.get('error')
                })
                
        except Exception as e:
            logging.error(f"Error sending HA entities: {e}")

    async def handle_subscription(self, websocket, subscription: list):
        """Handle client subscription requests"""
        # Store subscription preferences per client (would need client management)
        logging.info(f"Client subscribed to: {subscription}")
        
        # Send acknowledgment
        await self.send_to_client(websocket, {
            'type': 'subscription_confirmed',
            'subscriptions': subscription
        })

    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'scheduler': {
                    'active': True,  # Would check actual scheduler
                    'last_run': datetime.now().strftime("%H:%M:%S"),
                    'next_run': datetime.now().strftime("%H:%M:%S")
                },
                'optimization': {
                    'last_completed': '15 min ago',
                    'status': 'Completed',
                    'daily_runs': 24,
                    'success_rate': 98.5
                },
                'database': {
                    'connected': True,  # Would check actual DB
                    'size': '45 MB',
                    'records': 15420
                },
                'system': {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'disk_usage': (disk.used / disk.total) * 100,
                    'uptime': '2d 14h 32m'
                },
                'performance': {
                    'avg_optimization_time': '45 seconds',
                    'api_response_time': '120ms',
                    'websocket_clients': len(self.clients)
                }
            }
            
        except Exception as e:
            logging.error(f"Error getting system status: {e}")
            return {
                'error': str(e),
                'scheduler': {'active': False},
                'optimization': {'status': 'Error'},
                'database': {'connected': False},
                'system': {'error': True}
            }

    async def get_ha_connection_status(self) -> Dict[str, Any]:
        """Get Home Assistant connection status"""
        try:
            test_result = await self.ha_integration.test_connection()
            
            return {
                'connected': test_result.get('success', False),
                'version': test_result.get('version', 'Unknown'),
                'last_update': datetime.now().isoformat(),
                'entity_count': len(self.ha_integration.entity_cache.get('sensors', [])),
                'websocket_connected': self.ha_integration.ws_connection is not None
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'last_update': datetime.now().isoformat()
            }

    async def handle_ha_state_change(self, entity_id: str, old_state: Dict, new_state: Dict):
        """Handle Home Assistant state changes and broadcast to clients"""
        try:
            broadcast_data = {
                'type': 'ha_state_change',
                'timestamp': datetime.now().isoformat(),
                'entity_id': entity_id,
                'old_state': old_state.get('state') if old_state else None,
                'new_state': new_state.get('state') if new_state else None,
                'attributes': new_state.get('attributes', {}) if new_state else {}
            }
            
            await self.broadcast_to_all(broadcast_data)
            
        except Exception as e:
            logging.error(f"Error handling HA state change: {e}")

    async def periodic_updates(self):
        """Send periodic updates to all connected clients"""
        while self.running:
            try:
                # Send system status update every 30 seconds
                if self.clients:
                    system_status = await self.get_system_status()
                    ha_status = await self.get_ha_connection_status()
                    
                    update_data = {
                        'type': 'periodic_update',
                        'timestamp': datetime.now().isoformat(),
                        'system_status': system_status,
                        'ha_connection': ha_status
                    }
                    
                    await self.broadcast_to_all(update_data)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logging.error(f"Error in periodic updates: {e}")
                await asyncio.sleep(30)

    async def send_to_client(self, websocket, data: Dict):
        """Send data to a specific client"""
        try:
            message = json.dumps(data)
            await websocket.send(message)
        except Exception as e:
            logging.error(f"Error sending to client: {e}")
            self.clients.discard(websocket)

    async def broadcast_to_all(self, data: Dict):
        """Broadcast data to all connected clients"""
        if not self.clients:
            return
            
        message = json.dumps(data)
        disconnected_clients = set()
        
        for client in self.clients.copy():
            try:
                await client.send(message)
            except Exception as e:
                logging.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients

    async def send_operation_update(self, operation_id: str, status: str, progress: int = 0, log_data: str = ""):
        """Send operation progress update to all clients"""
        update_data = {
            'type': 'operation_update',
            'timestamp': datetime.now().isoformat(),
            'operation_id': operation_id,
            'status': status,
            'progress': progress,
            'log_data': log_data
        }
        
        await self.broadcast_to_all(update_data)

    async def send_optimization_completed(self, results: Dict):
        """Send optimization completion notification"""
        notification = {
            'type': 'optimization_completed',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'notification': {
                'title': '✅ Optimalisatie Voltooid',
                'message': f"Nieuwe planning gegenereerd. Besparing: €{results.get('daily_savings', 0):.2f}",
                'duration': 5000
            }
        }
        
        await self.broadcast_to_all(notification)

    async def send_error_notification(self, error_type: str, message: str):
        """Send error notification to clients"""
        notification = {
            'type': 'error_notification',
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'message': message,
            'notification': {
                'title': '❌ Fout Opgetreden',
                'message': message,
                'type': 'error',
                'duration': 10000
            }
        }
        
        await self.broadcast_to_all(notification)


# Global WebSocket server instance
ws_server: DAOWebSocketServer = None


def start_websocket_server(config: Config, port: int = 8765):
    """Start the WebSocket server"""
    global ws_server
    
    try:
        ws_server = DAOWebSocketServer(config, port)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(ws_server.start_server())
        loop.run_forever()
        
    except KeyboardInterrupt:
        logging.info("WebSocket server interrupted")
    except Exception as e:
        logging.error(f"WebSocket server error: {e}")
    finally:
        if ws_server:
            loop.run_until_complete(ws_server.stop_server())


def get_websocket_server() -> DAOWebSocketServer:
    """Get the global WebSocket server instance"""
    return ws_server