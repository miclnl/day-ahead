#!/usr/bin/env python3
"""
Start Smart DAO Services
Start alle smart optimization services als background processen
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, '/root/dao/prog')
sys.path.insert(0, '/root/dao')

from dao.prog.da_config import Config
from dao.prog.da_smart_integration import start_smart_dao_services, stop_smart_dao_services
from dao.webserver.da_websocket import DAOWebSocketServer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/dao/data/log/smart_services.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global variables
config = None
ws_server = None
running = True


async def shutdown(signal_received=None):
    """Shutdown all services gracefully"""
    global running, ws_server
    
    if signal_received:
        logger.info(f"Received signal {signal_received.name}, shutting down...")
    else:
        logger.info("Shutting down smart services...")
    
    running = False
    
    try:
        # Stop Smart DAO services
        await stop_smart_dao_services()
        
        # Stop WebSocket server
        if ws_server:
            await ws_server.stop_server()
            
        logger.info("All smart services stopped successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(shutdown(signal.Signals(s))))
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(shutdown(signal.Signals(s))))


async def start_services():
    """Start all smart services"""
    global config, ws_server
    
    try:
        # Load configuration
        config_path = "/root/dao/data/options.json"
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return False
            
        config = Config(config_path)
        logger.info("Configuration loaded successfully")
        
        # Start WebSocket server
        ws_server = DAOWebSocketServer(config, port=8765)
        await ws_server.start_server()
        logger.info("WebSocket server started on port 8765")
        
        # Start Smart DAO services
        await start_smart_dao_services(config)
        logger.info("Smart DAO services started successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start services: {e}")
        return False


async def main():
    """Main service loop"""
    global running
    
    logger.info("=== Starting DAO Smart Services ===")
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Start all services
    if not await start_services():
        logger.error("Failed to start services, exiting")
        sys.exit(1)
    
    logger.info("All services started successfully")
    logger.info("Smart DAO is ready for optimization!")
    
    # Main service loop
    try:
        while running:
            await asyncio.sleep(10)
            
            # Health check every 10 seconds
            if config and ws_server:
                # Check if WebSocket server is still running
                if not ws_server.running:
                    logger.warning("WebSocket server stopped unexpectedly, restarting...")
                    try:
                        await ws_server.start_server()
                        logger.info("WebSocket server restarted successfully")
                    except Exception as e:
                        logger.error(f"Failed to restart WebSocket server: {e}")
            
    except asyncio.CancelledError:
        logger.info("Main loop cancelled")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        await shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    
    logger.info("DAO Smart Services shutdown complete")