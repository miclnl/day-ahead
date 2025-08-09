"""
Simple WebSocket-like communication using HTTP Server-Sent Events (SSE)
Alternative to websockets library for real-time updates.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Any
import logging


class SSEHandler(BaseHTTPRequestHandler):
    """HTTP Server-Sent Events handler"""
    
    clients = []  # Store active connections
    
    def do_GET(self):
        """Handle SSE connections"""
        if self.path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Add client to list
            SSEHandler.clients.append(self)
            
            try:
                # Keep connection alive
                while True:
                    time.sleep(1)
                    # Send heartbeat
                    self.wfile.write(b"data: {\"type\":\"heartbeat\"}\n\n")
                    self.wfile.flush()
            except:
                # Remove client when connection closes
                if self in SSEHandler.clients:
                    SSEHandler.clients.remove(self)
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass
    
    @classmethod
    def broadcast(cls, event_type: str, data: Dict[str, Any]):
        """Broadcast to all connected clients"""
        message = f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"
        
        for client in cls.clients[:]:  # Copy list to avoid modification during iteration
            try:
                client.wfile.write(message.encode())
                client.wfile.flush()
            except:
                # Remove broken connections
                cls.clients.remove(client)


class SimpleEventServer:
    """Simple event server using Server-Sent Events"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False
    
    def start(self):
        """Start the SSE server"""
        try:
            self.server = HTTPServer(('0.0.0.0', self.port), SSEHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            self.running = True
            logging.info(f"Simple Event Server started on port {self.port}")
        except Exception as e:
            logging.error(f"Failed to start event server: {e}")
    
    def stop(self):
        """Stop the server"""
        if self.server:
            self.server.shutdown()
            self.running = False
            logging.info("Simple Event Server stopped")
    
    def broadcast(self, event_type: str, data: Dict[str, Any]):
        """Send event to all clients"""
        if self.running:
            SSEHandler.broadcast(event_type, data)


# Global server instance
_event_server = None


def get_event_server(port: int = 8765) -> SimpleEventServer:
    """Get or create event server instance"""
    global _event_server
    if _event_server is None:
        _event_server = SimpleEventServer(port)
        _event_server.start()
    return _event_server


def send_optimization_update(savings: float, next_run: str):
    """Send optimization update to clients"""
    server = get_event_server()
    server.broadcast('optimization_update', {
        'daily_savings': savings,
        'next_run': next_run,
        'timestamp': time.time()
    })


def send_status_update(status: str, message: str):
    """Send status update to clients"""
    server = get_event_server()
    server.broadcast('status_update', {
        'status': status,
        'message': message,
        'timestamp': time.time()
    })