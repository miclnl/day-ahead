"""
Home Assistant Token Helper for DAO Light

Automatically obtains HA token via environment variables or addon config.
"""

import os
import logging
from typing import Optional


class HATokenHelper:
    """Helper class to automatically obtain Home Assistant access token."""
    
    @staticmethod
    def get_auto_token() -> Optional[str]:
        """
        Get Home Assistant token automatically from various sources.
        
        Returns:
            str: The token if found, None otherwise
        """
        # Method 1: Environment variable (most reliable for addons)
        supervisor_token = os.getenv('SUPERVISOR_TOKEN')
        if supervisor_token:
            logging.info("Using SUPERVISOR_TOKEN for Home Assistant access")
            return supervisor_token
            
        # Method 2: HA addon environment 
        ha_token = os.getenv('HOMEASSISTANT_TOKEN')
        if ha_token:
            logging.info("Using HOMEASSISTANT_TOKEN for Home Assistant access")
            return ha_token
            
        # Method 3: Check for token file (some setups)
        token_paths = [
            '/var/run/secrets/homeassistant_token',
            '/data/homeassistant_token',
            '/config/homeassistant_token'
        ]
        
        for token_path in token_paths:
            if os.path.exists(token_path):
                try:
                    with open(token_path, 'r') as f:
                        token = f.read().strip()
                    if token:
                        logging.info(f"Using token from {token_path}")
                        return token
                except Exception as e:
                    logging.debug(f"Failed to read token from {token_path}: {e}")
                    
        logging.warning("No automatic HA token found - manual configuration required")
        return None
    
    @staticmethod
    def get_auto_url() -> str:
        """
        Get Home Assistant URL automatically.
        
        Returns:
            str: The URL for Home Assistant
        """
        # In addon context, supervisor is the standard internal URL
        if os.getenv('SUPERVISOR_TOKEN'):
            return "http://supervisor/core"
            
        # Fallback for other environments
        ha_url = os.getenv('HOMEASSISTANT_URL', 'http://supervisor/core')
        logging.info(f"Using Home Assistant URL: {ha_url}")
        return ha_url
    
    @staticmethod
    def is_addon_context() -> bool:
        """Check if we're running in a Home Assistant addon context."""
        return bool(os.getenv('SUPERVISOR_TOKEN'))