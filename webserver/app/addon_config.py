#!/usr/bin/env python3
"""
Home Assistant Add-on Configuration Manager
Leest configuratie uit /data/options.json en past deze toe op de applicatie
"""
import json
import os
import logging
from typing import Dict, Any, Optional

class AddonConfig:
    """Beheert de Home Assistant add-on configuratie"""

    def __init__(self):
        self.config_file = "/data/options.json"
        self.config = {}
        self.load_config()
        self.setup_logging()

    def load_config(self) -> None:
        """Laadt de configuratie uit /data/options.json"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                logging.debug(f"Add-on configuratie geladen uit {self.config_file}")
            else:
                logging.debug(f"Configuratie bestand {self.config_file} niet gevonden, gebruik standaard waarden")
                self.config = {}
        except Exception as e:
            logging.warning(f"Kon configuratie niet laden uit {self.config_file}: {e}")
            self.config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Haalt een configuratie waarde op"""
        return self.config.get(key, default)

    def get_log_level(self) -> str:
        """Haalt het loglevel op uit de configuratie"""
        # Probeer eerst de nieuwe add-on configuratie
        log_level = self.get("log_level", "info")

        # Fallback naar oude configuratie structuur
        if log_level == "info":
            log_level = self.config.get("logging level", "info")

        # Valideer en normaliseer
        valid_levels = ["debug", "info", "warning", "error"]
        if log_level.lower() not in valid_levels:
            log_level = "info"

        return log_level.lower()

    def get_database_engine(self) -> str:
        """Haalt de database engine op"""
        return self.get("database_engine", "sqlite")

    def get_homeassistant_token(self) -> str:
        """Haalt de Home Assistant API token op"""
        return self.get("homeassistant_token", "")

    def get_homeassistant_url(self) -> Optional[str]:
        """Haalt de Home Assistant URL op"""
        return self.get("homeassistant_url")

    def get_optimization_mode(self) -> str:
        """Haalt de optimalisatie modus op"""
        return self.get("optimization_mode", "statistical")

    def get_cloud_ai_enabled(self) -> bool:
        """Haalt de cloud AI instelling op"""
        return self.get("cloud_ai_enabled", False)

    def get_battery_entity_id(self) -> Optional[str]:
        """Haalt de Home Assistant batterij entity ID op"""
        return self.get("battery_entity_id")

    def get_battery_setpoint_entity_id(self) -> Optional[str]:
        """Haalt de Home Assistant batterij setpoint entity ID op"""
        return self.get("battery_setpoint_entity_id")

    def get_battery_control_enabled(self) -> bool:
        """Controleert of batterij controle is ingeschakeld"""
        return self.get("battery_control_enabled", False)

    def get_battery_min_soc(self) -> float:
        """Haalt de minimale batterij SoC op"""
        return self.get("battery_min_soc", 20.0)

    def get_battery_max_soc(self) -> float:
        """Haalt de maximale batterij SoC op"""
        return self.get("battery_max_soc", 90.0)

    def get_battery_max_charge_rate(self) -> float:
        """Haalt de maximale laad snelheid op in kW"""
        return self.get("battery_max_charge_rate", 5.0)

    def get_battery_max_discharge_rate(self) -> float:
        """Haalt de maximale ontlaad snelheid op in kW"""
        return self.get("battery_max_discharge_rate", 5.0)

    def setup_logging(self) -> None:
        """Stelt logging in op basis van de configuratie"""
        log_level = self.get_log_level()

        # Converteer naar Python logging niveau
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR
        }

        logging_level = level_map.get(log_level, logging.INFO)

        # Configureer root logger
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()  # Log naar STDOUT
            ]
        )

        # Stel Flask logger in
        flask_logger = logging.getLogger('werkzeug')
        flask_logger.setLevel(logging_level)

        # Stel SQLAlchemy logger in
        sqlalchemy_logger = logging.getLogger('sqlalchemy')
        sqlalchemy_logger.setLevel(logging_level)

        logging.debug(f"Logging ingesteld op niveau: {log_level.upper()}")

    def reload_config(self) -> None:
        """Herlaadt de configuratie"""
        self.load_config()
        self.setup_logging()
        logging.debug("Configuratie herladen en logging bijgewerkt")

# Globale instantie
addon_config = AddonConfig()
