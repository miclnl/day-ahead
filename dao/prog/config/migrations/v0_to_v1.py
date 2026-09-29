"""
Migration from configuration v0 to v1.

TEMPLATE: This file is commented out and serves as a template for future migrations.
Uncomment and modify when you need to create a real v0→v1 migration.
"""

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


def migrate_v0_to_v1(config: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate from v0 to v1.

    Changes in v1:
    - [DESCRIBE YOUR CHANGES HERE]
    - changed meteo_atteps -> meteo_attempts
    - changed [ev]["entity stop laden"] -> [ev]["entity_stop_charging"]

    Args:
        config: Version 0 configuration

    Returns:
        Version 1 configuration
    """
    # Deep copy: nested dicts (EVs) are modified in place below.
    migrated = copy.deepcopy(config)

    # meteoserver attempts: the v0 model spelled the field "meteoserver_attemps"
    # with alias "meteoserver-attemps"; v1 fixes the typo. Move whichever
    # spelling the document uses to the v1 alias, but never overwrite a value
    # that is already there under the new name.
    value = None
    for old_key in ("meteoserver-attemps", "meteoserver_attemps", "meteo_attemps", "meteo attemps"):
        if old_key in migrated:
            if value is None:
                value = migrated[old_key]
            del migrated[old_key]
    if value is not None and not (
        "meteoserver-attempts" in migrated or "meteoserver_attempts" in migrated
    ):
        migrated["meteoserver-attempts"] = value
        logger.info(f"changed meteoserver-attemps -> meteoserver-attempts ({value})")

    # entity_stop_charging
    if "electric_vehicle" in migrated and isinstance(
        migrated["electric_vehicle"], list
    ):
        for ev in migrated["electric_vehicle"]:
            value = None
            if "entity_stop_laden" in ev:
                value = ev["entity_stop_laden"]
                del ev["entity_stop_laden"]
            if "entity stop laden" in ev:
                value = ev["entity stop laden"]
                del ev["entity stop laden"]
            if value:
                ev["entity_stop_charging"] = value
                logger.info("changed entity_stop_laden -> entity_stop_charging")

    # Update version
    migrated["config_version"] = 1

    logger.info("Migrated configuration from v0 to v1")
    return migrated
