"""
Migration from configuration v0 to v1.

TEMPLATE: This file is commented out and serves as a template for future migrations.
Uncomment and modify when you need to create a real v0→v1 migration.
"""

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


def migrate_v1_to_v2(config: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate from v1 to v2.

    Changes in v2:
    - [DESCRIBE YOUR CHANGES HERE]
    - moved "entity_balance_switch" from battery to grid
    - moved "entity_grid_setpoint" from battery to grid

    Args:
        config: Version 1 configuration

    Returns:
        Version 2 configuration
    """
    # Deep copy: nested dicts (batteries, grid) are modified in place below.
    migrated = copy.deepcopy(config)

    if not isinstance(migrated.get("grid"), dict):
        migrated["grid"] = {}
    grid = migrated["grid"]
    if "battery" in migrated and isinstance(migrated["battery"], list):
        # Real configurations use the aliases with spaces ("entity balance
        # switch"); the snake_case spelling is what populate_by_name allows.
        # Both must be moved, otherwise the key silently lingers in the
        # battery as an unknown extra and the grid feature stays off.
        for snake, spaced in (
            ("entity_balance_switch", "entity balance switch"),
            ("entity_grid_setpoint", "entity grid setpoint"),
        ):
            already = grid.get(spaced) or grid.get(snake)
            for battery in migrated["battery"]:
                if not isinstance(battery, dict):
                    continue
                for key in (spaced, snake):
                    if key not in battery:
                        continue
                    value = battery.pop(key)
                    name = battery.get("name", "unknown")
                    if already is None and value:
                        grid[spaced] = value
                        already = value
                        logger.info(f"Moved '{spaced}' from battery {name} -> grid")
                    else:
                        logger.info(f"Removed '{key}' from battery {name}")

    # Update version
    migrated["config_version"] = 2

    logger.info("Migrated configuration from v1 to v2")
    return migrated
