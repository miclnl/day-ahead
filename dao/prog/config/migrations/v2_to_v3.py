"""
Migration from configuration v2 to v3.

``ml_prediction`` (a boolean) becomes ``model`` (physical | ml | auto), so
an installation can also be told to pick whichever of the two backtests
better. A configuration that already carries ``model`` was written against
v3's vocabulary and is left alone.
"""

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Every spelling of the old key a hand-written options.json might use.
_ML_PREDICTION_KEYS = ("ml_prediction", "ml prediction", "ml-prediction")


def _migrate_installation(installation: dict[str, Any]) -> bool:
    """Rewrite one installation in place. Returns whether anything changed."""
    present = [key for key in _ML_PREDICTION_KEYS if key in installation]
    if not present:
        return False

    was_ml = any(bool(installation.pop(key)) for key in present)
    # An explicit "model" was written against v3 and outranks the old flag,
    # which is only removed so it cannot contradict it later.
    if "model" not in installation:
        installation["model"] = "ml" if was_ml else "physical"
    return True


def migrate_v2_to_v3(config: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate from v2 to v3.

    Args:
        config: Version 2 configuration

    Returns:
        Version 3 configuration
    """
    migrated = copy.deepcopy(config)

    installations: list = []
    if isinstance(migrated.get("solar"), list):
        installations.extend(
            item for item in migrated["solar"] if isinstance(item, dict)
        )
    if isinstance(migrated.get("battery"), list):
        for battery in migrated["battery"]:
            if isinstance(battery, dict) and isinstance(battery.get("solar"), list):
                installations.extend(
                    item for item in battery["solar"] if isinstance(item, dict)
                )

    changed = sum(1 for item in installations if _migrate_installation(item))
    if changed:
        logger.info(
            f"Migrated {changed} solar installation(s) from ml_prediction to model"
        )

    migrated["config_version"] = 3
    logger.info("Migrated configuration from v2 to v3")
    return migrated
