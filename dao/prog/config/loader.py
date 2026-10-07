"""
Configuration loader with support for versioning, migration, and unknown key preservation.
"""

import copy
import os
import shutil
import json
import math
import logging
from pathlib import Path
from typing import Any, Optional, Type
from pydantic import BaseModel, ValidationError
import fcntl
from .migrations.migrator import migrate_config
from .models.base import FlexValue
from .versions.v0 import ConfigurationV0

# Uncomment when creating v1:
from .versions.v1 import ConfigurationV1
from .versions.v2 import ConfigurationV2
from .versions.v3 import ConfigurationV3

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when configuration fails Pydantic validation, with a human-readable message."""

    def __init__(self, error: ValidationError) -> None:
        lines = ["Configuration validation failed:"]
        for err in error.errors():
            path = " → ".join(str(p) for p in err["loc"])
            lines.append(f"  • {path}: {err['msg']}")
        super().__init__("\n".join(lines))


# Version models registry: maps version number -> Pydantic model class
VERSION_MODELS: dict[int, Type[BaseModel]] = {
    0: ConfigurationV0,
    1: ConfigurationV1,
    # Uncomment when creating v2:
    2: ConfigurationV2,
    3: ConfigurationV3,
}

# Derive current version from registry
CURRENT_VERSION = max(VERSION_MODELS.keys())


def validate_config_data(config_data: Any) -> BaseModel:
    """Migrate (in memory) and validate a configuration without touching disk.

    This is what every writer of options.json must call before saving: the
    web editors, the fast-control mode switch and the tests. It follows the
    same migration path as ConfigurationLoader, so a document with an older
    or missing config_version is judged the way the loader would judge it.

    Raises ConfigValidationError (a ValueError) with a readable message.
    """
    if not isinstance(config_data, dict):
        raise ValueError("De configuratie moet een JSON-object zijn")
    version = config_data.get("config_version")
    if version is not None and (isinstance(version, bool) or not isinstance(version, int)):
        raise ValueError(
            f"config_version moet een geheel getal zijn, niet {version!r}"
        )
    if version is None or version < CURRENT_VERSION:
        config_data = migrate_config(
            copy.deepcopy(config_data), target_version=CURRENT_VERSION
        )
    version = config_data.get("config_version", CURRENT_VERSION)
    if version not in VERSION_MODELS:
        raise ValueError(
            f"Onbekende config_version {version}; bekend zijn "
            f"{sorted(VERSION_MODELS.keys())}"
        )
    try:
        return VERSION_MODELS[version](**config_data)
    except ValidationError as e:
        raise ConfigValidationError(e) from e


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file so a reader never sees a half-written version.

    The content goes to a temporary file in the same directory, is flushed to
    disk and then renamed over the target. A crash in the middle leaves the
    old file untouched instead of a truncated options.json.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def json_safe(data: Any) -> Any:
    """``data`` with every non-finite float replaced by ``None``.

    Python's json module happily writes bare ``NaN``, ``Infinity`` and
    ``-Infinity``, which the JSON specification does not allow and
    JavaScript's ``JSON.parse`` rejects outright. A score for a candidate
    that could not be evaluated is genuinely NaN, so the artefacts do
    produce them; written as ``null`` the reader sees "not available"
    instead of failing to parse the whole document.
    """
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    if isinstance(data, dict):
        return {key: json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_safe(value) for value in data]
    return data


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(
        path, json.dumps(json_safe(data), indent=2, ensure_ascii=False) + "\n"
    )


FAST_CONTROL_MODES = ("off", "shadow", "active")


def set_fast_control_mode(config_path: Path, new_mode: str) -> None:
    """Change only the fast-control mode in options.json.

    Edits the raw document instead of dumping the whole validated model, so
    the user's key spelling, comments-as-extra-keys, explicit nulls and
    omitted defaults survive. The result is validated before it is written.

    Raises ValueError when the mode is invalid, when the mode is bound to a
    Home Assistant entity (it must then be changed in HA), or when the
    resulting configuration does not validate.
    """
    if new_mode not in FAST_CONTROL_MODES:
        raise ValueError(f"Ongeldige modus {new_mode!r}")
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("options.json moet een JSON-object zijn")
    key = "fast control" if "fast control" in raw or "fast_control" not in raw else "fast_control"
    section = raw.get(key)
    if not isinstance(section, dict):
        section = {}
    current = section.get("mode")
    if isinstance(current, dict):
        current = current.get("value")
    if FlexValue.is_entity_id(current):
        raise ValueError(
            f"De modus wordt gestuurd door Home Assistant entity {current}; "
            f"wijzig die in Home Assistant"
        )
    section["mode"] = new_mode
    raw[key] = section
    validate_config_data(raw)
    atomic_write_json(config_path, raw)


class ConfigurationLoader:
    """
    Loads and saves configuration files with migration and unknown key preservation.

    Features:
    - Automatic version detection and migration
    - Unknown key preservation (extra='allow')
    - Secret resolution from separate secrets.json
    - Backup creation before migration
    """

    def __init__(self, config_path: Path, secrets_path: Optional[Path] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to options.json
            secrets_path: Path to secrets.json (optional, auto-detected if omitted)
        """
        self.config_path = config_path
        self.secrets_path = secrets_path or config_path.parent / "secrets.json"
        self._raw_options: Optional[dict[str, Any]] = None
        self._secrets: Optional[dict[str, str]] = None

    def _load_secrets(self) -> dict[str, str]:
        """
        Load secrets from secrets.json.

        Returns:
            Dictionary of secret key->value pairs
        """
        if self._secrets is not None:
            return self._secrets

        if not self.secrets_path.exists():
            logger.warning("No secrets file found, secret resolution will fail")
            self._secrets = {}
            return self._secrets

        with open(self.secrets_path, "r", encoding="utf-8") as f:
            self._secrets = json.load(f)

        logger.info(f"Loaded {len(self._secrets)} secrets from {self.secrets_path}")
        return self._secrets

    def _load_and_migrate(self) -> dict[str, Any]:
        """
        Load configuration and apply migrations if needed.

        Returns:
            Migrated configuration (not yet validated with Pydantic)
        """
        with open(self.config_path, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            # Load raw config
            config_data = json.load(f)

            # Store original for unknown key preservation
            self._raw_options = config_data.copy()

            # Check if migration needed
            config_version = config_data.get("config_version")

            if config_version is None or config_version < CURRENT_VERSION:
                from_ver = (
                    "unversioned" if config_version is None else f"v{config_version}"
                )
                logger.info(
                    f"Configuration needs migration from {from_ver} to v{CURRENT_VERSION}"
                )

                # Save backup before migration
                backup_path = self.config_path.parent / f"options_{from_ver}.json"
                shutil.copy2(self.config_path, backup_path)
                logger.info(f"Saved backup configuration to {backup_path}")

                migrated_data = migrate_config(
                    config_data, target_version=CURRENT_VERSION
                )

                # Validate before anything is written; a migration that
                # produces an invalid document must not replace the file.
                version = migrated_data.get("config_version", CURRENT_VERSION)
                model_class = VERSION_MODELS[version]
                try:
                    model_class(**migrated_data)
                except ValidationError as e:
                    raise ConfigValidationError(e) from e

                # Write the migrated *document*, not a dump of the model. A
                # model dump renamed every key to its python name, froze every
                # default into the user's file and dropped explicit nulls, so
                # the file looked different depending on who saved it last and
                # later default changes never reached existing installations.
                self._raw_options = copy.deepcopy(migrated_data)
                atomic_write_json(self.config_path, migrated_data)
                logger.info(f"Saved migrated configuration to {self.config_path}")
            else:
                logger.debug("Configuration is up to date, no migration needed")
                migrated_data = config_data

            return migrated_data

    def load_and_validate(self) -> BaseModel:
        """
        Load configuration, apply migrations, and validate with Pydantic.

        This is the recommended way to load configuration - it automatically:
        1. Detects the current version
        2. Migrates to CURRENT_VERSION if needed
        3. Validates with the appropriate Pydantic model

        Returns:
            Validated Pydantic model (type depends on CURRENT_VERSION)
        """
        # Migrate to current version if required
        migrated_data = self._load_and_migrate()

        # Ensure secrets are loaded and available via self.secrets
        self._load_secrets()

        # Get the model class for current version
        version = migrated_data.get("config_version", CURRENT_VERSION)

        if version not in VERSION_MODELS:
            raise RuntimeError(
                f"No Pydantic model defined for version {version}. "
                f"Available versions: {list(VERSION_MODELS.keys())}"
            )

        model_class = VERSION_MODELS[version]
        logger.info(f"Validating configuration with {model_class.__name__}")

        # Validate and return; wrap pydantic's ValidationError to strip the noisy
        # input_value dumps and present only field path + message to the user.
        try:
            return model_class(**migrated_data)
        except ValidationError as e:
            raise ConfigValidationError(e) from e

    def save(
        self, config_data: dict[str, Any], save_path: Optional[Path] = None
    ) -> None:
        """
        Save configuration to disk, preserving unknown keys.

        Args:
            config_data: Configuration to save (can be Pydantic model dict or raw dict)
            save_path: Path to save to (defaults to self.config_path)
        """
        if save_path is None:
            save_path = self.config_path

        # Merge with raw options to preserve unknown keys
        if self._raw_options:
            # Start with raw options (includes unknown keys)
            merged = self._raw_options.copy()
            # Update with new values
            merged.update(config_data)
            save_data = merged
        else:
            save_data = config_data

        atomic_write_json(save_path, save_data)
        logger.info(f"Saved configuration to {save_path}")

    @property
    def secrets(self) -> dict[str, str]:
        """Get loaded secrets (lazy load)."""
        if self._secrets is None:
            self._load_secrets()
        return self._secrets
