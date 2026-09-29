"""
Tests for configuration loader.
"""

import json
import pytest
from pathlib import Path
from dao.prog.config.loader import CURRENT_VERSION, ConfigurationLoader


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory."""
    return tmp_path


@pytest.fixture
def sample_unversioned_config():
    """Sample unversioned configuration."""
    return {
        "database ha": {
            "engine": "sqlite",
            "database": "test.db"
        },
        "logging level": "info",
        "custom_field": "should_be_preserved",
        "meteoserver-key": "test_api_key"
    }


@pytest.fixture
def sample_secrets():
    """Sample secrets."""
    return {
        "db_password": "secret123",
        "api_key": "key456"
    }


class TestConfigurationLoader:
    """Test configuration loader functionality."""
    
    def test_load_nonexistent_config(self, temp_config_dir):
        """Test error when config file doesn't exist."""
        config_path = temp_config_dir / "nonexistent.json"
        loader = ConfigurationLoader(config_path)
        
        with pytest.raises(FileNotFoundError):
            loader.load_and_validate()
    
    def test_load_secrets(self, temp_config_dir, sample_secrets):
        """Test loading secrets via property."""
        config_path = temp_config_dir / "options.json"
        secrets_path = temp_config_dir / "secrets.json"
        
        config_path.write_text(json.dumps({}))
        secrets_path.write_text(json.dumps(sample_secrets))
        
        loader = ConfigurationLoader(config_path, secrets_path)
        secrets = loader.secrets
        
        assert secrets == sample_secrets
    
    def test_load_secrets_file_missing(self, temp_config_dir):
        """Test loading when secrets file doesn't exist."""
        config_path = temp_config_dir / "options.json"
        secrets_path = temp_config_dir / "nonexistent_secrets.json"
        
        config_path.write_text(json.dumps({}))
        
        loader = ConfigurationLoader(config_path, secrets_path)
        secrets = loader.secrets
        
        assert secrets == {}
    
    def test_migrate_unversioned_to_current(
        self, temp_config_dir, sample_unversioned_config
    ):
        """An unversioned config is migrated all the way to the current version."""
        config_path = temp_config_dir / "options.json"
        config_path.write_text(json.dumps(sample_unversioned_config))

        loader = ConfigurationLoader(config_path)
        loader.load_and_validate()

        # Config file should have been updated with a version field
        migrated = json.loads(config_path.read_text())
        assert migrated["config_version"] == CURRENT_VERSION

    def test_migration_writes_the_document_not_a_model_dump(
        self, temp_config_dir, sample_unversioned_config
    ):
        """The user's key spelling, unknown keys and explicit nulls survive a
        migration, and defaults are not frozen into the file."""
        config_path = temp_config_dir / "options.json"
        document = {
            **sample_unversioned_config,
            "//note": "kept as is",
            "history": {"save days": 7, "half life days": None},
        }
        config_path.write_text(json.dumps(document))

        ConfigurationLoader(config_path).load_and_validate()

        migrated = json.loads(config_path.read_text())
        assert migrated["config_version"] == CURRENT_VERSION
        assert migrated["//note"] == "kept as is"
        assert migrated["history"] == {"save days": 7, "half life days": None}
        # Aliases with spaces stay as the user wrote them.
        assert "meteoserver-key" in migrated and "meteoserver_key" not in migrated
        # Defaults that the user never set are not written.
        assert "fast control" not in migrated and "fast_control" not in migrated
        # Loading again is a no-op: same document, no second backup.
        before = config_path.read_text()
        ConfigurationLoader(config_path).load_and_validate()
        assert config_path.read_text() == before
    
    def test_backup_creation(self, temp_config_dir, sample_unversioned_config):
        """Test backup creation before migration."""
        config_path = temp_config_dir / "options.json"
        backup_path = temp_config_dir / "options_unversioned.json"
        
        config_path.write_text(json.dumps(sample_unversioned_config))
        
        loader = ConfigurationLoader(config_path)
        loader.load_and_validate()
        
        # Backup should exist with the pre-migration version name
        assert backup_path.exists()
        
        # Backup should contain original data
        backup_data = json.loads(backup_path.read_text())
        assert backup_data == sample_unversioned_config
    
    def test_save_preserves_unknown_keys(self, temp_config_dir):
        """Test that saving preserves unknown keys."""
        config_path = temp_config_dir / "options.json"
        
        original_data = {
            "meteoserver-key": "test_api_key",
            "unknown_field": "should_be_kept",
            "config_version": 0
        }
        
        config_path.write_text(json.dumps(original_data))
        
        loader = ConfigurationLoader(config_path)
        loader.load_and_validate()
        
        # Save with only known fields
        new_data = {
            "meteoserver-key": "updated_api_key",
            "config_version": 0
        }
        
        loader.save(new_data)
        
        # Read back and check
        saved_data = json.loads(config_path.read_text())
        
        assert saved_data["meteoserver-key"] == "updated_api_key"
        assert saved_data["unknown_field"] == "should_be_kept"
        assert saved_data["config_version"] == 0
