"""Test the configuration system."""

import os
import tempfile
import pytest
from unittest.mock import patch
from ipfs_huggingface_scraper_py.config import Config, get_config, DEFAULT_CONFIG

@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    # Create a temporary file
    fd, config_path = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    
    with open(config_path, 'w') as f:
        f.write("""
[scraper]
output_dir = "test_output"
max_models = 42
log_level = "DEBUG"

[api]
base_url = "https://test.huggingface.co"
api_token = "test_token"
authenticated = true

[storage]
use_ipfs = false
        """)
    
    yield config_path
    
    # Cleanup
    if os.path.exists(config_path):
        os.remove(config_path)

@pytest.fixture
def config_with_file(temp_config_file):
    """Create a Config instance with a file."""
    return Config(temp_config_file)

def test_default_config():
    """Test default configuration values."""
    # Create config without a file
    config = Config()
    
    # Check default values
    assert config.config["scraper"]["output_dir"] == "hf_model_data"
    assert config.config["scraper"]["max_models"] is None
    assert config.config["scraper"]["save_metadata"] is True
    assert config.config["api"]["authenticated"] is False
    assert config.config["storage"]["use_ipfs"] is True

def test_load_config_from_file(temp_config_file):
    """Test loading configuration from a file."""
    # Create config with a file
    config = Config(temp_config_file)
    
    # Check values from file
    assert config.config["scraper"]["output_dir"] == "test_output"
    assert config.config["scraper"]["max_models"] == 42
    assert config.config["scraper"]["log_level"] == "DEBUG"
    assert config.config["api"]["base_url"] == "https://test.huggingface.co"
    assert config.config["api"]["api_token"] == "test_token"
    assert config.config["api"]["authenticated"] is True
    assert config.config["storage"]["use_ipfs"] is False
    
    # Check that values not in file get defaults
    assert config.config["scraper"]["save_metadata"] is True
    assert config.config["storage"]["metadata_format"] == "parquet"

def test_get_set_config_values(config_with_file):
    """Test getting and setting configuration values."""
    # Get values
    assert config_with_file.get("scraper", "output_dir") == "test_output"
    assert config_with_file.get("api", "authenticated") is True
    
    # Test getting non-existent value with default
    assert config_with_file.get("nonexistent", "key", "default") == "default"
    
    # Set values
    config_with_file.set("scraper", "batch_size", 100)
    config_with_file.set("storage", "use_ipfs", True)
    
    # Check values were set
    assert config_with_file.get("scraper", "batch_size") == 100
    assert config_with_file.get("storage", "use_ipfs") is True

@patch.dict(os.environ, {
    "HF_API_TOKEN": "env_token",
    "HF_OUTPUT_DIR": "env_output",
    "HF_MAX_MODELS": "99",
    "HF_LOG_LEVEL": "WARNING"
})
def test_environment_variables():
    """Test that environment variables override config."""
    # Create config
    config = Config()
    
    # Check values from environment
    assert config.config["api"]["api_token"] == "env_token"
    assert config.config["api"]["authenticated"] is True
    assert config.config["scraper"]["output_dir"] == "env_output"
    assert config.config["scraper"]["max_models"] == 99
    assert config.config["scraper"]["log_level"] == "WARNING"

def test_save_config(temp_config_file):
    """Test saving configuration to a file."""
    # Create config with a file
    config = Config(temp_config_file)
    
    # Modify some values
    config.set("scraper", "max_models", 100)
    config.set("api", "timeout", 60)
    config.set("storage", "metadata_format", "json")
    
    # Save config to a new file
    new_config_path = temp_config_file + ".new"
    config.save(new_config_path)
    
    # Create a new config from the saved file
    new_config = Config(new_config_path)
    
    try:
        # Check that values were saved correctly
        assert new_config.get("scraper", "max_models") == 100
        assert new_config.get("api", "timeout") == 60
        assert new_config.get("storage", "metadata_format") == "json"
        
        # Check original values are still there
        assert new_config.get("scraper", "output_dir") == "test_output"
        assert new_config.get("api", "base_url") == "https://test.huggingface.co"
    finally:
        # Cleanup
        if os.path.exists(new_config_path):
            os.remove(new_config_path)

def test_export_config_template():
    """Test exporting a configuration template."""
    config = Config()
    
    # Create a temporary file for the template
    fd, template_path = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    
    try:
        # Export template
        config.export_config_template(template_path)
        
        # Check that file exists
        assert os.path.exists(template_path)
        
        # Load the template
        template_config = Config(template_path)
        
        # Check that it contains default values
        assert template_config.config == DEFAULT_CONFIG
    finally:
        # Cleanup
        if os.path.exists(template_path):
            os.remove(template_path)

def test_get_config_singleton():
    """Test that get_config returns a singleton."""
    # Get config twice
    config1 = get_config()
    config2 = get_config()
    
    # Should be the same instance
    assert config1 is config2
    
    # With a config path should create a new instance
    temp_fd, temp_path = tempfile.mkstemp(suffix=".toml")
    os.close(temp_fd)
    
    try:
        config3 = get_config(temp_path)
        # Should be a different instance
        assert config1 is not config3
        
        # But subsequent calls with the same path should return the same instance
        config4 = get_config(temp_path)
        assert config3 is config4
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_deep_update():
    """Test deep dictionary update."""
    config = Config()
    
    # Initial value
    assert config.config["scraper"]["max_models"] is None
    
    # Create a source dict with nested structure
    source = {
        "scraper": {
            "max_models": 42,
            "new_key": "value"
        },
        "new_section": {
            "key": "value"
        }
    }
    
    # Perform deep update
    config._deep_update(config.config, source)
    
    # Check updated values
    assert config.config["scraper"]["max_models"] == 42
    assert config.config["scraper"]["new_key"] == "value"
    assert config.config["new_section"]["key"] == "value"
    
    # Check that other values remain unchanged
    assert config.config["scraper"]["save_metadata"] is True
    assert config.config["api"]["base_url"] == "https://huggingface.co"

def test_make_serializable():
    """Test making objects serializable."""
    config = Config()
    
    # Create a test object with non-serializable types
    test_obj = {
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "set": {1, 2, 3},
        "int": 42,
        "str": "string",
        "nested": {
            "set": {4, 5, 6},
            "list": [4, 5, 6]
        }
    }
    
    # Make serializable
    serialized = config._make_serializable(test_obj)
    
    # Check types
    assert isinstance(serialized["list"], list)
    assert isinstance(serialized["dict"], dict)
    assert isinstance(serialized["set"], list)  # Set converted to list
    assert isinstance(serialized["nested"]["set"], list)  # Nested set converted
    
    # Check values
    assert sorted(serialized["set"]) == [1, 2, 3]
    assert sorted(serialized["nested"]["set"]) == [4, 5, 6]
    assert serialized["int"] == 42
    assert serialized["str"] == "string"

def test_logging_configuration():
    """Test logging configuration."""
    with patch('logging.basicConfig') as mock_basic_config:
        # Create config
        config = Config()
        
        # Check that logging was configured
        mock_basic_config.assert_called_once()
        
        # Check log level
        args, kwargs = mock_basic_config.call_args
        assert kwargs["level"] == logging.INFO  # Default level
    
    # Test with DEBUG level
    with patch('logging.basicConfig') as mock_basic_config:
        config = Config()
        config.config["scraper"]["log_level"] = "DEBUG"
        config._configure_logging()
        
        args, kwargs = mock_basic_config.call_args
        assert kwargs["level"] == logging.DEBUG

# Import logging here instead of at the top to avoid interfering with the tests
import logging