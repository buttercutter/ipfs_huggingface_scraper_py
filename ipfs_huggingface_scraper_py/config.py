import os
import logging
import tomli
import tomli_w
from typing import Dict, Any, Optional, List, Union

# Default configuration values
DEFAULT_CONFIG = {
    "scraper": {
        "output_dir": "hf_model_data",
        "max_models": None,  # None means no limit
        "max_datasets": None,  # None means no limit
        "max_spaces": None,  # None means no limit
        "entity_types": ["models"],  # Can include "models", "datasets", "spaces"
        "track_provenance": True,  # Track provenance information between entities
        "save_metadata": True,
        "filename_to_download": "config.json",
        "dataset_preview_max_rows": 100,  # Max rows for dataset preview
        "batch_size": 100,
        "skip_existing": True,
        "retry_delay_seconds": 5,
        "max_retries": 2,
        "log_level": "INFO",
        "metadata_fields": [
            "id", "modelId", "sha", "lastModified", "tags", 
            "pipeline_tag", "siblings", "config"
        ],
    },
    "api": {
        "base_url": "https://huggingface.co",
        "api_token": None,  # Should be set via environment or config file
        "authenticated": False,
        "anonymous_rate_limit": 5.0,  # requests per second
        "authenticated_rate_limit": 10.0,  # requests per second
        "daily_anonymous_quota": 300000,  # requests per day
        "daily_authenticated_quota": 1000000,  # requests per day
        "max_retries": 5,
        "timeout": 30,  # seconds
    },
    "storage": {
        "use_ipfs": True,
        "ipfs_add_options": {
            "pin": True,
            "wrap_with_directory": True,
            "chunker": "size-262144",
            "hash": "sha2-256"
        },
        "local_cache_max_size_gb": 10,
        "local_cache_retention_days": 30,
        "metadata_format": "parquet",  # Options: json, parquet
        "enable_knowledge_graph": True,  # Store relationships in knowledge graph
    },
    "state": {
        "state_dir": ".scraper_state",
        "checkpoint_interval": 50,  # Save checkpoint every N entities
        "auto_resume": True,
    },
    "provenance": {
        "extract_base_models": True,  # Extract base model information
        "extract_dataset_relationships": True,  # Extract dataset relationships
        "extract_evaluation_datasets": True,  # Extract evaluation dataset information
        "track_version_history": True,  # Track entity version history
        "max_relationship_depth": 3,  # Maximum depth for relationship traversal
    }
}

class Config:
    """
    Configuration manager for the HuggingFace scraper.
    
    Handles loading and saving configuration from TOML files,
    with fallback to environment variables and default values.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from a TOML file.
        
        Args:
            config_path: Path to the TOML configuration file. If None,
                         looks for 'config.toml' in the current directory.
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config_path = config_path or "config.toml"
        
        # Load configuration
        self._load_config()
        
        # Apply environment variables
        self._apply_env_vars()
        
        # Setup logging
        self._configure_logging()
    
    def _load_config(self) -> None:
        """Load configuration from TOML file if it exists."""
        try:
            if os.path.exists(self.config_path):
                logging.info(f"Loading configuration from {self.config_path}")
                with open(self.config_path, "rb") as f:
                    loaded_config = tomli.load(f)
                
                # Deep update config
                self._deep_update(self.config, loaded_config)
                logging.info("Configuration loaded successfully")
            else:
                logging.warning(f"Configuration file {self.config_path} not found. Using defaults.")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}. Using defaults.")
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _apply_env_vars(self) -> None:
        """Apply environment variables to override configuration."""
        # API token from environment
        api_token = os.environ.get("HF_API_TOKEN")
        if api_token:
            self.config["api"]["api_token"] = api_token
            self.config["api"]["authenticated"] = True
            logging.info("Using API token from environment")
        
        # Output directory from environment
        output_dir = os.environ.get("HF_OUTPUT_DIR")
        if output_dir:
            self.config["scraper"]["output_dir"] = output_dir
            logging.info(f"Using output directory from environment: {output_dir}")
        
        # Log level from environment
        log_level = os.environ.get("HF_LOG_LEVEL")
        if log_level:
            self.config["scraper"]["log_level"] = log_level
            logging.info(f"Using log level from environment: {log_level}")
        
        # Max models from environment
        max_models = os.environ.get("HF_MAX_MODELS")
        if max_models:
            try:
                max_models_int = int(max_models)
                self.config["scraper"]["max_models"] = max_models_int
                logging.info(f"Using max models from environment: {max_models_int}")
            except ValueError:
                logging.warning(f"Invalid value for HF_MAX_MODELS: {max_models}. Using default.")
    
    def _configure_logging(self) -> None:
        """Configure logging based on settings."""
        log_level_str = self.config["scraper"]["log_level"].upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        output_dir = self.config["scraper"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(output_dir, "scraper.log")),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"Logging configured with level {log_level_str}")
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to a TOML file.
        
        Args:
            path: Path to save to. If None, uses the path from initialization.
        """
        save_path = path or self.config_path
        try:
            # Convert sets to lists for serialization
            serializable_config = self._make_serializable(self.config)
            
            with open(save_path, "wb") as f:
                tomli_w.dump(serializable_config, f)
            logging.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Make an object serializable for TOML.
        
        Args:
            obj: Object to make serializable
            
        Returns:
            Serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def export_config_template(self, path: str) -> None:
        """
        Export a configuration template with default values and comments.
        
        Args:
            path: Path to save the template
        """
        try:
            with open(path, "wb") as f:
                tomli_w.dump(DEFAULT_CONFIG, f)
            logging.info(f"Configuration template exported to {path}")
        except Exception as e:
            logging.error(f"Error exporting configuration template: {e}")

# Global config instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to config file to use
        
    Returns:
        Global Config instance
    """
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = Config(config_path)
    return _config_instance