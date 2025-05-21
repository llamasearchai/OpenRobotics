"""
Configuration management for the OpenRobotics framework.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration paths
DEFAULT_CONFIG_DIR = Path.home() / ".openrobotics"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_MODELS_DIR = DEFAULT_CONFIG_DIR / "models"
DEFAULT_DATA_DIR = DEFAULT_CONFIG_DIR / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "robotics.db"

# Ensure directories exist
DEFAULT_CONFIG_DIR.mkdir(exist_ok=True)
DEFAULT_MODELS_DIR.mkdir(exist_ok=True)
DEFAULT_DATA_DIR.mkdir(exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "debug": False,
        "api_keys": [],
    },
    "llm": {
        "default_model": "gpt-4",
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "cache_dir": str(DEFAULT_DATA_DIR / "llm_cache"),
    },
    "mlx": {
        "models_dir": str(DEFAULT_MODELS_DIR),
        "default_dtype": "float16",
        "use_gpu": True,
    },
    "robotics": {
        "simulation_enabled": True,
        "safety_checks_enabled": True,
        "max_velocity": 1.0,
        "max_acceleration": 0.5,
    },
    "database": {
        "path": str(DEFAULT_DB_PATH),
        "datasette_enabled": True,
        "datasette_port": 8001,
    },
    "logging": {
        "level": "INFO",
        "file": str(DEFAULT_DATA_DIR / "openrobotics.log"),
        "console": True,
    },
}


class Config:
    """
    Configuration manager for OpenRobotics.
    
    Handles loading, saving, and accessing configuration settings.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def __init__(self):
        # Initialization happens in __new__
        pass
    
    def _load_config(self):
        """Load configuration from file or create default if not exists."""
        if DEFAULT_CONFIG_FILE.exists():
            with open(DEFAULT_CONFIG_FILE, 'r') as f:
                self._config = yaml.safe_load(f)
                
            # Update with any missing default values
            self._update_missing_defaults(self._config, DEFAULT_CONFIG)
        else:
            self._config = DEFAULT_CONFIG
            self.save()
    
    def _update_missing_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]):
        """Recursively update config with missing default values."""
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                self._update_missing_defaults(config[key], value)
    
    def save(self):
        """Save current configuration to file."""
        with open(DEFAULT_CONFIG_FILE, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def get(self, section: str, key: Optional[str] = None):
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key within section (if None, returns entire section)
            
        Returns:
            Configuration value or section dictionary
        """
        if section not in self._config:
            return None
        
        if key is None:
            return self._config[section]
        
        return self._config[section].get(key)
    
    def set(self, section: str, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key within section
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section][key] = value
        self.save()
    
    def update(self, section: str, values: Dict[str, Any]):
        """
        Update multiple values in a section.
        
        Args:
            section: Configuration section
            values: Dictionary of values to update
        """
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section].update(values)
        self.save()
    
    @property
    def all(self):
        """Get complete configuration dictionary."""
        return self._config


# Global configuration instance
config = Config()
