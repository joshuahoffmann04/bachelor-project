"""Configuration loader with environment variable substitution."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Global config cache
_config_cache: Dict[str, Any] = {}


def _substitute_env_vars(config: Any) -> Any:
    """Recursively substitute environment variables in config.

    Supports ${VAR_NAME} syntax in strings.
    """
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Replace ${VAR_NAME} with environment variable
        if config.startswith('${') and config.endswith('}'):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        return config
    return config


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load and parse YAML configuration file.

    Args:
        config_path: Path to config file (relative to project root)

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    # Find project root (contains config directory)
    current_dir = Path.cwd()
    config_file = None

    # Try current directory first
    if (current_dir / config_path).exists():
        config_file = current_dir / config_path
    # Try parent directories
    else:
        for parent in current_dir.parents:
            if (parent / config_path).exists():
                config_file = parent / config_path
                break

    if config_file is None:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load and parse YAML
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    config = _substitute_env_vars(config)

    # Cache the config
    global _config_cache
    _config_cache = config

    return config


def get_config() -> Dict[str, Any]:
    """Get cached configuration.

    Returns:
        Cached configuration dictionary

    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    if not _config_cache:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return _config_cache


def get_nested(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'llm.openai.model')
        default: Default value if path doesn't exist

    Returns:
        Configuration value or default

    Example:
        >>> config = {'llm': {'openai': {'model': 'gpt-4'}}}
        >>> get_nested(config, 'llm.openai.model')
        'gpt-4'
    """
    keys = path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
