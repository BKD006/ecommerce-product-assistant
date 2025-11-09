from pathlib import Path
import os
import yaml


def _project_root() -> Path:
    """
    Determine the project root directory.

    This helper function infers the project root path based on the location
    of this script (`config_loader.py`), assuming the following structure:

        product_assistant/
        ├── config/
        │   └── config.yaml
        └── utils/
            └── config_loader.py

    Returns:
        Path: The absolute path to the project root directory.
    """
    # Example: if file path is /home/user/project/utils/config_loader.py,
    # then parents[1] = /home/user/project
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | None = None) -> dict:
    """
    Load a YAML configuration file from a reliable path.

    The function resolves the configuration path based on a clear priority order:
        1. Explicit argument `config_path` (if provided)
        2. Environment variable `CONFIG_PATH`
        3. Default path: <project_root>/config/config.yaml

    It supports both absolute and relative paths, and ensures that the file exists
    before attempting to load it.

    Args:
        config_path (str | None, optional): 
            The path to the configuration file. If None, the function falls back 
            to the environment variable or default project config path.

    Returns:
        dict: Parsed configuration as a Python dictionary. Returns an empty dict if
              the YAML file is empty.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.

    Example:
        >>> # Load using default location
        >>> config = load_config()
        >>> # Load using explicit path
        >>> config = load_config("custom_configs/app.yaml")
        >>> # Load using environment variable
        >>> os.environ["CONFIG_PATH"] = "/etc/app/config.yaml"
        >>> config = load_config()
    """
    # Step 1: Get the CONFIG_PATH environment variable (if set)
    env_path = os.getenv("CONFIG_PATH")

    # Step 2: Determine which path to use (explicit arg > env var > default)
    if config_path is None:
        config_path = env_path or str(_project_root() / "config" / "config.yaml")

    # Step 3: Convert string to Path object
    path = Path(config_path)

    # Step 4: Resolve relative paths against the project root
    if not path.is_absolute():
        path = _project_root() / path

    # Step 5: Validate that the config file exists
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Step 6: Load YAML config safely (prevents code execution)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}