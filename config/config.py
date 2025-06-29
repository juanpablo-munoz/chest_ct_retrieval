import yaml
from pathlib import Path

def load_config(config_path: str | Path):
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
