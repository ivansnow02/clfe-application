from pathlib import Path
from typing import Any, Dict

import yaml

from .utils import dict_to_namespace


def load_config(path: str):
    """Load a YAML config file and convert to SimpleNamespace (nested)."""
    cfg_path = Path(path)
    with cfg_path.open() as f:
        cfg_dict: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)
    return dict_to_namespace(cfg_dict)
