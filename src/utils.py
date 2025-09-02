import yaml
from typing import Any


def load_config(path: str) -> dict[str, Any] | list[Any]:
    with open(path) as f:
        config = yaml.safe_load(f)

    return config
