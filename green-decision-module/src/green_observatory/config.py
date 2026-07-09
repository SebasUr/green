"""Lightweight YAML config loading.

Core numeric functions take explicit parameters; this layer only maps YAML into
those parameters, so the science stays fully testable without any config files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: Repository root (…/green-decision-module), two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict (empty dict for an empty file)."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_named(name: str) -> dict:
    """Load ``configs/<name>.yaml`` (the ``.yaml`` suffix is optional)."""
    p = Path(name)
    if not p.suffix:
        p = CONFIG_DIR / f"{name}.yaml"
    elif not p.is_absolute() and not p.exists():
        p = CONFIG_DIR / p
    return load_yaml(p)


def cfg_get(d: dict, dotted: str, default: Any = None) -> Any:
    """Fetch a nested value by dotted path, e.g. ``cfg_get(cfg, "model.random_state", 42)``."""
    cur: Any = d
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
