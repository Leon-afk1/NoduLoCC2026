"""Config loading utilities.

Supports:
- hierarchical config files through `base_config`
- runtime CLI overrides using dotted keys (`train.epochs=2`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def _parse_value(raw: str) -> Any:
    """Convert override string values to bool/int/float/None when possible."""
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set `cfg[a][b][c]` from a dotted key `a.b.c`."""
    parts = dotted_key.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def load_config(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load one YAML config, resolve base config, and apply CLI overrides.

    Args:
        config_path: Path to the task config file.
        overrides: Optional list like `["train.epochs=1", "model.backbone=tiny_cnn"]`.

    Returns:
        A merged dictionary ready to be consumed by the pipeline.
    """
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}

    base_cfg: dict[str, Any] = {}
    base_config = raw.get("base_config")
    if base_config:
        base_path = (path.parent / base_config).resolve()
        base_cfg = load_config(str(base_path), overrides=[])

    cfg = _merge_dicts(base_cfg, {k: v for k, v in raw.items() if k != "base_config"})

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, raw_value = item.split("=", 1)
        _set_nested(cfg, key, _parse_value(raw_value))

    cfg["_config_path"] = str(path)
    return cfg
