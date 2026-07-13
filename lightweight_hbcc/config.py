from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving optional ``_base_`` inheritance.

    Base paths are resolved relative to the YAML file that declares them. A
    string loads one base, while a list loads and merges bases from left to
    right. The child config is applied last. This lets controlled experiments
    share one data/training recipe instead of duplicating it across models.
    """

    return _load_config(Path(path).resolve(), stack=())


def _load_config(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Circular config inheritance detected: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        raw: dict[str, Any] = {}
    elif isinstance(loaded, dict):
        raw = loaded
    else:
        raise ValueError(f"Config root must be a mapping: {path}")

    base_value = raw.pop("_base_", None)
    if base_value is None:
        return raw
    base_items = base_value if isinstance(base_value, list) else [base_value]
    if not all(isinstance(item, str) for item in base_items):
        raise ValueError(f"_base_ must be a path or list of paths: {path}")

    merged: dict[str, Any] = {}
    next_stack = (*stack, path)
    for item in base_items:
        base_path = Path(item)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = deep_update(merged, _load_config(base_path.resolve(), next_stack))
    return deep_update(merged, raw)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_by_path(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_by_path(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_override(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        set_by_path(out, key, parse_override(raw))
    return out
