"""YAML config loading with attribute access and CLI overrides.

Design goals (per project requirements):
- Every experiment is one command away; configs are the single source of truth.
- No hardcoded constants inside algorithms — everything flows from a config.
- `--override a.b.c=value` lets sweeps run without editing files.

Example
-------
>>> cfg = load_config("scout_fl/configs/microbenchmark.yaml",
...                   overrides=["selection.budget=3", "seed=1"])
>>> cfg.selection.budget
3
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml

_NUMERIC_RE = re.compile(r"^[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$")


def _coerce(value: Any) -> Any:
    """Coerce numeric-looking strings to int/float.

    Works around PyYAML's quirk where an unsigned exponent (e.g. ``1.0e6``) is
    left as a string instead of parsed as a float, which would silently break
    arithmetic downstream.
    """
    if isinstance(value, str) and _NUMERIC_RE.match(value.strip()):
        text = value.strip()
        try:
            return float(text) if re.search(r"[.eE]", text) else int(text)
        except ValueError:
            return value
    return value


class Config(dict):
    """A dict with attribute access; nested dicts/lists are wrapped recursively."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, val in list(self.items()):
            super().__setitem__(key, self._wrap(val))

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, Config):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return _coerce(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def get(self, key: str, default: Any = None) -> Any:  # noqa: D102
        return self[key] if key in self else default


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML config file, apply ``key.path=value`` overrides, return a Config."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(raw)
    for override in overrides or []:
        if not override:
            continue
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Malformed override (expected key=value): {override!r}")
        _set_nested(cfg, key.strip(), _parse_scalar(value.strip()))
    cfg["_config_path"] = str(path)
    return cfg


def _set_nested(cfg: Config, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node: Any = cfg
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = Config._wrap(value)


def _parse_scalar(text: str) -> Any:
    """Parse an override value using YAML rules (so ints/floats/bools/null work)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def to_plain(cfg: Any) -> Any:
    """Recursively convert a Config back to plain dict/list (drops private keys)."""
    if isinstance(cfg, dict):
        return {k: to_plain(v) for k, v in cfg.items() if not k.startswith("_")}
    if isinstance(cfg, list):
        return [to_plain(v) for v in cfg]
    return cfg
