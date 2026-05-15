from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar
import tomllib


ConfigT = TypeVar("ConfigT")


def load_toml_section(path: str | Path, section: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    values = config.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"Config section [{section}] must be a table")
    return values


def dataclass_from_section(defaults: ConfigT, path: str | Path, section: str) -> ConfigT:
    if not is_dataclass(defaults):
        raise TypeError("defaults must be a dataclass instance")

    values = {field.name: getattr(defaults, field.name) for field in fields(defaults)}
    overrides = load_toml_section(path, section)
    unknown_keys = sorted(set(overrides) - set(values))
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unknown config key(s) in [{section}]: {joined}")

    values.update(overrides)
    return type(defaults)(**values)
