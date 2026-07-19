"""Shared configuration and utility helpers."""

from .config import AttackConfig, DefenseConfig, parse_csv_floats, parse_csv_ints, parse_csv_strings

_COMMON_EXPORTS = {
    "as_serializable_config",
    "log",
    "resolve_device",
    "save_npz",
    "set_seed",
    "write_csv",
    "write_json",
}

__all__ = [
    "AttackConfig",
    "DefenseConfig",
    "as_serializable_config",
    "log",
    "parse_csv_floats",
    "parse_csv_ints",
    "parse_csv_strings",
    "resolve_device",
    "save_npz",
    "set_seed",
    "write_csv",
    "write_json",
]


def __getattr__(name: str):
    if name in _COMMON_EXPORTS:
        from . import common

        return getattr(common, name)
    raise AttributeError(name)
