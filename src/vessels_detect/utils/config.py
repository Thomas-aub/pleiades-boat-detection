"""
src/utils/config.py
--------------------
Centralised YAML configuration loader for all pipeline stages.

Provides :class:`Config`, a thin, dict-backed wrapper that supports both
dictionary-style (``cfg["key"]``) and attribute-style (``cfg.key``) access,
recursive nesting via :class:`Config` sub-objects, and safe merging of
overrides at runtime.

All public methods carry Google-style docstrings and full type annotations.

Typical usage::

    from src.utils.config import Config

    cfg = Config.from_yaml("configs/train.yaml")
    print(cfg.training.epochs)          # attribute access
    print(cfg["training"]["batch"])     # dict access
    cfg_copy = cfg.merge({"training": {"epochs": 50}})
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Recursive dot-access wrapper around a YAML/dictionary configuration.

    Nested dictionaries are automatically promoted to :class:`Config`
    sub-objects so that deep keys are reachable as chained attributes.

    Args:
        data: The raw configuration dictionary.  Keys must be strings.

    Raises:
        TypeError: If *data* is not a :class:`dict`.

    Examples:
        >>> cfg = Config({"model": {"weights": "best.pt"}, "epochs": 100})
        >>> cfg.model.weights
        'best.pt'
        >>> cfg["epochs"]
        100
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError(f"Config requires a dict, got {type(data).__name__}.")
        object.__setattr__(self, "_data", data)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load a YAML file and return a :class:`Config` instance.

        Args:
            path: Path to the ``.yaml`` or ``.yml`` file.

        Returns:
            A :class:`Config` wrapping the parsed YAML content.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
            TypeError: If the top-level YAML value is not a mapping.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise TypeError(
                f"Expected a YAML mapping at the top level of '{path}', "
                f"got {type(raw).__name__}."
            )

        logger.debug("Loaded config from '%s'.", path)
        return cls(raw)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Wrap a plain dictionary as a :class:`Config`.

        Args:
            data: Source dictionary.

        Returns:
            A :class:`Config` instance backed by a deep copy of *data*.
        """
        return cls(copy.deepcopy(data))

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent.

        Args:
            key: Configuration key.
            default: Value returned when *key* is not present.

        Returns:
            The stored value (promoted to :class:`Config` if a dict), or
            *default*.
        """
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return Config(value)
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain (deep-copied) dictionary representation.

        Returns:
            A ``dict`` with all nested :class:`Config` objects unwrapped.
        """
        return copy.deepcopy(self._data)

    def merge(self, overrides: Dict[str, Any]) -> "Config":
        """Return a new :class:`Config` with *overrides* deep-merged in.

        Nested dictionaries are merged recursively; scalar values are
        replaced.

        Args:
            overrides: A flat or nested dictionary of values to override.

        Returns:
            A new :class:`Config` instance; the original is unchanged.
        """
        merged = _deep_merge(copy.deepcopy(self._data), overrides)
        return Config(merged)

    # ------------------------------------------------------------------
    # Dunder protocol — attribute and item access
    # ------------------------------------------------------------------

    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if key not in data:
            raise AttributeError(
                f"Config has no key '{key}'. "
                f"Available keys: {sorted(data.keys())}"
            )
        value = data[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*.

    Args:
        base: Base dictionary (mutated in-place).
        overrides: Overriding values.

    Returns:
        The mutated *base* dictionary.
    """
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: Union[str, Path]) -> Config:
    """Convenience wrapper around :meth:`Config.from_yaml`.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A :class:`Config` instance.
    """
    return Config.from_yaml(path)
