"""
src/vessels_detect/preprocessing/steps/base.py
-----------------------------------------------
Abstract base class that every pipeline step must implement.

The Registry/Manager pattern used in ``manager.py`` relies on this contract:
each step is a self-contained object that receives the resolved configuration
dictionary and exposes a single ``run()`` entry-point.

Implementing a new step
~~~~~~~~~~~~~~~~~~~~~~~
1.  Create a module under ``src/vessels_detect/preprocessing/steps/``.
2.  Subclass :class:`BaseStep` and implement :meth:`run`.
3.  Register the step in ``manager.py``'s ``STEP_REGISTRY``.

Example::

    from src.vessels_detect.preprocessing.steps.base import BaseStep

    class MyStep(BaseStep):
        NAME = "my_step"

        def run(self, cfg: dict) -> None:
            paths = cfg["paths"]
            params = cfg["my_step"]
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseStep(ABC):
    """Abstract base class for all preprocessing pipeline steps.

    Subclasses must define a class-level ``NAME`` string that matches the
    key used in the YAML ``stages`` list and ``STEP_REGISTRY`` mapping.
    """

    #: Registry key — must be unique across all steps.
    NAME: str = ""

    @abstractmethod
    def run(self, cfg: dict) -> None:
        """Execute the step.

        Args:
            cfg: Fully resolved configuration dictionary produced by
                :func:`manager.load_config`.  ``cfg["paths"]`` contains
                :class:`~pathlib.Path` objects; all other sections are plain
                Python scalars / dicts / lists as parsed from YAML.

        Raises:
            Any exception is allowed to propagate — the manager will catch
            it, log it, and abort the pipeline.
        """
