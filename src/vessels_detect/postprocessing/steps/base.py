"""
src/vessels_detect/postprocessing/steps/base.py
------------------------------------------------
Abstract base class for all postprocessing filter steps.

Every concrete step must:
  - declare a unique ``NAME`` class attribute.
  - implement :meth:`run`, which receives the fully resolved config dict
    and returns the number of polygons removed across all processed files.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseStep(ABC):
    """Contract for a single postprocessing filter stage.

    Subclasses are discovered and instantiated by
    :class:`~src.vessels_detect.postprocessing.manager.PostprocessingManager`
    via :data:`~src.vessels_detect.postprocessing.manager.STEP_REGISTRY`.
    """

    #: Unique stage name — must match the ``name`` key in the YAML config.
    NAME: str = ""

    @abstractmethod
    def run(self, cfg: dict) -> int:
        """Execute this filter step over the full dataset.

        Args:
            cfg: Fully resolved configuration dictionary produced by
                 :func:`~src.vessels_detect.postprocessing.manager.load_config`.

        Returns:
            Total number of polygons removed across all processed files.
        """