#!/usr/bin/env python3
"""
scripts/predict.py
-------------------
Entry point for the full prediction pipeline.

Loads ``configs/predict.yaml`` (or a path supplied via ``--config``),
instantiates :class:`~src.vessels_detect.manager.PredictManager`, and
runs the pipeline according to the configured mode.

Usage
~~~~~
.. code-block:: bash

    # Run with the default config (configs/predict.yaml):
    python scripts/predict.py

    # Override the config path:
    python scripts/predict.py --config configs/predict_eval.yaml

    # Set mode on the command line (overrides predict.yaml):
    python scripts/predict.py --mode evaluation

Exit codes
~~~~~~~~~~
``0`` - pipeline completed successfully.
``1`` - pipeline failed; details are in the log.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when the script is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vessels_detect.manager import PredictManager


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the vessel detection prediction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/predict.yaml"),
        help="Path to the predict.yaml configuration file.",
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "evaluation"],
        default=None,
        help=(
            "Override pipeline.mode from the config file.  "
            "Use 'inference' to skip evaluation, 'evaluation' to enable it."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("predict.py - config: %s", args.config.resolve())

    # Optionally patch the mode before the manager loads the config.
    # We do this by temporarily rewriting the YAML in memory via a thin
    # wrapper rather than touching the file on disk.
    config_path = args.config
    if args.mode is not None:
        config_path = _patched_config(args.config, mode=args.mode)

    manager = PredictManager(config_path=config_path)
    return manager.run()


def _patched_config(original: Path, *, mode: str) -> Path:
    """Return a temporary config path with ``pipeline.mode`` overridden.

    Writes a modified copy to a temp file in the same directory.  The
    original file is never modified.

    Args:
        original: Path to the original ``predict.yaml``.
        mode:     Mode string to inject.

    Returns:
        Path to the patched temporary file.
    """
    import tempfile
    import yaml

    with open(original, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg.setdefault("pipeline", {})["mode"] = mode

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        dir=original.parent,
        delete=False,
        encoding="utf-8",
    )
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return Path(tmp.name)


if __name__ == "__main__":
    sys.exit(main())
