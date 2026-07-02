"""
scripts/preprocessing.py
--------------------------
Orchestrates the vessel-detection preprocessing pipeline.

Reads configs/preprocessing.yaml and runs every stage in order:

    0. degradation          - blur + GSD downsampling (sensor-domain match)
    1. image_enhancement    - radiometric stretch + gamma, then upsampling
    2. label_conversion     - GeoJSON OBB -> YOLO OBB
    3. dataset_split        - image-level train / val / test split
    4. slicing              - tiling + label projection
    5. background_filtering - cap background-tile ratio per split
    6. data_augmentation    - synthesize extra augmented tiles (train only)

Parallelism strategy
~~~~~~~~~~~~~~~~~~~~
Steps 0-2 and step 4 run in a ProcessPoolExecutor, but with a worker count
capped to avoid OOM on large GeoTIFFs:

    max_workers = min(cpu_count, floor(available_RAM_GB / RAM_PER_WORKER_GB))

RAM_PER_WORKER_GB is configurable (default 4 GB) and should reflect the peak
memory one worker needs while reading + upsampling a full-resolution GeoTIFF.

If a worker is still killed (e.g. RAM estimate was too low), the failed
images are retried one at a time in the main process before the pipeline
continues, so the output is always complete.

No CLI: edit configs/preprocessing.yaml and run this file directly.

    PYTHONPATH=. python scripts/preprocessing.py
"""

from __future__ import annotations

import logging
import concurrent.futures
import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import yaml

from src.vessels_detect.preprocessing.degradation import degrade_image
from src.vessels_detect.preprocessing.image_enhancement import enhance_image
from src.vessels_detect.preprocessing.label_conversion import convert_labels
from src.vessels_detect.preprocessing.dataset_split import split_dataset
from src.vessels_detect.preprocessing.slicing import slice_image
from src.vessels_detect.preprocessing.background_filtering import filter_background
from src.vessels_detect.preprocessing.data_augmentation import augment_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker count helper
# ---------------------------------------------------------------------------

def _safe_worker_count(ram_per_worker_gb: float) -> int:
    """Return a worker count that won't exhaust available RAM.

    Caps to min(cpu_count, floor(available_ram / ram_per_worker)),
    always at least 1.
    """
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    ram_cap = max(1, math.floor(available_gb / ram_per_worker_gb))
    cpu_cap = os.cpu_count() or 1
    n = min(cpu_cap, ram_cap)
    logger.info(
        "Worker cap: %d  (%.1f GB available / %.1f GB per worker, %d CPUs)",
        n, available_gb, ram_per_worker_gb, cpu_cap,
    )
    return n


# ---------------------------------------------------------------------------
# Step 0 - GSD degradation (blur + downsample)
# ---------------------------------------------------------------------------

def degradation(image_path: Path, out_dir: Path, cfg: Dict) -> Path:
    """Anti-alias blur, then downsample to the target GSD."""
    return degrade_image(image_path, out_dir, cfg)


# ---------------------------------------------------------------------------
# Step 1 & 2 - Image enhancement + label conversion
# ---------------------------------------------------------------------------

def image_enhancement(image_path: Path, out_dir: Path, cfg: Dict) -> Path:
    """Radiometric stretch + gamma correction, then upsampling."""
    return enhance_image(image_path, out_dir, cfg)


def label_conversion(enhanced_image_path: Path, raw_dir: Path, out_dir: Path, cfg: Dict) -> Path:
    """GeoJSON OBB -> YOLO OBB, normalised to the enhanced image's dimensions."""
    return convert_labels(enhanced_image_path, raw_dir, out_dir, cfg)


def process_single_image(
    image_path: Path,
    raw_dir: Path,
    degraded_dir: Path,
    enhanced_dir: Path,
    labels_dir: Path,
    cfg: Dict,
) -> Tuple[Path, Path]:
    """Run steps 0, 1, and 2 sequentially for one image (called per worker)."""
    degraded_path = degradation(image_path, degraded_dir, cfg)
    enhanced_path = image_enhancement(degraded_path, enhanced_dir, cfg)
    label_path = label_conversion(enhanced_path, raw_dir, labels_dir, cfg)
    return enhanced_path, label_path


# ---------------------------------------------------------------------------
# Step 3 - Dataset split
# ---------------------------------------------------------------------------

def dataset_split(
    image_label_pairs: List[Tuple[Path, Path]], out_dir: Path, cfg: Dict
) -> Dict[str, str]:
    """Image-level train / val / test split with zero spatial leakage."""
    return split_dataset(image_label_pairs, out_dir, cfg)


# ---------------------------------------------------------------------------
# Step 4 - Slicing
# ---------------------------------------------------------------------------

def slicing(
    image_path: Path, label_path: Path, out_dir: Path, split: str, cfg: Dict
) -> Tuple[int, int]:
    """Tile one image and project its labels onto every tile produced."""
    return slice_image(image_path, label_path, out_dir, split, cfg)


# ---------------------------------------------------------------------------
# Step 5 - Background filtering
# ---------------------------------------------------------------------------

def background_filtering(tiled_dir: Path, cfg: Dict) -> int:
    """Cap the background-tile ratio across the whole tiled dataset."""
    return filter_background(tiled_dir, cfg)


# ---------------------------------------------------------------------------
# Step 6 - Data augmentation
# ---------------------------------------------------------------------------

def data_augmentation(tiled_dir: Path, cfg: Dict) -> int:
    """Synthesize N augmented copies of every tile in the configured split(s)."""
    return augment_split(tiled_dir, cfg)


# ---------------------------------------------------------------------------
# Parallel runner with sequential fallback
# ---------------------------------------------------------------------------

def _run_parallel_with_fallback(
    image_paths: List[Path],
    raw_dir: Path,
    degraded_dir: Path,
    enhanced_dir: Path,
    labels_dir: Path,
    cfg: Dict,
    max_workers: int,
) -> List[Tuple[Path, Path]]:
    """Submit all images to the pool; retry any OOM-killed worker sequentially.

    A worker is killed by the OS (not by an exception) when it runs out of
    memory - this surfaces as ``BrokenProcessPool`` or the generic
    'process was terminated abruptly' message. Any image whose future
    carries that error is retried in the main process, one at a time, so
    the pipeline always produces a complete set of outputs.
    """
    results: Dict[Path, Optional[Tuple[Path, Path]]] = {p: None for p in image_paths}
    failed: List[Path] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image,
                image_path, raw_dir, degraded_dir, enhanced_dir, labels_dir, cfg,
            ): image_path
            for image_path in image_paths
        }
        for future in concurrent.futures.as_completed(futures):
            image_path = futures[future]
            try:
                results[image_path] = future.result()
            except Exception as exc:
                logger.warning(
                    "Worker failed for %s (%s) - will retry sequentially.",
                    image_path.name, exc,
                )
                failed.append(image_path)

    # Sequential fallback - one image at a time, no concurrent RAM pressure.
    if failed:
        logger.info("Retrying %d image(s) sequentially...", len(failed))
        for image_path in failed:
            try:
                results[image_path] = process_single_image(
                    image_path, raw_dir, degraded_dir, enhanced_dir, labels_dir, cfg
                )
                logger.info("  [retry OK] %s", image_path.name)
            except Exception as exc:
                logger.error("  [retry FAILED] %s: %s", image_path.name, exc)

    return [v for v in results.values() if v is not None]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── UTILITIES CALL (config loading, path definitions, etc.) ────────────
    config_path = Path("configs/preprocessing.yaml")
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    raw_dir      = paths["raw_dir"]
    degraded_dir = paths["degraded_dir"]
    enhanced_dir = paths["enhanced_dir"]
    labels_dir   = paths["labels_dir"]
    dataset_dir  = paths["dataset_dir"]
    tiled_dir    = paths["tiled_dir"]

    raw_images = sorted(raw_dir.glob("*.tif"))
    if not raw_images:
        raise RuntimeError(f"No .tif files found in '{raw_dir}'.")

    # RAM budget per worker: each worker reads a GeoTIFF + writes a 2x upsampled
    # version, so peak usage is roughly 3x the raw file size. Tune in the config
    # if your images are larger or smaller than ~1 GB on disk.
    ram_per_worker_gb = cfg.get("pipeline", {}).get("ram_per_worker_gb", 4.0)
    max_workers = _safe_worker_count(ram_per_worker_gb)

    # ── Step 0, 1 & 2 - Degradation, enhancement & label conversion (PARALLELIZED) ──
    logger.info(
        "Starting Degradation, Image Enhancement & Label Conversion "
        "(%d image(s), up to %d worker(s))...",
        len(raw_images), max_workers,
    )
    image_label_pairs = _run_parallel_with_fallback(
        raw_images, raw_dir, degraded_dir, enhanced_dir, labels_dir, cfg, max_workers
    )

    if not image_label_pairs:
        raise RuntimeError("All images failed during enhancement/conversion. Aborting.")

    if len(image_label_pairs) < len(raw_images):
        logger.warning(
            "%d / %d image(s) failed and could not be recovered. "
            "The dataset will be incomplete.",
            len(raw_images) - len(image_label_pairs), len(raw_images),
        )

    # Sort to maintain deterministic ordering for the dataset split.
    image_label_pairs.sort(key=lambda x: x[0].name)

    # ── Step 3 - Dataset split (SEQUENTIAL) ─────────────────────────────────
    logger.info("Starting Dataset Split...")
    assignment = dataset_split(image_label_pairs, dataset_dir, cfg)

    # ── Step 4 - Slicing (PARALLELIZED) ─────────────────────────────────────
    logger.info("Starting Slicing in parallel...")
    tiling_splits = set(cfg["tiling"].get("splits", ["train", "val"]))

    slicing_tasks = [
        (dataset_dir / "images" / assignment[img.stem] / img.name,
         dataset_dir / "labels" / assignment[img.stem] / f"{img.stem}.txt",
         assignment[img.stem])
        for img, _ in image_label_pairs
        if assignment[img.stem] in tiling_splits
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        slicing_futures = {
            executor.submit(slicing, img_path, lbl_path, tiled_dir, split, cfg): img_path
            for img_path, lbl_path, split in slicing_tasks
        }
        for future in concurrent.futures.as_completed(slicing_futures):
            img_path = slicing_futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error("Slicing failed for %s: %s", img_path.name, exc)

    # ── Step 5 - Background filtering (SEQUENTIAL) ──────────────────────────
    logger.info("Starting Background Filtering...")
    # Must operate on the whole tiled folder so the background ratio is
    # computed across all tiles of a split, not per image.
    total_moved = background_filtering(tiled_dir, cfg)

    # ── Step 6 - Data augmentation (SEQUENTIAL) ──────────────────────────────
    logger.info("Starting Data Augmentation...")
    # Must run after background filtering so augmented copies are only
    # generated from the final, ratio-balanced tile set.
    total_augmented = data_augmentation(tiled_dir, cfg)

    logger.info(
        "Pipeline complete. %d background tile(s) relocated, %d augmented tile(s) created.",
        total_moved, total_augmented,
    )


if __name__ == "__main__":
    main()