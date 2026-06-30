"""
scripts/preprocessing.py
--------------------------
Orchestrates the vessel-detection preprocessing pipeline.

Reads configs/preprocessing.yaml and runs every stage in order:

    1. image_enhancement   - radiometric stretch + gamma, then upsampling
    2. label_conversion    - GeoJSON OBB -> YOLO OBB
    3. dataset_split       - image-level train / val / test split
    4. slicing             - tiling + label projection
    5. background_filtering- cap background-tile ratio per split

This version has been optimized to use ProcessPoolExecutor for concurrent 
processing of heavy I/O and CPU tasks.

No CLI: edit configs/preprocessing.yaml and run this file directly.

    PYTHONPATH=. python scripts/preprocessing.py
"""

from __future__ import annotations

import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.vessels_detect.preprocessing.image_enhancement import enhance_image
from src.vessels_detect.preprocessing.label_conversion import convert_labels
from src.vessels_detect.preprocessing.dataset_split import split_dataset
from src.vessels_detect.preprocessing.slicing import slice_image
from src.vessels_detect.preprocessing.background_filtering import filter_background

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 & 2 - Image enhancement + label conversion
# ---------------------------------------------------------------------------

def image_enhancement(image_path: Path, out_dir: Path, cfg: Dict) -> Path:
    """Radiometric stretch + gamma correction, then upsampling."""
    return enhance_image(image_path, out_dir, cfg)


def label_conversion(enhanced_image_path: Path, raw_dir: Path, out_dir: Path, cfg: Dict) -> Path:
    """GeoJSON OBB -> YOLO OBB, normalised to the enhanced image's dimensions."""
    return convert_labels(enhanced_image_path, raw_dir, out_dir, cfg)

def process_single_image(image_path: Path, raw_dir: Path, enhanced_dir: Path, labels_dir: Path, cfg: Dict) -> Tuple[Path, Path]:
    """Helper function to run steps 1 and 2 sequentially for a single image, enabling parallelization."""
    enhanced_path = image_enhancement(image_path, enhanced_dir, cfg)
    label_path = label_conversion(enhanced_path, raw_dir, labels_dir, cfg)
    return enhanced_path, label_path


# ---------------------------------------------------------------------------
# Step 3 - Dataset split
# ---------------------------------------------------------------------------

def dataset_split(image_label_pairs: List[Tuple[Path, Path]], out_dir: Path, cfg: Dict) -> Dict[str, str]:
    """Image-level train / val / test split with zero spatial leakage."""
    return split_dataset(image_label_pairs, out_dir, cfg)


# ---------------------------------------------------------------------------
# Step 4 - Slicing
# ---------------------------------------------------------------------------

def slicing(image_path: Path, label_path: Path, out_dir: Path, split: str, cfg: Dict) -> Tuple[int, int]:
    """Tile one image and project its labels onto every tile produced."""
    return slice_image(image_path, label_path, out_dir, split, cfg)


# ---------------------------------------------------------------------------
# Step 5 - Background filtering
# ---------------------------------------------------------------------------

def background_filtering(tiled_dir: Path, cfg: Dict) -> int:
    """Cap the background-tile ratio across the whole tiled dataset."""
    return filter_background(tiled_dir, cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── UTILITIES CALL (config loading, path definitions, etc.) ────────────
    config_path = Path("configs/preprocessing.yaml")
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    raw_dir = paths["raw_dir"]
    enhanced_dir = paths["enhanced_dir"]
    labels_dir = paths["labels_dir"]
    dataset_dir = paths["dataset_dir"]
    tiled_dir = paths["tiled_dir"]

    raw_images = sorted(raw_dir.glob("*.tif"))
    if not raw_images:
        raise RuntimeError(f"No .tif files found in '{raw_dir}'.")

    image_label_pairs: List[Tuple[Path, Path]] = []

    # ── Step 1 & 2 - Image enhancement & label conversion (PARALLELIZED) ────
    logger.info("Starting Image Enhancement & Label Conversion in parallel...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all image processing tasks to the process pool
        futures = {
            executor.submit(process_single_image, image_path, raw_dir, enhanced_dir, labels_dir, cfg): image_path
            for image_path in raw_images
        }
        
        # Gather results as they complete
        for future in concurrent.futures.as_completed(futures):
            image_path = futures[future]
            try:
                result = future.result()
                image_label_pairs.append(result)
            except Exception as exc:
                logger.error(f"Image {image_path.name} generated an exception during enhancement/conversion: {exc}")

    # Sort the pairs to maintain deterministic ordering for the dataset split
    image_label_pairs.sort(key=lambda x: x[0].name)

    # ── Step 3 - Dataset split (SEQUENTIAL) ─────────────────────────────────
    logger.info("Starting Dataset Split...")
    assignment = dataset_split(image_label_pairs, dataset_dir, cfg)

    # ── Step 4 - Slicing (PARALLELIZED) ─────────────────────────────────────
    logger.info("Starting Slicing in parallel...")
    tiling_splits = set(cfg["tiling"].get("splits", ["train", "val"]))
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        slicing_futures = []
        for image_path, _ in image_label_pairs:
            split = assignment[image_path.stem]
            if split not in tiling_splits:
                continue

            split_image_path = dataset_dir / "images" / split / image_path.name
            split_label_path = dataset_dir / "labels" / split / f"{image_path.stem}.txt"

            slicing_futures.append(
                executor.submit(slicing, split_image_path, split_label_path, tiled_dir, split, cfg)
            )
        
        # Ensure all slicing tasks finish and catch any errors
        for future in concurrent.futures.as_completed(slicing_futures):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"An exception occurred during slicing: {exc}")

    # ── Step 5 - Background filtering (SEQUENTIAL) ──────────────────────────
    logger.info("Starting Background Filtering...")
    # Must operate on the whole tiled folder so the background ratio is
    # computed across all tiles of a split, not per image.
    total_moved = background_filtering(tiled_dir, cfg)

    logger.info("Pipeline complete. %d background tile(s) relocated.", total_moved)


if __name__ == "__main__":
    main()