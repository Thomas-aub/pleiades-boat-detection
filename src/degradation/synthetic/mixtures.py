"""
mixtures.py — Tile collection and degradation-mixture assignment.

Produces a CSV mapping every valid GeoTIFF tile to N randomised degradation
mixtures (PSF σ, SNR dB, step order) for sensor-emulation training.

Pipeline (Orchestrator-Worker):
    generate_pool → collect_tiles → filter_nodata
        → [reduce_background] → assign_mixtures → export_csv
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

from pyproj import Transformer

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)

_IMAGE_GLOBS        = ("*.tif", "*.tiff")


# ============================================================================
# Tools — pure functions
# ============================================================================

def _load_annotations(labels_dir: Path, image_stem: str) -> List[List[tuple]]:
    """Load polygon coordinate rings (WGS84) from the GeoJSON for this image."""
    p = labels_dir / f"{image_stem}.geojson"
    if not p.exists():
        return []
    return [f["geometry"]["coordinates"][0]
            for f in json.loads(p.read_text(encoding="utf-8"))["features"]]


def _tile_has_annotation(annots: List[List[tuple]], tile: dict, tf: Transformer) -> bool:
    """Return True if any annotation centroid (reprojected to native CRS) falls inside the tile bbox."""
    for ring in annots:
        cx = sum(c[0] for c in ring) / len(ring)
        cy = sum(c[1] for c in ring) / len(ring)
        x, y = tf.transform(cx, cy)
        if tile["min_x"] <= x <= tile["max_x"] and tile["min_y"] <= y <= tile["max_y"]:
            return True
    return False


def _bg_keep(n_pos: int, ratio: float) -> int:
    """Return how many background tiles to retain given positives count and target ratio."""
    return int(n_pos * ratio / (1.0 - ratio))


def _tile_windows(tif_path: Path, tile_size: int, gsd_out: float) -> List[dict]:
    """Enumerate all non-overlapping tile-metadata dicts for a single GeoTIFF."""
    tiles = []
    with rasterio.open(tif_path) as ds:
        if ds.crs is None:
            return tiles
        t = ds.transform
        gsd_x, gsd_y = abs(t.a), abs(t.e)
        for row in range(0, ds.height, tile_size):
            for col in range(0, ds.width, tile_size):
                c1, r1        = min(col + tile_size, ds.width), min(row + tile_size, ds.height)
                left, top     = t * (col, row)
                right, bottom = t * (c1, r1)
                tiles.append({
                    "_tif_path":       tif_path,
                    "_window":         Window(col, row, c1 - col, r1 - row),
                    "image_id":        tif_path.name,
                    "tile_id":         f"{row}_{col}",
                    "min_x":           min(left, right),
                    "max_x":           max(left, right),
                    "min_y":           min(top, bottom),
                    "max_y":           max(top, bottom),
                    "tile_size_px":    tile_size,
                    "length_x_meters": tile_size * gsd_x,
                    "length_y_meters": tile_size * gsd_y,
                    "native_crs":      ds.crs.to_string(),
                    "GSD_input":       round((gsd_x + gsd_y) / 2.0, 4),
                    "GSD_output":      gsd_out,
                })
    return tiles


def _sample_mixture(psf: list, snr: list, i: int) -> dict:
    """Sample one randomised degradation configuration (PSF σ, SNR dB, step order)."""
    steps = ["B", "D", "N"]
    random.shuffle(steps)
    return {
        "Transform_id": i,
        "step_order":   steps,
        "snr_db":       round(random.uniform(*snr), 2),
        "psf_size":     round(random.uniform(*psf), 2),
    }


def _build_record(tile: dict, mix: dict) -> dict:
    """Merge tile metadata and one mixture dict into a single flat output row."""
    return {
        **{k: v for k, v in tile.items() if not k.startswith("_")},
        "Transform_id": mix["Transform_id"],
        "Order":        "".join(mix["step_order"]),
        "SNR (dB)":     mix["snr_db"],
        "PSF":          mix["psf_size"],
    }


# ============================================================================
# Workers
# ============================================================================

def generate_pool(cfg: dict) -> List[dict]:
    """Generate a global pool of N randomised degradation mixtures."""
    random.seed(cfg["random_seed"])
    pool = [_sample_mixture(cfg["psf_limits"], cfg["snr_limits"], i)
            for i in range(cfg["n_mixtures"])]
    logger.info("Generated %d mixtures.", cfg["n_mixtures"])
    return pool


def collect_tiles(cfg: dict) -> List[dict]:
    """Scan input directory and collect tile-metadata dicts from all GeoTIFFs."""
    input_dir = Path(cfg["input_directory"])
    tif_files = sorted(p for g in _IMAGE_GLOBS for p in input_dir.glob(g))
    tiles = []
    for tif in tif_files:
        logger.info("Collecting tiles from %s...", tif.name)
        tiles.extend(_tile_windows(tif, cfg["tile_size"], cfg["GSD_output"]))
    logger.info("Collected %d tiles from %d file(s).", len(tiles), len(tif_files))
    return tiles


def filter_nodata(tiles: List[dict], cfg: dict) -> List[dict]:
    """Drop tiles exceeding the nodata threshold or with uniform (zero-information) content."""
    if cfg["max_nodata_pct"] >= 1.0:
        return tiles

    by_file: Dict[Path, List[dict]] = {}
    for t in tiles:
        by_file.setdefault(t["_tif_path"], []).append(t)

    kept = []
    for tif_path, group in by_file.items():
        with rasterio.open(tif_path) as ds:
            nd = ds.nodata or 0
            for tile in group:
                raw = ds.read(window=tile["_window"])
                # Use first 3 bands (RGB) for nodata check if available
                check_raw = raw[:3] if raw.shape[0] >= 3 else raw
                nd_pct = np.all(check_raw == nd, axis=0).sum() / (raw.shape[1] * raw.shape[2])
                
                if nd_pct > cfg["max_nodata_pct"]:
                    continue
                if (check_raw.max() == check_raw.min()):
                    continue
                kept.append(tile)

    logger.info("Nodata filter: kept %d/%d tiles.", len(kept), len(tiles))
    return kept


def reduce_background(tiles: List[dict], cfg: dict) -> List[dict]:
    """Cap background-tile fraction at cfg['target_bg_ratio'] using per-image GeoJSON label files."""
    if not cfg.get("labels_dir"):
        return tiles

    labels_dir = Path(cfg["labels_dir"])

    # Group by image to build one Transformer and load annotations once per TIF.
    by_image: Dict[str, List[dict]] = {}
    for tile in tiles:
        by_image.setdefault(tile["image_id"], []).append(tile)

    pos, bg = [], []
    for image_id, img_tiles in by_image.items():
        annots = _load_annotations(labels_dir, Path(image_id).stem)
        tf     = Transformer.from_crs("EPSG:4326", img_tiles[0]["native_crs"], always_xy=True)
        for tile in img_tiles:
            (pos if _tile_has_annotation(annots, tile, tf) else bg).append(tile)

    if not pos:
        logger.warning("No positive tiles found — skipping background reduction.")
        return tiles

    keep = _bg_keep(len(pos), cfg["target_bg_ratio"])
    random.seed(cfg["random_seed"])
    random.shuffle(bg)

    logger.info(
        "Background reduction: %d pos + %d/%d bg kept (target %.0f%%).",
        len(pos), keep, len(bg), cfg["target_bg_ratio"] * 100,
    )
    return pos + bg[:keep]


def assign_mixtures(tiles: List[dict], pool: List[dict], cfg: dict) -> List[dict]:
    """Assign n_mixtures_per_tile randomly sampled mixtures to each tile."""
    records = [
        _build_record(tile, mix)
        for tile in tiles
        for mix in random.sample(pool, k=cfg["n_mixtures_per_tile"])
    ]
    logger.info("Assigned %d records (%d tiles x %d mixtures).",
                len(records), len(tiles), cfg["n_mixtures_per_tile"])
    return records


def export_csv(records: List[dict], cfg: dict) -> pd.DataFrame:
    """Write tile-mixture records to CSV and return the DataFrame."""
    df = pd.DataFrame(records)
    df.to_csv(cfg["output_csv"], index=False)
    logger.info("Exported %d rows -> %s", len(df), cfg["output_csv"])
    return df


# ============================================================================
# Orchestrator
# ============================================================================

def run(cfg: dict) -> List[dict]:
    """Execute the full pipeline: pool -> tiles -> filter -> reduce -> assign -> export."""
    pool    = generate_pool(cfg)
    tiles   = collect_tiles(cfg)
    tiles   = filter_nodata(tiles, cfg)
    tiles   = reduce_background(tiles, cfg)
    records = assign_mixtures(tiles, pool, cfg)
    export_csv(records, cfg)
    return records


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = {
        "input_directory":     "/home/thomas/Documents/code/pleiades-boat-detection/data/raw",
        "output_csv":          "tile_mixture_assignments.csv",
        "GSD_output":          0.5,
        "tile_size":           1024,
        "psf_limits":          [0.48, 0.51],
        "snr_limits":          [42, 44],
        "n_mixtures":          50,
        "n_mixtures_per_tile": 3,
        "max_nodata_pct":      0.50,
        # Set labels_dir to activate background reduction:
        "labels_dir":          None,
        "target_bg_ratio":     0.20,
        "random_seed":         42,
    }

    run(cfg)