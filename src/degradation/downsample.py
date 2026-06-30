"""Pléiades Neo → Pléiades baseline: pure spatial decimation, no blur/noise.

Control condition for the physics/stochastic degradation pipelines. Skips any
modeling of the sensor chain (MTF, SNR, SRF) and only resamples the GSD, since
both Neo and Pléiades are delivered post-deconvolution/denoising — see
discussion on why naive resampling alone lands closer to the real Pléiades
domain than a from-scratch sensor degradation does.

GeoJSON labels are georeferenced (real-world CRS coordinates), so resampling
the raster changes pixel grid only — annotations are copied as-is.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from tqdm import tqdm

NEO_GSD = 0.3   # Pléiades Neo PMS, m/px
PHR_GSD = 0.5   # Pléiades PMS, m/px
GSD_RATIO = PHR_GSD / NEO_GSD  # ≈ 1.667, output has fewer pixels


def downsample_tif(in_path: Path, out_path: Path) -> None:
    with rasterio.open(in_path) as src:
        data = src.read()  # (bands, H, W)
        out_w = max(1, round(src.width / GSD_RATIO))
        out_h = max(1, round(src.height / GSD_RATIO))
        out_transform = from_bounds(*src.bounds, width=out_w, height=out_h)

        # INTER_AREA averages source pixels per output pixel — the correct
        # choice for downscaling (avoids aliasing that INTER_LINEAR/CUBIC
        # would introduce here).
        resampled = np.stack(
            [
                cv2.resize(band, (out_w, out_h), interpolation=cv2.INTER_AREA)
                for band in data
            ]
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=out_h,
            width=out_w,
            count=src.count,
            dtype=resampled.dtype,
            crs=src.crs,
            transform=out_transform,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(resampled)
            dst.update_tags(SOURCE_GSD_M=str(NEO_GSD), OUTPUT_GSD_M=str(PHR_GSD))


def process_folder(input_dir: Path, output_dir: Path) -> None:
    tif_files = sorted(input_dir.glob("*.tif"))
    if not tif_files:
        print(f"No .tif files found in {input_dir}")
        return

    for in_path in tqdm(tif_files, desc="Downsampling", unit="file"):
        out_path = output_dir / in_path.name
        downsample_tif(in_path, out_path)

        geojson_src = in_path.with_suffix(".geojson")
        if geojson_src.exists():
            shutil.copy2(geojson_src, out_path.with_suffix(".geojson"))
        else:
            tqdm.write(f"No GeoJSON for {in_path.name}, skipping.")


if __name__ == "__main__":
    process_folder(
        input_dir=Path("data/raw"),
        output_dir=Path("data/pleiades_synthetic/downsampled"),
    )