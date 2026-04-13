# pip install git+https://github.com/JulioContrerasH/satharmony

"""Pléiades Neo → Pléiades (PHR 1A/1B) sensor emulation.

Transforms Pléiades Neo MS imagery to the radiometric and spatial
characteristics of the Pléiades (PHR 1A/1B) MS product.


--------------------------------------------------------------------
Pléiades (PHR 1A/1B) — source: Airbus User Guide (https://www.engesat.com.br/wp-content/uploads/PleiadesUserGuide-17062019.pdf)

* GSD: 0.5 m PAN, 2.0 m MS
* Spectral bands: B, G, R, NIR (4 bands)
* Quantization: 12-bit
* Swath: 20 km
* PSF: ~1.0–1.3 px FWHM
* MTF @ Nyquist: ~0.1 (PAN), ~0.15 (MS)
* SNR: ~200 (typical)
* Dynamic range: 0–4095 DN
--------------------------------------------------------------------
Pléiades Neo (3/4) — source: Airbus User Guide (https://wp-cdn.apollomapping.com/web_assets/user_uploads/2021/11/08103301/2021.10_PleiadesNeo_UserGuide-EarlyRelease_20211015.pdf)

* GSD: 0.3 m PAN, 1.2 m MS (4-band), 0.9 m MS (6-band: CA, B, G, Y, R, RE, NIR)
* Spectral bands: 6 MS (Deep Blue/Coastal Aerosol, B, G, Yellow, R, Red Edge, NIR) + PAN
* Quantization: 12-bit
* PSF sigma: tighter (~0.8–1.1 px FWHM) — better MTF than PHR
* MTF @ Nyquist: ~0.15 (PAN), ~0.20 (MS) — slightly better than PHR
* SNR: ~250 (higher sensitivity)
* Pixel size detector: 4 μm (vs 13 μm PHR → enables 0.3m from ~620 km)
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from satharmony import MSSEmulator, PipelineConfig


# ---------------------------------------------------------------------------
# GSD constants
# ---------------------------------------------------------------------------

_PHR_MS_GSD = 0.5   # metres — PHR MS bundle product target
_NEO_MS_GSD = 0.3   # metres — Native GSD of your input image (updated)
_GSD_RATIO  = _PHR_MS_GSD / _NEO_MS_GSD 


# ---------------------------------------------------------------------------
# Emulation config
# ---------------------------------------------------------------------------

def config_neo_to_phr() -> PipelineConfig:
    config = PipelineConfig()

    # -- Spectral -------------------------------------------------------
    config.spectral.enabled = True
    config.spectral.s2_bands = [0, 1, 2]  # Only emulate the first 3 bands
    config.spectral.srf_adjustment = True
    config.spectral.srf_noise_std.min = 0.005
    config.spectral.srf_noise_std.max = 0.04
    config.spectral.band_scale_factors = [1.02, 1.0, 0.98] # Apply to the 3 bands

    # -- Spatial --------------------------------------------------------
    config.spatial.enabled = True
    config.spatial.input_gsd = _NEO_MS_GSD
    config.spatial.target_gsd.min = _PHR_MS_GSD
    config.spatial.target_gsd.max = _PHR_MS_GSD
    config.spatial.psf_sigma.min = 0.25
    config.spatial.psf_sigma.max = 0.45

    # -- Radiometric ----------------------------------------------------
    config.radiometric.enabled = True
    config.radiometric.quantization_bits = 12   
    config.radiometric.sqrt_compression = False  
    config.radiometric.sqrt_bands = []
    config.radiometric.saturation_threshold.min = 0.97
    config.radiometric.saturation_threshold.max = 1.0
    config.radiometric.reflectance_boost.min = 0.98
    config.radiometric.reflectance_boost.max = 1.02
    config.radiometric.reflectance_boost_prob = 0.3

    # -- Noise ----------------------------------------------------------
    config.random_noise.enabled = True
    config.random_noise.probability = 1.0
    config.random_noise.snr_db.min = 42   
    config.random_noise.snr_db.max = 47   
    config.random_noise.noise_type = "poisson"
    config.random_noise.poisson_weight.min = 0.4
    config.random_noise.poisson_weight.max = 0.7

    # -- Artifacts ------------------------------------------------------
    config.striping.enabled = False
    config.memory_effect.enabled = False
    config.coherent_noise.enabled = False
    config.scan_artifacts.enabled = False

    return config


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------

def transform_neo_to_phr(in_path: str | Path, out_path: str | Path) -> None:
    in_path, out_path = Path(in_path), Path(out_path)

    with rasterio.open(in_path) as src:
        neo_crs = src.crs
        # Capture the exact geographical bounding box of the original image
        neo_bounds = src.bounds 
        
        # Dead-simple band management: strictly read the first 3 bands.
        neo_data = src.read()[:3].astype(np.float32)

    # Per-band normalization to [0, 1]
    band_min = neo_data.min(axis=(1, 2), keepdims=True)
    band_max = neo_data.max(axis=(1, 2), keepdims=True)
    neo_data = (neo_data - band_min) / (band_max - band_min + 1e-6)

    # Run through emulator
    config = config_neo_to_phr()
    emulator = MSSEmulator(config)
    phr_data = emulator(neo_data)

    # Scale to 12-bit DN (PHR standard range 0–4095)
    phr_dn = (phr_data * 4095).clip(0, 4095).astype(np.uint16)

    # --- THE FIX ---
    # Generate a new transform by forcing the newly sized array 
    # to stretch exactly over the original geographic bounding box.
    phr_transform = from_bounds(
        *neo_bounds, 
        width=phr_dn.shape[2], 
        height=phr_dn.shape[1]
    )

    print(
        f"Neo input  : {neo_data.dtype}, shape {neo_data.shape}, GSD {_NEO_MS_GSD} m\n"
        f"PHR output : {phr_dn.dtype}, shape {phr_dn.shape}, GSD {_PHR_MS_GSD} m"
    )

    # Save emulated image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=phr_dn.shape[1],
        width=phr_dn.shape[2],
        count=phr_dn.shape[0],
        dtype=np.uint16,
        crs=neo_crs,
        transform=phr_transform,
        compress="deflate",
        predictor=2,        
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(phr_dn)
        dst.update_tags(
            EMULATION="PleiadesNeo_to_PHR",
            INPUT_GSD_M=str(_NEO_MS_GSD),
            OUTPUT_GSD_M=str(_PHR_MS_GSD),
            BANDS="1, 2, 3",
        )


def main() -> None:
    transform_neo_to_phr(
        in_path="/home/thomas/Documents/code/pleiades-boat-detection/data/raw/IMG_PNEO3_202310130650460_MS-FS_ORT_PWOI_000279689_1_1_F_1_R1C2.tif",
        out_path="/home/thomas/Documents/code/pleiades-boat-detection/data/generated/emulated_phr.tif",
    )


if __name__ == "__main__":
    main()