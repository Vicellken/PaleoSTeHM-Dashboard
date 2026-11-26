"""Live spatial mask preview for the Results step."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stgp_dashboard.paths import ensure_import_paths

ensure_import_paths()

from stgp_pipeline import data, export, visualize  # noqa: E402
from stgp_pipeline.visualize import SpatialPlotMaskSettings  # noqa: E402


def mask_settings_from_ui(
    *,
    enabled: bool,
    calibration_factor: float,
) -> SpatialPlotMaskSettings:
    return SpatialPlotMaskSettings(
        enabled=bool(enabled),
        calibration_factor=float(calibration_factor),
    )


def _load_paleo_rsl(cfg: Mapping[str, Any]) -> Optional[Any]:
    paths = cfg.get("paths") or {}
    paleo = paths.get("paleo_rsl")
    if not paleo:
        return None
    max_age_bp = int((cfg.get("analysis") or {}).get("max_age_bp", 16000))
    try:
        return data.load_and_filter_rsl(str(paleo), max_age_bp=max_age_bp, config=dict(cfg))
    except Exception:  # noqa: BLE001
        return None


def _spatial_slice_at_age(
    scenario_dir: Path,
    age_bp: float,
) -> Optional[Any]:
    spatial_csv = scenario_dir / "spatial_predictions.csv"
    if not spatial_csv.is_file():
        return None
    df = pd.read_csv(spatial_csv)
    comp_labels = export._infer_spatial_component_labels(df)
    slices = export.spatial_slices_from_dataframe(
        df, [float(age_bp)], comp_labels=comp_labels or None
    )
    return slices[0] if slices else None


def compute_mask_stats(
    spatial_slice: Any,
    rsl_data: Optional[Any],
    mask_settings: SpatialPlotMaskSettings,
) -> Dict[str, float]:
    """Return threshold, median slip SD, and masked fraction for total mean panel."""
    std_2d = spatial_slice.total_std
    if not mask_settings.enabled:
        return {
            "threshold_m": float("nan"),
            "median_slip_sd_m": float("nan"),
            "masked_fraction_pct": 0.0,
        }

    if rsl_data is not None:
        slip_sds = visualize._slip_grid_sds(
            rsl_data.rsl_lat,
            rsl_data.rsl_lon,
            spatial_slice.lat_mat,
            spatial_slice.lon_mat,
            std_2d,
        )
        median_slip_sd = float(np.median(slip_sds))
        threshold = visualize._calibrate_threshold(
            slip_sds, factor=mask_settings.calibration_factor
        )
    else:
        median_slip_sd = float("nan")
        threshold = mask_settings.rsl_std_fallback_m

    mask = np.asarray(std_2d) > threshold
    masked_fraction = float(np.mean(mask) * 100.0)
    return {
        "threshold_m": float(threshold),
        "median_slip_sd_m": median_slip_sd,
        "masked_fraction_pct": masked_fraction,
    }


def render_mask_preview_figure(
    spatial_slice: Any,
    rsl_data: Optional[Any],
    mask_settings: SpatialPlotMaskSettings,
) -> plt.Figure:
    """Three-panel preview: SD, mask map, masked total mean."""
    import cartopy.crs as ccrs

    stats = compute_mask_stats(spatial_slice, rsl_data, mask_settings)
    threshold = stats["threshold_m"]

    if mask_settings.enabled and rsl_data is not None:
        slip_sds = visualize._slip_grid_sds(
            rsl_data.rsl_lat,
            rsl_data.rsl_lon,
            spatial_slice.lat_mat,
            spatial_slice.lon_mat,
            spatial_slice.total_std,
        )
        threshold = visualize._calibrate_threshold(
            slip_sds, factor=mask_settings.calibration_factor
        )
    elif not mask_settings.enabled:
        threshold = np.inf

    masked_mean = visualize._apply_std_mask(
        spatial_slice.total_mean,
        spatial_slice.total_std,
        threshold,
        mask_settings,
    )
    mask_map = np.asarray(spatial_slice.total_std) > threshold
    if not mask_settings.enabled:
        mask_map = np.zeros_like(mask_map, dtype=bool)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im0 = axes[0].pcolormesh(
        spatial_slice.lon_mat,
        spatial_slice.lat_mat,
        spatial_slice.total_std,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
    )
    visualize._add_spatial_basemap(axes[0])
    axes[0].set_title("Posterior SD (m)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].pcolormesh(
        spatial_slice.lon_mat,
        spatial_slice.lat_mat,
        mask_map.astype(float),
        cmap="Greys_r",
        vmin=0,
        vmax=1,
        transform=ccrs.PlateCarree(),
    )
    visualize._add_spatial_basemap(axes[1])
    axes[1].set_title("Masked cells (1 = hidden)")

    vabs = max(
        abs(float(np.nanmin(spatial_slice.total_mean))),
        abs(float(np.nanmax(spatial_slice.total_mean))),
        0.01,
    )
    im2 = axes[2].pcolormesh(
        spatial_slice.lon_mat,
        spatial_slice.lat_mat,
        masked_mean,
        cmap="RdBu_r",
        vmin=-vabs,
        vmax=vabs,
        transform=ccrs.PlateCarree(),
    )
    visualize._add_spatial_basemap(axes[2])
    axes[2].set_title("Total mean (masked)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    age_bp = int(round(spatial_slice.age_bp))
    factor = mask_settings.calibration_factor
    if mask_settings.enabled and np.isfinite(threshold):
        suptitle = (
            f"Mask preview at {age_bp} BP — "
            f"τ = {threshold:.3g} m ({factor:g}× median slip SD)"
        )
    elif mask_settings.enabled:
        suptitle = f"Mask preview at {age_bp} BP — factor {factor:g}× (fallback threshold)"
    else:
        suptitle = f"Mask preview at {age_bp} BP — masking disabled"

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig
