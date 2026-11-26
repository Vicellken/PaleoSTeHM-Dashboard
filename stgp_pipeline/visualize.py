"""
All plotting functions for the STGP + GIA pipeline.

Every function returns a ``matplotlib.figure.Figure`` — no file I/O.
The notebook calls ``ctx.save_figure(fig, name)`` for persistence.

Cartopy is imported at function level (lazy) to avoid import overhead
when only non-spatial functions are needed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .types import (
    RSLData,
    GIAResult,
    TrainedModel,
    RegionResult,
    ProjectedRegionData,
    SpatialSlice,
    SpatialRateResult,
)

# Shared styling constants
COMP_COLORS = {
    "global": "C0",
    "global_fast": "C7",
    "regional_linear": "C1",
    "regional_nl": "C2",
    "regional_nl_fast": "C6",
    "merged_nl": "C4",
    "local_nl": "C3",
    "local_nl_fast": "C8",
}
COMP_TITLES = {
    "global": "Global temporal",
    "global_fast": "Global temporal (fast)",
    "regional_linear": "Regional linear",
    "regional_nl": "Regional nonlinear",
    "regional_nl_fast": "Regional nonlinear (fast)",
    "merged_nl": "Merged regional+local NL",
    "local_nl": "Local nonlinear",
    "local_nl_fast": "Local nonlinear (fast)",
}
MEAN_CMAPS = {
    "global": "PuOr_r",
    "global_fast": "PuOr_r",
    "regional_linear": "PuOr_r",
    "regional_nl": "PuOr_r",
    "regional_nl_fast": "PuOr_r",
    "merged_nl": "PuOr_r",
    "local_nl": "BrBG",
    "local_nl_fast": "BrBG",
}
SPATIAL_COMP_TITLES = {
    "global": "Global",
    "global_fast": "Global (fast)",
    "regional_linear": "Regional linear",
    "regional_nl": "Regional NL",
    "regional_nl_fast": "Regional NL (fast)",
    "merged_nl": "Merged NL",
    "local_nl": "Local NL",
    "local_nl_fast": "Local NL (fast)",
}
RATE_CMAPS = {
    "total": "coolwarm",
    "global": "coolwarm",
    "global_fast": "coolwarm",
    "regional_linear": "PuOr_r",
    "regional_nl": "RdBu_r",
    "regional_nl_fast": "RdBu_r",
    "merged_nl": "RdBu_r",
    "local_nl": "PRGn_r",
    "local_nl_fast": "PRGn_r",
}


def plot_data_overview(rsl_data: RSLData) -> matplotlib.figure.Figure:
    """Temporal uncertainty boxes + spatial site map (2 panels)."""
    import cartopy
    import cartopy.crs as ccrs

    import PSTHM.plotting

    age = rsl_data.X[:, 0]
    y = rsl_data.y
    age_sigma = rsl_data.age_sigma
    y_sigma = rsl_data.y_sigma
    rsl_lon = rsl_data.rsl_lon
    rsl_lat = rsl_data.rsl_lat
    rsl_region = rsl_data.rsl_region

    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 2)

    ax = fig.add_subplot(gs[0, 0])
    PSTHM.plotting.plot_uncertainty_boxes(
        age, y, age_sigma * 2, y_sigma * 2, ax=ax)
    ax.set_xlim(float(age.max()) + 500, -200)
    ax.set_ylim(-60, 25)
    ax.set_title("Temporal SLIP data distribution")
    ax.set_xlabel("Age (BP)")
    ax.set_ylabel("RSL (m)")

    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax2.add_feature(cartopy.feature.LAND, edgecolor="black", zorder=0)
    ax2.set_extent(
        [
            np.min(rsl_lon) - 2,
            np.max(rsl_lon) + 2,
            np.min(rsl_lat) - 2,
            np.max(rsl_lat) + 2,
        ]
    )
    rsl_region_index = [np.where(rsl_region == i)[0]
                        for i in np.unique(rsl_region)]
    ax2.scatter(
        [np.mean(rsl_lon[x]) for x in rsl_region_index],
        [np.mean(rsl_lat[x]) for x in rsl_region_index],
        transform=ccrs.PlateCarree(),
        s=[len(x) * 20 for x in rsl_region_index],
        marker="o",
        facecolor="none",
        edgecolor="C0",
        linewidth=2,
    )
    for idx_list in rsl_region_index:
        ax2.scatter(
            rsl_lon[idx_list],
            rsl_lat[idx_list],
            transform=ccrs.PlateCarree(),
            s=10,
            alpha=0.5,
        )
    ax2.set_title("Spatial distribution")
    plt.tight_layout()
    return fig


def plot_gia_comparison(
    rsl_data: RSLData,
    gia_result: GIAResult,
    max_age_bp: int = 14000,
    bp_min: float = 0.0,
    bp_max: float = 21000.0,
) -> matplotlib.figure.Figure:
    """3-panel: SLIP+GIA, GIA curves per region, residuals."""
    import PSTHM.plotting

    X = rsl_data.X
    y = rsl_data.y
    age_sigma = rsl_data.age_sigma
    y_sigma = rsl_data.y_sigma
    rsl_lat = rsl_data.rsl_lat
    rsl_lon = rsl_data.rsl_lon
    rsl_region = rsl_data.rsl_region
    gia_pred = gia_result.gia_mean_at_obs
    gia_cov = gia_result.gia_cov_at_obs
    interp_funcs = gia_result.interp_funcs
    weights = gia_result.weights

    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    # Panel 1: SLIP + GIA prediction
    ax = axes[0]
    PSTHM.plotting.plot_uncertainty_boxes(
        X[:, 0], y, age_sigma * 2, y_sigma * 2, ax=ax)
    gia_std_pts = np.sqrt(np.diag(gia_cov))
    ax.scatter(
        X[:, 0].numpy(), gia_pred, c="r", s=30, zorder=5, label="GIA weighted mean"
    )
    ax.errorbar(
        X[:, 0].numpy(),
        gia_pred,
        yerr=2 * gia_std_pts,
        fmt="none",
        ecolor="r",
        alpha=0.3,
        zorder=4,
        label="GIA ±2σ (ensemble)",
    )
    ax.set_xlabel("Age (BP)")
    ax.set_ylabel("RSL (m)")
    ax.set_title("SLIP data + GIA prediction")
    ax.legend(fontsize=10)
    ax.invert_xaxis()

    # Panel 2: GIA per region centroid
    ax = axes[1]
    test_ages_plot = np.linspace(bp_min, min(bp_max, max_age_bp), 200)
    for rn in np.unique(rsl_region):
        region_mask = rsl_region == rn
        mean_lat = np.mean(rsl_lat[region_mask])
        mean_lon = np.mean(rsl_lon[region_mask]) % 360
        n_t = len(test_ages_plot)
        X_ts = np.column_stack(
            [test_ages_plot, np.full(n_t, mean_lat), np.full(n_t, mean_lon)]
        )
        preds = np.array([f(X_ts) for f in interp_funcs])
        from .gia_setup_3d import _compute_gia_mean_from_preds, _effective_mean_indices

        mean_idx = _effective_mean_indices(
            gia_result.mean_model_indices, gia_result.indices_3d
        )
        gia_ts = _compute_gia_mean_from_preds(
            preds, weights, mean_idx, gia_result.mean_weighting
        )
        ax.plot(test_ages_plot, gia_ts, alpha=0.6, label=f"R{rn}")
    ax.set_xlabel("Age (BP)")
    ax.set_ylabel("GIA RSL (m)")
    ax.set_title("GIA weighted mean per region centroid")
    ax.legend(fontsize=8, ncol=3)
    ax.invert_xaxis()

    # Panel 3: Residuals
    ax = axes[2]
    residual = y.numpy() - gia_pred
    ax.scatter(X[:, 0].numpy(), residual, c="k", marker="*", s=30)
    ax.axhline(0, color="gray", ls="--")
    ax.set_xlabel("Age (BP)")
    ax.set_ylabel("RSL residual (m)")
    ax.set_title("Data − GIA weighted mean")
    ax.invert_xaxis()

    plt.tight_layout()
    return fig


def plot_optimization_track(track_list: List[Any]) -> matplotlib.figure.Figure:
    """SVI optimization loss curve."""
    import PSTHM.plotting

    axes = PSTHM.plotting.plot_track_list(track_list)
    return plt.gcf()


def plot_temporal_all_regions(
    results: Dict[int, RegionResult],
    rsl_data: RSLData,
) -> matplotlib.figure.Figure:
    """N-panel temporal RSL predictions, one per region."""
    import PSTHM.plotting

    unique_regions = sorted(results.keys())
    n_regions = len(unique_regions)
    n_cols = 4
    n_rows = int(np.ceil(n_regions / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5))
    axes = axes.flatten()

    for i, rn in enumerate(unique_regions):
        ax = axes[i]
        res = results[rn]
        t = res.test_age

        ax.plot(t, res.y_total, "C0", linewidth=2)
        ax.fill_between(
            t,
            res.y_total - 2 * res.y_std,
            res.y_total + 2 * res.y_std,
            alpha=0.2,
            color="C0",
        )
        ax.plot(t, res.gia_ts, "r--", linewidth=1, alpha=0.7, label="GIA mean")

        region_mask = rsl_data.rsl_region == rn
        PSTHM.plotting.plot_uncertainty_boxes(
            rsl_data.X[region_mask, 0],
            rsl_data.y[region_mask],
            rsl_data.age_sigma[region_mask] * 2,
            rsl_data.y_sigma[region_mask] * 2,
            ax=ax,
        )

        ax.set_title(f"R{rn}: {res.region_name}", fontsize=12)
        ax.set_xlabel("Age (BP)")
        ax.set_ylabel("RSL (m)")
        ax.invert_xaxis()
        if i == 0:
            ax.legend(fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def plot_rate_all_regions(
    results: Dict[int, RegionResult],
) -> matplotlib.figure.Figure:
    """N-panel rate predictions, one per region."""
    unique_regions = sorted(results.keys())
    n_regions = len(unique_regions)
    n_cols = 4
    n_rows = int(np.ceil(n_regions / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5))
    axes = axes.flatten()

    for i, rn in enumerate(unique_regions):
        ax = axes[i]
        res = results[rn]
        t = res.rsl_time

        ax.plot(t, -res.rsl_rate * 1000, "C0", linewidth=2, label="Total rate")
        ax.fill_between(
            t,
            -(res.rsl_rate - 2 * res.rsl_rate_sd) * 1000,
            -(res.rsl_rate + 2 * res.rsl_rate_sd) * 1000,
            alpha=0.2,
            color="C0",
        )
        ax.plot(
            t,
            -res.gia_rate * 1000,
            "r--",
            linewidth=1,
            alpha=0.7,
            label="GIA mean rate",
        )
        ax.axhline(0, color="gray", ls="--")

        ax.set_title(f"R{rn}: {res.region_name}", fontsize=12)
        ax.set_xlabel("Age (BP)")
        ax.set_ylabel("Rate (mm/yr)")
        ax.invert_xaxis()
        if i == 0:
            ax.legend(fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def plot_temporal_decomposition(
    rn: int,
    result: RegionResult,
    rsl_data: RSLData,
    comp_labels: Optional[List[str]] = None,
    *,
    overlay_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    2 x (1 + n_components) panel figure for one region: components + rates.

    Row 1: Total (GP+GIA), then one panel per component in ``comp_labels`` order.
    Row 2: Total rate, then matching component rate panels.

    The grid width adapts to len(comp_labels) so an extra component
    (e.g. ``regional_nl_fast`` from the kernel-sensitivity test) lays out
    cleanly without overflowing into the rate row.

    ``overlay_mask`` / ``title`` override the Total-panel data overlay and the
    figure title (defaults reproduce the region behavior: all SLIP with
    ``rsl_region == rn`` and a ``"R{rn}: ..."`` title). They let the same plot
    be reused for a site-anchored prediction — see
    :func:`plot_temporal_decomposition_at_site`.
    """
    import PSTHM.plotting

    if comp_labels is None:
        comp_labels = list(COMP_COLORS.keys())

    t = result.test_age
    comps = result.components
    n_comp = len(comp_labels)
    n_cols = 1 + n_comp  # +1 for the Total panel

    fig = plt.figure(figsize=(10 * n_cols, 20))

    # Row 1, Panel 1: Total
    ax = plt.subplot(2, n_cols, 1)
    mask = overlay_mask if overlay_mask is not None else (
        rsl_data.rsl_region == rn)
    PSTHM.plotting.plot_uncertainty_boxes(
        rsl_data.X[mask, 0],
        rsl_data.y[mask],
        rsl_data.age_sigma[mask] * 2,
        rsl_data.y_sigma[mask] * 2,
        ax=ax,
    )
    ax.plot(t, result.y_total, "C5", linewidth=3, label="Total prediction")
    ax.fill_between(
        t,
        result.y_total - 2 * result.y_std,
        result.y_total + 2 * result.y_std,
        color="C5",
        alpha=0.3,
        label="95% CI",
    )
    gia_ts = np.asarray(result.gia_ts, dtype=float)
    if gia_ts.size and np.any(np.isfinite(gia_ts)):
        ax.plot(t, gia_ts, "r--", linewidth=2, alpha=0.7, label="GIA mean")
    ax.set_xlabel("Age (BP)")
    ax.set_ylabel("RSL (m)")
    ax.set_xlim(float(t.max()) + 500, -200)
    ax.legend(fontsize=14)

    # Row 1, Panels 2..n_cols: Components
    for j, label in enumerate(comp_labels):
        ax = plt.subplot(2, n_cols, j + 2)
        c = comps[label]
        color = COMP_COLORS.get(label, "C0")
        ax.plot(
            t,
            c["mean"],
            color,
            linewidth=3,
            label=COMP_TITLES.get(label, label),
        )
        ax.fill_between(
            t,
            c["mean"] - 2 * c["std"],
            c["mean"] + 2 * c["std"],
            color=color,
            alpha=0.3,
            label="95% CI",
        )
        ax.axhline(0, color="gray", ls="--")
        ax.set_xlabel("Age (BP)")
        ax.set_ylabel("RSL (m)")
        ax.set_xlim(float(t.max()) + 500, -200)
        ax.legend(fontsize=14)

    # Row 2, Panel 1: Total rate
    ax = plt.subplot(2, n_cols, n_cols + 1)
    ax.plot(
        result.rsl_time, -result.rsl_rate * 1000, "C5", linewidth=3, label="Total rate"
    )
    ax.fill_between(
        result.rsl_time,
        -(result.rsl_rate - 2 * result.rsl_rate_sd) * 1000,
        -(result.rsl_rate + 2 * result.rsl_rate_sd) * 1000,
        color="C5",
        alpha=0.3,
        label="95% CI",
    )
    gia_rate = np.asarray(result.gia_rate, dtype=float)
    if gia_rate.size and np.any(np.isfinite(gia_rate)):
        ax.plot(
            result.rsl_time,
            -gia_rate * 1000,
            "r--",
            linewidth=2,
            alpha=0.7,
            label="GIA mean rate",
        )
    ax.set_xlabel("Age (BP)")
    ax.set_ylabel("Rate (mm/yr)")
    ax.set_xlim(float(t.max()) + 500, -200)
    ax.legend(fontsize=14)

    # Row 2, Panels 2..n_cols: Component rates
    for j, label in enumerate(comp_labels):
        ax = plt.subplot(2, n_cols, n_cols + j + 2)
        c = comps[label]
        color = COMP_COLORS.get(label, "C0")
        ax.plot(
            result.rsl_time,
            -c["rate"] * 1000,
            color,
            linewidth=3,
            label=f"{COMP_TITLES.get(label, label)} rate",
        )
        ax.fill_between(
            result.rsl_time,
            -(c["rate"] - 2 * c["rate_sd"]) * 1000,
            -(c["rate"] + 2 * c["rate_sd"]) * 1000,
            color=color,
            alpha=0.3,
            label="95% CI",
        )
        ax.set_xlabel("Age (BP)")
        ax.set_ylabel("Rate (mm/yr)")
        ax.set_xlim(float(t.max()) + 500, -200)
        ax.legend(fontsize=14)

    plt.suptitle(
        title if title is not None else f"R{rn}: {result.region_name}", fontsize=30
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """Great-circle distance in km from (lat2, lon2) to each (lat1, lon1) point."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _plot_site_posterior_axes(
    ax: matplotlib.axes.Axes,
    result: RegionResult,
    rsl_data: RSLData,
    site_lat: float,
    site_lon: float,
    *,
    site_label: str = "",
    radius_km: float = 2.0,
) -> None:
    """Draw total RSL posterior + local SLIP on one axes."""
    import PSTHM.plotting

    t = result.test_age
    dist_km = _haversine_km(
        rsl_data.rsl_lat, rsl_data.rsl_lon, site_lat, site_lon)
    local_mask = dist_km <= radius_km

    if np.any(local_mask):
        PSTHM.plotting.plot_uncertainty_boxes(
            rsl_data.X[local_mask, 0],
            rsl_data.y[local_mask],
            rsl_data.age_sigma[local_mask] * 2,
            rsl_data.y_sigma[local_mask] * 2,
            ax=ax,
        )

    ax.plot(t, result.y_total, "C5", linewidth=3, label="Total prediction")
    ax.fill_between(
        t,
        result.y_total - 2 * result.y_std,
        result.y_total + 2 * result.y_std,
        color="C5",
        alpha=0.3,
        label="95% CI",
    )
    ax.plot(t, result.gia_ts, "r--", linewidth=2, alpha=0.7, label="GIA mean")
    ax.set_xlabel("Age (BP)")
    ax.set_ylabel("RSL (m)")
    ax.set_xlim(float(t.max()) + 500, -200)
    ax.legend(fontsize=12)
    label = site_label or result.region_name
    ax.set_title(
        f"{label}\n({site_lat:.4f}°N, {site_lon:.4f}°E; "
        f"{int(np.sum(local_mask))} SLIP within {radius_km:g} km)"
    )


def plot_site_posterior(
    result: RegionResult,
    rsl_data: RSLData,
    site_lat: float,
    site_lon: float,
    *,
    site_label: str = "",
    radius_km: float = 2.0,
) -> matplotlib.figure.Figure:
    """Total RSL posterior at an exact site with nearby SLIP data overlaid."""
    fig, ax = plt.subplots(figsize=(12, 7))
    _plot_site_posterior_axes(
        ax,
        result,
        rsl_data,
        site_lat,
        site_lon,
        site_label=site_label,
        radius_km=radius_km,
    )
    plt.tight_layout()
    return fig


def plot_site_posterior_comparison(
    results: Dict[str, RegionResult],
    rsl_data: RSLData,
    sites: List[Tuple[str, float, float]],
    *,
    radius_km: float = 2.0,
) -> matplotlib.figure.Figure:
    """Side-by-side total RSL posteriors for multiple sites."""
    n_sites = len(sites)
    fig, axes = plt.subplots(1, n_sites, figsize=(
        12 * n_sites, 7), squeeze=False)
    for ax, (name, lat, lon) in zip(axes[0], sites):
        _plot_site_posterior_axes(
            ax,
            results[name],
            rsl_data,
            lat,
            lon,
            site_label=name,
            radius_km=radius_km,
        )
    plt.tight_layout()
    return fig


def plot_temporal_decomposition_at_site(
    result: RegionResult,
    rsl_data: RSLData,
    site_lat: float,
    site_lon: float,
    *,
    site_label: str = "",
    radius_km: float = 2.0,
    comp_labels: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """
    Full temporal decomposition (Total + components + rates) anchored at a site.

    Same panels as :func:`plot_temporal_decomposition`, but the curve is the
    site-level prediction (from :func:`predict.temporal_at_sites`) and the data
    overlay shows SLIP within ``radius_km`` of the site instead of the whole
    region. Use this to show a single site's RSL history — e.g. a mid-Holocene
    highstand that the region-centroid curve averages away.
    """
    dist_km = _haversine_km(
        rsl_data.rsl_lat, rsl_data.rsl_lon, site_lat, site_lon)
    mask = dist_km <= radius_km
    label = site_label or result.region_name
    title = (
        f"{label}\n({site_lat:.4f}°N, {site_lon:.4f}°E; "
        f"{int(np.sum(mask))} SLIP within {radius_km:g} km)"
    )
    return plot_temporal_decomposition(
        result.region_num,
        result,
        rsl_data,
        comp_labels,
        overlay_mask=mask,
        title=title,
    )


def plot_projected_region(
    ref_result: RegionResult,
    projected: ProjectedRegionData,
    *,
    show_original: bool = True,
) -> matplotlib.figure.Figure:
    """
    Region SLIP data projected to a reference location, over the reference curve.

    Plots the reference-location Total RSL curve (from
    :func:`predict.temporal_at_sites` at the reference) with the projected data
    overlaid. When ``show_original`` the unprojected observations are drawn faded
    so the GP-applied shift is visible (cf. the before/after figures in the
    projection toy example). The reference curve itself is unchanged by the
    projection — it comes from the model; only the data move.
    """
    import PSTHM.plotting

    fig, ax = plt.subplots(figsize=(12, 7))
    t = ref_result.test_age

    if show_original:
        PSTHM.plotting.plot_uncertainty_boxes(
            projected.age,
            projected.orig_y,
            projected.age_sigma * 2,
            projected.y_sigma * 2,
            fc=(0.85, 0.85, 0.85, 0.25),
            ec=(0.5, 0.5, 0.5, 0.6),
            ax=ax,
            label="Observed (at own site)",
        )
    PSTHM.plotting.plot_uncertainty_boxes(
        projected.age,
        projected.projected_y,
        projected.age_sigma * 2,
        projected.y_sigma * 2,
        fc=(0.12, 0.47, 0.71, 0.20),
        ec=(0.12, 0.47, 0.71, 0.85),
        ax=ax,
        label="Projected to reference",
    )

    ax.plot(
        t, ref_result.y_total, "C5", linewidth=3, label="Total prediction", zorder=4
    )
    ax.fill_between(
        t,
        ref_result.y_total - 2 * ref_result.y_std,
        ref_result.y_total + 2 * ref_result.y_std,
        color="C5",
        alpha=0.3,
        label="95% CI",
        zorder=1,
    )
    ax.plot(t, ref_result.gia_ts, "r--",
            linewidth=2, alpha=0.7, label="GIA mean")
    ax.set_xlabel("Age (BP)")
    ax.set_ylabel("RSL (m)")
    ax.set_xlim(float(t.max()) + 500, -200)
    ax.legend(fontsize=12)
    rlabel = projected.ref_name or ref_result.region_name
    region_tag = f"R{projected.region_num} " if projected.region_num is not None else ""
    ax.set_title(
        f"{region_tag}data projected to {rlabel}\n"
        f"({projected.ref_lat:.4f}°N, {projected.ref_lon:.4f}°E; "
        f"{len(projected.projected_y)} SLIP)"
    )
    plt.tight_layout()
    return fig


# Defaults when spatial_plot_mask is absent from config (backward compatible).
_SPATIAL_RSL_STD_FALLBACK_M = 2.0
_SPATIAL_RATE_SD_FALLBACK_MM_YR = 50.0
_SPATIAL_CALIBRATION_FACTOR = 2.0


@dataclass(frozen=True)
class SpatialPlotMaskSettings:
    """Visualization-only SD masking for spatial mean/rate panels."""

    enabled: bool = True
    calibration_factor: float = _SPATIAL_CALIBRATION_FACTOR
    rsl_std_fallback_m: float = _SPATIAL_RSL_STD_FALLBACK_M
    rate_sd_fallback_mm_yr: float = _SPATIAL_RATE_SD_FALLBACK_MM_YR


def spatial_plot_mask_settings(
    cfg: Optional[Mapping[str, Any]] = None,
) -> SpatialPlotMaskSettings:
    """Parse ``spatial_plot_mask`` from a study YAML config."""
    if not cfg:
        return SpatialPlotMaskSettings()
    raw = cfg.get("spatial_plot_mask", {}) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("spatial_plot_mask must be a mapping")
    factor = float(raw.get("calibration_factor", _SPATIAL_CALIBRATION_FACTOR))
    if factor <= 0:
        raise ValueError(
            f"spatial_plot_mask.calibration_factor must be > 0, got {factor}"
        )
    return SpatialPlotMaskSettings(
        enabled=bool(raw.get("enabled", True)),
        calibration_factor=factor,
        rsl_std_fallback_m=float(
            raw.get("rsl_std_fallback_m", _SPATIAL_RSL_STD_FALLBACK_M)
        ),
        rate_sd_fallback_mm_yr=float(
            raw.get("rate_sd_fallback_mm_yr", _SPATIAL_RATE_SD_FALLBACK_MM_YR)
        ),
    )


def _resolve_mask_settings(
    mask_settings: Optional[SpatialPlotMaskSettings],
    cfg: Optional[Mapping[str, Any]],
) -> SpatialPlotMaskSettings:
    if mask_settings is not None:
        return mask_settings
    return spatial_plot_mask_settings(cfg)


def _slip_grid_sds(
    slip_lat: np.ndarray,
    slip_lon: np.ndarray,
    lat_mat: np.ndarray,
    lon_mat: np.ndarray,
    sd_2d: np.ndarray,
) -> np.ndarray:
    """Posterior SD at the nearest spatial grid cell for each SLIP."""
    lat_flat = lat_mat.ravel()
    lon_flat = lon_mat.ravel()
    sd_flat = np.asarray(sd_2d).ravel()
    sds = np.empty(len(slip_lat), dtype=float)
    for k, (lat, lon) in enumerate(zip(slip_lat, slip_lon)):
        dist = _haversine_km(lat_flat, lon_flat,
                             float(lat), float(lon) % 360.0)
        sds[k] = sd_flat[int(np.argmin(dist))]
    return sds


def _calibrate_threshold(
    slip_sds: np.ndarray, factor: float = _SPATIAL_CALIBRATION_FACTOR
) -> float:
    return float(np.median(slip_sds) * factor)


def _mask_by_std(
    data: np.ndarray, std: np.ndarray, threshold: float
) -> np.ma.MaskedArray:
    return np.ma.masked_array(np.asarray(data), np.asarray(std) > threshold)


def _apply_std_mask(
    data: np.ndarray,
    std: np.ndarray,
    threshold: float,
    mask_settings: SpatialPlotMaskSettings,
) -> np.ma.MaskedArray:
    if not mask_settings.enabled:
        return np.ma.masked_array(
            np.asarray(data),
            mask=np.zeros(np.asarray(data).shape, dtype=bool),
        )
    return _mask_by_std(data, std, threshold)


def _resolve_std_threshold(
    *,
    rsl_data: Optional[RSLData],
    lat_mat: np.ndarray,
    lon_mat: np.ndarray,
    std_2d: np.ndarray,
    explicit_threshold: Optional[float],
    by_key: Optional[Dict[str, float]],
    key: Optional[str],
    unit: str,
    panel_name: str,
    fallback: float,
    mask_settings: SpatialPlotMaskSettings,
) -> float:
    if by_key and key is not None and key in by_key:
        threshold = by_key[key]
        print(
            f"{panel_name}: masking where SD > {threshold:.4g} {unit} (manual override)"
        )
        return threshold
    if explicit_threshold is not None:
        print(
            f"{panel_name}: masking where SD > {explicit_threshold:.4g} {unit} (global override)"
        )
        return explicit_threshold
    if rsl_data is not None:
        slip_sds = _slip_grid_sds(
            rsl_data.rsl_lat, rsl_data.rsl_lon, lat_mat, lon_mat, std_2d
        )
        threshold = _calibrate_threshold(
            slip_sds, factor=mask_settings.calibration_factor
        )
        print(
            f"{panel_name}: masking where SD > {threshold:.4g} {unit} "
            f"({mask_settings.calibration_factor:g}× median slip SD)"
        )
        return threshold
    warnings.warn(
        f"{panel_name}: rsl_data not provided; using fallback threshold "
        f"{fallback:.4g} {unit}",
        stacklevel=2,
    )
    return fallback


def spatial_total_trust_mask(
    spatial_slice: SpatialSlice,
    rsl_data: Optional[RSLData] = None,
    std_threshold_m: Optional[float] = None,
    std_threshold_m_by_comp: Optional[Dict[str, float]] = None,
    *,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
    panel_name: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """Return the cells retained by total-RSL maps and their SD threshold."""
    ms = _resolve_mask_settings(mask_settings, cfg)
    s = spatial_slice
    threshold = _resolve_std_threshold(
        rsl_data=rsl_data,
        lat_mat=s.lat_mat,
        lon_mat=s.lon_mat,
        std_2d=s.total_std,
        explicit_threshold=std_threshold_m,
        by_key=std_threshold_m_by_comp,
        key="total",
        unit="m",
        panel_name=panel_name or f"Total RSL at {s.age_bp:.0f} BP",
        fallback=ms.rsl_std_fallback_m,
        mask_settings=ms,
    )
    trusted = np.ones(np.asarray(s.total_std).shape, dtype=bool)
    if ms.enabled:
        trusted = ~(np.asarray(s.total_std) > threshold)
    return trusted, threshold


def _add_spatial_basemap(ax: matplotlib.axes.Axes) -> None:
    ax.coastlines()


def plot_spatial_total(
    spatial_slice: SpatialSlice,
    rsl_data: Optional[RSLData] = None,
    std_threshold_m: Optional[float] = None,
    *,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> matplotlib.figure.Figure:
    """2-panel RSL + uncertainty map at one age."""
    import cartopy.crs as ccrs

    ms = _resolve_mask_settings(mask_settings, cfg)
    s = spatial_slice
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(22, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    trusted, _total_threshold = spatial_total_trust_mask(
        s,
        rsl_data=rsl_data,
        std_threshold_m=std_threshold_m,
        mask_settings=ms,
        panel_name=f"Total RSL at {s.age_bp:.0f} BP",
    )
    masked_mean = np.ma.masked_array(np.asarray(s.total_mean), mask=~trusted)

    vabs = max(abs(np.nanmin(s.total_mean)), abs(np.nanmax(s.total_mean)))
    if vabs < 0.01:
        vabs = 1.0
    im1 = ax1.pcolormesh(
        s.lon_mat,
        s.lat_mat,
        masked_mean,
        cmap="RdBu_r",
        vmin=-vabs,
        vmax=vabs,
        transform=ccrs.PlateCarree(),
    )
    _add_spatial_basemap(ax1)
    plt.colorbar(im1, ax=ax1, label="RSL (m)")
    ax1.set_title(f"Total RSL at {s.age_bp:.0f} BP")

    im2 = ax2.pcolormesh(
        s.lon_mat,
        s.lat_mat,
        s.total_std,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
    )
    _add_spatial_basemap(ax2)
    plt.colorbar(im2, ax=ax2, label="Std (m)")
    ax2.set_title(f"Total uncertainty (GP + GIA) at {s.age_bp:.0f} BP")

    plt.tight_layout()
    return fig


def _floor_spatial_vabs(vabs: float) -> float:
    return 1.0 if vabs < 0.01 else float(vabs)


def _vabs_from_masked(masked: np.ma.MaskedArray) -> float:
    if masked.count() == 0:
        return 0.0
    return float(np.nanmax(np.abs(masked.compressed())))


def _panel_vabs(
    vabs_by_panel: Optional[Dict[str, float]],
    key: str,
    data: np.ndarray,
) -> float:
    if vabs_by_panel is not None and key in vabs_by_panel:
        return vabs_by_panel[key]
    vabs = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    return _floor_spatial_vabs(vabs)


def _rate_interval_age_labels(
    rate_result: SpatialRateResult,
    interval_idx: int,
) -> Tuple[int, int]:
    r = rate_result
    i = interval_idx
    n_intervals = len(r.rate_midpoints)
    if len(r.spatial_ages) > n_intervals + 1:
        return int(r.spatial_ages[0]), int(r.spatial_ages[-1])
    return int(r.spatial_ages[i]), int(r.spatial_ages[i + 1])


def _spatial_rate_panel_arrays(
    rate_result: SpatialRateResult,
    interval_idx: int,
    key: str,
    comp_labels: List[str],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Rate and SD grids (mm/yr) plus display title for one panel key."""
    r = rate_result
    i = interval_idx
    lat_shape = r.lat_mat.shape
    if key == "total":
        return (
            (-r.rate_total[i] * 1000).reshape(lat_shape),
            (r.sd_total[i] * 1000).reshape(lat_shape),
            "Total (GP+GIA)",
        )
    if key == "gia":
        return (
            (-r.rate_gia[i] * 1000).reshape(lat_shape),
            (r.sd_gia[i] * 1000).reshape(lat_shape),
            "GIA",
        )
    if key not in comp_labels:
        raise KeyError(f"Unknown spatial rate panel key {key!r}")
    return (
        (-r.rate_comp[key][i] * 1000).reshape(lat_shape),
        (r.sd_comp[key][i] * 1000).reshape(lat_shape),
        SPATIAL_COMP_TITLES[key],
    )


def _masked_spatial_rate_panel(
    rate_result: SpatialRateResult,
    interval_idx: int,
    key: str,
    comp_labels: List[str],
    rsl_data: Optional[RSLData] = None,
    sd_threshold_mm_yr: Optional[float] = None,
    sd_threshold_mm_yr_by_comp: Optional[Dict[str, float]] = None,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> np.ma.MaskedArray:
    ms = _resolve_mask_settings(mask_settings, cfg)
    younger, older = _rate_interval_age_labels(rate_result, interval_idx)
    data_2d, sd_2d, title = _spatial_rate_panel_arrays(
        rate_result, interval_idx, key, comp_labels
    )
    threshold = _resolve_std_threshold(
        rsl_data=rsl_data,
        lat_mat=rate_result.lat_mat,
        lon_mat=rate_result.lon_mat,
        std_2d=sd_2d,
        explicit_threshold=sd_threshold_mm_yr,
        by_key=sd_threshold_mm_yr_by_comp,
        key=key,
        unit="mm/yr",
        panel_name=f"{title} rate ({younger}--{older} BP)",
        fallback=ms.rate_sd_fallback_mm_yr,
        mask_settings=ms,
    )
    return _apply_std_mask(data_2d, sd_2d, threshold, ms)


def spatial_rate_vabs_by_panel(
    rate_result: SpatialRateResult,
    comp_labels: Optional[List[str]] = None,
    rsl_data: Optional[RSLData] = None,
    sd_threshold_mm_yr: Optional[float] = None,
    sd_threshold_mm_yr_by_comp: Optional[Dict[str, float]] = None,
    *,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Symmetric color limits (mm/yr) per panel, max over all intervals."""
    if comp_labels is None:
        comp_labels = list(SPATIAL_COMP_TITLES.keys())
    keys = ["total", "gia"] + list(comp_labels)
    vabs: Dict[str, float] = {k: 0.0 for k in keys}
    for i in range(len(rate_result.rate_midpoints)):
        for key in keys:
            masked = _masked_spatial_rate_panel(
                rate_result,
                i,
                key,
                comp_labels,
                rsl_data=rsl_data,
                sd_threshold_mm_yr=sd_threshold_mm_yr,
                sd_threshold_mm_yr_by_comp=sd_threshold_mm_yr_by_comp,
                mask_settings=mask_settings,
                cfg=cfg,
            )
            vabs[key] = max(vabs[key], _vabs_from_masked(masked))
    return {k: _floor_spatial_vabs(v) for k, v in vabs.items()}


def _masked_spatial_decomposition_mean_panel(
    spatial_slice: SpatialSlice,
    key: str,
    comp_labels: List[str],
    rsl_data: Optional[RSLData] = None,
    std_threshold_m: Optional[float] = None,
    std_threshold_m_by_comp: Optional[Dict[str, float]] = None,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> np.ma.MaskedArray:
    ms = _resolve_mask_settings(mask_settings, cfg)
    s = spatial_slice
    if key == "total":
        data = s.total_mean
        panel_name = f"Total mean (GP+GIA) at {s.age_bp:.0f} BP"
        trusted, _threshold = spatial_total_trust_mask(
            s,
            rsl_data=rsl_data,
            std_threshold_m=std_threshold_m,
            std_threshold_m_by_comp=std_threshold_m_by_comp,
            mask_settings=ms,
            panel_name=panel_name,
        )
        return np.ma.masked_array(np.asarray(data), mask=~trusted)
    else:
        if key not in comp_labels:
            raise KeyError(f"Unknown decomposition panel key {key!r}")
        data = s.components[key]["mean"]
        std_2d = s.components[key]["std"]
        panel_name = f"{SPATIAL_COMP_TITLES[key]} mean at {s.age_bp:.0f} BP"
    threshold = _resolve_std_threshold(
        rsl_data=rsl_data,
        lat_mat=s.lat_mat,
        lon_mat=s.lon_mat,
        std_2d=std_2d,
        explicit_threshold=std_threshold_m,
        by_key=std_threshold_m_by_comp,
        key=key,
        unit="m",
        panel_name=panel_name,
        fallback=ms.rsl_std_fallback_m,
        mask_settings=ms,
    )
    return _apply_std_mask(data, std_2d, threshold, ms)


def spatial_decomposition_vabs_by_panel(
    spatial_slices: Sequence[SpatialSlice],
    comp_labels: Optional[List[str]] = None,
    rsl_data: Optional[RSLData] = None,
    std_threshold_m: Optional[float] = None,
    std_threshold_m_by_comp: Optional[Dict[str, float]] = None,
    *,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Symmetric color limits (m) per mean panel, max over all ages."""
    if comp_labels is None:
        comp_labels = list(SPATIAL_COMP_TITLES.keys())
    keys = ["total"] + list(comp_labels)
    vabs: Dict[str, float] = {k: 0.0 for k in keys}
    for s in spatial_slices:
        for key in keys:
            masked = _masked_spatial_decomposition_mean_panel(
                s,
                key,
                comp_labels,
                rsl_data=rsl_data,
                std_threshold_m=std_threshold_m,
                std_threshold_m_by_comp=std_threshold_m_by_comp,
                mask_settings=mask_settings,
                cfg=cfg,
            )
            vabs[key] = max(vabs[key], _vabs_from_masked(masked))
    return {k: _floor_spatial_vabs(v) for k, v in vabs.items()}


def _masked_total_and_regional_linear(
    spatial_slice: SpatialSlice,
    rsl_data: Optional[RSLData] = None,
    std_threshold_m: Optional[float] = None,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """Masked total and regional-linear mean grids for one spatial slice."""
    ms = _resolve_mask_settings(mask_settings, cfg)
    s = spatial_slice
    if "regional_linear" not in s.components:
        raise KeyError(
            "spatial_slice.components missing 'regional_linear'; "
            "is regional_linear enabled in the kernel stack?"
        )

    total_trusted, _total_threshold = spatial_total_trust_mask(
        s,
        rsl_data=rsl_data,
        std_threshold_m=std_threshold_m,
        mask_settings=ms,
        panel_name=f"Total mean (GP+GIA) at {s.age_bp:.0f} BP",
    )
    masked_total = np.ma.masked_array(
        np.asarray(s.total_mean), mask=~total_trusted
    )

    rl_mean = s.components["regional_linear"]["mean"]
    rl_std = s.components["regional_linear"]["std"]
    rl_threshold = _resolve_std_threshold(
        rsl_data=rsl_data,
        lat_mat=s.lat_mat,
        lon_mat=s.lon_mat,
        std_2d=rl_std,
        explicit_threshold=std_threshold_m,
        by_key=None,
        key="regional_linear",
        unit="m",
        panel_name=f"{SPATIAL_COMP_TITLES['regional_linear']} mean at {s.age_bp:.0f} BP",
        fallback=ms.rsl_std_fallback_m,
        mask_settings=ms,
    )
    masked_rl = _apply_std_mask(rl_mean, rl_std, rl_threshold, ms)
    return masked_total, masked_rl


def spatial_total_vs_regional_linear_vabs(
    spatial_slices: Sequence[SpatialSlice],
    rsl_data: Optional[RSLData] = None,
    std_threshold_m: Optional[float] = None,
    *,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, float]:
    """
    Global symmetric color limits (m) for total vs regional-linear comparison maps.

    Scans all ages so every figure in a time series shares the same vmin/vmax per
    panel type.
    """
    vabs_total = 0.0
    vabs_rl = 0.0
    for s in spatial_slices:
        masked_total, masked_rl = _masked_total_and_regional_linear(
            s,
            rsl_data=rsl_data,
            std_threshold_m=std_threshold_m,
            mask_settings=mask_settings,
            cfg=cfg,
        )
        vabs_total = max(vabs_total, _vabs_from_masked(masked_total))
        vabs_rl = max(vabs_rl, _vabs_from_masked(masked_rl))
    return _floor_spatial_vabs(vabs_total), _floor_spatial_vabs(vabs_rl)


def plot_spatial_total_vs_regional_linear(
    spatial_slice: SpatialSlice,
    rsl_data: Optional[RSLData] = None,
    std_threshold_m: Optional[float] = None,
    vabs_total: Optional[float] = None,
    vabs_rl: Optional[float] = None,
    *,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> matplotlib.figure.Figure:
    """1x2 mean maps: total (GP+GIA) vs regional linear component at one age."""
    import cartopy.crs as ccrs

    s = spatial_slice
    masked_total, masked_rl = _masked_total_and_regional_linear(
        s,
        rsl_data=rsl_data,
        std_threshold_m=std_threshold_m,
        mask_settings=mask_settings,
        cfg=cfg,
    )

    if vabs_total is None:
        vabs_total = _floor_spatial_vabs(_vabs_from_masked(masked_total))
    if vabs_rl is None:
        vabs_rl = _floor_spatial_vabs(_vabs_from_masked(masked_rl))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(22, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im1 = ax1.pcolormesh(
        s.lon_mat,
        s.lat_mat,
        masked_total,
        cmap="RdBu_r",
        vmin=-vabs_total,
        vmax=vabs_total,
        transform=ccrs.PlateCarree(),
    )
    _add_spatial_basemap(ax1)
    plt.colorbar(im1, ax=ax1, label="m")
    ax1.set_title("Total mean (GP+GIA)")

    im2 = ax2.pcolormesh(
        s.lon_mat,
        s.lat_mat,
        masked_rl,
        cmap=MEAN_CMAPS["regional_linear"],
        vmin=-vabs_rl,
        vmax=vabs_rl,
        transform=ccrs.PlateCarree(),
    )
    _add_spatial_basemap(ax2)
    plt.colorbar(im2, ax=ax2, label="m")
    ax2.set_title(f"{SPATIAL_COMP_TITLES['regional_linear']} mean")

    fig.suptitle(
        f"Total vs regional linear at {s.age_bp:.0f} BP "
        f"(limits: total ±{vabs_total:.1f} m, RL ±{vabs_rl:.1f} m)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_spatial_decomposition(
    spatial_slice: SpatialSlice,
    comp_labels: Optional[List[str]] = None,
    rsl_data: Optional[RSLData] = None,
    std_threshold_m: Optional[float] = None,
    std_threshold_m_by_comp: Optional[Dict[str, float]] = None,
    vabs_by_panel: Optional[Dict[str, float]] = None,
    *,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> matplotlib.figure.Figure:
    """2 x (1 + n_components) component maps: means (row 1) and stds (row 2)."""
    import cartopy.crs as ccrs

    ms = _resolve_mask_settings(mask_settings, cfg)
    if comp_labels is None:
        comp_labels = list(SPATIAL_COMP_TITLES.keys())

    s = spatial_slice
    n_comp = len(comp_labels)
    n_cols = 1 + n_comp

    fig = plt.figure(figsize=(10 * n_cols, 20))

    # Row 1: Means
    total_trusted, _total_threshold = spatial_total_trust_mask(
        s,
        rsl_data=rsl_data,
        std_threshold_m=std_threshold_m,
        std_threshold_m_by_comp=std_threshold_m_by_comp,
        mask_settings=ms,
        panel_name=f"Total mean (GP+GIA) at {s.age_bp:.0f} BP",
    )
    ax = fig.add_subplot(2, n_cols, 1, projection=ccrs.PlateCarree())
    total_vabs = _panel_vabs(vabs_by_panel, "total", s.total_mean)
    im = ax.pcolormesh(
        s.lon_mat,
        s.lat_mat,
        np.ma.masked_array(np.asarray(s.total_mean), mask=~total_trusted),
        cmap="RdBu_r",
        vmin=-total_vabs,
        vmax=total_vabs,
        transform=ccrs.PlateCarree(),
    )
    _add_spatial_basemap(ax)
    plt.colorbar(im, ax=ax, label="m")
    ax.set_title("Total mean (GP+GIA)")

    for j, label in enumerate(comp_labels):
        comp_std = s.components[label]["std"]
        comp_threshold = _resolve_std_threshold(
            rsl_data=rsl_data,
            lat_mat=s.lat_mat,
            lon_mat=s.lon_mat,
            std_2d=comp_std,
            explicit_threshold=std_threshold_m,
            by_key=std_threshold_m_by_comp,
            key=label,
            unit="m",
            panel_name=f"{SPATIAL_COMP_TITLES[label]} mean at {s.age_bp:.0f} BP",
            fallback=ms.rsl_std_fallback_m,
            mask_settings=ms,
        )
        ax = fig.add_subplot(2, n_cols, j + 2, projection=ccrs.PlateCarree())
        cmap = MEAN_CMAPS[label]
        data = s.components[label]["mean"]
        vabs = _panel_vabs(vabs_by_panel, label, data)
        im = ax.pcolormesh(
            s.lon_mat,
            s.lat_mat,
            _apply_std_mask(data, comp_std, comp_threshold, ms),
            cmap=cmap,
            vmin=-vabs,
            vmax=vabs,
            transform=ccrs.PlateCarree(),
        )
        _add_spatial_basemap(ax)
        plt.colorbar(im, ax=ax, label="m")
        ax.set_title(f"{SPATIAL_COMP_TITLES[label]} mean")

    # Row 2: Stds — unified colorbar range
    all_std_vals = np.concatenate(
        [s.total_std.flatten()]
        + [s.components[l]["std"].flatten() for l in comp_labels]
    )
    std_vmin, std_vmax = np.min(all_std_vals), np.max(all_std_vals)

    ax = fig.add_subplot(2, n_cols, n_cols + 1, projection=ccrs.PlateCarree())
    im = ax.pcolormesh(
        s.lon_mat,
        s.lat_mat,
        s.total_std,
        cmap="magma",
        vmin=std_vmin,
        vmax=std_vmax,
        transform=ccrs.PlateCarree(),
    )
    _add_spatial_basemap(ax)
    plt.colorbar(im, ax=ax, label="m")
    ax.set_title("Total std (GP+GIA)")

    for j, label in enumerate(comp_labels):
        ax = fig.add_subplot(2, n_cols, n_cols + j + 2,
                             projection=ccrs.PlateCarree())
        im = ax.pcolormesh(
            s.lon_mat,
            s.lat_mat,
            s.components[label]["std"],
            cmap="magma",
            vmin=std_vmin,
            vmax=std_vmax,
            transform=ccrs.PlateCarree(),
        )
        _add_spatial_basemap(ax)
        plt.colorbar(im, ax=ax, label="m")
        ax.set_title(f"{SPATIAL_COMP_TITLES[label]} std")

    suptitle = f"Spatial kernel decomposition at {s.age_bp:.0f} BP"
    if vabs_by_panel:
        suptitle += " (fixed scales across ages)"
    plt.suptitle(suptitle, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_spatial_rate(
    rate_result: SpatialRateResult,
    interval_idx: int,
    comp_labels: Optional[List[str]] = None,
    rsl_data: Optional[RSLData] = None,
    sd_threshold_mm_yr: Optional[float] = None,
    sd_threshold_mm_yr_by_comp: Optional[Dict[str, float]] = None,
    vabs_by_panel: Optional[Dict[str, float]] = None,
    *,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> matplotlib.figure.Figure:
    """2 x (2 + n_components) rate + SD maps for one time interval."""
    import cartopy.crs as ccrs

    ms = _resolve_mask_settings(mask_settings, cfg)
    if comp_labels is None:
        comp_labels = list(SPATIAL_COMP_TITLES.keys())

    r = rate_result
    i = interval_idx
    younger, older = _rate_interval_age_labels(r, i)
    midpt = r.rate_midpoints[i]
    lat_mat = r.lat_mat
    lon_mat = r.lon_mat

    n_cols = 2 + len(comp_labels)  # Total + GIA + each component
    fig = plt.figure(figsize=(10 * n_cols, 20))

    # Row 1: Rate panels (display mm/yr; SD arrays stored in m/yr)
    rate_panel_keys = ["total", "gia"] + list(comp_labels)

    for j, key in enumerate(rate_panel_keys):
        data_2d, sd_2d, title = _spatial_rate_panel_arrays(
            r, i, key, comp_labels)
        ax = fig.add_subplot(2, n_cols, j + 1, projection=ccrs.PlateCarree())
        threshold = _resolve_std_threshold(
            rsl_data=rsl_data,
            lat_mat=lat_mat,
            lon_mat=lon_mat,
            std_2d=sd_2d,
            explicit_threshold=sd_threshold_mm_yr,
            by_key=sd_threshold_mm_yr_by_comp,
            key=key,
            unit="mm/yr",
            panel_name=f"{title} rate ({younger}--{older} BP)",
            fallback=ms.rate_sd_fallback_mm_yr,
            mask_settings=ms,
        )
        masked_data = _apply_std_mask(data_2d, sd_2d, threshold, ms)
        vabs = _panel_vabs(vabs_by_panel, key, data_2d)
        im = ax.pcolormesh(
            lon_mat,
            lat_mat,
            masked_data,
            cmap=RATE_CMAPS.get(key, "coolwarm"),
            vmin=-vabs,
            vmax=vabs,
            transform=ccrs.PlateCarree(),
        )
        _add_spatial_basemap(ax)
        plt.colorbar(im, ax=ax, label="mm/yr")
        ax.set_title(title)

    # Row 2: SD panels
    sd_panels = [
        ("Total SD", r.sd_total[i] * 1000),
        ("GIA SD", r.sd_gia[i] * 1000),
    ] + [(f"{SPATIAL_COMP_TITLES[l]} SD", r.sd_comp[l][i] * 1000) for l in comp_labels]
    all_sd = np.concatenate([d.reshape(-1) for _, d in sd_panels])
    sd_vmax = np.max(all_sd)

    for j, (title, data) in enumerate(sd_panels):
        ax = fig.add_subplot(2, n_cols, n_cols + j + 1,
                             projection=ccrs.PlateCarree())
        data_2d = data.reshape(lat_mat.shape)
        im = ax.pcolormesh(
            lon_mat,
            lat_mat,
            data_2d,
            cmap="magma",
            vmin=0,
            vmax=sd_vmax,
            transform=ccrs.PlateCarree(),
        )
        _add_spatial_basemap(ax)
        plt.colorbar(im, ax=ax, label="mm/yr")
        ax.set_title(title)

    suptitle = f"Spatial rate: {younger}--{older} BP (midpoint {midpt:.0f})"
    if vabs_by_panel:
        suptitle += " [fixed scales across intervals]"
    plt.suptitle(suptitle, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _masked_total_and_regional_linear_rate(
    rate_result: SpatialRateResult,
    interval_idx: int,
    rsl_data: Optional[RSLData] = None,
    sd_threshold_mm_yr: Optional[float] = None,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """Masked total and regional-linear rate grids (mm/yr) for one interval."""
    ms = _resolve_mask_settings(mask_settings, cfg)
    r = rate_result
    i = interval_idx
    n_intervals = len(r.rate_midpoints)
    if len(r.spatial_ages) > n_intervals + 1:
        younger = int(r.spatial_ages[0])
        older = int(r.spatial_ages[-1])
    else:
        younger = int(r.spatial_ages[i])
        older = int(r.spatial_ages[i + 1])
    lat_mat = r.lat_mat
    lon_mat = r.lon_mat

    if "regional_linear" not in r.rate_comp:
        raise KeyError(
            "rate_result.rate_comp missing 'regional_linear'; "
            "is regional_linear enabled in the kernel stack?"
        )

    total_rate = (-r.rate_total[i] * 1000).reshape(lat_mat.shape)
    total_sd = (r.sd_total[i] * 1000).reshape(lat_mat.shape)
    total_threshold = _resolve_std_threshold(
        rsl_data=rsl_data,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        std_2d=total_sd,
        explicit_threshold=sd_threshold_mm_yr,
        by_key=None,
        key="total",
        unit="mm/yr",
        panel_name=f"Total (GP+GIA) rate ({younger}--{older} BP)",
        fallback=ms.rate_sd_fallback_mm_yr,
        mask_settings=ms,
    )
    masked_total = _apply_std_mask(total_rate, total_sd, total_threshold, ms)

    rl_rate = (-r.rate_comp["regional_linear"]
               [i] * 1000).reshape(lat_mat.shape)
    rl_sd = (r.sd_comp["regional_linear"][i] * 1000).reshape(lat_mat.shape)
    rl_threshold = _resolve_std_threshold(
        rsl_data=rsl_data,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        std_2d=rl_sd,
        explicit_threshold=sd_threshold_mm_yr,
        by_key=None,
        key="regional_linear",
        unit="mm/yr",
        panel_name=(
            f"{SPATIAL_COMP_TITLES['regional_linear']} rate ({younger}--{older} BP)"
        ),
        fallback=ms.rate_sd_fallback_mm_yr,
        mask_settings=ms,
    )
    masked_rl = _apply_std_mask(rl_rate, rl_sd, rl_threshold, ms)
    return masked_total, masked_rl


def spatial_total_vs_regional_linear_rate_vabs(
    rate_result: SpatialRateResult,
    interval_idx: int,
    rsl_data: Optional[RSLData] = None,
    sd_threshold_mm_yr: Optional[float] = None,
    *,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> float:
    """Symmetric color limit (mm/yr) shared by total vs regional-linear rate maps."""
    masked_total, masked_rl = _masked_total_and_regional_linear_rate(
        rate_result,
        interval_idx,
        rsl_data=rsl_data,
        sd_threshold_mm_yr=sd_threshold_mm_yr,
        mask_settings=mask_settings,
        cfg=cfg,
    )
    vabs = max(_vabs_from_masked(masked_total), _vabs_from_masked(masked_rl))
    return _floor_spatial_vabs(vabs)


def plot_spatial_total_vs_regional_linear_rate(
    rate_result: SpatialRateResult,
    interval_idx: int,
    rsl_data: Optional[RSLData] = None,
    sd_threshold_mm_yr: Optional[float] = None,
    vabs: Optional[float] = None,
    *,
    mask_settings: Optional[SpatialPlotMaskSettings] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> matplotlib.figure.Figure:
    """1x2 rate maps: total (GP+GIA) vs regional linear with a shared color scale."""
    import cartopy.crs as ccrs

    r = rate_result
    i = interval_idx
    n_intervals = len(r.rate_midpoints)
    if len(r.spatial_ages) > n_intervals + 1:
        younger = int(r.spatial_ages[0])
        older = int(r.spatial_ages[-1])
    else:
        younger = int(r.spatial_ages[i])
        older = int(r.spatial_ages[i + 1])
    midpt = r.rate_midpoints[i]

    masked_total, masked_rl = _masked_total_and_regional_linear_rate(
        r,
        i,
        rsl_data=rsl_data,
        sd_threshold_mm_yr=sd_threshold_mm_yr,
        mask_settings=mask_settings,
        cfg=cfg,
    )

    if vabs is None:
        vabs = spatial_total_vs_regional_linear_rate_vabs(
            r,
            i,
            rsl_data=rsl_data,
            sd_threshold_mm_yr=sd_threshold_mm_yr,
            mask_settings=mask_settings,
            cfg=cfg,
        )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(22, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im1 = ax1.pcolormesh(
        r.lon_mat,
        r.lat_mat,
        masked_total,
        cmap=RATE_CMAPS["total"],
        vmin=-vabs,
        vmax=vabs,
        transform=ccrs.PlateCarree(),
    )
    _add_spatial_basemap(ax1)
    plt.colorbar(im1, ax=ax1, label="mm/yr")
    ax1.set_title("Total (GP+GIA) rate")

    im2 = ax2.pcolormesh(
        r.lon_mat,
        r.lat_mat,
        masked_rl,
        cmap=RATE_CMAPS["regional_linear"],
        vmin=-vabs,
        vmax=vabs,
        transform=ccrs.PlateCarree(),
    )
    _add_spatial_basemap(ax2)
    plt.colorbar(im2, ax=ax2, label="mm/yr")
    ax2.set_title(f"{SPATIAL_COMP_TITLES['regional_linear']} rate")

    fig.suptitle(
        f"Total vs regional linear rate: {younger}--{older} BP "
        f"(midpoint {midpt:.0f} BP, shared scale ±{vabs:.2f} mm/yr)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
