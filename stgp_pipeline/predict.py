"""
All prediction workflows: temporal, spatial, and rate predictions.

**Math annotations:**

Total prediction at test point x*:
    y_total(x*) = y_GP_residual(x*) + g_bar_w(x*)
    sigma^2_total(x*) = sigma^2_GP(x*) + sigma^2_GIA(x*)

Rate uncertainty propagation (finite difference):
    sigma^2_dot_y = [sigma^2(t1) + sigma^2(t2) - 2*Cov(y(t1), y(t2))] / dt^2

Bessel-corrected GIA variance at test points:
    sigma^2_GIA(x) = [sum_k w_k * (g_k(x) - g_bar(x))^2]
                     / [sum(w) - sum(w^2)/sum(w)]

Cholesky precomputation:
    Lff = cholesky(Kff + noise + jitter)
    Reused across all grid points and time slices for efficiency.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pyro.contrib.gp.util import conditional

import PSTHM.post

from .types import (
    RSLData,
    GIAResult,
    TrainedModel,
    RegionResult,
    ProjectedRegionData,
    SpatialSlice,
    SpatialRateResult,
    RegionalMeanRateResult,
)
from . import gia_setup


def _gia_mean_kwargs(gia_result: GIAResult) -> Dict[str, Any]:
    """Keyword args for gia_setup_3d test-point mean (when using 3D ensemble)."""
    return {
        "mean_weighting": gia_result.mean_weighting,
        "mean_model_indices": gia_result.mean_model_indices,
    }


def build_spatial_grid(
    rsl_data: RSLData,
    cfg: Dict[str, Any],
    padding: float = 2.0,
    resolution: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a regular lat/lon grid around the data extent.

    Reads ``cfg["spatial_grid"]`` if present, falling back to the
    provided ``padding`` and ``resolution`` defaults.

    Returns
    -------
    lat_grid : 1-D array
    lon_grid : 1-D array
    lat_mat : 2-D meshgrid
    lon_mat : 2-D meshgrid
    """
    sg = cfg.get("spatial_grid", {}) or {}
    pad = sg.get("lat_padding", padding)
    lon_pad = sg.get("lon_padding", padding)
    res = sg.get("resolution", resolution)

    lat_grid = np.arange(
        np.min(rsl_data.rsl_lat) - pad, np.max(rsl_data.rsl_lat) + pad, res
    )
    lon_grid = np.arange(
        np.min(rsl_data.rsl_lon) - lon_pad, np.max(rsl_data.rsl_lon) + lon_pad, res
    )

    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)

    print(f"Spatial grid: {len(lat_grid)} lat x {len(lon_grid)} lon")
    return lat_grid, lon_grid, lat_mat, lon_mat


def _get_temporal_test_ages(cfg: Dict[str, Any], max_age_bp: int) -> np.ndarray:
    """Extract temporal test age array from config."""
    tc = cfg.get("temporal_test_age", {})
    test_age = np.arange(
        tc.get("start", 0),
        tc.get("end", max_age_bp) + 1,
        tc.get("interval", 200),
    )
    return test_age


def _region_centroid(
    rsl_lat: np.ndarray,
    rsl_lon: np.ndarray,
    mode: str,
    grid_n: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Return (lat_arr, lon_arr, is_grid) for a region under the chosen centroid mode.

    Modes:
      - ``data_mean`` (default, legacy): unweighted arithmetic mean of every
        observation's coordinate. Pulled toward data-dense corners.
      - ``bbox_center``: midpoint of the lat/lon bounding box. Pure geometry,
        independent of data density.
      - ``grid_mean``: ``grid_n`` x ``grid_n`` regular lat/lon grid spanning
        the bbox. Downstream code area-averages the predictions, removing
        both centroid bias and GP-kernel bias.

    Longitudes are normalized to [0, 360) before any min/max/mean, so regions
    that straddle 0 lon still work for grids defined in that convention.
    """
    lon_adj = np.asarray(rsl_lon) % 360
    lat = np.asarray(rsl_lat)

    if mode == "data_mean":
        return np.array([np.mean(lat)]), np.array([np.mean(lon_adj)]), False
    if mode == "bbox_center":
        lat_c = 0.5 * (np.min(lat) + np.max(lat))
        lon_c = 0.5 * (np.min(lon_adj) + np.max(lon_adj))
        return np.array([lat_c]), np.array([lon_c]), False
    if mode == "grid_mean":
        lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
        lon_min, lon_max = float(np.min(lon_adj)), float(np.max(lon_adj))
        # Degenerate bboxes (single site, or collinear) collapse to length-1 arrays
        n_lat = 1 if lat_min == lat_max else int(grid_n)
        n_lon = 1 if lon_min == lon_max else int(grid_n)
        lat_arr = (
            np.array([lat_min]) if n_lat == 1 else np.linspace(lat_min, lat_max, n_lat)
        )
        lon_arr = (
            np.array([lon_min]) if n_lon == 1 else np.linspace(lon_min, lon_max, n_lon)
        )
        return lat_arr, lon_arr, (n_lat * n_lon) > 1
    raise ValueError(
        f"Unknown temporal_centroid mode: {mode!r}. "
        "Expected one of: data_mean, bbox_center, grid_mean."
    )


def _tile_spatial_for_gia(
    test_age: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tile (age, lat, lon) arrays to match the row layout of
    ``PSTHM.post.gen_pred_matrix`` (age varies fastest, then lon, then lat).

    Returns flat arrays of length ``len(lat_arr) * len(lon_arr) * len(test_age)``.
    """
    n_a = len(test_age)
    n_la = len(lat_arr)
    n_lo = len(lon_arr)
    age_flat = np.tile(test_age, n_la * n_lo)
    lat_flat = np.repeat(lat_arr, n_lo * n_a)
    lon_flat = np.tile(np.repeat(lon_arr, n_a), n_la)
    return age_flat, lat_flat, lon_flat


def _area_average_mean(flat: np.ndarray, n_la: int, n_lo: int, n_a: int) -> np.ndarray:
    """Reshape a flat (n_la*n_lo*n_a,) prediction to (n_la, n_lo, n_a) and average spatially."""
    return np.asarray(flat).reshape(n_la, n_lo, n_a).mean(axis=(0, 1))


def _area_average_cov(
    flat_cov: np.ndarray, n_la: int, n_lo: int, n_a: int
) -> np.ndarray:
    """
    Area-average a (N, N) covariance (N = n_la*n_lo*n_a) to the covariance of
    the spatial mean: Cov(y_bar)[k1, k2] = mean_{i1,j1,i2,j2} Cov(y)[i1,j1,k1, i2,j2,k2].
    """
    cov6 = np.asarray(flat_cov).reshape(n_la, n_lo, n_a, n_la, n_lo, n_a)
    return cov6.mean(axis=(0, 1, 3, 4))


def _get_spatial_test_ages(cfg: Dict[str, Any], max_age_bp: int) -> np.ndarray:
    """Extract spatial test age array from config."""
    sc = cfg.get("spatial_test_age", {})
    spatial_ages = np.arange(
        sc.get("start", 0),
        sc.get("end", max_age_bp) + 1,
        sc.get("interval", 1000),
    )
    return spatial_ages


def _temporal_at_coords(
    gpr: Any,
    decomp_kernels: List[Any],
    comp_labels: List[str],
    interp_funcs: List[Any],
    weights: np.ndarray,
    indices_3d: Optional[np.ndarray],
    test_age: np.ndarray,
    temporal_interval: int,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    is_grid: bool,
    region_num: int,
    region_name: str,
    *,
    mean_weighting: str = "equal",
    mean_model_indices: Optional[np.ndarray] = None,
) -> RegionResult:
    """Compute temporal GP+GIA predictions at one or more spatial coordinates."""
    n_la, n_lo = len(lat_arr), len(lon_arr)
    n_t = len(test_age)

    pred_matrix = PSTHM.post.gen_pred_matrix(test_age, lat_arr, lon_arr)

    with torch.no_grad():
        y_mean_res, y_var_res = gpr(pred_matrix.double(), full_cov=True)

    age_flat, lat_flat, lon_flat = _tile_spatial_for_gia(test_age, lat_arr, lon_arr)
    if indices_3d is not None:
        from . import gia_setup_3d

        gia_ts_flat, gia_var_flat = gia_setup_3d.gia_at_test_points_3d(
            interp_funcs,
            weights,
            indices_3d,
            age_flat,
            lat_flat,
            lon_flat,
            mean_weighting=mean_weighting,
            mean_model_indices=mean_model_indices,
        )
        _, gia_cov_flat = gia_setup_3d.gia_covariance_at_test_points_3d(
            interp_funcs,
            weights,
            indices_3d,
            age_flat,
            lat_flat,
            lon_flat,
            mean_weighting=mean_weighting,
            mean_model_indices=mean_model_indices,
        )
    else:
        gia_ts_flat, gia_var_flat = gia_setup.gia_at_test_points(
            interp_funcs,
            weights,
            age_flat,
            lat_flat,
            lon_flat,
        )
        _, gia_cov_flat = gia_setup.gia_covariance_at_test_points(
            interp_funcs,
            weights,
            age_flat,
            lat_flat,
            lon_flat,
        )

    y_mean_res_np = y_mean_res.detach().numpy().flatten()
    y_var_res_np = y_var_res.cpu().detach().numpy()

    if is_grid:
        y_mean_res_avg = _area_average_mean(y_mean_res_np, n_la, n_lo, n_t)
        y_var_res_avg = _area_average_cov(y_var_res_np, n_la, n_lo, n_t)
        gia_ts = _area_average_mean(gia_ts_flat, n_la, n_lo, n_t)
        gia_cov_test = _area_average_cov(gia_cov_flat, n_la, n_lo, n_t)
        gia_var_test = np.diag(gia_cov_test)
    else:
        y_mean_res_avg = y_mean_res_np
        y_var_res_avg = y_var_res_np
        gia_ts = gia_ts_flat
        gia_var_test = gia_var_flat
        gia_cov_test = gia_cov_flat

    y_total = y_mean_res_avg + gia_ts
    gp_var = np.diag(y_var_res_avg)
    y_std = np.sqrt(gp_var + gia_var_test)

    _, gia_rate, gia_rate_sd = PSTHM.post.cal_rate_var(
        test_age, gia_cov_test, gia_ts, difftimestep=temporal_interval
    )

    total_cov = y_var_res_avg + gia_cov_test
    rsl_time, rsl_rate, rsl_rate_sd = PSTHM.post.cal_rate_var(
        test_age, total_cov, y_total, difftimestep=temporal_interval
    )

    decomp_results = PSTHM.post.decompose_kernels(gpr, pred_matrix, decomp_kernels)
    components = {}
    for label, (d_mean, d_cov) in zip(comp_labels, decomp_results):
        d_mean_np = d_mean.detach().numpy().flatten()
        d_cov_np = d_cov.detach().numpy()
        if is_grid:
            d_mean_np = _area_average_mean(d_mean_np, n_la, n_lo, n_t)
            d_cov_np = _area_average_cov(d_cov_np, n_la, n_lo, n_t)
        d_std_np = np.sqrt(np.diag(d_cov_np))
        _, d_rate, d_rate_sd = PSTHM.post.cal_rate_var(
            test_age, d_cov_np, d_mean_np, difftimestep=temporal_interval
        )
        components[label] = {
            "mean": d_mean_np,
            "std": d_std_np,
            "cov": d_cov_np,
            "rate": d_rate,
            "rate_sd": d_rate_sd,
        }

    return RegionResult(
        region_num=region_num,
        region_name=region_name,
        test_age=test_age,
        y_total=y_total,
        y_std=y_std,
        y_res_mean=y_mean_res_avg,
        gia_ts=gia_ts,
        gia_var=gia_var_test,
        gia_cov_test=gia_cov_test,
        rsl_time=rsl_time,
        rsl_rate=rsl_rate,
        rsl_rate_sd=rsl_rate_sd,
        gia_rate=gia_rate,
        gia_rate_sd=gia_rate_sd,
        pred_matrix=pred_matrix,
        y_var_res=torch.as_tensor(y_var_res_avg),
        components=components,
    )


def temporal_at_sites(
    trained: TrainedModel,
    gia_result: GIAResult,
    cfg: Dict[str, Any],
    sites: List[Tuple[str, float, float]],
    max_age_bp: int = 14000,
    bp_min: float = 0.0,
    bp_max: float = 21000.0,
) -> Dict[str, RegionResult]:
    """
    Compute temporal total-RSL predictions at exact site coordinates.

    Parameters
    ----------
    sites : list of (name, lat, lon)

    Returns
    -------
    dict : site name → RegionResult (no spatial averaging)
    """
    test_age = _get_temporal_test_ages(cfg, max_age_bp)
    test_age = test_age[(test_age >= bp_min) & (test_age <= bp_max)]
    temporal_interval = int(test_age[1] - test_age[0]) if len(test_age) > 1 else 200

    gpr = trained.gpr
    interp_funcs = gia_result.interp_funcs
    weights = gia_result.weights
    indices_3d = gia_result.indices_3d
    gia_mean_kw = _gia_mean_kwargs(gia_result)

    results: Dict[str, RegionResult] = {}
    for i, (name, lat, lon) in enumerate(sites):
        lat_arr = np.array([lat])
        lon_arr = np.array([lon % 360])
        results[name] = _temporal_at_coords(
            gpr,
            trained.decomp_kernels,
            trained.comp_labels,
            interp_funcs,
            weights,
            indices_3d,
            test_age,
            temporal_interval,
            lat_arr,
            lon_arr,
            is_grid=False,
            region_num=-(i + 1),
            region_name=name,
            **gia_mean_kw,
        )

    print(f"Computed temporal predictions for {len(results)} sites.")
    return results


def _apply_projection(
    y: np.ndarray, total_site: np.ndarray, total_ref: np.ndarray
) -> np.ndarray:
    """Project observed RSL to the reference frame: ``y - (Total_site - Total_ref)``.

    Pure arithmetic, isolated from the GP/GIA evaluation so the projection math
    can be unit-tested without a trained model. At the reference location
    ``total_site == total_ref`` so the correction is zero and ``projected == y``.
    """
    return y - (total_site - total_ref)


def project_region_to_reference(
    trained: TrainedModel,
    rsl_data: RSLData,
    gia_result: GIAResult,
    cfg: Dict[str, Any],
    ref_lat: float,
    ref_lon: float,
    *,
    region_num: Optional[int] = None,
    ref_name: str = "",
) -> ProjectedRegionData:
    """
    Project a region's SLIP data to a reference location using the trained GP+GIA.

    The GP generalization of the classic "isobase" projection. For each
    datum the model's full Total = GP residual + weighted GIA mean is evaluated
    at both the datum's own coordinate and the reference; the difference is
    subtracted from the observed value. This brings multi-site data into one
    reference frame so they can be plotted against the reference-location curve.

    The reference choice is the key knob: project to a highstand site (e.g. BWB)
    and the reference curve shows the highstand; project to a density-weighted
    centroid and it will not.

    Only the datum mean is shifted; each datum keeps its own sigma (the
    correction's own GP uncertainty is not propagated, matching standard isobase
    practice). ``cfg`` is accepted for signature parity with the other predict
    entry points (currently unused — data ages are used directly).

    Parameters
    ----------
    region_num : int, optional
        Restrict to one region (``rsl_region == region_num``); ``None`` uses all SLIPs.
    """
    if region_num is None:
        mask = np.ones(len(rsl_data.rsl_region), dtype=bool)
    else:
        mask = rsl_data.rsl_region == region_num

    # Datum coordinates straight from the training inputs (exact lon convention).
    X_np = rsl_data.X.detach().cpu().numpy()
    age = X_np[mask, 0]
    lat = X_np[mask, 1]
    lon = X_np[mask, 2]
    y = rsl_data.y.detach().cpu().numpy()[mask]
    y_sigma = rsl_data.y_sigma.detach().cpu().numpy()[mask]
    age_sigma = rsl_data.age_sigma.detach().cpu().numpy()[mask]

    # GP residual mean at each datum and at the reference, both at the datum's age.
    X_site_t = torch.as_tensor(X_np[mask]).double()
    X_ref_t = torch.as_tensor(
        np.column_stack([age, np.full(age.shape, ref_lat), np.full(age.shape, ref_lon)])
    ).double()
    gpr = trained.gpr
    with torch.no_grad():
        gp_site_mean, _ = gpr(X_site_t, full_cov=False)
        gp_ref_mean, _ = gpr(X_ref_t, full_cov=False)
    gp_site = gp_site_mean.detach().cpu().numpy().flatten()
    gp_ref = gp_ref_mean.detach().cpu().numpy().flatten()

    # GIA weighted mean at each datum and at the reference (lon %360 handled inside).
    interp_funcs = gia_result.interp_funcs
    weights = gia_result.weights
    indices_3d = gia_result.indices_3d
    ref_lat_arr = np.full(age.shape, ref_lat)
    ref_lon_arr = np.full(age.shape, ref_lon)
    if indices_3d is not None:
        from . import gia_setup_3d

        gia_mean_kw = _gia_mean_kwargs(gia_result)
        gia_site, _ = gia_setup_3d.gia_at_test_points_3d(
            interp_funcs, weights, indices_3d, age, lat, lon, **gia_mean_kw
        )
        gia_ref, _ = gia_setup_3d.gia_at_test_points_3d(
            interp_funcs,
            weights,
            indices_3d,
            age,
            ref_lat_arr,
            ref_lon_arr,
            **gia_mean_kw,
        )
    else:
        gia_site, _ = gia_setup.gia_at_test_points(interp_funcs, weights, age, lat, lon)
        gia_ref, _ = gia_setup.gia_at_test_points(
            interp_funcs, weights, age, ref_lat_arr, ref_lon_arr
        )

    total_site = gp_site + gia_site
    total_ref = gp_ref + gia_ref
    correction = total_site - total_ref
    projected_y = _apply_projection(y, total_site, total_ref)

    region_tag = f" (region {region_num})" if region_num is not None else ""
    print(
        f"Projected {int(mask.sum())} SLIP{region_tag} to "
        f"{ref_name or 'reference'} ({ref_lat:.4f}, {ref_lon:.4f})."
    )

    return ProjectedRegionData(
        region_num=region_num,
        ref_name=ref_name,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        age=age,
        orig_y=y,
        projected_y=projected_y,
        y_sigma=y_sigma,
        age_sigma=age_sigma,
        correction=correction,
    )


def temporal_all_regions(
    trained: TrainedModel,
    rsl_data: RSLData,
    gia_result: GIAResult,
    cfg: Dict[str, Any],
    max_age_bp: int = 14000,
    bp_min: float = 0.0,
    bp_max: float = 21000.0,
) -> Dict[int, RegionResult]:
    """
    Compute temporal predictions for all regions.

    For each region, predicts at the region centroid, adds GIA weighted mean,
    computes total uncertainty (GP + GIA), and decomposes by kernel component.

    Returns
    -------
    dict : region_num → RegionResult
    """
    test_age = _get_temporal_test_ages(cfg, max_age_bp)
    test_age = test_age[(test_age >= bp_min) & (test_age <= bp_max)]

    unique_regions = rsl_data.unique_regions
    gpr = trained.gpr
    decomp_kernels = trained.decomp_kernels
    comp_labels = trained.comp_labels
    interp_funcs = gia_result.interp_funcs
    weights = gia_result.weights
    indices_3d = gia_result.indices_3d
    gia_mean_kw = _gia_mean_kwargs(gia_result)

    # Regional centroid / averaging mode (opt-in via cfg). Default preserves
    # legacy behavior (data_mean = density-weighted arithmetic mean).
    centroid_cfg = cfg.get("temporal_centroid", {}) or {}
    centroid_mode = centroid_cfg.get("mode", "data_mean")
    grid_n = int(centroid_cfg.get("grid_n", 6))

    # Derive difftimestep from the test age interval (not hardcoded default)
    temporal_interval = int(test_age[1] - test_age[0]) if len(test_age) > 1 else 200

    print(
        f"Test age: {test_age[0]} – {test_age[-1]} BP, {len(test_age)} points, dt={temporal_interval}"
    )
    print(f"Regions to predict: {unique_regions}")
    if centroid_mode == "grid_mean":
        print(f"Regional temporal centroid: grid_mean ({grid_n}x{grid_n})")
    else:
        print(f"Regional temporal centroid: {centroid_mode}")

    results = {}

    for rn in unique_regions:
        region_mask = rsl_data.rsl_region == rn
        lat_arr, lon_arr, is_grid = _region_centroid(
            rsl_data.rsl_lat[region_mask],
            rsl_data.rsl_lon[region_mask],
            centroid_mode,
            grid_n,
        )
        name = rsl_data.region_num_to_name.get(int(rn), f"Region {rn}")
        results[rn] = _temporal_at_coords(
            gpr,
            decomp_kernels,
            comp_labels,
            interp_funcs,
            weights,
            indices_3d,
            test_age,
            temporal_interval,
            lat_arr,
            lon_arr,
            is_grid,
            int(rn),
            name,
            **gia_mean_kw,
        )

    print(f"Computed temporal predictions for {len(results)} regions.")
    return results


def _precompute_cholesky(gpr: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute Cholesky factor Lff and centered observations for reuse
    across spatial predictions.

    Returns
    -------
    Lff : Tensor (N, N)
    y_res_cached : Tensor (N,)
    """
    with torch.no_grad():
        N = gpr.X.size(0)
        Kff = gpr.kernel(gpr.X).contiguous()
        if gpr.noise.dim() <= 1:
            Kff.view(-1)[:: N + 1] += gpr.jitter + gpr.noise
        elif gpr.noise.dim() == 2:
            Kff = Kff + gpr.noise
            Kff.view(-1)[:: N + 1] += gpr.jitter
        Lff = torch.linalg.cholesky(Kff)
        y_res_cached = gpr.y - gpr.mean_function(gpr.X)
    return Lff, y_res_cached


def spatial_all_ages(
    trained: TrainedModel,
    rsl_data: RSLData,
    gia_result: GIAResult,
    cfg: Dict[str, Any],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    max_age_bp: int = 14000,
    bp_min: float = 0.0,
    bp_max: float = 21000.0,
    ages_bp: Optional[Sequence[float]] = None,
) -> List[SpatialSlice]:
    """
    Compute spatial RSL maps at selected time slices.

    Precomputes Cholesky factor Lff once and reuses across all ages.

    Parameters
    ----------
    ages_bp : sequence of float, optional
        Explicit BP ages to evaluate. When provided, overrides
        ``spatial_test_age`` from config.

    Returns
    -------
    list of SpatialSlice
    """
    if ages_bp is not None:
        spatial_ages = np.asarray(ages_bp, dtype=float)
    else:
        spatial_ages = _get_spatial_test_ages(cfg, max_age_bp)
    spatial_ages = spatial_ages[(spatial_ages >= bp_min) & (spatial_ages <= bp_max)]

    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    n_grid = lat_mat.size
    lat_flat = lat_mat.flatten()
    lon_flat = lon_mat.flatten() % 360

    gpr = trained.gpr
    interp_funcs = gia_result.interp_funcs
    weights = gia_result.weights
    indices_3d = gia_result.indices_3d
    gia_mean_kw = _gia_mean_kwargs(gia_result)
    M_models = len(interp_funcs)
    decomp_kernels = trained.decomp_kernels
    comp_labels = trained.comp_labels

    Lff, y_res_cached = _precompute_cholesky(gpr)

    print(f"Spatial ages: {spatial_ages}")
    slices = []

    for age_bp in spatial_ages:
        pred_matrix_sp = PSTHM.post.gen_pred_matrix(
            np.array([age_bp]), lat_grid, lon_grid
        )

        with torch.no_grad():
            loc, var = conditional(
                pred_matrix_sp,
                gpr.X,
                gpr.kernel,
                y_res_cached,
                None,
                Lff,
                full_cov=False,
                jitter=gpr.jitter,
            )
            y_mean_sp = loc + gpr.mean_function(pred_matrix_sp)

        gp_var_np = var.detach().cpu().numpy().reshape(lat_mat.shape)

        # GIA at every grid point
        if indices_3d is not None:
            from . import gia_setup_3d

            gia_mean_sp, gia_var_sp = gia_setup_3d.gia_at_test_points_3d(
                interp_funcs,
                weights,
                indices_3d,
                np.full(n_grid, age_bp),
                lat_flat,
                lon_flat,
                **gia_mean_kw,
            )
        else:
            gia_mean_sp, gia_var_sp = gia_setup.gia_at_test_points(
                interp_funcs,
                weights,
                np.full(n_grid, age_bp),
                lat_flat,
                lon_flat,
            )
        gia_mean_2d = gia_mean_sp.reshape(lat_mat.shape)
        gia_var_2d = gia_var_sp.reshape(lat_mat.shape)

        y_mean_np = y_mean_sp.detach().cpu().numpy().reshape(lat_mat.shape)
        total_mean = y_mean_np + gia_mean_2d
        total_std = np.sqrt(gp_var_np + gia_var_2d)

        # Kernel decomposition — diagonal-only path to avoid the O(M^2)
        # allocation inside PSTHM.post.decompose_kernels. Algebraically
        # identical to calling decompose_kernels() and taking .diag() on
        # the returned covariance, but never materializes the full M x M
        # tensors.
        #
        #   d_mean      = W @ v_sol
        #   diag(d_cov) = diag(Kss) - diag(W Wᵀ) = diag(Kss) - sum_k W[:, k]^2
        N_train = gpr.X.size(0)
        M_pred = pred_matrix_sp.size(0)
        f_loc_2D = y_res_cached.reshape(N_train, 1)

        with torch.no_grad():
            comp_data = {}
            for label, kernel in zip(comp_labels, decomp_kernels):
                Kfs = kernel(gpr.X, pred_matrix_sp)
                pack = torch.cat((f_loc_2D, Kfs), dim=1)
                Lffinv_pack = pack.triangular_solve(Lff, upper=False)[0]
                v_sol = Lffinv_pack[:, :1]
                W_t = Lffinv_pack[:, 1:]
                d_mean = (W_t.t() @ v_sol).reshape(M_pred)
                Qss_diag = (W_t * W_t).sum(dim=0)
                Kss_diag = kernel(pred_matrix_sp, diag=True)
                d_cov_diag = Kss_diag - Qss_diag

                del Kfs, pack, Lffinv_pack, W_t, v_sol, Qss_diag, Kss_diag

                # clamp_min(0) guards against tiny negative roundoff on the
                # diagonal before sqrt (same guard the GP path implicitly relies on).
                comp_data[label] = {
                    "mean": d_mean.detach().numpy().reshape(lat_mat.shape),
                    "std": d_cov_diag.clamp_min(0)
                    .sqrt()
                    .detach()
                    .numpy()
                    .reshape(lat_mat.shape),
                }

        slices.append(
            SpatialSlice(
                age_bp=float(age_bp),
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                lat_mat=lat_mat,
                lon_mat=lon_mat,
                total_mean=total_mean,
                total_std=total_std,
                gp_mean=y_mean_np,
                gp_var=gp_var_np,
                gia_mean=gia_mean_2d,
                gia_var=gia_var_2d,
                components=comp_data,
            )
        )

    print(f"Generated {len(slices)} spatial slices.")
    return slices


def spatial_rates(
    trained: TrainedModel,
    rsl_data: RSLData,
    gia_result: GIAResult,
    cfg: Dict[str, Any],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    max_age_bp: int = 14000,
    bp_min: float = 0.0,
    bp_max: float = 21000.0,
    ages_bp: Optional[Sequence[float]] = None,
) -> SpatialRateResult:
    """
    Compute spatial rate maps using full temporal covariance at each grid point.

    Precomputes Lff once. For each grid point, computes the full temporal
    covariance across all ages, then applies ``cal_rate_var()`` for GP, GIA,
    total, and per-component rates.

    Uses manual ``triangular_solve`` per kernel (matching the notebook's
    cell 35 optimization) instead of calling ``decompose_kernels()``
    per grid point.

    Parameters
    ----------
    ages_bp : sequence of float, optional
        Explicit BP ages defining consecutive rate windows. When provided,
        overrides ``spatial_test_age`` from config. Rates are formed between
        consecutive ages (``n_ages - 1`` intervals).

    Returns
    -------
    SpatialRateResult
    """
    if ages_bp is not None:
        spatial_ages = np.asarray(ages_bp, dtype=float)
    else:
        spatial_ages = _get_spatial_test_ages(cfg, max_age_bp)
    spatial_ages = spatial_ages[(spatial_ages >= bp_min) & (spatial_ages <= bp_max)]
    if spatial_ages.size < 2:
        raise ValueError(
            "spatial_rates needs at least two ages to form rate intervals; "
            f"got {spatial_ages.size}"
        )

    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    n_grid = lat_mat.size
    lat_flat = lat_mat.flatten()
    lon_flat = lon_mat.flatten() % 360

    n_ages = len(spatial_ages)
    n_intervals = n_ages - 1
    spatial_interval = int(spatial_ages[1] - spatial_ages[0])
    rate_midpoints = (spatial_ages[:-1] + spatial_ages[1:]) / 2.0

    gpr = trained.gpr
    interp_funcs = gia_result.interp_funcs
    weights = gia_result.weights
    indices_3d = gia_result.indices_3d
    gia_mean_kw = _gia_mean_kwargs(gia_result)
    M_models = len(interp_funcs)
    decomp_kernels = trained.decomp_kernels
    comp_labels = trained.comp_labels
    N_train = gpr.X.size(0)

    Lff, y_res_cached = _precompute_cholesky(gpr)
    f_loc_2D = y_res_cached.reshape(N_train, 1)

    # Output arrays
    rate_gp = np.zeros((n_intervals, n_grid))
    sd_gp = np.zeros((n_intervals, n_grid))
    rate_total = np.zeros((n_intervals, n_grid))
    sd_total = np.zeros((n_intervals, n_grid))
    rate_gia = np.zeros((n_intervals, n_grid))
    sd_gia = np.zeros((n_intervals, n_grid))
    rate_comp = {l: np.zeros((n_intervals, n_grid)) for l in comp_labels}
    sd_comp = {l: np.zeros((n_intervals, n_grid)) for l in comp_labels}

    t_start = time.time()

    for idx in range(n_grid):
        pred_pt = PSTHM.post.gen_pred_matrix(
            spatial_ages, np.array([lat_flat[idx]]), np.array([lon_flat[idx]])
        )
        M_pred = pred_pt.size(0)

        with torch.no_grad():
            # GP full covariance at this grid point across all ages
            loc, cov = conditional(
                pred_pt,
                gpr.X,
                gpr.kernel,
                y_res_cached,
                None,
                Lff,
                full_cov=True,
                jitter=gpr.jitter,
            )
            gp_mean = (loc + gpr.mean_function(pred_pt)).detach().cpu().numpy()
            gp_cov = cov.detach().cpu().numpy()

            # Per-kernel decomposition via manual triangular_solve
            comp_covs = {}
            comp_means = {}
            for label, kernel in zip(comp_labels, decomp_kernels):
                Kfs = kernel(gpr.X, pred_pt)
                pack = torch.cat((f_loc_2D, Kfs), dim=1)
                Lffinv_pack = pack.triangular_solve(Lff, upper=False)[0]
                W = Lffinv_pack[:, 1 : 1 + M_pred].t()
                Qss = W.matmul(W.t())
                v_sol = Lffinv_pack[:, :1]
                d_loc = W.matmul(v_sol).t().reshape(M_pred)
                Kss = kernel(pred_pt)
                d_cov = Kss - Qss
                comp_means[label] = d_loc.detach().numpy()
                comp_covs[label] = d_cov.detach().numpy()

        # GIA covariance at this point across all ages
        if indices_3d is not None:
            from . import gia_setup_3d

            gia_mean_pt, gia_cov_pt = gia_setup_3d.gia_covariance_at_test_points_3d(
                interp_funcs,
                weights,
                indices_3d,
                spatial_ages,
                np.full(n_ages, lat_flat[idx]),
                np.full(n_ages, lon_flat[idx]),
                **gia_mean_kw,
            )
        else:
            gia_mean_pt, gia_cov_pt = gia_setup.gia_covariance_at_test_points(
                interp_funcs,
                weights,
                spatial_ages,
                np.full(n_ages, lat_flat[idx]),
                np.full(n_ages, lon_flat[idx]),
            )

        # GP-only rates
        _, r_gp, s_gp = PSTHM.post.cal_rate_var(
            spatial_ages, gp_cov, gp_mean, difftimestep=spatial_interval
        )
        rate_gp[:, idx] = r_gp
        sd_gp[:, idx] = s_gp

        # Total rates (GP + GIA)
        _, r_tot, s_tot = PSTHM.post.cal_rate_var(
            spatial_ages,
            gp_cov + gia_cov_pt,
            gp_mean + gia_mean_pt,
            difftimestep=spatial_interval,
        )
        rate_total[:, idx] = r_tot
        sd_total[:, idx] = s_tot

        # GIA-only rates
        _, r_g, s_g = PSTHM.post.cal_rate_var(
            spatial_ages,
            gia_cov_pt,
            gia_mean_pt,
            difftimestep=spatial_interval,
        )
        rate_gia[:, idx] = r_g
        sd_gia[:, idx] = s_g

        # Per-component rates
        for label in comp_labels:
            _, d_r, d_sd = PSTHM.post.cal_rate_var(
                spatial_ages,
                comp_covs[label],
                comp_means[label],
                difftimestep=spatial_interval,
            )
            rate_comp[label][:, idx] = d_r
            sd_comp[label][:, idx] = d_sd

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            print(f"  {idx + 1}/{n_grid} ({elapsed:.0f}s)")

    elapsed = time.time() - t_start
    print(f"Spatial rates: {n_grid} grid points in {elapsed:.1f}s")

    return SpatialRateResult(
        spatial_ages=spatial_ages,
        rate_midpoints=rate_midpoints,
        lat_flat=lat_flat,
        lon_flat=lon_flat,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        rate_total=rate_total,
        sd_total=sd_total,
        rate_gp=rate_gp,
        sd_gp=sd_gp,
        rate_gia=rate_gia,
        sd_gia=sd_gia,
        rate_comp=rate_comp,
        sd_comp=sd_comp,
    )


def spatial_rate_interval(
    trained: TrainedModel,
    rsl_data: RSLData,
    gia_result: GIAResult,
    cfg: Dict[str, Any],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    younger_bp: int,
    older_bp: int,
) -> SpatialRateResult:
    """
    Compute spatial rate maps for one custom averaging window.

    Uses ``difftimestep = older_bp - younger_bp`` (e.g. 2000 yr for
    1000--3000 BP with midpoint 2000 BP).
    """
    if older_bp <= younger_bp:
        raise ValueError(
            f"older_bp must exceed younger_bp, got {younger_bp} and {older_bp}"
        )

    midpoint = (younger_bp + older_bp) / 2.0
    spatial_ages = np.array([younger_bp, midpoint, older_bp], dtype=float)
    difftimestep = int(older_bp - younger_bp)
    n_intervals = 1
    rate_midpoints = np.array([midpoint])

    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    n_grid = lat_mat.size
    lat_flat = lat_mat.flatten()
    lon_flat = lon_mat.flatten() % 360
    n_ages = len(spatial_ages)

    gpr = trained.gpr
    interp_funcs = gia_result.interp_funcs
    weights = gia_result.weights
    indices_3d = gia_result.indices_3d
    gia_mean_kw = _gia_mean_kwargs(gia_result)
    decomp_kernels = trained.decomp_kernels
    comp_labels = trained.comp_labels
    N_train = gpr.X.size(0)

    Lff, y_res_cached = _precompute_cholesky(gpr)
    f_loc_2D = y_res_cached.reshape(N_train, 1)

    rate_gp = np.zeros((n_intervals, n_grid))
    sd_gp = np.zeros((n_intervals, n_grid))
    rate_total = np.zeros((n_intervals, n_grid))
    sd_total = np.zeros((n_intervals, n_grid))
    rate_gia = np.zeros((n_intervals, n_grid))
    sd_gia = np.zeros((n_intervals, n_grid))
    rate_comp = {l: np.zeros((n_intervals, n_grid)) for l in comp_labels}
    sd_comp = {l: np.zeros((n_intervals, n_grid)) for l in comp_labels}

    t_start = time.time()

    for idx in range(n_grid):
        pred_pt = PSTHM.post.gen_pred_matrix(
            spatial_ages, np.array([lat_flat[idx]]), np.array([lon_flat[idx]])
        )
        M_pred = pred_pt.size(0)

        with torch.no_grad():
            loc, cov = conditional(
                pred_pt,
                gpr.X,
                gpr.kernel,
                y_res_cached,
                None,
                Lff,
                full_cov=True,
                jitter=gpr.jitter,
            )
            gp_mean = (loc + gpr.mean_function(pred_pt)).detach().cpu().numpy()
            gp_cov = cov.detach().cpu().numpy()

            comp_covs = {}
            comp_means = {}
            for label, kernel in zip(comp_labels, decomp_kernels):
                Kfs = kernel(gpr.X, pred_pt)
                pack = torch.cat((f_loc_2D, Kfs), dim=1)
                Lffinv_pack = pack.triangular_solve(Lff, upper=False)[0]
                W = Lffinv_pack[:, 1 : 1 + M_pred].t()
                Qss = W.matmul(W.t())
                v_sol = Lffinv_pack[:, :1]
                d_loc = W.matmul(v_sol).t().reshape(M_pred)
                Kss = kernel(pred_pt)
                d_cov = Kss - Qss
                comp_means[label] = d_loc.detach().numpy()
                comp_covs[label] = d_cov.detach().numpy()

        if indices_3d is not None:
            from . import gia_setup_3d

            gia_mean_pt, gia_cov_pt = gia_setup_3d.gia_covariance_at_test_points_3d(
                interp_funcs,
                weights,
                indices_3d,
                spatial_ages,
                np.full(n_ages, lat_flat[idx]),
                np.full(n_ages, lon_flat[idx]),
                **gia_mean_kw,
            )
        else:
            gia_mean_pt, gia_cov_pt = gia_setup.gia_covariance_at_test_points(
                interp_funcs,
                weights,
                spatial_ages,
                np.full(n_ages, lat_flat[idx]),
                np.full(n_ages, lon_flat[idx]),
            )

        _, r_gp, s_gp = PSTHM.post.cal_rate_var(
            spatial_ages, gp_cov, gp_mean, difftimestep=difftimestep
        )
        rate_gp[:, idx] = r_gp
        sd_gp[:, idx] = s_gp

        _, r_tot, s_tot = PSTHM.post.cal_rate_var(
            spatial_ages,
            gp_cov + gia_cov_pt,
            gp_mean + gia_mean_pt,
            difftimestep=difftimestep,
        )
        rate_total[:, idx] = r_tot
        sd_total[:, idx] = s_tot

        _, r_g, s_g = PSTHM.post.cal_rate_var(
            spatial_ages,
            gia_cov_pt,
            gia_mean_pt,
            difftimestep=difftimestep,
        )
        rate_gia[:, idx] = r_g
        sd_gia[:, idx] = s_g

        for label in comp_labels:
            _, d_r, d_sd = PSTHM.post.cal_rate_var(
                spatial_ages,
                comp_covs[label],
                comp_means[label],
                difftimestep=difftimestep,
            )
            rate_comp[label][:, idx] = d_r
            sd_comp[label][:, idx] = d_sd

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            print(f"  {idx + 1}/{n_grid} ({elapsed:.0f}s)")

    elapsed = time.time() - t_start
    print(
        f"Spatial rate ({younger_bp}--{older_bp} BP): "
        f"{n_grid} grid points in {elapsed:.1f}s"
    )

    return SpatialRateResult(
        spatial_ages=spatial_ages,
        rate_midpoints=rate_midpoints,
        lat_flat=lat_flat,
        lon_flat=lon_flat,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        rate_total=rate_total,
        sd_total=sd_total,
        rate_gp=rate_gp,
        sd_gp=sd_gp,
        rate_gia=rate_gia,
        sd_gia=sd_gia,
        rate_comp=rate_comp,
        sd_comp=sd_comp,
    )


def _paired_location_matrix(
    age_bp: float,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Prediction matrix for paired coordinates, without a lat/lon cross product."""
    values = np.column_stack(
        [
            np.full(len(lat), float(age_bp)),
            np.asarray(lat, dtype=float),
            np.asarray(lon, dtype=float) % 360.0,
        ]
    )
    return torch.as_tensor(
        values,
        dtype=reference.dtype,
        device=reference.device,
    )


def _weighted_kernel_prior_mean(
    kernel: Any,
    left: torch.Tensor,
    right: torch.Tensor,
    left_weights: torch.Tensor,
    right_weights: torch.Tensor,
    *,
    block_size: int,
) -> float:
    """Return ``left_weights.T @ K(left, right) @ right_weights`` in blocks."""
    total = torch.zeros((), dtype=left.dtype, device=left.device)
    with torch.no_grad():
        for i0 in range(0, left.size(0), block_size):
            i1 = min(i0 + block_size, left.size(0))
            wi = left_weights[i0:i1]
            for j0 in range(0, right.size(0), block_size):
                j1 = min(j0 + block_size, right.size(0))
                wj = right_weights[j0:j1]
                total = total + wi @ kernel(left[i0:i1], right[j0:j1]) @ wj
    return float(total.detach().cpu())


def _aggregate_component_posterior(
    gpr: Any,
    kernel: Any,
    Lff: torch.Tensor,
    y_res_cached: torch.Tensor,
    ages_bp: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    spatial_weights: np.ndarray,
    *,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior mean/covariance of an equal-location weighted kernel component."""
    pred_by_age = [
        _paired_location_matrix(age, lat, lon, reference=gpr.X)
        for age in ages_bp
    ]
    weights_t = torch.as_tensor(
        spatial_weights,
        dtype=gpr.X.dtype,
        device=gpr.X.device,
    )
    v_sol = torch.linalg.solve_triangular(
        Lff,
        y_res_cached.reshape(-1, 1),
        upper=False,
    ).reshape(-1)

    whitened_cross_means: List[torch.Tensor] = []
    component_mean = np.zeros(len(ages_bp), dtype=np.float64)
    with torch.no_grad():
        for age_index, pred_age in enumerate(pred_by_age):
            cross_mean = torch.zeros(
                gpr.X.size(0),
                dtype=gpr.X.dtype,
                device=gpr.X.device,
            )
            for j0 in range(0, pred_age.size(0), block_size):
                j1 = min(j0 + block_size, pred_age.size(0))
                cross_mean = cross_mean + (
                    kernel(gpr.X, pred_age[j0:j1]) @ weights_t[j0:j1]
                )
            whitened = torch.linalg.solve_triangular(
                Lff,
                cross_mean.reshape(-1, 1),
                upper=False,
            ).reshape(-1)
            whitened_cross_means.append(whitened)
            component_mean[age_index] = float(
                (whitened @ v_sol).detach().cpu()
            )

    component_cov = np.zeros((len(ages_bp), len(ages_bp)), dtype=np.float64)
    for i, left in enumerate(pred_by_age):
        for j in range(i, len(pred_by_age)):
            prior = _weighted_kernel_prior_mean(
                kernel,
                left,
                pred_by_age[j],
                weights_t,
                weights_t,
                block_size=block_size,
            )
            correction = float(
                (whitened_cross_means[i] @ whitened_cross_means[j])
                .detach()
                .cpu()
            )
            value = prior - correction
            component_cov[i, j] = value
            component_cov[j, i] = value
    return component_mean, component_cov


def regional_mean_rate_interval(
    trained: TrainedModel,
    rsl_data: RSLData,
    gia_result: GIAResult,
    cfg: Dict[str, Any],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    younger_bp: int,
    older_bp: int,
    *,
    component_label: str = "regional_linear",
    block_size: int = 256,
) -> List[RegionalMeanRateResult]:
    """Equal-grid-cell regional mean rates with full spatial covariance.

    Every spatial grid cell is assigned to its nearest SLIP region. Means and
    covariance are spatially averaged before applying the 0--4 ka finite
    difference, yielding the uncertainty of the regional mean rather than a
    summary of pointwise uncertainties.
    """
    if older_bp <= younger_bp:
        raise ValueError(
            f"older_bp must exceed younger_bp, got {younger_bp} and {older_bp}"
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if component_label not in trained.comp_labels:
        raise KeyError(
            f"{component_label!r} is not in model components {trained.comp_labels}"
        )

    from . import spatial_summaries

    midpoint_bp = 0.5 * (younger_bp + older_bp)
    ages_bp = np.array([younger_bp, midpoint_bp, older_bp], dtype=np.float64)
    difftimestep = int(older_bp - younger_bp)

    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    lat_flat = lat_mat.ravel()
    lon_flat = lon_mat.ravel() % 360.0
    assigned = spatial_summaries.assign_grid_to_nearest_slip_region(
        lat_flat,
        lon_flat,
        rsl_data.rsl_lat,
        rsl_data.rsl_lon,
        rsl_data.rsl_region,
    )

    component_index = trained.comp_labels.index(component_label)
    component_kernel = trained.decomp_kernels[component_index]
    Lff, y_res_cached = _precompute_cholesky(trained.gpr)
    results: List[RegionalMeanRateResult] = []

    for region_num in rsl_data.unique_regions:
        region_mask = assigned == int(region_num)
        region_lat = lat_flat[region_mask]
        region_lon = lon_flat[region_mask]
        n_pixels = int(region_lat.size)
        if n_pixels == 0:
            continue
        spatial_weights = np.full(
            n_pixels,
            1.0 / n_pixels,
            dtype=np.float64,
        )

        if gia_result.indices_3d is not None:
            from . import gia_setup_3d

            gia_mean, gia_cov = (
                gia_setup_3d.gia_spatial_mean_covariance_at_ages_3d(
                    gia_result.interp_funcs,
                    gia_result.weights,
                    gia_result.indices_3d,
                    ages_bp,
                    region_lat,
                    region_lon,
                    spatial_weights=spatial_weights,
                    **_gia_mean_kwargs(gia_result),
                )
            )
        else:
            gia_mean, gia_cov = gia_setup.gia_spatial_mean_covariance_at_ages(
                gia_result.interp_funcs,
                gia_result.weights,
                ages_bp,
                region_lat,
                region_lon,
                spatial_weights=spatial_weights,
            )

        component_mean, component_cov = _aggregate_component_posterior(
            trained.gpr,
            component_kernel,
            Lff,
            y_res_cached,
            ages_bp,
            region_lat,
            region_lon,
            spatial_weights,
            block_size=block_size,
        )

        _, gia_rate, gia_rate_sd = PSTHM.post.cal_rate_var(
            ages_bp,
            gia_cov,
            gia_mean,
            difftimestep=difftimestep,
        )
        _, component_rate, component_rate_sd = PSTHM.post.cal_rate_var(
            ages_bp,
            component_cov,
            component_mean,
            difftimestep=difftimestep,
        )
        _, combined_rate, combined_rate_sd = PSTHM.post.cal_rate_var(
            ages_bp,
            gia_cov + component_cov,
            gia_mean + component_mean,
            difftimestep=difftimestep,
        )

        results.append(
            RegionalMeanRateResult(
                region_num=int(region_num),
                region_name=rsl_data.region_num_to_name.get(
                    int(region_num), f"Region {region_num}"
                ),
                n_pixels=n_pixels,
                mask_mode="nearest_slip",
                weighting="equal_grid_cell",
                younger_bp=float(younger_bp),
                older_bp=float(older_bp),
                midpoint_bp=float(midpoint_bp),
                gia_rate=float(gia_rate[0]),
                gia_rate_sd=float(gia_rate_sd[0]),
                component_rates={component_label: float(component_rate[0])},
                component_rate_sds={
                    component_label: float(component_rate_sd[0])
                },
                combined_rate=float(combined_rate[0]),
                combined_rate_sd=float(combined_rate_sd[0]),
            )
        )

    return results
