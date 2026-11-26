"""
GIA ensemble loading, weighting, covariance, and residual computation.

**Math annotations:**

WRSS/erf formula (Creel et al. 2024; Briggs & Tarasov 2013):

For each limiting observation n (flag c_n = -1 marine, +1 terrestrial) and
GIA model m, search a 2-sigma age window for the minimum weighted residual:

    d_nm = min_{t_j in [a_n - 2*sigma_a, a_n + 2*sigma_a]}
           [ (2(t_j - a_n)/sigma_a)^2
             - 2 ln( 0.5 + 0.5 erf( c_n * (y_n - g_m(t_j, phi_n, lambda_n)) / sigma_y ) ) ]

Aggregate misfit:  D_m = sum_n d_nm
Inverse weights:   w_m = (1/D_m) / sum_k (1/D_k)

Bessel-corrected weighted ensemble covariance:

    Sigma^GIA_ij = [ sum_k w_k * delta_g_k(x_i) * delta_g_k(x_j) ]
                   / [ sum(w) - sum(w^2)/sum(w) ]

where delta_g_k = g_k - g_weighted_mean.

Noise assembly:  Sigma_total = diag(sigma^2_obs) + Sigma^GIA
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from gia_covariance_tools import (
    load_mat_gia_ensemble,
    compute_wrss_weights_continuous,
    extract_gia_covariance,
)

from .types import RSLData, GIAResult, ResidualData


def load_gia_ensemble(
    gia_mat_dir: str,
    exclude_models: Optional[List[str]] = None,
) -> Tuple[List[Any], List[str]]:
    """
    Load continuous GIA model grids (MAT format) and build interpolators.

    Returns
    -------
    interp_funcs : list of RegularGridInterpolator
    model_names : list of str
    """
    if exclude_models is None:
        exclude_models = []

    interp_funcs, model_names = load_mat_gia_ensemble(
        str(gia_mat_dir),
        exclude_models=exclude_models,
    )

    print(f"Loaded {len(interp_funcs)} continuous GIA models:")
    for name in model_names:
        print(f"  {name}")

    return interp_funcs, model_names


def compute_weights_and_covariance(
    interp_funcs: List[Any],
    model_names: List[str],
    rsl_data: RSLData,
    cfg: Dict[str, Any],
    bp_min: float = 0.0,
    bp_max: float = 21000.0,
) -> GIAResult:
    """
    Compute WRSS/erf model weights from limiting data and ensemble covariance
    at SLIP observation points.

    Parameters
    ----------
    interp_funcs : list
        GIA model interpolators.
    model_names : list of str
        Model identifiers.
    rsl_data : RSLData
        Filtered RSL data with limiting arrays.
    cfg : dict
        Full YAML config dict (reads gia.n_search).
    bp_min, bp_max : float
        GIA time coverage bounds.

    Returns
    -------
    GIAResult
    """
    n_search = int(cfg.get("gia", {}).get("n_search", 50))

    weights, misfits = compute_wrss_weights_continuous(
        interp_funcs,
        rsl_data.lim_age,
        rsl_data.lim_lat,
        rsl_data.lim_lon,
        rsl_data.lim_y,
        rsl_data.lim_y_sigma,
        rsl_data.lim_age_sigma,
        rsl_data.lim_flags,
        bp_min=bp_min,
        bp_max=bp_max,
        n_search=n_search,
    )

    print(
        f"Model weights (WRSS/erf, {len(rsl_data.lim_age)} limiting points, n_search={n_search}):"
    )
    for m, name in enumerate(model_names):
        print(f"  {name}: weight={weights[m]:.4f}  misfit={misfits[m]:.2f}")

    # Covariance and weighted mean at SLIP training points
    ages_bp = rsl_data.X[:, 0].numpy()
    lats = rsl_data.X[:, 1].numpy()
    lons = rsl_data.X[:, 2].numpy()

    X_gia = np.column_stack([ages_bp, lats, lons % 360])
    gia_cov, gia_mean = extract_gia_covariance(
        interp_funcs,
        X_gia,
        weights=weights,
        regularize=True,
        return_mean=True,
    )

    eigvals = np.linalg.eigvalsh(gia_cov)
    print(f"\nGIA covariance: shape={gia_cov.shape}")
    print(
        f"  Diagonal (std) range: {np.sqrt(np.diag(gia_cov)).min():.4f} – {np.sqrt(np.diag(gia_cov)).max():.4f} m"
    )
    print(f"  Eigenvalue range: [{eigvals.min():.2e}, {eigvals.max():.2e}]")
    print(f"  Positive definite: {eigvals.min() > 0}")

    return GIAResult(
        interp_funcs=interp_funcs,
        model_names=model_names,
        weights=weights,
        misfits=misfits,
        gia_mean_at_obs=gia_mean,
        gia_cov_at_obs=gia_cov,
    )


def compute_residuals_and_noise(
    rsl_data: RSLData,
    gia_result: GIAResult,
) -> ResidualData:
    """
    Compute GP training residuals and total noise matrix.

    y_residual = y_obs - GIA_weighted_mean
    noise_total = diag(sigma_obs^2) + Sigma^GIA

    Returns
    -------
    ResidualData
    """
    y_np = (
        rsl_data.y.numpy()
        if isinstance(rsl_data.y, torch.Tensor)
        else np.asarray(rsl_data.y)
    )
    y_sig_np = (
        rsl_data.y_sigma.numpy()
        if isinstance(rsl_data.y_sigma, torch.Tensor)
        else np.asarray(rsl_data.y_sigma)
    )

    y_residual = torch.tensor(y_np - gia_result.gia_mean_at_obs).double()

    noise_obs = torch.diag(torch.tensor(y_sig_np**2, dtype=torch.float64))
    noise_gia = torch.tensor(gia_result.gia_cov_at_obs, dtype=torch.float64)
    noise_total = noise_obs + noise_gia

    print(f"Residual range: {y_residual.min():.2f} to {y_residual.max():.2f} m")
    print(f"Residual std:   {y_residual.std():.2f} m")
    print(
        f"Noise matrix shape: {noise_total.shape} (observation diag + GIA covariance)"
    )

    return ResidualData(
        y_residual=y_residual,
        noise_total=noise_total,
    )


def gia_spatial_mean_covariance_at_ages(
    interp_funcs: List[Any],
    weights: np.ndarray,
    test_ages: np.ndarray,
    spatial_lat: np.ndarray,
    spatial_lon: np.ndarray,
    *,
    spatial_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted GIA mean/covariance after spatially averaging each model."""
    ages = np.asarray(test_ages, dtype=np.float64).reshape(-1)
    lat = np.asarray(spatial_lat, dtype=np.float64).reshape(-1)
    lon = np.asarray(spatial_lon, dtype=np.float64).reshape(-1) % 360.0
    if lat.size == 0 or lat.size != lon.size:
        raise ValueError("spatial_lat and spatial_lon must have equal non-zero size")

    if spatial_weights is None:
        point_weights = np.full(lat.size, 1.0 / lat.size, dtype=np.float64)
    else:
        point_weights = np.asarray(spatial_weights, dtype=np.float64).reshape(-1)
        if point_weights.size != lat.size:
            raise ValueError(
                "spatial_weights must have one value per spatial coordinate"
            )
        if not np.isfinite(point_weights).all() or np.any(point_weights < 0):
            raise ValueError("spatial_weights must be finite and non-negative")
        weight_sum = float(point_weights.sum())
        if weight_sum <= 0:
            raise ValueError("spatial_weights must sum to a positive value")
        point_weights = point_weights / weight_sum

    model_age_means = np.empty(
        (len(interp_funcs), ages.size), dtype=np.float64
    )
    for model_index, interp in enumerate(interp_funcs):
        for age_index, age_bp in enumerate(ages):
            X_test = np.column_stack([np.full(lat.size, age_bp), lat, lon])
            values = np.asarray(interp(X_test), dtype=np.float64).reshape(-1)
            if values.size != lat.size:
                raise ValueError(
                    "GIA interpolator returned an unexpected prediction size"
                )
            model_age_means[model_index, age_index] = float(
                point_weights @ values
            )

    model_weights = np.asarray(weights, dtype=np.float64)
    gia_mean = np.average(model_age_means, axis=0, weights=model_weights)
    if len(interp_funcs) < 2:
        return gia_mean, np.zeros((ages.size, ages.size), dtype=np.float64)

    centered = model_age_means - gia_mean
    gia_cov = np.sum(
        model_weights[:, None, None]
        * (centered[:, :, None] * centered[:, None, :]),
        axis=0,
    )
    denominator = model_weights.sum() - model_weights.sum() / len(interp_funcs)
    if denominator <= 0:
        raise ValueError("GIA covariance correction denominator must be positive")
    gia_cov /= denominator
    return gia_mean, 0.5 * (gia_cov + gia_cov.T)


def gia_at_test_points(
    interp_funcs: List[Any],
    weights: np.ndarray,
    test_ages: np.ndarray,
    test_lat: np.ndarray,
    test_lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate weighted GIA ensemble mean and Bessel-corrected variance at
    arbitrary test coordinates.

    Bessel-corrected weighted ensemble variance:
        sigma^2_GIA(x) = [ sum_k w_k * (g_k(x) - g_bar(x))^2 ]
                         / [ sum(w) - sum(w^2)/sum(w) ]

    Parameters
    ----------
    interp_funcs : list
        GIA model interpolators.
    weights : ndarray (M,)
    test_ages, test_lat, test_lon : ndarray
        All same length, or broadcastable.

    Returns
    -------
    gia_mean : ndarray — weighted mean
    gia_var : ndarray — Bessel-corrected weighted variance
    """
    M = len(interp_funcs)
    test_lon_360 = np.asarray(test_lon) % 360
    X_test = np.column_stack([test_ages, test_lat, test_lon_360])

    preds = np.array([f(X_test) for f in interp_funcs])  # (M, n_pts)
    gia_mean = np.average(preds, axis=0, weights=weights)
    centered = preds - gia_mean

    gia_var = np.zeros(len(test_ages))
    for m in range(M):
        gia_var += weights[m] * centered[m] ** 2
    gia_var /= weights.sum() - weights.sum() / M

    return gia_mean, gia_var


def gia_covariance_at_test_points(
    interp_funcs: List[Any],
    weights: np.ndarray,
    test_ages: np.ndarray,
    test_lat: np.ndarray,
    test_lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate weighted GIA ensemble mean and full Bessel-corrected covariance
    at test coordinates.

    Returns
    -------
    gia_mean : ndarray (n,) — weighted mean
    gia_cov : ndarray (n, n) — Bessel-corrected weighted covariance
    """
    M = len(interp_funcs)
    test_lon_360 = np.asarray(test_lon) % 360
    X_test = np.column_stack([test_ages, test_lat, test_lon_360])

    preds = np.array([f(X_test) for f in interp_funcs])  # (M, n_pts)
    gia_mean = np.average(preds, axis=0, weights=weights)
    centered = preds - gia_mean

    gia_cov = np.sum(
        weights[:, None, None] * (centered[:, :, None] * centered[:, None, :]),
        axis=0,
    )
    gia_cov /= weights.sum() - weights.sum() / M

    return gia_mean, gia_cov
