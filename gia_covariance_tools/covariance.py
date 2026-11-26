"""
Main Covariance Computation Functions for GIA Models
======================================================

This module provides the primary functions for extracting and computing
covariances from multiple Glacial Isostatic Adjustment (GIA) model ensembles.

Key functions include:
- Empirical covariance extraction
- Weighted covariance computation
- Spatial/temporal covariance structure analysis
- PSTHM kernel creation for GP regression
- Covariance validation utilities

Date: 2026-03-02
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Callable, Dict
import warnings

from .core_math import (
    compute_weighted_mean,
    compute_weighted_covariance,
    center_data,
    regularize_covariance_matrix,
    compute_correlation_from_covariance,
    compute_taper_matrix,
    compute_eigen_decomposition,
    compute_condition_number,
    compute_log_likelihood,
    validate_covariance_input,
)
from .config import (
    DEFAULT_TEMPORAL_TAPER_TAU_KA,
    DEFAULT_REGULARIZATION_EPSILON,
    DEFAULT_MIN_VARIANCE,
    DEFAULT_KERNEL_LENGTHSCALE,
    DEFAULT_KERNEL_VARIANCE,
)


# =============================================================================
# Main Covariance Extraction Functions
# =============================================================================


def extract_gia_covariance(
    gia_interp_list: List[Callable],
    X_points: np.ndarray,
    weights: Optional[np.ndarray] = None,
    regularize: bool = True,
    epsilon: float = DEFAULT_REGULARIZATION_EPSILON,
    return_mean: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract covariance matrix from multiple GIA model predictions.

    Mathematical Formulation:
    --------------------------
    Given M GIA models providing predictions at N points, the ensemble
    covariance is:

    $$\\Sigma_{ij} = \\frac{1}{M-1} \\sum_{k=1}^{M} (g_k(x_i) - \\bar{g}(x_i))(g_k(x_j) - \\bar{g}(x_j))$$

    where $\\bar{g}(x) = \\frac{1}{M}\\sum_{k=1}^{M} g_k(x)$ is the ensemble mean.

    For weighted covariance:

    $$\\Sigma_{ij} = \\frac{1}{\\sum_k w_k - 1} \\sum_{k=1}^{M} w_k (g_k(x_i) - \\bar{g}_w(x_i))(g_k(x_j) - \\bar{g}_w(x_j))$$

    where $\\bar{g}_w(x) = \\sum_k w_k g_k(x) / \\sum_k w_k$.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of interpolation functions, each taking coordinates (N, 3) and
        returning predictions of shape (N,)
    X_points : np.ndarray
        Prediction points of shape (N, 3) where columns are [time_BP, lat, lon]
    weights : np.ndarray, optional
        Model weights of shape (M,). If None, uses uniform weights.
    regularize : bool
        Whether to regularize the covariance matrix
    epsilon : float
        Regularization parameter
    return_mean : bool
        Whether to also return the mean predictions

    Returns:
    --------
    cov_matrix : np.ndarray
        Covariance matrix of shape (N, N)
    mean_predictions : np.ndarray, optional
        Mean predictions of shape (N,) if return_mean=True

    Examples:
    ---------
    >>> # With PSTHM GIAGrid objects
    >>> grids, names = load_ensemble_from_directory('./gia_models')
    >>> interp_funcs = [lambda x, g=g: g.predict_bp(x) for g in grids]
    >>> X_test = np.array([[10000, 40, 290], [8000, 40, 290]])
    >>> cov, mean = extract_gia_covariance(interp_funcs, X_test, return_mean=True)
    """
    n_models = len(gia_interp_list)
    n_points = X_points.shape[0]

    # Get predictions from each model
    predictions = np.zeros((n_models, n_points))

    for i, interp_func in enumerate(gia_interp_list):
        predictions[i] = np.asarray(interp_func(X_points)).flatten()

    # Validate and normalize weights
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n_models:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of models ({n_models})"
            )
    else:
        weights = np.ones(n_models) / n_models

    # Compute weighted mean
    mean_pred = compute_weighted_mean(predictions, weights, axis=0)

    # Center predictions
    centered = predictions - mean_pred

    # Compute weighted covariance with Bessel-like correction:
    # denominator = sum(w_i) - sum(w_i) / M  (equivalent to sum(w_i) * (M-1)/M)
    weight_sum = weights.sum()
    ddof = weight_sum / n_models
    cov_matrix = compute_weighted_covariance(
        centered, weights=weights, center=False, ddof=ddof
    )

    # Regularize if requested
    if regularize:
        cov_matrix = regularize_covariance_matrix(cov_matrix, epsilon=epsilon)

    if return_mean:
        return cov_matrix, mean_pred
    else:
        return cov_matrix


def extract_gia_covariance_weighted(
    gia_interp_list: List[Callable],
    X_points: np.ndarray,
    weights: np.ndarray,
    regularize: bool = True,
    epsilon: float = DEFAULT_REGULARIZATION_EPSILON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract weighted covariance from GIA model ensemble.

    This is an alias for extract_gia_covariance with explicit weights.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of interpolation functions
    X_points : np.ndarray
        Prediction points (N, 3)
    weights : np.ndarray
        Model weights (M,)
    regularize : bool
        Whether to regularize
    epsilon : float
        Regularization parameter

    Returns:
    --------
    tuple
        (covariance matrix, mean predictions)
    """
    return extract_gia_covariance(
        gia_interp_list,
        X_points,
        weights=weights,
        regularize=regularize,
        epsilon=epsilon,
        return_mean=True,
    )


# =============================================================================
# Spatial and Temporal Covariance Functions
# =============================================================================


def compute_spatial_covariance_at_time(
    gia_interp_list: List[Callable],
    time_bp: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    weights: Optional[np.ndarray] = None,
    regularize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial covariance matrix at a specific time.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of GIA model interpolation functions
    time_bp : float
        Time in years BP
    lat_grid : np.ndarray
        Latitude grid
    lon_grid : np.ndarray
        Longitude grid
    weights : np.ndarray, optional
        Model weights
    regularize : bool
        Whether to regularize

    Returns:
    --------
    tuple
        (covariance matrix of shape (n_locations, n_locations),
         mean predictions of shape (n_locations,))

    Examples:
    ---------
    >>> lat = np.linspace(30, 50, 11)
    >>> lon = np.linspace(280, 320, 11)
    >>> cov, mean = compute_spatial_covariance_at_time(interps, 6000, lat, lon)
    """
    # Create coordinate array
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    coords = np.column_stack(
        [np.full(lat_mesh.size, time_bp), lat_mesh.flatten(), lon_mesh.flatten()]
    )

    return extract_gia_covariance(
        gia_interp_list,
        coords,
        weights=weights,
        regularize=regularize,
        return_mean=True,
    )


def compute_temporal_covariance_at_location(
    gia_interp_list: List[Callable],
    lat: float,
    lon: float,
    times_bp: np.ndarray,
    weights: Optional[np.ndarray] = None,
    regularize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute temporal covariance at a specific location.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of GIA model interpolation functions
    lat : float
        Latitude
    lon : float
        Longitude (0-360)
    times_bp : np.ndarray
        Times in years BP
    weights : np.ndarray, optional
        Model weights
    regularize : bool
        Whether to regularize

    Returns:
    --------
    tuple
        (covariance matrix, mean predictions)

    Examples:
    ---------
    >>> times = np.arange(0, 26000, 1000)
    >>> cov, mean = compute_temporal_covariance_at_location(interps, 40, 290, times)
    """
    # Create coordinate array
    coords = np.column_stack(
        [times_bp, np.full(len(times_bp), lat), np.full(len(times_bp), lon)]
    )

    return extract_gia_covariance(
        gia_interp_list,
        coords,
        weights=weights,
        regularize=regularize,
        return_mean=True,
    )


def compute_empirical_gia_covariance_3d(
    gia_interp_list: List[Callable],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    times_ka: np.ndarray,
    weights: Optional[np.ndarray] = None,
    flatten: bool = True,
) -> Dict[str, np.ndarray]:
    """
        Compute full 3D (time, space) covariance tensor.

        Mathematical Formulation:
        --------------------------
        The 3D covariance tensor has dimensions (N_time, N_lat, N_lon, N_time, N_lat, N_lon)
        and captures covariance between all spacetime points:

        $$\\Sigma_{ijk, lmn} = \\text{Cov}[g(t_i, \\phi_j, \\lambda_k), g(t_l, \\phi_m, \\lambda_n)]$$

        For memory efficiency, this function can return the flattened 2D matrix.

        Parameters:
        -----------
    gia_interp_list : list of callable
            List of GIA model interpolation functions
        lat_grid : np.ndarray
            Latitude grid
        lon_grid : np.ndarray
            Longitude grid
        times_ka : np.ndarray
            Time points in ka BP
        weights : np.ndarray, optional
            Model weights
        flatten : bool
            If True, return flattened 2D matrix instead of 3D tensor

        Returns:
        --------
        dict
            Dictionary containing:
            - 'covariance': covariance matrix or tensor
            - 'mean': mean predictions
            - 'coords': coordinate array
            - 'shape': original shape tuple

        Examples:
        ---------
        >>> lat = np.linspace(-60, 60, 13)
        >>> lon = np.linspace(0, 360, 37)
        >>> times = np.arange(0, 26, 1)
        >>> result = compute_empirical_gia_covariance_3d(interps, lat, lon, times)
    """
    # Create coordinate arrays
    times_bp = times_ka * 1000  # Convert to years BP

    lon_mesh, lat_mesh, time_mesh = np.meshgrid(lon_grid, lat_grid, times_bp)
    coords = np.column_stack(
        [time_mesh.flatten(), lat_mesh.flatten(), lon_mesh.flatten()]
    )

    # Compute covariance
    cov_matrix, mean_pred = extract_gia_covariance(
        gia_interp_list, coords, weights=weights, regularize=True, return_mean=True
    )

    n_times = len(times_ka)
    n_lat = len(lat_grid)
    n_lon = len(lon_grid)
    n_points = n_times * n_lat * n_lon

    if flatten:
        return {
            "covariance": cov_matrix,
            "mean": mean_pred,
            "coords": coords,
            "shape": (n_times, n_lat, n_lon),
        }
    else:
        # Reshape to 3D tensor
        cov_3d = cov_matrix.reshape(n_times, n_lat, n_lon, n_times, n_lat, n_lon)
        return {
            "covariance": cov_3d,
            "mean": mean_pred.reshape(n_times, n_lat, n_lon),
            "coords": coords,
            "shape": (n_times, n_lat, n_lon),
        }


def compute_spatial_coherence_map(
    gia_interp_list: List[Callable],
    time_bp: float,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    resolution: float = 2.0,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spatial map of GIA model disagreement.

    This computes the variance across models at each location, showing
    where GIA models agree or disagree.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of GIA model interpolation functions
    time_bp : float
        Time in years BP
    lat_range : tuple
        (min, max) latitude
    lon_range : tuple
        (min, max) longitude
    resolution : float
        Spatial resolution in degrees
    weights : np.ndarray, optional
        Model weights

    Returns:
    --------
    tuple
        (lat_grid, lon_grid, disagreement_map)

    Examples:
    ---------
    >>> lat, lon, disagreement = compute_spatial_coherence_map(
    ...     interps, 6000, (-60, 60), (0, 360), resolution=5.0
    ... )
    """
    lat_grid = np.arange(lat_range[0], lat_range[1] + 1, resolution)
    lon_grid = np.arange(lon_range[0], lon_range[1] + 1, resolution)

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    coords = np.column_stack(
        [np.full(lat_mesh.size, time_bp), lat_mesh.flatten(), lon_mesh.flatten()]
    )

    # Get predictions
    predictions = np.array(
        [np.asarray(interp(coords)).flatten() for interp in gia_interp_list]
    )

    # Compute weighted variance (disagreement)
    if weights is not None:
        weights = np.asarray(weights)
        mean_pred = np.average(predictions, axis=0, weights=weights)
        centered = predictions - mean_pred
        disagreement = np.average(centered**2, axis=0, weights=weights)
    else:
        disagreement = np.var(predictions, axis=0)

    disagreement_map = disagreement.reshape(lat_mesh.shape)

    return lat_grid, lon_grid, disagreement_map


def compute_temporal_structure_function(
    gia_interp_list: List[Callable],
    lat: float,
    lon: float,
    times_ka: np.ndarray,
    weights: Optional[np.ndarray] = None,
    regularize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute temporal covariance structure at a fixed location.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of GIA model interpolation functions
    lat : float
        Latitude
    lon : float
        Longitude (0-360)
    times_ka : np.ndarray
        Time points in ka BP
    weights : np.ndarray, optional
        Model weights
    regularize : bool
        Whether to regularize

    Returns:
    --------
    tuple
        (times_years_bp, covariance_matrix, mean_predictions)

    Examples:
    ---------
    >>> times = np.arange(0, 26, 0.5)
    >>> times_bp, cov, mean = compute_temporal_structure_function(
    ...     interps, 40, 290, times
    ... )
    """
    times_bp = times_ka * 1000  # Convert to years

    return compute_temporal_covariance_at_location(
        gia_interp_list, lat, lon, times_bp, weights=weights, regularize=regularize
    )


# =============================================================================
# Advanced Analysis Functions
# =============================================================================


def compute_gia_model_disagreement(
    gia_interp_list: List[Callable],
    X_points: np.ndarray,
    weights: Optional[np.ndarray] = None,
    metric: str = "variance",
) -> np.ndarray:
    """
    Compute model disagreement at each point.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of GIA model interpolation functions
    X_points : np.ndarray
        Prediction points (N, 3)
    weights : np.ndarray, optional
        Model weights
    metric : str
        Disagreement metric:
        - 'variance': Variance across models
        - 'std': Standard deviation across models
        - 'range': Range (max - min) across models
        - 'iqr': Interquartile range across models

    Returns:
    --------
    np.ndarray
        Disagreement values of shape (N,)
    """
    # Get predictions
    predictions = np.array(
        [np.asarray(interp(X_points)).flatten() for interp in gia_interp_list]
    )

    if metric == "variance":
        if weights is not None:
            mean_pred = np.average(predictions, axis=0, weights=weights)
            centered = predictions - mean_pred
            disagreement = np.average(centered**2, axis=0, weights=weights)
        else:
            disagreement = np.var(predictions, axis=0)

    elif metric == "std":
        disagreement = compute_gia_model_disagreement(
            gia_interp_list, X_points, weights, metric="variance"
        )
        disagreement = np.sqrt(disagreement)

    elif metric == "range":
        disagreement = np.max(predictions, axis=0) - np.min(predictions, axis=0)

    elif metric == "iqr":
        q75 = np.percentile(predictions, 75, axis=0)
        q25 = np.percentile(predictions, 25, axis=0)
        disagreement = q75 - q25

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return disagreement


def combine_gia_observation_covariance(
    gia_cov: np.ndarray,
    obs_cov: Union[np.ndarray, float],
    observation_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine GIA ensemble covariance with observation-specific uncertainties.

    Mathematical Formulation:
    --------------------------
    The total covariance is the sum of GIA model covariance and
    observation uncertainty:

    $$\\Sigma_{total} = \\Sigma_{GIA} + \\Sigma_{obs}$$

    Parameters:
    -----------
    gia_cov : np.ndarray
        GIA ensemble covariance matrix
    obs_cov : np.ndarray or float
        Observation covariance matrix or variance (if scalar)
    observation_indices : np.ndarray, optional
        Indices of observations in the GIA covariance matrix

    Returns:
    --------
    np.ndarray
        Combined covariance matrix

    Examples:
    ---------
    >>> # With scalar variance
    >>> combined = combine_gia_observation_covariance(gia_cov, 0.25)
    >>> # With full observation covariance
    >>> combined = combine_gia_observation_covariance(gia_cov, obs_cov)
    """
    gia_cov = np.asarray(gia_cov)
    n_obs = gia_cov.shape[0]

    if observation_indices is not None:
        # Subset to observation indices
        gia_obs = gia_cov[np.ix_(observation_indices, observation_indices)]
        n_obs = len(observation_indices)
    else:
        gia_obs = gia_cov

    # Add observation uncertainty
    if isinstance(obs_cov, (int, float)):
        combined = gia_obs + obs_cov * np.eye(n_obs)
    else:
        obs_cov = np.asarray(obs_cov)
        if obs_cov.shape != (n_obs, n_obs):
            raise ValueError(
                f"Observation covariance shape {obs_cov.shape} "
                f"doesn't match {n_obs}x{n_obs}"
            )
        combined = gia_obs + obs_cov

    return combined


def validate_covariance_matrix(
    cov_matrix: np.ndarray,
    check_symmetry: bool = True,
    check_positive_definite: bool = True,
    check_finite: bool = True,
    epsilon: float = DEFAULT_REGULARIZATION_EPSILON,
) -> Dict[str, Union[bool, float, str]]:
    """
    Validate a covariance matrix.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix to validate
    check_symmetry : bool
        Check if matrix is symmetric
    check_positive_definite : bool
        Check if matrix is positive definite
    check_finite : bool
        Check if all values are finite
    epsilon : float
        Tolerance for positive definiteness check

    Returns:
    --------
    dict
        Validation results with keys:
        - 'is_valid': Overall validity
        - 'is_symmetric': Symmetry check result
        - 'is_positive_definite': PD check result
        - 'has_finite_values': Finite values check result
        - 'condition_number': Matrix condition number
        - 'min_eigenvalue': Minimum eigenvalue
        - 'max_eigenvalue': Maximum eigenvalue
        - 'warnings': List of warning messages

    Examples:
    ---------
    >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> validation = validate_covariance_matrix(cov)
    """
    results = {"is_valid": True, "warnings": []}

    # Check shape
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        results["is_valid"] = False
        results["warnings"].append("Matrix is not square")
        return results

    n = cov_matrix.shape[0]

    # Check symmetry
    if check_symmetry:
        is_symmetric = np.allclose(cov_matrix, cov_matrix.T, rtol=1e-10)
        results["is_symmetric"] = is_symmetric
        if not is_symmetric:
            results["is_valid"] = False
            results["warnings"].append("Matrix is not symmetric")

    # Check positive definiteness
    if check_positive_definite:
        try:
            eigenvalues, _ = compute_eigen_decomposition(cov_matrix, regularize=False)
            min_eig = np.min(eigenvalues)
            max_eig = np.max(eigenvalues)

            results["min_eigenvalue"] = float(min_eig)
            results["max_eigenvalue"] = float(max_eig)

            is_pd = min_eig > -epsilon * abs(max_eig)
            results["is_positive_definite"] = is_pd

            if not is_pd:
                results["warnings"].append(
                    f"Matrix is not positive definite "
                    f"(min eigenvalue: {min_eig:.2e})"
                )
        except Exception as e:
            results["is_positive_definite"] = False
            results["warnings"].append(f"Could not check PD: {str(e)}")

    # Check finite values
    if check_finite:
        has_finite = np.all(np.isfinite(cov_matrix))
        results["has_finite_values"] = has_finite
        if not has_finite:
            results["is_valid"] = False
            results["warnings"].append("Matrix contains non-finite values")

    # Compute condition number
    cond = compute_condition_number(cov_matrix, regularize=True)
    results["condition_number"] = cond

    if cond > 1e12:
        results["warnings"].append(f"Matrix is ill-conditioned (cond={cond:.2e})")

    return results


def compute_covariance_spectrum(
    cov_matrix: np.ndarray, n_components: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute spectral decomposition of covariance matrix.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix
    n_components : int, optional
        Number of components to return

    Returns:
    --------
    dict
        Dictionary containing:
        - 'eigenvalues': Eigenvalues
        - 'eigenvectors': Eigenvectors
        - 'explained_variance': Fraction of variance explained
        - 'cumulative_variance': Cumulative explained variance
        - 'n_components_95': Components needed for 95% variance

    Examples:
    ---------
    >>> spectrum = compute_covariance_spectrum(cov_matrix)
    >>> print(f"95% variance in {spectrum['n_components_95']} components")
    """
    eigenvalues, eigenvectors = compute_eigen_decomposition(cov_matrix)

    # Explained variance
    total_var = np.sum(eigenvalues)
    explained_var = eigenvalues / total_var
    cumulative_var = np.cumsum(explained_var)

    # Number of components for 95% variance
    n_comp_95 = np.searchsorted(cumulative_var, 0.95) + 1

    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
        explained_var = explained_var[:n_components]
        cumulative_var = cumulative_var[:n_components]

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance": explained_var,
        "cumulative_variance": cumulative_var,
        "n_components_95": n_comp_95,
        "total_variance": total_var,
    }


# =============================================================================
# PSTHM Integration Functions
# =============================================================================


def create_sampling_kernel(
    gia_interp_list: List[Callable],
    tau: Optional[float] = DEFAULT_TEMPORAL_TAPER_TAU_KA * 1000,
    weights: Optional[np.ndarray] = None,
) -> "Sampling":
    """
    Create a PSTHM Sampling kernel from GIA model ensemble.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of GIA model interpolation functions
    tau : float, optional
        Temporal taper length in years
    weights : np.ndarray, optional
        Model weights

    Returns:
    --------
    Sampling
        PSTHM Sampling kernel object

    Examples:
    ---------
    >>> kernel = create_sampling_kernel(interps, tau=5000)
    >>> combined = PSTHM.kernels.Sum(kernel, PSTHM.kernels.Matern32(3))
    """
    try:
        from PSTHM.kernels import Sampling
    except ImportError:
        raise ImportError("PSTHM must be installed to use this function")

    # Create sampling kernel
    kernel = Sampling(
        input_dim=3,  # time, lat, lon
        model_ensemble=gia_interp_list,
        weights=weights,
        tau=tau,
    )

    return kernel


def create_gia_ensemble_mean_function(
    gia_interp_list: List[Callable], weights: Optional[np.ndarray] = None
) -> "GIA_ensemble":
    """
    Create a PSTHM GIA ensemble mean function.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of GIA model interpolation functions
    weights : np.ndarray, optional
        Model weights

    Returns:
    --------
    GIA_ensemble
        PSTHM GIA ensemble kernel

    Examples:
    ---------
    >>> mean_func = create_gia_ensemble_mean_function(interps)
    >>> gpr = PSTHM.GPRegression_V(X, y, kernel, mean_function=mean_func)
    """
    try:
        from PSTHM.model import GIA_ensemble
    except ImportError:
        raise ImportError("PSTHM must be installed to use this function")

    mean_func = GIA_ensemble(
        GIA_model_interp=gia_interp_list,
        input_dim=1,  # Usually used with temporal input
    )

    if weights is not None:
        mean_func.w = weights

    return mean_func


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_covariance(
    gia_grids: List, X_points: np.ndarray, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Quick covariance computation from GIAGrid objects.

    This is a convenience function that handles the common case of
    computing covariance from PSTHM GIAGrid objects directly.

    Parameters:
    -----------
    gia_grids : list
        List of PSTHM GIAGrid objects
    X_points : np.ndarray
        Prediction points (N, 3) as [time_BP, lat, lon]
    weights : np.ndarray, optional
        Model weights

    Returns:
    --------
    np.ndarray
        Covariance matrix

    Examples:
    ---------
    >>> grids = [gia_grid_ice7g, gia_grid_ice6g, gia_grid_anu]
    >>> X = np.array([[10000, 40, 290], [8000, 40, 290]])
    >>> cov = quick_covariance(grids, X)
    """
    # Create interpolation functions
    interp_funcs = [lambda x, g=g: g.predict_bp(x) for g in gia_grids]

    # Compute covariance
    return extract_gia_covariance(
        interp_funcs, X_points, weights=weights, regularize=True, return_mean=False
    )
