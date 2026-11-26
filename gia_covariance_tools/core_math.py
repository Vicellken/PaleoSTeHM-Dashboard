"""
Core Mathematical Operations for GIA Covariance Calculations
===========================================================

Provides fundamental mathematical operations for computing covariances
from GIA model ensembles, including weighted statistics, matrix operations,
and numerical utilities.

Date: 2026-03-02
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings

from .config import (
    DEFAULT_REGULARIZATION_EPSILON,
    DEFAULT_MIN_VARIANCE,
    EIGENVALUE_TOLERANCE,
    CONDITION_NUMBER_WARNING,
)


# =============================================================================
# Basic Statistical Functions
# =============================================================================


def compute_weighted_mean(
    data: np.ndarray, weights: Optional[np.ndarray] = None, axis: int = 0
) -> np.ndarray:
    """
    Compute weighted mean of data along specified axis.

    Mathematical Formulation:
    --------------------------
    For a dataset with M models and N points:

    $$\\bar{x}_j = \\frac{\\sum_{i=1}^{M} w_i x_{ij}}{\\sum_{i=1}^{M} w_i}$$

    where $x_{ij}$ is the prediction from model $i$ at point $j$.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (M, N) where M is number of models
    weights : np.ndarray, optional
        Weights of shape (M,). If None, uses uniform weights.
    axis : int
        Axis along which to compute mean

    Returns:
    --------
    np.ndarray
        Weighted mean

    Examples:
    ---------
    >>> predictions = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]])
    >>> weights = np.array([0.2, 0.5, 0.3])
    >>> mean = compute_weighted_mean(predictions, weights)
    """
    if weights is None:
        return np.mean(data, axis=axis)

    weights = np.asarray(weights)
    if weights.ndim == 1:
        weights = weights / weights.sum()

        if axis == 0:
            return np.sum(data * weights[:, np.newaxis], axis=0)
        elif axis == 1:
            return np.sum(data * weights[np.newaxis, :], axis=1)
        else:
            raise ValueError(f"Invalid axis: {axis}")
    else:
        raise ValueError("Weights must be 1-dimensional")


def compute_weighted_covariance(
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    center: bool = True,
    ddof: int = 1,
) -> np.ndarray:
    """
    Compute weighted covariance matrix.

    Mathematical Formulation:
    --------------------------
    The weighted covariance between points $k$ and $l$ is:

    $$\\Sigma_{kl} = \\frac{1}{\\sum_i w_i - ddof} \\sum_i w_i (x_{ik} - \\bar{x}_k)(x_{il} - \\bar{x}_l)$$

    For unweighted case with ddof=1:

    $$\\Sigma_{kl} = \\frac{1}{M-1} \\sum_{i=1}^{M} (x_{ik} - \\bar{x}_k)(x_{il} - \\bar{x}_l)$$

    Parameters:
    -----------
    data : np.ndarray
        Data of shape (M, N) where M is number of models, N is number of points
    weights : np.ndarray, optional
        Weights of shape (M,). If None, uses uniform weights.
    center : bool
        If True, center data by subtracting mean before computing covariance
    ddof : int
        Delta degrees of freedom for normalization

    Returns:
    --------
    np.ndarray
        Covariance matrix of shape (N, N)

    Examples:
    ---------
    >>> predictions = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]])
    >>> cov = compute_weighted_covariance(predictions)
    """
    data = np.asarray(data)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    M, N = data.shape

    # Compute weights
    if weights is None:
        weights = np.ones(M)

    weights = np.asarray(weights, dtype=float)
    weight_sum = weights.sum()

    # Center data
    if center:
        mean = compute_weighted_mean(data, weights, axis=0)
        data_centered = data - mean
    else:
        data_centered = data

    # Compute weighted covariance
    if weights.ndim == 1:
        # Use the formula for weighted covariance
        # Sigma = sum_i w_i * (x_i - mean) * (x_i - mean)^T / (sum_i w_i - ddof)
        cov = np.zeros((N, N))

        for i in range(M):
            diff = data_centered[i, :, np.newaxis]  # (N, 1)
            diff_t = data_centered[i, np.newaxis, :]  # (1, N)
            cov += weights[i] * (diff @ diff_t)

        cov = cov / (weight_sum - ddof)

    return cov


def center_data(
    data: np.ndarray, weights: Optional[np.ndarray] = None, axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center data by subtracting weighted mean.

    Parameters:
    -----------
    data : np.ndarray
        Input data
    weights : np.ndarray, optional
        Weights for computing mean
    axis : int
        Axis along which to center

    Returns:
    --------
    tuple
        (centered_data, mean)

    Examples:
    ---------
    >>> data = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]])
    >>> centered, mean = center_data(data)
    """
    mean = compute_weighted_mean(data, weights, axis=axis)

    if axis == 0:
        centered = data - mean
    elif axis == 1:
        centered = data - mean[:, np.newaxis]
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return centered, mean


# =============================================================================
# Matrix Operations
# =============================================================================


def regularize_covariance_matrix(
    cov_matrix: np.ndarray,
    epsilon: float = DEFAULT_REGULARIZATION_EPSILON,
    min_variance: float = DEFAULT_MIN_VARIANCE,
    method: str = "diagonal",
) -> np.ndarray:
    """
    Regularize a covariance matrix to ensure positive definiteness.

    Mathematical Formulation:
    --------------------------
    For diagonal regularization:

    $$\\Sigma_{reg} = \\Sigma + \\max(\\epsilon \\cdot \\text{tr}(\\Sigma)/N, \\min_variance) \\cdot \\mathbf{I}$$

    This adds a small value to the diagonal to ensure numerical stability
    while preserving the correlation structure.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Input covariance matrix
    epsilon : float
        Regularization parameter (fraction of mean variance)
    min_variance : float
        Minimum variance to add to diagonal
    method : str
        Regularization method:
        - 'diagonal': Add to diagonal only
        - 'scaled': Scale eigenvalues
        - 'nugget': Add scaled identity

    Returns:
    --------
    np.ndarray
        Regularized covariance matrix

    Examples:
    ---------
    >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> cov_reg = regularize_covariance_matrix(cov, epsilon=1e-6)
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    n = cov_matrix.shape[0]

    if method == "diagonal":
        # Compute the regularization amount
        diagonal = np.diag(cov_matrix)

        # Use either a fraction of mean variance or minimum variance
        reg_value = max(epsilon * np.mean(diagonal), min_variance)

        # Add to diagonal
        reg_matrix = reg_value * np.eye(n)
        cov_regularized = cov_matrix + reg_matrix

    elif method == "scaled":
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Add small value to negative/zero eigenvalues
        eigenvalues = np.maximum(eigenvalues, min_variance)

        # Reconstruct
        cov_regularized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    elif method == "nugget":
        # Add scaled identity matrix
        nugget = epsilon * np.trace(cov_matrix) / n
        nugget = max(nugget, min_variance)
        cov_regularized = cov_matrix + nugget * np.eye(n)

    else:
        raise ValueError(f"Unknown regularization method: {method}")

    return cov_regularized


def compute_correlation_from_covariance(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix.

    Mathematical Formulation:
    --------------------------
    $$R_{ij} = \\frac{\\Sigma_{ij}}{\\sqrt{\\Sigma_{ii} \\cdot \\Sigma_{jj}}}$$

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix

    Returns:
    --------
    np.ndarray
        Correlation matrix

    Examples:
    ---------
    >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> corr = compute_correlation_from_covariance(cov)
    """
    # Compute standard deviations
    std = np.sqrt(np.diag(cov_matrix))

    # Handle zero variances
    std[std == 0] = 1.0

    # Compute correlation
    corr = cov_matrix / np.outer(std, std)

    return corr


def invert_covariance_matrix(
    cov_matrix: np.ndarray,
    regularize: bool = True,
    epsilon: float = DEFAULT_REGULARIZATION_EPSILON,
) -> np.ndarray:
    """
    Invert a covariance matrix with optional regularization.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix to invert
    regularize : bool
        Whether to regularize before inversion
    epsilon : float
        Regularization parameter

    Returns:
    --------
    np.ndarray
        Inverse covariance matrix (precision matrix)

    Examples:
    ---------
    >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> inv_cov = invert_covariance_matrix(cov)
    """
    if regularize:
        cov_matrix = regularize_covariance_matrix(cov_matrix, epsilon=epsilon)

    # Use Cholesky decomposition for better numerical stability
    try:
        L = np.linalg.cholesky(cov_matrix)
        inv_cov = np.linalg.inv(L).T @ np.linalg.inv(L)
    except np.linalg.LinAlgError:
        # Fall back to standard inverse
        warnings.warn("Cholesky decomposition failed, using standard inverse")
        inv_cov = np.linalg.inv(cov_matrix)

    return inv_cov


def compute_eigen_decomposition(
    cov_matrix: np.ndarray, regularize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of covariance matrix.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix
    regularize : bool
        Whether to regularize before decomposition

    Returns:
    --------
    tuple
        (eigenvalues, eigenvectors)

    Examples:
    ---------
    >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> eigenvalues, eigenvectors = compute_eigen_decomposition(cov)
    """
    if regularize:
        cov_matrix = regularize_covariance_matrix(cov_matrix)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def compute_condition_number(cov_matrix: np.ndarray, regularize: bool = True) -> float:
    """
    Compute condition number of covariance matrix.

    The condition number indicates numerical stability:
    - < 10^6: Well-conditioned
    - 10^6 to 10^12: Moderately ill-conditioned
    - > 10^12: Ill-conditioned (may cause issues in inversion)

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix
    regularize : bool
        Whether to regularize before computation

    Returns:
    --------
    float
        Condition number (ratio of max to min eigenvalue)
    """
    if regularize:
        cov_matrix = regularize_covariance_matrix(cov_matrix)

    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Remove zeros

    if len(eigenvalues) == 0:
        return np.inf

    cond = np.max(eigenvalues) / np.min(eigenvalues)

    if cond > CONDITION_NUMBER_WARNING:
        warnings.warn(f"Matrix is ill-conditioned: {cond:.2e}")

    return cond


# =============================================================================
# Advanced Operations
# =============================================================================


def compute_mahalanobis_distance(
    observations: np.ndarray, predictions: np.ndarray, cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute Mahalanobis distance from predictions to observations.

    Mathematical Formulation:
    --------------------------
    $$D_M = \\sqrt{(y - \\mu)^T \\Sigma^{-1} (y - \\mu)}$$

    Parameters:
    -----------
    observations : np.ndarray
        Observed values (n,)
    predictions : np.ndarray
        Predicted values (n,)
    cov_matrix : np.ndarray
        Covariance matrix (n, n)

    Returns:
    --------
    np.ndarray
        Mahalanobis distances

    Examples:
    ---------
    >>> obs = np.array([1.0, 2.0])
    >>> pred = np.array([1.1, 2.1])
    >>> cov = np.array([[0.1, 0.05], [0.05, 0.1]])
    >>> dist = compute_mahalanobis_distance(obs, pred, cov)
    """
    residuals = observations - predictions

    # Compute precision matrix
    precision = invert_covariance_matrix(cov_matrix)

    # Compute Mahalanobis distance
    mahal = np.sqrt(residuals @ precision @ residuals)

    return mahal


def compute_chi_squared(
    observations: np.ndarray, predictions: np.ndarray, cov_matrix: np.ndarray
) -> float:
    """
    Compute chi-squared statistic for model evaluation.

    Mathematical Formulation:
    --------------------------
    $$\\chi^2 = (y - \\mu)^T \\Sigma^{-1} (y - \\mu)$$

    Parameters:
    -----------
    observations : np.ndarray
        Observed values
    predictions : np.ndarray
        Predicted values
    cov_matrix : np.ndarray
        Covariance matrix

    Returns:
    --------
    float
        Chi-squared value
    """
    mahal = compute_mahalanobis_distance(observations, predictions, cov_matrix)
    return mahal**2


def compute_log_likelihood(
    observations: np.ndarray, predictions: np.ndarray, cov_matrix: np.ndarray
) -> float:
    """
    Compute log-likelihood of observations given predictions and covariance.

    Mathematical Formulation:
    --------------------------
    $$\\log p(y | \\mu, \\Sigma) = -\\frac{1}{2} \\left[ \\chi^2 + \\log|\\Sigma| + n \\log(2\\pi) \\right]$$

    Parameters:
    -----------
    observations : np.ndarray
        Observed values
    predictions : np.ndarray
        Predicted values
    cov_matrix : np.ndarray
        Covariance matrix

    Returns:
    --------
    float
        Log-likelihood

    Examples:
    ---------
    >>> obs = np.array([1.0, 2.0, 3.0])
    >>> pred = np.array([1.1, 2.1, 2.9])
    >>> cov = np.eye(3) * 0.1
    >>> ll = compute_log_likelihood(obs, pred, cov)
    """
    residuals = observations - predictions
    n = len(observations)

    # Compute log determinant
    sign, logdet = np.linalg.slogdet(cov_matrix)
    if sign <= 0:
        warnings.warn("Covariance matrix has non-positive determinant")
        logdet = np.log(np.linalg.det(cov_matrix + 1e-10 * np.eye(n)))

    # Compute chi-squared
    chi2 = compute_chi_squared(observations, predictions, cov_matrix)

    # Compute log-likelihood
    log_likelihood = -0.5 * (chi2 + logdet + n * np.log(2 * np.pi))

    return log_likelihood


def compute_distance_matrix(
    coords: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute pairwise distance matrix for coordinates.

    Parameters:
    -----------
    coords : np.ndarray
        Coordinates of shape (N, D) where D is dimension
    metric : str
        Distance metric ('euclidean', 'haversine', 'great_circle')

    Returns:
    --------
    np.ndarray
        Distance matrix of shape (N, N)
    """
    if metric == "euclidean":
        from scipy.spatial.distance import cdist

        return cdist(coords, coords)

    elif metric == "haversine":
        # For 3D coordinates (lat, lon, time)
        # Convert to radians
        coords_rad = np.radians(coords)

        n = coords.shape[0]
        dist = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Haversine formula for first two dimensions
                lat1, lon1 = coords_rad[i, 0], coords_rad[i, 1]
                lat2, lon2 = coords_rad[j, 0], coords_rad[j, 1]

                dlat = lat2 - lat1
                dlon = lon2 - lon1

                a = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                )
                c = 2 * np.arcsin(np.sqrt(a))

                dist[i, j] = c
                dist[j, i] = c

        return dist

    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_taper_matrix(
    coords: np.ndarray, tau: float, dimension: int = 0, metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute Gaussian taper matrix for covariance localization.

    Mathematical Formulation:
    --------------------------
    $$T_{ij} = \\exp\\left(-\\frac{d_{ij}^2}{2\\tau^2}\\right)$$

    where $d_{ij}$ is the distance between points $i$ and $j$ along
    the specified dimension.

    Parameters:
    -----------
    coords : np.ndarray
        Coordinates
    tau : float
        Correlation length scale
    dimension : int
        Which dimension to use for tapering (0=time, 1=lat, 2=lon)
    metric : str
        Distance metric

    Returns:
    --------
    np.ndarray
        Taper matrix

    Examples:
    ---------
    >>> times = np.array([0, 1000, 2000])
    >>> taper = compute_taper_matrix(times[:, np.newaxis], tau=5000, dimension=0)
    """
    # Extract the relevant dimension
    coord_dim = coords[:, dimension]

    # Compute distance matrix for this dimension
    dist = np.abs(coord_dim[:, np.newaxis] - coord_dim[np.newaxis, :])

    # Apply Gaussian taper
    taper = np.exp(-(dist**2) / (2 * tau**2))

    return taper


# =============================================================================
# Validation Functions
# =============================================================================


def validate_covariance_input(
    data: np.ndarray, weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and preprocess inputs for covariance computation.

    Parameters:
    -----------
    data : np.ndarray
        Input data
    weights : np.ndarray, optional
        Weights

    Returns:
    --------
    tuple
        (validated_data, validated_weights)

    Raises:
    -------
    ValueError
        If inputs are invalid
    """
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.ndim != 2:
        raise ValueError(f"Data must be 1D or 2D, got {data.ndim}D")

    M, N = data.shape

    if M < 2:
        raise ValueError(f"Need at least 2 models, got {M}")

    # Validate weights
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != M:
            raise ValueError(f"Weights length {len(weights)} doesn't match models {M}")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        if weights.sum() == 0:
            raise ValueError("Weights sum to zero")
    else:
        weights = np.ones(M) / M

    return data, weights
