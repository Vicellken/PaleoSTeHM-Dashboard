"""
Visualization Functions for GIA Covariance Analysis
====================================================

Provides plotting functions for covariance matrices, model comparisons,
and spatial/temporal analysis of GIA ensembles.

Date: 2026-03-02
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from typing import Optional, Tuple, List, Dict
import warnings

# Try to import cartopy for geographic plots
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    warnings.warn("Cartopy not available. Geographic plots will be limited.")

from .config import (
    DEFAULT_FIGURE_SIZE,
    DEFAULT_DPI,
    DEFAULT_CMAP_DIVERGENT,
    DEFAULT_CMAP_SEQUENTIAL,
    DEFAULT_CMAP_DIVERGENCE,
    MODEL_COLORS,
)


# =============================================================================
# Covariance Matrix Plots
# =============================================================================


def plot_covariance_matrix(
    cov_matrix: np.ndarray,
    coords: Optional[np.ndarray] = None,
    title: str = "GIA Model Covariance Matrix",
    cmap: str = DEFAULT_CMAP_DIVERGENT,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGURE_SIZE,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a covariance matrix as a heatmap.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix (N, N)
    coords : np.ndarray, optional
        Coordinates (N, 3) for axis labels
    title : str
        Plot title
    cmap : str
        Colormap
    vmin, vmax : float, optional
        Color limits
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine color limits
    if vmin is None or vmax is None:
        vmax = np.max(np.abs(cov_matrix))
        if vmin is None:
            vmin = -vmax

    # Create norm for divergent colormap
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plot
    im = ax.imshow(cov_matrix, cmap=cmap, norm=norm, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Covariance (m²)")

    # Add coordinate labels if provided
    if coords is not None:
        n = len(coords)
        if n <= 20:
            tick_step = max(1, n // 10)
            ticks = range(0, n, tick_step)
            tick_labels = [
                f"{coords[i, 0]/1000:.1f}ka\n{coords[i, 1]:.0f}°N" for i in ticks
            ]
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels)

    ax.set_title(title)
    ax.set_xlabel("Prediction Point")
    ax.set_ylabel("Prediction Point")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


def plot_correlation_matrix(
    cov_matrix: np.ndarray,
    title: str = "GIA Model Correlation Matrix",
    cmap: str = DEFAULT_CMAP_DIVERGENT,
    figsize: Tuple[int, int] = DEFAULT_FIGURE_SIZE,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot correlation matrix derived from covariance.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix
    title : str
        Plot title
    cmap : str
        Colormap
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
    """
    from .core_math import compute_correlation_from_covariance

    # Convert to correlation
    corr = compute_correlation_from_covariance(cov_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot with fixed range [-1, 1]
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr, cmap=cmap, norm=norm, aspect="auto")

    cbar = plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title(title)
    ax.set_xlabel("Prediction Point")
    ax.set_ylabel("Prediction Point")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


def plot_covariance_spectrum(
    cov_matrix: np.ndarray,
    n_components_show: int = 20,
    title: str = "Covariance Spectrum",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot eigenvalues (spectrum) of covariance matrix.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix
    n_components_show : int
        Number of components to show
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
    """
    from .core_math import compute_eigen_decomposition

    eigenvalues, _ = compute_eigen_decomposition(cov_matrix)
    eigenvalues = np.abs(eigenvalues)  # Ensure positive

    # Compute explained variance
    total_var = np.sum(eigenvalues)
    explained_var = eigenvalues / total_var
    cumulative_var = np.cumsum(explained_var)

    n_show = min(n_components_show, len(eigenvalues))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Individual eigenvalues (log scale)
    ax1 = axes[0]
    ax1.semilogy(range(1, n_show + 1), eigenvalues[:n_show], "o-", markersize=4)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Eigenvalue (log scale)")
    ax1.set_title("Eigenvalues")
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    ax2 = axes[1]
    ax2.plot(range(1, n_show + 1), cumulative_var[:n_show] * 100, "o-", markersize=4)
    ax2.axhline(y=95, color="r", linestyle="--", label="95% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Explained Variance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Find 95% component
    n_95 = np.searchsorted(cumulative_var, 0.95) + 1
    ax2.annotate(
        f"{n_95} components\nfor 95%",
        xy=(n_95, 95),
        xytext=(n_95 + 2, 80),
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


# =============================================================================
# Spatial Plots
# =============================================================================


def plot_spatial_disagreement_map(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    disagreement: np.ndarray,
    title: str = "GIA Model Disagreement",
    cmap: str = DEFAULT_CMAP_DIVERGENCE,
    projection: str = "PlateCarree",
    add_coastline: bool = True,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot spatial map of model disagreement.

    Parameters:
    -----------
    lat_grid : np.ndarray
        Latitude grid
    lon_grid : np.ndarray
        Longitude grid
    disagreement : np.ndarray
        Disagreement values (same shape as meshgrid)
    title : str
        Plot title
    cmap : str
        Colormap
    projection : str
        Cartopy projection name
    add_coastline : bool
        Whether to add coastline
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
    """
    if CARTOPY_AVAILABLE:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=getattr(ccrs, projection)())

        # Convert lon to 0-360 if needed
        lon_plot = lon_grid.copy()

        # Create meshgrid for plotting
        lon_mesh, lat_mesh = np.meshgrid(lon_plot, lat_grid)

        # Plot
        im = ax.pcolormesh(
            lon_mesh,
            lat_mesh,
            disagreement,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )

        if add_coastline:
            ax.coastlines()
            ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)

        cbar = plt.colorbar(
            im, ax=ax, orientation="vertical", shrink=0.7, label="Model Variance (m²)"
        )

        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    else:
        # Fallback without cartopy
        fig, ax = plt.subplots(figsize=figsize)

        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        im = ax.pcolormesh(lon_mesh, lat_mesh, disagreement, cmap=cmap, shading="auto")

        cbar = plt.colorbar(im, ax=ax, label="Model Variance (m²)")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


def plot_spatial_covariance_slice(
    cov_matrix: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    time_index: int,
    times_bp: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    cmap: str = DEFAULT_CMAP_DIVERGENT,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot spatial slice of covariance matrix at a specific time.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Full covariance matrix
    lat_grid : np.ndarray
        Latitude grid
    lon_grid : np.ndarray
        Longitude grid
    time_index : int
        Time index to slice at
    times_bp : np.ndarray
        Times in years BP
    vmin, vmax : float, optional
        Color limits
    title : str, optional
        Plot title
    cmap : str
        Colormap
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
    """
    n_lat = len(lat_grid)
    n_lon = len(lon_grid)

    # Extract spatial slice
    start_idx = time_index * n_lat * n_lon
    end_idx = start_idx + n_lat * n_lon

    # Reshape to spatial grid
    cov_slice = cov_matrix[start_idx:end_idx, start_idx:end_idx]

    if title is None:
        title = f"Spatial Covariance at {times_bp[time_index]/1000:.1f} ka BP"

    return plot_covariance_matrix(
        cov_slice,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        save_path=save_path,
    )


# =============================================================================
# Temporal Plots
# =============================================================================


def plot_temporal_structure(
    times_bp: np.ndarray,
    cov_matrix: np.ndarray,
    location_label: str = "",
    title: Optional[str] = None,
    cmap: str = DEFAULT_CMAP_DIVERGENT,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot temporal covariance structure as heatmap.

    Parameters:
    -----------
    times_bp : np.ndarray
        Times in years BP
    cov_matrix : np.ndarray
        Temporal covariance matrix
    location_label : str
        Label for location
    title : str, optional
        Plot title
    cmap : str
        Colormap
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
    """
    if title is None:
        title = f"Temporal Covariance Structure {location_label}"

    # Convert times to ka for labels
    times_ka = times_bp / 1000

    fig, ax = plt.subplots(figsize=figsize)

    # Determine color limits
    vmax = np.max(np.abs(cov_matrix))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(
        cov_matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        extent=[times_ka[0], times_ka[-1], times_ka[-1], times_ka[0]],
    )

    cbar = plt.colorbar(im, ax=ax, label="Covariance (m²)")

    ax.set_xlabel("Time (ka BP)")
    ax.set_ylabel("Time (ka BP)")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


def plot_temporal_variance(
    times_bp: np.ndarray,
    cov_matrix: np.ndarray,
    location_label: str = "",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot temporal evolution of variance.

    Parameters:
    -----------
    times_bp : np.ndarray
        Times in years BP
    cov_matrix : np.ndarray
        Temporal covariance matrix
    location_label : str
        Label for location
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
    """
    if title is None:
        title = f"Temporal Variance {location_label}"

    # Extract diagonal (variances)
    variances = np.diag(cov_matrix)
    std = np.sqrt(np.maximum(variances, 0))  # Ensure positive

    times_ka = times_bp / 1000

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(times_ka, std, "o-", markersize=4, linewidth=1.5)
    ax.fill_between(times_ka, 0, std, alpha=0.3)

    ax.set_xlabel("Time (ka BP)")
    ax.set_ylabel("Standard Deviation (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Time going back into past

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


# =============================================================================
# Model Comparison Plots
# =============================================================================


def plot_model_comparison(
    gia_interp_list: List[Callable],
    X_points: np.ndarray,
    model_names: Optional[List[str]] = None,
    title: str = "GIA Model Predictions",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predictions from different GIA models.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of GIA model interpolation functions
    X_points : np.ndarray
        Prediction points
    model_names : list of str, optional
        Names for each model
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
    """
    n_models = len(gia_interp_list)

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]

    # Get predictions
    predictions = np.array(
        [np.asarray(interp(X_points)).flatten() for interp in gia_interp_list]
    )

    # Assume time is first coordinate
    times = X_points[:, 0]

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_models):
        ax.plot(
            times / 1000,
            predictions[i],
            "-",
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            label=model_names[i],
            linewidth=1.5,
            alpha=0.8,
        )

    # Add mean
    mean_pred = np.mean(predictions, axis=0)
    ax.plot(times / 1000, mean_pred, "k--", label="Ensemble Mean", linewidth=2)

    ax.set_xlabel("Time (ka BP)")
    ax.set_ylabel("RSL (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


def plot_model_difference_heatmap(
    gia_interp_list: List[Callable],
    X_points: np.ndarray,
    reference_model_index: int = 0,
    model_names: Optional[List[str]] = None,
    title: str = "Model Differences from Reference",
    cmap: str = DEFAULT_CMAP_DIVERGENT,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot heatmap of differences between models.

    Parameters:
    -----------
    gia_interp_list : list of callable
        List of GIA model interpolation functions
    X_points : np.ndarray
        Prediction points
    reference_model_index : int
        Index of reference model
    model_names : list of str, optional
        Names for each model
    title : str
        Plot title
    cmap : str
        Colormap
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
    """
    n_models = len(gia_interp_list)

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]

    # Get predictions
    predictions = np.array(
        [np.asarray(interp(X_points)).flatten() for interp in gia_interp_list]
    )

    # Compute differences from reference
    reference = predictions[reference_model_index]
    differences = predictions - reference

    # Times for x-axis
    times = X_points[:, 0]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        differences,
        cmap=cmap,
        aspect="auto",
        extent=[times[0] / 1000, times[-1] / 1000, n_models - 0.5, -0.5],
    )

    # Set y-ticks
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names)

    cbar = plt.colorbar(im, ax=ax, label="Difference from Reference (m)")

    ax.set_xlabel("Time (ka BP)")
    ax.set_title(title)
    ax.invert_xaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


# =============================================================================
# Summary Plots
# =============================================================================


def plot_validation_summary(
    validation_results: Dict,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot summary of covariance validation results.

    Parameters:
    -----------
    validation_results : dict
        Results from validate_covariance_matrix
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Symmetry
    ax1 = axes[0]
    is_sym = validation_results.get("is_symmetric", None)
    if is_sym is not None:
        color = "green" if is_sym else "red"
        ax1.bar(["Symmetric"], [1], color=color)
        ax1.set_title("Symmetry Check")
        ax1.set_ylim(0, 1.5)
        ax1.text(
            0,
            1.2,
            "PASS" if is_sym else "FAIL",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    # Positive Definiteness
    ax2 = axes[1]
    is_pd = validation_results.get("is_positive_definite", None)
    if is_pd is not None:
        color = "green" if is_pd else "orange"
        ax2.bar(["Positive Definite"], [1], color=color)
        ax2.set_title("Positive Definiteness")
        ax2.set_ylim(0, 1.5)
        ax2.text(
            0,
            1.2,
            "PASS" if is_pd else "WARNING",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    # Condition Number
    ax3 = axes[2]
    cond = validation_results.get("condition_number", None)
    if cond is not None:
        log_cond = np.log10(cond) if cond > 0 else 0
        ax3.bar(
            ["Condition Number"], [log_cond], color="orange" if log_cond < 12 else "red"
        )
        ax3.set_title("Condition Number (log10)")
        ax3.set_ylim(0, 16)
        ax3.axhline(y=12, color="r", linestyle="--", label="Warning threshold")
        ax3.text(0, 14, f"{cond:.2e}", ha="center", fontsize=10)
        ax3.legend()

    # Warnings
    warnings_list = validation_results.get("warnings", [])
    if warnings_list:
        fig.text(
            0.5,
            0.02,
            "\n".join([f"Warning: {w}" for w in warnings_list]),
            ha="center",
            fontsize=10,
            color="orange",
        )

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig
