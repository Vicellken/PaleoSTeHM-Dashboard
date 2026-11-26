"""
Input/Output Handlers for GIA Covariance Tools
==============================================

Functions for loading GIA model data from various formats and exporting
covariance matrices and results.

Supported formats:
- NumPy binary (.npy, .npz)
- CSV files (.csv)
- NetCDF (.nc)
- PSTHM GIAGrid objects

Date: 2026-03-02
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Tuple, Callable
import warnings

try:
    from scipy.io import loadmat

    SCIPY_IO_AVAILABLE = True
except ImportError:
    SCIPY_IO_AVAILABLE = False

# Try to import PSTHM for GIAGrid support
try:
    from PSTHM import gia as psthm_gia

    PSTHM_AVAILABLE = True
except ImportError:
    PSTHM_AVAILABLE = False
    warnings.warn("PSTHM not available. Some loading functions will be limited.")

from .config import (
    DEFAULT_LAT_RANGE,
    DEFAULT_LON_RANGE,
    DEFAULT_TIME_KA_RANGE,
    DEFAULT_TIME_RESOLUTION_KA,
    REFERENCE_YEAR_CE,
    PRESENT_YEAR_CE,
    SUPPORTED_INPUT_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
)


# =============================================================================
# GIA Grid Loading Functions
# =============================================================================


def load_gia_grid(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    lat_range: Tuple[float, float] = DEFAULT_LAT_RANGE,
    lon_range: Tuple[float, float] = DEFAULT_LON_RANGE,
    time_ka_range: Tuple[float, float] = DEFAULT_TIME_KA_RANGE,
    time_resolution_ka: float = DEFAULT_TIME_RESOLUTION_KA,
    reference_year: int = REFERENCE_YEAR_CE,
    present_year: int = PRESENT_YEAR_CE,
    **kwargs,
) -> "GIAGrid":
    """
    Load a single GIA model grid from file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the GIA model file
    file_format : str, optional
        File format ('npy', 'csv', 'cube'). If None, inferred from extension.
    lat_range : tuple
        (min, max) latitude range for grid
    lon_range : tuple
        (min, max) longitude range for grid
    time_ka_range : tuple
        (min, max) time range in ka BP
    time_resolution_ka : float
        Time resolution in ka
    reference_year : int
        Reference year for BP/CE conversion (default: 1950)
    present_year : int
        Present year in CE (default: 2017)
    **kwargs
        Additional arguments passed to PSTHM GIAGrid loaders

    Returns:
    --------
    GIAGrid
        Loaded GIA grid object

    Examples:
    ---------
    >>> grid = load_gia_grid('ice7g.npy', lat_range=(-90, 90),
    ...                       lon_range=(0, 360), time_ka_range=(0, 26))
    """
    if not PSTHM_AVAILABLE:
        raise ImportError(
            "PSTHM is required for loading GIA grids. Install with: pip install PSTHM"
        )

    file_path = Path(file_path)

    if file_format is None:
        file_format = file_path.suffix.lower().replace(".", "")

    # Generate coordinate arrays
    lat = np.arange(lat_range[0], lat_range[1] + 1, 1)  # 1 degree resolution
    lon = np.arange(lon_range[0], lon_range[1] + 1, 1)
    times_ka = np.arange(time_ka_range[0], time_ka_range[1] + 1, time_resolution_ka)

    if file_format == "npy":
        grid = psthm_gia.GIAGrid.from_npy(
            file_path,
            lat=lat,
            lon=lon,
            times_ka=times_ka,
            reference_year=reference_year,
            present_ce=present_year,
            **kwargs,
        )
    elif file_format == "csv":
        grid = psthm_gia.GIAGrid.from_csv(
            file_path,
            lat_col="Lat",
            lon_col="Long",
            time_suffix="ka",
            reference_year=reference_year,
            present_ce=present_year,
            **kwargs,
        )
    elif file_format == "cube":
        grid = psthm_gia.GIAGrid.from_npy_cube(
            file_path,
            lon_range=(lon_range[0], lon_range[1] + 1, 1),
            lat_range=(lat_range[0], lat_range[1] + 1, 1),
            time_bp_range=(
                time_ka_range[1] * 1000,
                time_ka_range[0] * 1000,
                -time_resolution_ka * 1000,
            ),
            reference_year=reference_year,
            present_ce=present_year,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return grid


def load_ensemble_from_directory(
    directory: Union[str, Path],
    model_names: Optional[List[str]] = None,
    file_pattern: str = "*.npy",
    lat_range: Tuple[float, float] = DEFAULT_LAT_RANGE,
    lon_range: Tuple[float, float] = DEFAULT_LON_RANGE,
    time_ka_range: Tuple[float, float] = DEFAULT_TIME_KA_RANGE,
    **kwargs,
) -> Tuple[List, List[str]]:
    """
    Load multiple GIA model grids from a directory.

    Parameters:
    -----------
    directory : str or Path
        Directory containing GIA model files
    model_names : list of str, optional
        Specific model names to load. If None, loads all matching files.
    file_pattern : str
        Glob pattern for file matching
    lat_range, lon_range, time_ka_range
        Grid parameters (see load_gia_grid)
    **kwargs
        Additional arguments passed to load_gia_grid

    Returns:
    --------
    tuple
        (list of GIAGrid objects, list of model names)

    Examples:
    ---------
    >>> grids, names = load_ensemble_from_directory('./gia_models',
    ...                                             model_names=['ICE7G', 'ICE6G'])
    """
    directory = Path(directory)

    # Find all matching files
    if model_names is None:
        files = sorted(directory.glob(file_pattern))
    else:
        files = []
        for name in model_names:
            # Try different extensions
            for ext in ["", ".npy", ".csv", ".npz"]:
                f = directory / f"{name}{ext}"
                if f.exists():
                    files.append(f)
                    break

    grids = []
    names = []

    for file_path in files:
        try:
            grid = load_gia_grid(
                file_path,
                lat_range=lat_range,
                lon_range=lon_range,
                time_ka_range=time_ka_range,
                **kwargs,
            )
            grids.append(grid)
            names.append(file_path.stem)
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {e}")

    return grids, names


def load_mat_gia_grid(
    rsl_path: Union[str, Path],
    lat_path: Union[str, Path],
    lon_path: Union[str, Path],
    time_path: Union[str, Path],
    rsl_var_name: Optional[str] = None,
) -> Callable:
    """Load a single GIA RSL .mat file and return an interpolation callable.

    Builds a ``RegularGridInterpolator`` over (time_BP, lat, lon) from the
    MAT-format 3-D RSL cube and shared coordinate grids.

    Parameters
    ----------
    rsl_path : str or Path
        Path to the RSL ``.mat`` file (e.g. ``48p210.mat``).
    lat_path, lon_path, time_path : str or Path
        Paths to LatitudeGrid.mat, LongitudeGrid.mat, time.mat.
    rsl_var_name : str, optional
        Variable name inside the RSL file.  If *None*, the first 3-D array
        whose key does not start with ``__`` is used.

    Returns
    -------
    callable
        ``predict_fn(X) -> ndarray`` where *X* is (N, 3) as
        ``[time_BP, lat, lon]`` (lon in [0, 360)).
    """
    if not SCIPY_IO_AVAILABLE:
        raise ImportError("scipy is required for loading .mat files")

    from scipy.interpolate import RegularGridInterpolator

    # --- Load coordinate grids ---
    lat_mat = loadmat(str(lat_path))
    lon_mat = loadmat(str(lon_path))
    time_mat = loadmat(str(time_path))

    lat_1d = lat_mat["lat_out"][:, 0].astype(float)  # (256,)
    lon_1d = lon_mat["lon_out"][0, :].astype(float)  # (512,)
    time_ka = time_mat["ice_time_new"].flatten().astype(float)  # (42,) in ka
    time_bp = time_ka * 1000.0  # convert to years BP

    # --- Load RSL cube ---
    rsl_mat = loadmat(str(rsl_path))
    if rsl_var_name is not None:
        rsl_cube = rsl_mat[rsl_var_name]
    else:
        rsl_cube = None
        for key, val in rsl_mat.items():
            if key.startswith("__"):
                continue
            if hasattr(val, "ndim") and val.ndim == 3:
                rsl_cube = val
                break
        if rsl_cube is None:
            raise ValueError(f"No 3D array found in {rsl_path}")

    # rsl_cube is (lat=256, lon=512, time=42) → transpose to (time, lat, lon)
    rsl_cube = rsl_cube.astype(float).transpose(2, 0, 1)  # (42, 256, 512)

    # --- Ensure ascending axes ---
    # Time: ka goes 0→21 already ascending after *1000 → 0→21000 BP → ascending
    if time_bp[0] > time_bp[-1]:
        time_bp = time_bp[::-1]
        rsl_cube = rsl_cube[::-1, :, :]

    if lat_1d[0] > lat_1d[-1]:
        lat_1d = lat_1d[::-1]
        rsl_cube = rsl_cube[:, ::-1, :]

    if lon_1d[0] > lon_1d[-1]:
        lon_1d = lon_1d[::-1]
        rsl_cube = rsl_cube[:, :, ::-1]

    interp = RegularGridInterpolator(
        (time_bp, lat_1d, lon_1d),
        rsl_cube,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    def predict_fn(X):
        X = np.asarray(X, dtype=float)
        pts = X.copy()
        pts[:, 2] = pts[:, 2] % 360.0  # wrap lon to [0, 360)
        return interp(pts).flatten()

    return predict_fn


def load_mat_gia_ensemble(
    gia_dir: Union[str, Path],
    exclude_models: Optional[List[str]] = None,
    rsl_var_name: Optional[str] = None,
) -> Tuple[List[Callable], List[str]]:
    """Load all GIA RSL .mat files from a directory as interpolation callables.

    Parameters
    ----------
    gia_dir : str or Path
        Directory containing ``*.mat`` RSL files plus shared coordinate files
        (``LatitudeGrid.mat``, ``LongitudeGrid.mat``, ``time.mat``).
    exclude_models : list of str, optional
        Model stems to skip (e.g. ``["14055"]``).
    rsl_var_name : str, optional
        Passed to :func:`load_mat_gia_grid`.

    Returns
    -------
    interp_funcs : list of callable
        One interpolation function per model (same interface as
        ``create_interpolation_functions`` output).
    model_names : list of str
        Stem names of loaded models.
    """
    gia_dir = Path(gia_dir)
    coord_files = {"LatitudeGrid", "LongitudeGrid", "time"}
    exclude_set = set(exclude_models) if exclude_models else set()

    lat_path = gia_dir / "LatitudeGrid.mat"
    lon_path = gia_dir / "LongitudeGrid.mat"
    time_path = gia_dir / "time.mat"

    mat_files = sorted(gia_dir.glob("*.mat"))
    interp_funcs = []
    model_names = []

    for f in mat_files:
        stem = f.stem
        if stem in coord_files or stem in exclude_set:
            continue
        try:
            fn = load_mat_gia_grid(
                f, lat_path, lon_path, time_path, rsl_var_name=rsl_var_name
            )
            interp_funcs.append(fn)
            model_names.append(stem)
        except Exception as e:
            warnings.warn(f"Failed to load {f}: {e}")

    return interp_funcs, model_names


def compute_min_distance_weights_continuous(
    interp_funcs: List[Callable],
    ages_bp: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    y_obs: np.ndarray,
    y_sigma: np.ndarray,
    age_sigma: np.ndarray,
    bp_min: float = 0.0,
    bp_max: float = 21000.0,
    n_search: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute model weights via minimum-distance chi metric (Lin et al. 2025).

    Operates on continuous (lat, lon) interpolation functions.

    For each observation *i* and model *j*, the combined distance at
    candidate time *t* within a 2-sigma age window is:

    .. math::

        d_i(t) = \\frac{(y_i - g_j(x_i, t))^2}{\\sigma_{y_i}^2}
                 + \\frac{(\\hat{t}_i - t)^2}{\\sigma_{t_i}^2}

    The optimal *t** minimises *d_i*.  The per-model chi metric is:

    .. math::

        \\chi_j = \\frac{1}{n} \\sqrt{\\sum_{i=1}^{n} d_i(t_i^*)}

    Weights follow the exponential scheme:

    .. math::

        \\alpha_j = \\frac{\\exp(-0.5\\,\\chi_j^2)}
                         {\\sum_k \\exp(-0.5\\,\\chi_k^2)}

    Parameters
    ----------
    interp_funcs : list of callable
        One function per model, each accepting (N, 3) as [time_BP, lat, lon].
    ages_bp, lats, lons : array-like
        Observation coordinates (length N).
    y_obs, y_sigma, age_sigma : array-like
        Observation values and uncertainties (length N).
    bp_min, bp_max : float
        Bounds for the age search window.
    n_search : int
        Number of search points within each age window.

    Returns
    -------
    weights : np.ndarray, shape (M,)
        Normalised model weights.
    chi : np.ndarray, shape (M,)
        Per-model chi values.
    """
    ages_bp = np.asarray(ages_bp, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float) % 360.0
    y_obs = np.asarray(y_obs, dtype=float)
    y_sigma = np.asarray(y_sigma, dtype=float)
    age_sigma = np.asarray(age_sigma, dtype=float)
    N = len(ages_bp)
    M = len(interp_funcs)
    raw_sums = np.zeros(M)

    for m, fn in enumerate(interp_funcs):
        total = 0.0
        for i in range(N):
            t_lo = max(ages_bp[i] - 2 * age_sigma[i], bp_min)
            t_hi = min(ages_bp[i] + 2 * age_sigma[i], bp_max)
            t_search = np.linspace(t_lo, t_hi, n_search)
            X_search = np.column_stack(
                [t_search, np.full(n_search, lats[i]), np.full(n_search, lons[i])]
            )
            g_vals = fn(X_search)
            rsl_term = ((y_obs[i] - g_vals) / max(y_sigma[i], 1e-6)) ** 2
            age_term = ((ages_bp[i] - t_search) / max(age_sigma[i], 1e-6)) ** 2
            total += np.min(rsl_term + age_term)
        raw_sums[m] = total

    chi = (1.0 / N) * np.sqrt(raw_sums)

    log_w = -0.5 * chi**2
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    weights /= weights.sum()
    return weights, chi


def compute_wrss_weights_continuous(
    interp_funcs: List[Callable],
    ages_bp: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    y_obs: np.ndarray,
    y_sigma: np.ndarray,
    age_sigma: np.ndarray,
    limiting_flags: np.ndarray,
    bp_min: float = 0.0,
    bp_max: float = 21000.0,
    n_search: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GIA model weights using the WRSS/erf formulation.

    Implements the Weighted Residual Sum of Squares approach of
    Creel et al. (2024) and Briggs & Tarasov (2013).  Index points
    use a standard squared-residual score; limiting points use an
    error-function formulation that penalises models on the
    physically wrong side of a one-way constraint.

    For each observation *n* and model *m* the per-search-point score
    within the 2-sigma age window is:

    * Index (c = 0):
      W = (2 r_t / sigma_a)^2  +  (2 r_y / sigma_y)^2

    * Limiting (c = +/-1):
      W = (2 r_t / sigma_a)^2  -  2 ln(0.5 + 0.5 erf(c r_y / sigma_y))

    where r_y = y_obs - g_model  and  r_t = t_search - a_obs.

    The minimum W over the age window is taken per observation, then
    summed over all observations to give the aggregate misfit D_m.
    Weights are 1/D_m, normalised to sum to 1.

    Parameters
    ----------
    interp_funcs : list of callable
        One function per GIA model, each accepting (N, 3) arrays of
        [time_BP, lat, lon] and returning (N,) RSL predictions.
    ages_bp, lats, lons : array-like, shape (N,)
        Observation coordinates.
    y_obs, y_sigma, age_sigma : array-like, shape (N,)
        RSL observations, 1-sigma RSL uncertainties, and 1-sigma age
        uncertainties.
    limiting_flags : array-like, shape (N,)
        -1 for marine limiting, 0 for index, +1 for terrestrial
        limiting.
    bp_min, bp_max : float
        Hard bounds for the age search window (yr BP).
    n_search : int
        Number of equally spaced ages sampled within each 2-sigma
        age window.

    Returns
    -------
    weights : np.ndarray, shape (M,)
        Normalised inverse-misfit model weights.
    misfits : np.ndarray, shape (M,)
        Raw aggregate WRSS misfit per model.
    """
    from scipy.special import erf as _erf

    ages_bp = np.asarray(ages_bp, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float) % 360.0
    y_obs = np.asarray(y_obs, dtype=float)
    y_sigma = np.asarray(y_sigma, dtype=float)
    age_sigma = np.asarray(age_sigma, dtype=float)
    limiting_flags = np.asarray(limiting_flags, dtype=float)

    N = len(ages_bp)
    M = len(interp_funcs)
    misfits = np.zeros(M)

    _LN_FLOOR = 1e-30  # prevent log(0)

    for m, fn in enumerate(interp_funcs):
        total = 0.0
        for i in range(N):
            t_lo = max(ages_bp[i] - 2.0 * age_sigma[i], bp_min)
            t_hi = min(ages_bp[i] + 2.0 * age_sigma[i], bp_max)
            t_search = np.linspace(t_lo, t_hi, n_search)

            X_search = np.column_stack(
                [t_search, np.full(n_search, lats[i]), np.full(n_search, lons[i])]
            )
            g_vals = np.asarray(fn(X_search)).flatten()

            r_y = y_obs[i] - g_vals  # (n_search,)
            r_t = t_search - ages_bp[i]  # (n_search,)

            sig_y = max(y_sigma[i], 1e-6)
            sig_a = max(age_sigma[i], 1e-6)

            age_term = (2.0 * r_t / sig_a) ** 2

            c = limiting_flags[i]
            if c == 0:
                rsl_term = (2.0 * r_y / sig_y) ** 2
            else:
                arg = 0.5 + 0.5 * _erf(c * r_y / sig_y)
                arg = np.maximum(arg, _LN_FLOOR)
                rsl_term = -2.0 * np.log(arg)

            W = age_term + rsl_term
            total += float(np.min(W))

        misfits[m] = total

    weights = 1.0 / (misfits + 1e-10)
    weights /= weights.sum()
    return weights, misfits


def create_interpolation_functions(
    grids: List, names: List[str]
) -> Tuple[List[Callable], List[str]]:
    """
    Create interpolation functions from loaded GIA grids.

    Parameters:
    -----------
    grids : list
        List of GIAGrid objects
    names : list of str
        Names corresponding to each grid

    Returns:
    --------
    tuple
        (list of interpolation functions, list of names)

    Examples:
    ---------
    >>> grids, names = load_ensemble_from_directory('./gia_models')
    >>> interp_funcs, model_names = create_interpolation_functions(grids, names)
    >>> rsl = interp_funcs[0](np.array([[10000, 40, 290]]))
    """
    interp_functions = []

    for grid in grids:

        def make_interp_func(g):
            def interp_func(x):
                """Wrapper for GIAGrid prediction in BP format."""
                return g.predict_bp(x)

            return interp_func

        interp_functions.append(make_interp_func(grid))

    return interp_functions, names


# =============================================================================
# Prediction Functions
# =============================================================================


def predict_gia_at_points(
    grids: List,
    times_bp: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    use_ce: bool = False,
) -> np.ndarray:
    """
    Get GIA model predictions at specified points.

    Parameters:
    -----------
    grids : list
        List of GIAGrid objects
    times_bp : np.ndarray
        Times in years BP
    lats : np.ndarray
        Latitudes
    lons : np.ndarray
        Longitudes (0-360)
    use_ce : bool
        If True, times are in CE rather than BP

    Returns:
    --------
    np.ndarray
        Array of shape (n_models, n_points) with predictions
    """
    n_models = len(grids)

    # Create coordinate array
    times = times_bp if not use_ce else (REFERENCE_YEAR_CE - times_bp)
    coords = np.column_stack([times, lats, lons])

    predictions = np.zeros((n_models, len(times)))

    for i, grid in enumerate(grids):
        if use_ce:
            predictions[i] = grid.predict_ce(coords).flatten()
        else:
            predictions[i] = grid.predict_bp(coords).flatten()

    return predictions


def create_prediction_grid(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    time_ka_range: Tuple[float, float],
    resolution: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a regular grid for predictions.

    Parameters:
    -----------
    lat_range : tuple
        (min, max) latitude
    lon_range : tuple
        (min, max) longitude
    time_ka_range : tuple
        (min, max) time in ka BP
    resolution : dict, optional
        Resolution parameters with keys 'lat', 'lon', 'time_ka'

    Returns:
    --------
    tuple
        (lat_grid, lon_grid, time_ka_grid, coords_array)
    """
    if resolution is None:
        resolution = {"lat": 1, "lon": 1, "time_ka": 1}

    lat = np.arange(lat_range[0], lat_range[1] + 1, resolution.get("lat", 1))
    lon = np.arange(lon_range[0], lon_range[1] + 1, resolution.get("lon", 1))
    time_ka = np.arange(
        time_ka_range[0], time_ka_range[1] + 1, resolution.get("time_ka", 1)
    )

    # Create meshgrid
    lon_mesh, lat_mesh, time_mesh = np.meshgrid(lon, lat, time_ka)

    # Create coordinate array (time in years BP)
    coords = np.column_stack(
        [
            time_mesh.flatten() * 1000,  # Convert ka to years BP
            lat_mesh.flatten(),
            lon_mesh.flatten(),
        ]
    )

    return lat, lon, time_ka, coords


# =============================================================================
# Export Functions
# =============================================================================


def export_covariance_matrix(
    cov_matrix: np.ndarray,
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Export a covariance matrix to file.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix to export
    file_path : str or Path
        Output file path
    file_format : str, optional
        Output format ('npy', 'csv', 'npz'). If None, inferred from extension.
    metadata : dict, optional
        Additional metadata to save (for npz format)

    Examples:
    ---------
    >>> export_covariance_matrix(cov_matrix, 'cov.npy')
    >>> export_covariance_matrix(cov_matrix, 'cov.csv')
    """
    file_path = Path(file_path)

    if file_format is None:
        file_format = file_path.suffix.lower().replace(".", "")

    if file_format == "npy":
        np.save(file_path, cov_matrix)
    elif file_format == "csv":
        np.savetxt(file_path, cov_matrix, delimiter=",")
    elif file_format == "npz":
        data = {"covariance": cov_matrix}
        if metadata:
            data.update(metadata)
        np.savez(file_path, **data)
    else:
        raise ValueError(f"Unsupported output format: {file_format}")


def export_results(
    results: dict, output_dir: Union[str, Path], prefix: str = "gia_cov"
) -> dict:
    """
    Export all results from covariance analysis.

    Parameters:
    -----------
    results : dict
        Dictionary containing:
        - 'covariance': covariance matrix
        - 'mean': mean predictions
        - 'coords': coordinate array
        - 'metadata': analysis metadata
    output_dir : str or Path
        Output directory
    prefix : str
        Filename prefix

    Returns:
    --------
    dict
        Dictionary of exported file paths

    Examples:
    ---------
    >>> results = {
    ...     'covariance': cov_matrix,
    ...     'mean': mean_pred,
    ...     'coords': coords,
    ...     'metadata': {'n_models': 3, 'time_range': [0, 26]}
    ... }
    >>> exported = export_results(results, './output', 'ice7g_ice6g')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    # Export covariance matrix
    if "covariance" in results:
        cov_path = output_dir / f"{prefix}_covariance.npy"
        export_covariance_matrix(
            results["covariance"], cov_path, metadata=results.get("metadata")
        )
        exported["covariance"] = str(cov_path)

    # Export mean predictions
    if "mean" in results:
        mean_path = output_dir / f"{prefix}_mean.npy"
        np.save(mean_path, results["mean"])
        exported["mean"] = str(mean_path)

    # Export coordinates
    if "coords" in results:
        coords_path = output_dir / f"{prefix}_coords.npy"
        np.save(coords_path, results["coords"])
        exported["coords"] = str(coords_path)

    # Export metadata as JSON
    if "metadata" in results:
        import json

        metadata_path = output_dir / f"{prefix}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(results["metadata"], f, indent=2)
        exported["metadata"] = str(metadata_path)

    return exported


# =============================================================================
# Validation Data Loading
# =============================================================================


def load_validation_data(file_path: Union[str, Path], format: str = "csv") -> dict:
    """
    Load validation/observation data for GIA model comparison.

    Parameters:
    -----------
    file_path : str or Path
        Path to validation data file
    format : str
        Data format ('csv', 'npy')

    Returns:
    --------
    dict
        Dictionary with keys:
        - 'coords': (n, 3) array of [time_BP, lat, lon]
        - 'rsl': (n,) array of RSL observations
        - 'rsl_unc': (n,) array of RSL uncertainties (if available)
    """
    file_path = Path(file_path)

    if format == "csv" or file_path.suffix == ".csv":
        df = pd.read_csv(file_path)

        # Try to identify column names
        coord_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if "time" in col_lower or "age" in col_lower:
                coord_cols.append(col)
            elif "lat" in col_lower:
                lat_col = col
            elif "lon" in col_lower or "long" in col_lower:
                lon_col = col

        # Build coordinate array
        coords = np.column_stack(
            [
                df[coord_cols[0]].values if coord_cols else np.zeros(len(df)),
                df[lat_col].values,
                df[lon_col].values,
            ]
        )

        # Find RSL column
        rsl_col = None
        for col in df.columns:
            if "rsl" in col.lower() or "sea" in col.lower():
                rsl_col = col
                break

        if rsl_col is None:
            raise ValueError("Could not identify RSL column in data")

        result = {
            "coords": coords,
            "rsl": df[rsl_col].values,
        }

        # Try to find uncertainty column
        for col in df.columns:
            if "unc" in col.lower() or "err" in col.lower() or "sigma" in col.lower():
                result["rsl_unc"] = df[col].values
                break

        return result

    elif format == "npy" or file_path.suffix == ".npy":
        data = np.load(file_path, allow_pickle=True)
        if isinstance(data, np.ndarray):
            return {"data": data}
        return dict(data.item())

    else:
        raise ValueError(f"Unsupported format: {format}")


def compute_model_weights_from_validation(
    grids: List, validation_data: dict, method: str = "inverse_mse"
) -> np.ndarray:
    """
    Compute weights for GIA models based on validation data.

    Parameters:
    -----------
    grids : list
        List of GIAGrid objects
    validation_data : dict
        Validation data from load_validation_data
    method : str
        Weighting method:
        - 'inverse_mse': Inverse mean squared error
        - 'inverse_mae': Inverse mean absolute error
        - 'log_likelihood': Based on observation uncertainties

    Returns:
    --------
    np.ndarray
        Weights for each model (sum to 1)
    """
    coords = validation_data["coords"]
    observations = validation_data["rsl"]
    uncertainties = validation_data.get(
        "rsl_unc", np.ones(len(observations)) * 5.0
    )  # Default 5m uncertainty

    n_models = len(grids)
    errors = np.zeros(n_models)

    for i, grid in enumerate(grids):
        predictions = grid.predict_bp(coords).flatten()
        residuals = observations - predictions

        if method == "inverse_mse":
            errors[i] = np.mean(residuals**2)
        elif method == "inverse_mae":
            errors[i] = np.mean(np.abs(residuals))
        elif method == "log_likelihood":
            # Compute log likelihood assuming Gaussian errors
            variances = uncertainties**2
            log_likelihood = -0.5 * np.sum(
                residuals**2 / variances + np.log(2 * np.pi * variances)
            )
            errors[i] = -log_likelihood  # Negative for minimization

    # Convert to weights (inverse of errors)
    weights = 1.0 / (errors + 1e-10)  # Add small constant to avoid division by zero
    weights = weights / weights.sum()  # Normalize

    return weights


# =============================================================================
# Region-Based GIA Model Adapter
# =============================================================================


def make_region_interp_funcs(
    all_model_interps: List[dict],
    region_centroids: dict,
) -> List[Callable]:
    """Create (N,3)-callable interpolation functions from region-based GIA models.

    This adapter wraps region-based GIA model interpolators (keyed by region
    number) so they accept the standard ``(N, 3)`` coordinate format
    ``[time_BP, lat, lon]`` used throughout gia_covariance_tools.  Spatial
    points are assigned to the nearest region centroid.

    Parameters
    ----------
    all_model_interps : list of dict
        One dict per GIA model.  Each dict maps ``region_num -> interp1d``
        (a callable that takes an array of ages in BP and returns RSL).
    region_centroids : dict
        Maps ``region_num -> (lat, lon)`` giving the centroid of each region.

    Returns
    -------
    list of callable
        One function per model, each accepting ``X`` of shape ``(N, 3)`` as
        ``[time_BP, lat, lon]`` and returning an ``(N,)`` prediction array.

    Examples
    --------
    >>> centroids = {1: (22.0, 114.0), 2: (10.0, 108.0)}
    >>> funcs = make_region_interp_funcs(gia._all_model_interps, centroids)
    >>> X_test = np.array([[5000, 22.1, 114.2], [3000, 10.5, 108.1]])
    >>> preds = funcs[0](X_test)
    """
    centroid_rns = np.array(list(region_centroids.keys()))
    centroid_lats = np.array([region_centroids[rn][0] for rn in centroid_rns])
    centroid_lons = np.array([region_centroids[rn][1] for rn in centroid_rns])

    def _assign_regions(lats, lons):
        """Assign each (lat, lon) to the nearest region centroid."""
        d2 = (lats[:, None] - centroid_lats[None, :]) ** 2 + (
            lons[:, None] - centroid_lons[None, :]
        ) ** 2
        return centroid_rns[np.argmin(d2, axis=1)]

    funcs = []
    for model_interps in all_model_interps:

        def predict_fn(X, mi=model_interps):
            X = np.asarray(X)
            ages = X[:, 0]
            lats = X[:, 1]
            lons = X[:, 2]
            rns = _assign_regions(lats, lons)
            result = np.zeros(len(X))
            for rn in np.unique(rns):
                mask = rns == rn
                result[mask] = mi[rn](ages[mask])
            return result

        funcs.append(predict_fn)
    return funcs


def compute_min_distance_weights(
    all_model_interps: List[dict],
    ages_bp: np.ndarray,
    region_nums: np.ndarray,
    y_obs: np.ndarray,
    y_sigma: np.ndarray,
    age_sigma: np.ndarray,
    bp_min: float = 0.0,
    bp_max: float = 26000.0,
    n_search: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute model weights via minimum-distance chi metric (Lin et al. 2025).

    Region-based variant that operates on ``region_num -> interp1d`` dicts.

    For each observation *i* and model *k*, the combined distance at
    candidate time *t* within a 2-sigma age window is:

    .. math::

        d_i(t) = \\frac{(y_i - g_k(t, \\text{region}_i))^2}{\\sigma_{y_i}^2}
                 + \\frac{(\\hat{t}_i - t)^2}{\\sigma_{t_i}^2}

    The optimal *t** minimises *d_i*.  The per-model chi metric is:

    .. math::

        \\chi_k = \\frac{1}{n} \\sqrt{\\sum_{i=1}^{n} d_i(t_i^*)}

    Weights follow the exponential scheme:

    .. math::

        \\alpha_k = \\frac{\\exp(-0.5\\,\\chi_k^2)}
                         {\\sum_j \\exp(-0.5\\,\\chi_j^2)}

    Parameters
    ----------
    all_model_interps : list of dict
        One dict per model, mapping ``region_num -> interp1d``.
    ages_bp, region_nums, y_obs, y_sigma, age_sigma : array-like
        Observation arrays (length N).
    bp_min, bp_max : float
        Bounds for the age search window.
    n_search : int
        Number of search points within each age window.

    Returns
    -------
    weights : np.ndarray, shape (M,)
        Normalised model weights.
    chi : np.ndarray, shape (M,)
        Per-model chi values.
    """
    ages_bp = np.asarray(ages_bp, dtype=float)
    region_nums = np.asarray(region_nums, dtype=int)
    y_obs = np.asarray(y_obs, dtype=float)
    y_sigma = np.asarray(y_sigma, dtype=float)
    age_sigma = np.asarray(age_sigma, dtype=float)
    N = len(ages_bp)
    M = len(all_model_interps)
    raw_sums = np.zeros(M)

    for m, model_interps in enumerate(all_model_interps):
        total = 0.0
        for i in range(N):
            t_lo = max(ages_bp[i] - 2 * age_sigma[i], bp_min)
            t_hi = min(ages_bp[i] + 2 * age_sigma[i], bp_max)
            t_search = np.linspace(t_lo, t_hi, n_search)
            g_vals = model_interps[region_nums[i]](t_search)
            rsl_term = ((y_obs[i] - g_vals) / max(y_sigma[i], 1e-6)) ** 2
            age_term = ((ages_bp[i] - t_search) / max(age_sigma[i], 1e-6)) ** 2
            total += np.min(rsl_term + age_term)
        raw_sums[m] = total

    chi = (1.0 / N) * np.sqrt(raw_sums)

    log_w = -0.5 * chi**2
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    weights /= weights.sum()
    return weights, chi
