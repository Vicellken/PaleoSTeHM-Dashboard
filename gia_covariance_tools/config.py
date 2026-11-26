"""
Configuration Module for GIA Covariance Tools
============================================

Contains default parameters, constants, and configuration settings
for GIA model covariance calculations.

Date: 2026-03-02
"""

import numpy as np
from pathlib import Path

# =============================================================================
# Physical Constants
# =============================================================================

# Earth parameters
EARTH_RADIUS_KM = 6371.0  # Mean Earth radius in kilometers
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000  # in meters

# Time conversions
YEARS_PER_KA = 1000  # 1 ka = 1000 years
YEARS_PER_MA = 1_000_000  # 1 Ma = 1,000,000 years

# =============================================================================
# Default Reference Years
# =============================================================================

REFERENCE_YEAR_CE = 1950  # Standard reference year for BP/CE conversion
PRESENT_YEAR_CE = 2017  # Default present year for CE time axis

# =============================================================================
# Default GIA Grid Parameters
# =============================================================================

DEFAULT_LAT_RANGE = (-90, 90)
DEFAULT_LON_RANGE = (0, 360)
DEFAULT_TIME_KA_RANGE = (0, 26)  # 0-26 ka BP
DEFAULT_TIME_RESOLUTION_KA = 1.0  # 1 ka resolution
DEFAULT_SPATIAL_RESOLUTION_DEG = 1.0  # 1 degree resolution

# =============================================================================
# Covariance Computation Parameters
# =============================================================================

# Regularization
DEFAULT_REGULARIZATION_EPSILON = 1e-8  # For matrix stabilization
DEFAULT_MIN_VARIANCE = 1e-12  # Minimum variance on diagonal

# Taper parameters
DEFAULT_TEMPORAL_TAPER_TAU_KA = 5.0  # 5 ka default temporal taper
DEFAULT_SPATIAL_TAPER_TAU_DEG = 10.0  # 10 degree default spatial taper

# Numerical tolerances
EIGENVALUE_TOLERANCE = 1e-10  # For positive-definiteness check
CONDITION_NUMBER_WARNING = 1e12  # Warning threshold for ill-conditioning
CORRELATION_ABS_TOLERANCE = 1.0 - 1e-10  # For near-singular correlation detection

# =============================================================================
# Supported Data Formats
# =============================================================================

SUPPORTED_INPUT_FORMATS = [
    ".npy",  # NumPy binary format
    ".npz",  # NumPy archive
    ".csv",  # CSV files
    ".nc",  # NetCDF
    ".h5",  # HDF5
    ".mat",  # MATLAB MAT files
]

SUPPORTED_OUTPUT_FORMATS = [
    ".npy",  # NumPy binary
    ".csv",  # CSV text
    ".npz",  # NumPy archive
]

# =============================================================================
# PSTHM Integration Settings
# =============================================================================

# Default kernel parameters for GIA integration
DEFAULT_KERNEL_LENGTHSCALE = 5000.0  # years
DEFAULT_KERNEL_VARIANCE = 100.0  # m²

# GP optimization settings
DEFAULT_NUM_ITERATIONS = 1000
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_JITTER = 1e-6

# =============================================================================
# Visualization Settings
# =============================================================================

# Plotting defaults
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 100
DEFAULT_CMAP_DIVERGENT = "RdBu_r"
DEFAULT_CMAP_SEQUENTIAL = "viridis"
DEFAULT_CMAP_DIVERGENCE = "YlOrRd"

# Color schemes for model comparison
MODEL_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# =============================================================================
# File Path Configuration
# =============================================================================

# Default data directories (can be overridden)
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_OUTPUT_DIR = Path("./output")
DEFAULT_MODEL_DIR = Path("./models")

# Default file naming
COVARIANCE_MATRIX_FILENAME = "gia_covariance_matrix"
MEAN_PREDICTION_FILENAME = "gia_mean_prediction"
VALIDATION_REPORT_FILENAME = "covariance_validation_report"

# =============================================================================
# Validation Settings
# =============================================================================

# Covariance matrix validation
VALIDATION_CHECKS = {
    "symmetry": True,
    "positive_definiteness": True,
    "diagonal_positivity": True,
    "finite_values": True,
    "shape_consistency": True,
}

# Warning thresholds
MAX_CONDITION_NUMBER = 1e15
MIN_EIGENVALUE = -1e-10  # Negative eigenvalues indicate non-positive-definiteness

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# Earth Model Names (Common GIA Models)
# =============================================================================

KNOWN_GIA_MODELS = {
    "ICE7G": "ICE-7G (VM7)",
    "ICE6G": "ICE-6G (VM5a)",
    "ANU": "ANU",
    "K29": "K29",
    "JB": "JB",
    "RR": "Rayner et al.",
    "MIT": "MIT",
    "GLO": "Global",
}

# =============================================================================
# Utility Functions
# =============================================================================


def get_default_config():
    """
    Returns a dictionary of all default configuration parameters.

    Returns:
    --------
    dict
        Dictionary containing all default settings
    """
    return {
        "earth_radius_km": EARTH_RADIUS_KM,
        "reference_year_ce": REFERENCE_YEAR_CE,
        "present_year_ce": PRESENT_YEAR_CE,
        "default_temporal_taper_ka": DEFAULT_TEMPORAL_TAPER_TAU_KA,
        "regularization_epsilon": DEFAULT_REGULARIZATION_EPSILON,
        "min_variance": DEFAULT_MIN_VARIANCE,
        "default_kernel_lengthscale": DEFAULT_KERNEL_LENGTHSCALE,
        "default_kernel_variance": DEFAULT_KERNEL_VARIANCE,
        "validation_checks": VALIDATION_CHECKS.copy(),
    }


def update_config(**kwargs):
    """
    Update configuration parameters.

    Parameters:
    -----------
    **kwargs
        Key-value pairs to update in the configuration

    Returns:
    --------
    dict
        Updated configuration dictionary
    """
    config = get_default_config()
    config.update(kwargs)
    return config


# =============================================================================
# Unit Conversion Functions
# =============================================================================


def ka_to_years_bp(ka):
    """Convert ka to years BP."""
    return np.asarray(ka) * YEARS_PER_KA


def years_bp_to_ka(years_bp):
    """Convert years BP to ka."""
    return np.asarray(years_bp) / YEARS_PER_KA


def ce_to_bp(years_ce, reference_year=REFERENCE_YEAR_CE):
    """Convert Common Era (CE) years to Before Present (BP)."""
    return reference_year - np.asarray(years_ce)


def bp_to_ce(years_bp, reference_year=REFERENCE_YEAR_CE):
    """Convert Before Present (BP) to Common Era (CE) years."""
    return reference_year - np.asarray(years_bp)


def km_to_degrees(km, latitude=45):
    """
    Convert kilometers to degrees at a given latitude.

    Parameters:
    -----------
    km : float or array
        Distance in kilometers
    latitude : float
        Reference latitude in degrees

    Returns:
    --------
    float or array
        Distance in degrees
    """
    # At equator: 1 degree ≈ 111 km
    # At latitude θ: 1 degree ≈ 111 * cos(θ) km
    lat_rad = np.radians(latitude)
    return km / (EARTH_RADIUS_KM * np.pi / 180 * np.cos(lat_rad))


def degrees_to_km(degrees, latitude=45):
    """
    Convert degrees to kilometers at a given latitude.

    Parameters:
    -----------
    degrees : float or array
        Distance in degrees
    latitude : float
        Reference latitude in degrees

    Returns:
    --------
    float or array
        Distance in kilometers
    """
    lat_rad = np.radians(latitude)
    return degrees * (EARTH_RADIUS_KM * np.pi / 180 * np.cos(lat_rad))
