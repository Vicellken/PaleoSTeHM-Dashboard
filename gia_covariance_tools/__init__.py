"""
GIA Covariance Tools
====================

A comprehensive toolkit for extracting and computing covariances from multiple
Glacial Isostatic Adjustment (GIA) model ensembles.

This module provides functions for:
- Loading GIA model data from various formats
- Computing ensemble statistics (mean, covariance)
- Advanced covariance operations (temporal/spatial structure)
- Visualization of covariance matrices
- Integration with PSTHM Gaussian Process framework

Date: 2026-03-02
"""

from .config import *
from .io_handlers import (
    load_gia_grid,
    load_ensemble_from_directory,
    load_mat_gia_grid,
    load_mat_gia_ensemble,
    compute_min_distance_weights_continuous,
    compute_wrss_weights_continuous,
    export_covariance_matrix,
    export_results,
    make_region_interp_funcs,
    compute_min_distance_weights,
)
from .core_math import (
    compute_weighted_mean,
    compute_weighted_covariance,
    center_data,
    regularize_covariance_matrix,
    compute_correlation_from_covariance,
    invert_covariance_matrix,
    compute_eigen_decomposition,
    compute_condition_number,
)
from .covariance import (
    extract_gia_covariance,
    extract_gia_covariance_weighted,
    compute_spatial_covariance_at_time,
    compute_temporal_covariance_at_location,
    compute_empirical_gia_covariance_3d,
    compute_spatial_coherence_map,
    compute_temporal_structure_function,
    compute_gia_model_disagreement,
    combine_gia_observation_covariance,
    create_sampling_kernel,
    create_gia_ensemble_mean_function,
    validate_covariance_matrix,
    compute_covariance_spectrum,
)

__version__ = "0.1.0"
__author__ = "Yi-Fei Gu"

__all__ = [
    # I/O functions
    "load_gia_grid",
    "load_ensemble_from_directory",
    "export_covariance_matrix",
    "export_results",
    # Core math functions
    "compute_weighted_mean",
    "compute_weighted_covariance",
    "center_data",
    "regularize_covariance_matrix",
    "compute_correlation_from_covariance",
    "invert_covariance_matrix",
    "compute_eigen_decomposition",
    "compute_condition_number",
    # MAT-format continuous GIA loaders
    "load_mat_gia_grid",
    "load_mat_gia_ensemble",
    "compute_min_distance_weights_continuous",
    "compute_wrss_weights_continuous",
    # Region-based adapters
    "make_region_interp_funcs",
    "compute_min_distance_weights",
    # Main covariance functions
    "extract_gia_covariance",
    "extract_gia_covariance_weighted",
    "compute_spatial_covariance_at_time",
    "compute_temporal_covariance_at_location",
    "compute_empirical_gia_covariance_3d",
    "compute_spatial_coherence_map",
    "compute_temporal_structure_function",
    "compute_gia_model_disagreement",
    "combine_gia_observation_covariance",
    "create_sampling_kernel",
    "create_gia_ensemble_mean_function",
    "validate_covariance_matrix",
    "compute_covariance_spectrum",
]
