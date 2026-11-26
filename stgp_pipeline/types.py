"""
Shared dataclass definitions for the STGP + GIA pipeline.

Placed in a separate module to prevent circular imports between
data, gia_setup, training, predict, export, and visualize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class RSLData:
    """Filtered SLIP + limiting data ready for GP training and GIA weighting."""

    # SLIP training data (filtered)
    X: torch.Tensor  # (N, 3) — [age_bp, lat, lon]
    y: torch.Tensor  # (N,)
    y_sigma: torch.Tensor  # (N,)
    age_sigma: torch.Tensor  # (N,)
    rsl_lat: np.ndarray  # (N,)
    rsl_lon: np.ndarray  # (N,)
    rsl_region: np.ndarray  # (N,) int region IDs
    rsl_region_name: np.ndarray  # (N,) string region names

    # Combined limiting data (ML + TL, filtered)
    lim_age: np.ndarray  # (L,)
    lim_y: np.ndarray  # (L,)
    lim_y_sigma: np.ndarray  # (L,)
    lim_age_sigma: np.ndarray  # (L,)
    lim_lat: np.ndarray  # (L,)
    lim_lon: np.ndarray  # (L,)
    lim_flags: np.ndarray  # (L,) — -1 marine, +1 terrestrial
    lim_region: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    lim_region_name: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=object)
    )

    # Convenience
    region_num_to_name: Dict[int, str] = field(default_factory=dict)

    @property
    def unique_regions(self) -> np.ndarray:
        return np.sort(np.unique(self.rsl_region))

    @property
    def n_slip(self) -> int:
        return len(self.y)


@dataclass
class GIAResult:
    """Outputs of GIA ensemble weighting and covariance computation."""

    interp_funcs: List[Any]  # RegularGridInterpolator per model
    model_names: List[str]
    weights: np.ndarray  # (M,)
    misfits: np.ndarray  # (M,)
    gia_mean_at_obs: np.ndarray  # (N,) weighted mean at SLIP points
    gia_cov_at_obs: np.ndarray  # (N, N) weighted ensemble covariance
    indices_3d: Optional[np.ndarray] = None  # indices of 3D models in interp_funcs
    mean_weighting: str = "equal"  # equal | wrss_3d | wrss_all
    mean_model_indices: Optional[np.ndarray] = None  # pool for equal / wrss_3d


@dataclass
class ResidualData:
    """GP training inputs: residuals and total noise matrix."""

    y_residual: torch.Tensor  # (N,) — y_obs - GIA_mean
    noise_total: torch.Tensor  # (N, N) — diag(sigma_obs^2) + GIA_cov


@dataclass
class TrainedModel:
    """Trained GP model with named kernel components."""

    gpr: Any  # GPRegression_V instance
    kernel_components: Dict[str, Any]  # name → kernel object
    decomp_kernels: List[Any]  # ordered list for decompose_kernels()
    comp_labels: List[str]  # labels matching decomp_kernels order
    track_list: Optional[List[Any]] = None  # SVI optimization track


@dataclass
class RegionResult:
    """Per-region temporal prediction results."""

    region_num: int
    region_name: str
    test_age: np.ndarray  # (T,)
    y_total: np.ndarray  # (T,)
    y_std: np.ndarray  # (T,)
    y_res_mean: np.ndarray  # (T,) GP residual mean
    gia_ts: np.ndarray  # (T,) GIA weighted mean at centroid
    gia_var: np.ndarray  # (T,) GIA variance at centroid
    gia_cov_test: np.ndarray  # (T, T) GIA covariance at centroid
    rsl_time: np.ndarray  # (T-1,) rate midpoints
    rsl_rate: np.ndarray  # (T-1,)
    rsl_rate_sd: np.ndarray  # (T-1,)
    gia_rate: np.ndarray  # (T-1,)
    gia_rate_sd: np.ndarray  # (T-1,)
    pred_matrix: torch.Tensor  # prediction matrix used
    y_var_res: torch.Tensor  # (T, T) GP residual covariance
    components: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    # components[label] = {"mean", "std", "cov", "rate", "rate_sd"}


@dataclass
class ProjectedRegionData:
    """RSL data projected to a reference location via the trained GP+GIA model.

    For each datum i at (lat_i, lon_i, age_i):
        correction_i = Total(lat_i, lon_i, age_i) - Total(ref_lat, ref_lon, age_i)
        projected_y_i = y_i - correction_i
    where Total = GP residual + weighted GIA mean. This brings multi-site data
    into a single reference frame so they can be plotted against the
    reference-location curve — the GP generalization of the classic "isobase"
    projection. The reference curve itself is unchanged (it comes from the model);
    only the data points move.
    """

    region_num: Optional[int]
    ref_name: str
    ref_lat: float
    ref_lon: float
    age: np.ndarray  # (N,)
    orig_y: np.ndarray  # (N,)
    projected_y: np.ndarray  # (N,)
    y_sigma: np.ndarray  # (N,)
    age_sigma: np.ndarray  # (N,)
    correction: np.ndarray  # (N,)


@dataclass
class SpatialSlice:
    """Spatial map at one age."""

    age_bp: float
    lat_grid: np.ndarray  # 1-D
    lon_grid: np.ndarray  # 1-D
    lat_mat: np.ndarray  # 2-D meshgrid
    lon_mat: np.ndarray  # 2-D meshgrid
    total_mean: np.ndarray  # 2-D (nlat, nlon)
    total_std: np.ndarray  # 2-D
    gp_mean: np.ndarray  # 2-D GP residual
    gp_var: np.ndarray  # 2-D GP variance
    gia_mean: np.ndarray  # 2-D
    gia_var: np.ndarray  # 2-D
    components: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    # components[label] = {"mean": 2-D, "std": 2-D}


@dataclass
class SpatialRateResult:
    """Spatial rates for all time intervals."""

    spatial_ages: np.ndarray  # (n_ages,)
    rate_midpoints: np.ndarray  # (n_intervals,)
    lat_flat: np.ndarray  # (n_grid,)
    lon_flat: np.ndarray  # (n_grid,)
    lat_mat: np.ndarray  # 2-D
    lon_mat: np.ndarray  # 2-D

    rate_total: np.ndarray  # (n_intervals, n_grid)
    sd_total: np.ndarray
    rate_gp: np.ndarray
    sd_gp: np.ndarray
    rate_gia: np.ndarray
    sd_gia: np.ndarray
    rate_comp: Dict[str, np.ndarray] = field(default_factory=dict)
    sd_comp: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RegionalMeanRateResult:
    """One interval of equal-grid-cell regional mean rates.

    Rates use the pipeline's internal m/yr, increasing-age-BP convention.
    """

    region_num: int
    region_name: str
    n_pixels: int
    mask_mode: str
    weighting: str
    younger_bp: float
    older_bp: float
    midpoint_bp: float
    gia_rate: float
    gia_rate_sd: float
    component_rates: Dict[str, float] = field(default_factory=dict)
    component_rate_sds: Dict[str, float] = field(default_factory=dict)
    combined_rate: Optional[float] = None
    combined_rate_sd: Optional[float] = None
