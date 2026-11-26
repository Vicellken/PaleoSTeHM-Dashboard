"""Tests for bound-proximity reporting (post-optimization diagnostics)."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PSTHM.kernels_factory import build_spatiotemporal_kernel
from stgp_pipeline.sensitivity.workflow_common import (
    boundary_proximity_from_notebook_params,
    compute_boundary_proximity,
)


def test_set_prior_registers_pyro_sample_not_pyro_param():
    """Prior setup must keep Pyro GP PyroSample machinery intact."""
    kernel = build_spatiotemporal_kernel(
        {
            "kernels": {
                "global_temporal": {
                    "enabled": True,
                    "type": "Matern32",
                    "lengthscale_prior": [50.0, 10000.0],
                    "variance_prior": [10.0, 400.0],
                }
            }
        }
    )
    global_kernel = kernel.component_kernels["global_temporal"]
    assert "lengthscale" in global_kernel._priors
    assert "variance" in global_kernel._priors
    assert hasattr(global_kernel, "lengthscale_map")
    assert hasattr(global_kernel, "variance_map")
    # Prior draw should not be pinned to the lower bound (regression check).
    ls = global_kernel.lengthscale.item()
    assert 50.0 < ls < 10000.0


def test_notebook_boundary_proximity_for_sample_config():
    cfg = {
        "gp_model": {
            "kernels": {
                "global_temporal": {
                    "enabled": True,
                    "lengthscale_prior": [100.0, 20000.0],
                    "variance_prior": [1.0e-5, 1000.0],
                },
                "regional_linear": {
                    "enabled": True,
                    "temporal_variance_prior": [1.0e-12, 1.0e-6],
                    "spatial_lengthscale_prior": [0.0001, 0.01],
                },
                "regional_nonlinear": {
                    "enabled": True,
                    "temporal_variance_prior": [1.0e-6, 100.0],
                    "temporal_lengthscale_prior": [100.0, 20000.0],
                    "spatial_lengthscale_prior": [0.0001, 0.01],
                },
            }
        }
    }
    optimized = {
        "global_kernel_lengthscale (yr)": 6738.648652,
        "global_kernel_variance (m)": 13.572316,
        "regional_linearl_kernel_spatil_lengthscale (km)": 62.272927,
        "regional_nl_kernel_spatial_lengthscale (km)": 63.536837,
    }
    df = boundary_proximity_from_notebook_params(optimized, cfg)
    assert not df.empty
    assert set(df["parameter"]) >= set(optimized.keys())

    for _, row in df.iterrows():
        assert row["bound_low"] <= row["value"] <= row["bound_high"]

    rnl_row = df.loc[
        df["parameter"] == "regional_nl_kernel_spatial_lengthscale (km)"
    ].iloc[0]
    pct, _ = compute_boundary_proximity(
        float(rnl_row["value"]),
        float(rnl_row["bound_low"]),
        float(rnl_row["bound_high"]),
    )
    assert abs(float(rnl_row["proximity_pct"]) - pct) < 1e-6
    assert float(rnl_row["proximity_pct"]) > 90.0
