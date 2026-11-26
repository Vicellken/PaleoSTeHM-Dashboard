"""Device-consistency tests for PSTHM.opti.SVI_NI_optm."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pyro
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import PSTHM
from PSTHM.opti import SVI_NI_optm


def _build_tiny_gpr(n: int = 12):
    torch.manual_seed(0)
    pyro.set_rng_seed(0)

    age = torch.linspace(100.0, 5000.0, n, dtype=torch.float64)
    lat = torch.full((n,), 30.0, dtype=torch.float64)
    lon = torch.linspace(110.0, 120.0, n, dtype=torch.float64)
    X = torch.stack([age, lat, lon], dim=1)
    y = 0.001 * age + 0.05 * torch.randn(n, dtype=torch.float64)
    y_sigma = torch.full((n,), 0.5, dtype=torch.float64)
    noise = torch.diag(y_sigma**2)

    pyro.clear_param_store()
    kernel = PSTHM.kernels.Matern32(input_dim=1)
    gpr = PSTHM.model.GPRegression_V(X, y, kernel, noise=noise, jitter=1e-5)
    x_sigma = torch.full((n,), 10.0, dtype=torch.float64)
    return gpr, x_sigma


class TestSviDevice(unittest.TestCase):
    def test_svi_ni_optm_cpu_returns_model_on_cpu(self):
        gpr, x_sigma = _build_tiny_gpr()
        gpr_out, track_df = SVI_NI_optm(
            gpr,
            x_sigma=x_sigma,
            num_iteration=5,
            lr=0.1,
            decay_r=1.0,
            step_size=5,
            gpu=False,
            show_progress=False,
        )

        self.assertEqual(gpr_out.X.device.type, "cpu")
        self.assertEqual(gpr_out.y.device.type, "cpu")
        self.assertEqual(gpr_out.noise.device.type, "cpu")
        self.assertIn("loss", track_df.columns)
        self.assertTrue(np.all(np.isfinite(track_df["loss"].values)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_svi_ni_optm_gpu_returns_model_on_cpu(self):
        gpr, x_sigma = _build_tiny_gpr()
        gpr_out, track_df = SVI_NI_optm(
            gpr,
            x_sigma=x_sigma,
            num_iteration=5,
            lr=0.1,
            decay_r=1.0,
            step_size=5,
            gpu=True,
            show_progress=False,
        )

        self.assertEqual(gpr_out.X.device.type, "cpu")
        self.assertEqual(gpr_out.y.device.type, "cpu")
        self.assertEqual(gpr_out.noise.device.type, "cpu")
        self.assertIn("loss", track_df.columns)
        self.assertTrue(np.all(np.isfinite(track_df["loss"].values)))


if __name__ == "__main__":
    unittest.main()
