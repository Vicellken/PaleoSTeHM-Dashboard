"""Tests for gia.mean_weighting (equal / wrss_3d / wrss_all)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock

# gia_setup_3d -> types imports torch; stub only when PyTorch is absent.
try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = MagicMock()

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stgp_pipeline.gia_setup_3d import (
    MEAN_WEIGHTING_CHOICES,
    _compute_gia_mean_from_preds,
    _parse_mean_weighting,
    _resolve_mean_model_indices,
)


def _constant_interp(value: float):
    def fn(x):
        n = len(x) if hasattr(x, "__len__") else 1
        return np.full(n, value, dtype=np.float64)

    return fn


class TestComputeGiaMeanFromPreds:
    """Unit tests for per-point GIA mean combination."""

    WEIGHTS = np.array([0.2, 0.5, 0.3])
    # Model 0 -> 10 m, model 1 -> 20 m, model 2 -> 30 m at each point
    PREDS = np.array(
        [
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
        ]
    )
    POOL = np.array([0, 1], dtype=int)

    def test_equal_arithmetic_mean(self):
        mean = _compute_gia_mean_from_preds(
            self.PREDS, self.WEIGHTS, self.POOL, "equal"
        )
        np.testing.assert_allclose(mean, [15.0, 15.0])

    def test_wrss_3d_renormalized_weights(self):
        mean = _compute_gia_mean_from_preds(
            self.PREDS, self.WEIGHTS, self.POOL, "wrss_3d"
        )
        w0, w1 = 0.2 / 0.7, 0.5 / 0.7
        expected = w0 * 10.0 + w1 * 20.0
        np.testing.assert_allclose(mean, [expected, expected])

    def test_wrss_all_full_ensemble(self):
        mean = _compute_gia_mean_from_preds(
            self.PREDS, self.WEIGHTS, self.POOL, "wrss_all"
        )
        expected = np.average(self.PREDS, axis=0, weights=self.WEIGHTS)
        np.testing.assert_allclose(mean, expected)

    def test_invalid_mean_weighting_raises(self):
        with pytest.raises(ValueError, match="mean_weighting"):
            _compute_gia_mean_from_preds(self.PREDS, self.WEIGHTS, self.POOL, "invalid")


class TestParseMeanWeighting:
    def test_default_equal(self):
        assert _parse_mean_weighting({}) == "equal"

    def test_accepts_wrss_3d(self):
        assert _parse_mean_weighting({"mean_weighting": "wrss_3d"}) == "wrss_3d"

    def test_rejects_unknown(self):
        with pytest.raises(ValueError, match="mean_weighting"):
            _parse_mean_weighting({"mean_weighting": "bad"})

    def test_choices_tuple_documents_options(self):
        assert set(MEAN_WEIGHTING_CHOICES) == {"equal", "wrss_3d", "wrss_all"}


class TestResolveMeanModelIndices:
    def test_defaults_to_indices_3d(self):
        idx = _resolve_mean_model_indices(["a", "b", "c"], np.array([0, 1]), None)
        np.testing.assert_array_equal(idx, [0, 1])

    def test_mean_model_names_override(self):
        idx = _resolve_mean_model_indices(["a", "b", "c"], np.array([0, 1]), ["c", "a"])
        np.testing.assert_array_equal(idx, [2, 0])

    def test_unknown_mean_model_raises(self):
        with pytest.raises(KeyError, match="mean_model_names"):
            _resolve_mean_model_indices(["a", "b"], np.array([0]), ["missing"])


class TestGiaAtTestPoints3d:
    def test_matches_helper_for_each_mode(self):
        from stgp_pipeline.gia_setup_3d import gia_at_test_points_3d

        interps = [_constant_interp(v) for v in (10.0, 20.0, 30.0)]
        weights = TestComputeGiaMeanFromPreds.WEIGHTS
        indices_3d = np.array([0, 1], dtype=int)
        ages = np.array([1000.0])
        lats = np.array([20.0])
        lons = np.array([110.0])

        for mode in MEAN_WEIGHTING_CHOICES:
            gia_mean, _ = gia_at_test_points_3d(
                interps,
                weights,
                indices_3d,
                ages,
                lats,
                lons,
                mean_weighting=mode,
                mean_model_indices=indices_3d,
            )
            preds = np.array([[10.0], [20.0], [30.0]])
            expected = _compute_gia_mean_from_preds(preds, weights, indices_3d, mode)
            np.testing.assert_allclose(gia_mean, expected.flatten())

    def test_spatial_mean_covariance_averages_models_before_covariance(self):
        from stgp_pipeline.gia_setup_3d import (
            gia_spatial_mean_covariance_at_ages_3d,
        )

        def model(scale):
            def predict(x):
                return scale * (x[:, 0] / 1000.0 + x[:, 1])

            return predict

        mean, covariance = gia_spatial_mean_covariance_at_ages_3d(
            [model(1.0), model(2.0)],
            weights=np.array([0.5, 0.5]),
            indices_3d=np.array([0, 1]),
            test_ages=np.array([0.0, 2000.0]),
            spatial_lat=np.array([0.0, 2.0]),
            spatial_lon=np.array([110.0, 110.0]),
            mean_weighting="wrss_all",
        )

        np.testing.assert_allclose(mean, [1.5, 4.5])
        np.testing.assert_allclose(
            covariance,
            [[0.5, 1.5], [1.5, 4.5]],
        )
