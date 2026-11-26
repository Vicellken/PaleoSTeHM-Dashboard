"""Tests for ANU-ICE 1D GIA txt loader (gia_setup_3d)."""

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

from stgp_pipeline import gia_setup_3d

ANU_1D_DIR = _REPO_ROOT / "Data" / "ANU_1D"
REAL_MODEL = ANU_1D_DIR / "RSL_ANU-ICE_EM48p210_21ka.txt"
GIA_3D_DIR = _REPO_ROOT / "Data" / "3D_GIA"


def _write_synthetic_anu_1d_txt(path: Path, rsl_value: float = 42.0) -> None:
    """Minimal 2×2 lon/lat grid with constant RSL across 34 time columns."""
    rows = []
    for lon in (100.0, 100.5):
        for lat in (10.0, 10.5):
            rsl_cols = " ".join(f"{rsl_value:.2f}" for _ in range(34))
            rows.append(f"{lon:7.2f} {lat:7.2f} {rsl_cols}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class TestAnu1dModelStem:
    def test_stem_normalization(self):
        stem = gia_setup_3d._anu_1d_model_stem(Path("RSL_ANU-ICE_EM48p210_21ka.txt"))
        assert stem == "48p210"


class TestLoadAnu1dGiaEnsemble:
    def test_synthetic_interpolation(self, tmp_path: Path):
        txt = tmp_path / "RSL_ANU-ICE_EM48p210_21ka.txt"
        _write_synthetic_anu_1d_txt(txt, rsl_value=42.0)
        funcs, names = gia_setup_3d.load_anu_1d_gia_ensemble(str(tmp_path))
        assert names == ["48p210"]
        X = np.array([[10000.0, 10.25, 100.25]])
        pred = funcs[0](X)
        np.testing.assert_allclose(pred, [42.0], rtol=0, atol=1e-6)

    def test_exclude_models(self, tmp_path: Path):
        txt = tmp_path / "RSL_ANU-ICE_EM48p210_21ka.txt"
        _write_synthetic_anu_1d_txt(txt)
        funcs, names = gia_setup_3d.load_anu_1d_gia_ensemble(
            str(tmp_path), exclude_models=["48p210"]
        )
        assert funcs == []
        assert names == []

    @pytest.mark.skipif(not REAL_MODEL.exists(), reason="Data/ANU_1D not present")
    def test_real_file_finite_at_obs_point(self):
        fn, name = gia_setup_3d.load_txt_gia_model(
            REAL_MODEL, time_bp=gia_setup_3d.ANU_1D_TIME_BP.copy()
        )
        assert name == REAL_MODEL.stem
        X = np.array([[10000.0, 15.0, 110.0]])
        pred = fn(X)
        assert np.isfinite(pred[0])

    def test_name_suffix(self, tmp_path: Path):
        txt = tmp_path / "RSL_ANU-ICE_EM48p210_21ka.txt"
        _write_synthetic_anu_1d_txt(txt)
        _, names = gia_setup_3d.load_anu_1d_gia_ensemble(
            str(tmp_path), name_suffix="_anu"
        )
        assert names == ["48p210_anu"]

    @pytest.mark.skipif(
        not REAL_MODEL.exists() or not GIA_3D_DIR.exists(),
        reason="ANU_1D or 3D_GIA_combined data missing",
    )
    def test_load_combined_ensemble_anu_only(self):
        funcs, names, indices_3d = gia_setup_3d.load_combined_ensemble(
            str(GIA_3D_DIR),
            gia_anu_dir=str(ANU_1D_DIR),
            one_d_source="anu",
        )
        assert len(indices_3d) == 2
        assert len(funcs) == 14
        assert len(names) == 14
        assert "48p210" in names


class TestResolve1dSource:
    def test_explicit_mat(self):
        source = gia_setup_3d.resolve_1d_source(
            {"1d_source": "mat"},
            {"gia_mat_dir": "/mat", "gia_ANU_dir": "/anu"},
        )
        assert source == "mat"

    def test_infers_anu_when_only_anu_path(self):
        source = gia_setup_3d.resolve_1d_source({}, {"gia_ANU_dir": "/anu"})
        assert source == "anu"

    def test_infers_mat_when_only_mat_path(self):
        source = gia_setup_3d.resolve_1d_source({}, {"gia_mat_dir": "/mat"})
        assert source == "mat"

    def test_defaults_to_anu_when_both_paths(self):
        source = gia_setup_3d.resolve_1d_source(
            {},
            {"gia_mat_dir": "/mat", "gia_ANU_dir": "/anu"},
        )
        assert source == "anu"

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="1d_source"):
            gia_setup_3d.resolve_1d_source({"1d_source": "npy"}, {})


class TestResolveGia1dDirs:
    def test_resolves_anu_from_cfg(self, tmp_path: Path):
        anu_dir = tmp_path / "anu"
        anu_dir.mkdir()
        cfg = {
            "paths": {"gia_ANU_dir": str(anu_dir)},
            "gia": {"1d_source": "anu"},
        }
        mat, anu, source = gia_setup_3d.resolve_gia_1d_dirs(cfg)
        assert mat is None
        assert anu == str(anu_dir.resolve())
        assert source == "anu"

    def test_both_requires_both_paths(self):
        cfg = {
            "paths": {"gia_mat_dir": "/mat"},
            "gia": {"1d_source": "both"},
        }
        with pytest.raises(KeyError, match="both"):
            gia_setup_3d.resolve_gia_1d_dirs(cfg)


class TestLoadCombinedBothMode:
    def test_both_mode_suffixes(self, tmp_path: Path, monkeypatch):
        anu_dir = tmp_path / "anu"
        mat_dir = tmp_path / "mat"
        anu_dir.mkdir()
        mat_dir.mkdir()
        _write_synthetic_anu_1d_txt(anu_dir / "RSL_ANU-ICE_EM48p210_21ka.txt")

        def fake_3d(_gia_3d_dir, time_start=16000.0):
            return [lambda X: np.array([0.0])], ["ANU_3D"]

        def fake_mat(_gia_mat_dir, exclude_models=None, name_suffix=""):
            return [lambda X: np.array([1.0])], [f"48p210{name_suffix}"]

        monkeypatch.setattr(gia_setup_3d, "load_3d_gia_ensemble", fake_3d)
        monkeypatch.setattr(gia_setup_3d, "load_mat_1d_gia_ensemble", fake_mat)

        funcs, names, indices_3d = gia_setup_3d.load_combined_ensemble(
            str(tmp_path / "3d"),
            gia_mat_dir=str(mat_dir),
            gia_anu_dir=str(anu_dir),
            one_d_source="both",
        )
        assert list(indices_3d) == [0]
        assert names == ["ANU_3D", "48p210_mat", "48p210_anu"]
        assert len(funcs) == 3


class TestLoadEnsembleFromCfg1dSource:
    def test_resolves_relative_gia_3d_dir_against_base_dir(
        self, tmp_path: Path, monkeypatch
    ):
        cfg_dir = tmp_path / "studies" / "region_a" / "configs"
        cfg_dir.mkdir(parents=True)
        gia_3d_dir = tmp_path / "studies" / "Data" / "3D_GIA"
        gia_3d_dir.mkdir(parents=True)
        (gia_3d_dir / "RSL_ANU_3D_sample_21ka.csv").write_text(
            "lon lat 10000\n100 10 0\n", encoding="utf-8"
        )

        seen_3d_dir: list[str] = []

        def fake_3d(resolved_gia_3d_dir, time_start=16000.0):
            seen_3d_dir.append(resolved_gia_3d_dir)
            return [lambda X: np.array([0.0])], ["ANU_3D"]

        monkeypatch.setattr(gia_setup_3d, "load_3d_gia_ensemble", fake_3d)

        anu_dir = tmp_path / "Data" / "ANU_1D"
        seen_anu_dir: list[str] = []

        def fake_anu(resolved_anu_dir, exclude_models=None, name_suffix=""):
            seen_anu_dir.append(resolved_anu_dir)
            return [lambda X: np.array([1.0])], ["48p210"]

        monkeypatch.setattr(gia_setup_3d, "load_anu_1d_gia_ensemble", fake_anu)

        cfg = {
            "paths": {
                "gia_3d_dir": "../../Data/3D_GIA",
                "gia_ANU_dir": "../../../Data/ANU_1D",
            },
            "gia": {"ensemble": "all", "1d_source": "anu"},
        }
        gia_setup_3d.load_ensemble_from_cfg(cfg, base_dir=cfg_dir)
        assert seen_3d_dir == [str(gia_3d_dir.resolve())]
        assert seen_anu_dir == [str(anu_dir.resolve())]

    def test_reads_1d_source_from_cfg(self, tmp_path: Path, monkeypatch):
        anu_dir = tmp_path / "anu"
        anu_dir.mkdir()
        _write_synthetic_anu_1d_txt(anu_dir / "RSL_ANU-ICE_EM48p210_21ka.txt")

        def fake_3d(_gia_3d_dir, time_start=16000.0):
            return [lambda X: np.array([0.0])], ["ANU_3D"]

        monkeypatch.setattr(gia_setup_3d, "load_3d_gia_ensemble", fake_3d)

        cfg = {
            "paths": {
                "gia_3d_dir": str(tmp_path / "3d"),
                "gia_ANU_dir": str(anu_dir),
            },
            "gia": {"ensemble": "all", "1d_source": "anu"},
        }
        funcs, names, indices_3d = gia_setup_3d.load_ensemble_from_cfg(
            cfg, base_dir=tmp_path
        )
        assert list(indices_3d) == [0]
        assert "48p210" in names
        assert len(funcs) == 2


class TestResolved1dDirFromPaths:
    def test_defaults_to_anu_when_both_paths(self):
        mat, anu = gia_setup_3d.resolved_1d_dir_from_paths(
            {"gia_mat_dir": "/mat", "gia_ANU_dir": "/anu"}
        )
        assert mat is None
        assert anu == "/anu"

    def test_explicit_both_returns_both(self):
        mat, anu = gia_setup_3d.resolved_1d_dir_from_paths(
            {"gia_mat_dir": "/mat", "gia_ANU_dir": "/anu"},
            gia_cfg={"1d_source": "both"},
        )
        assert mat == "/mat"
        assert anu == "/anu"

    def test_falls_back_to_mat(self):
        mat, anu = gia_setup_3d.resolved_1d_dir_from_paths({"gia_mat_dir": "/mat"})
        assert mat == "/mat"
        assert anu is None
