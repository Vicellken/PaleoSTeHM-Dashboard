from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse=True)
def _clean_stgp_pipeline_import():
    """Ensure ``stgp_pipeline`` resolves to the real package, not a namespace stub.

    During a full-suite run, sibling test modules can leave ``stgp_pipeline``
    cached in ``sys.modules`` as a namespace package (``__file__ is None``),
    which breaks our submodule imports. Drop any such poisoned entry before each
    test so the real package (on the path inserted above) is re-imported.
    """
    pkg = sys.modules.get("stgp_pipeline")
    if pkg is not None and getattr(pkg, "__file__", None) is None:
        for name in [
            k
            for k in list(sys.modules)
            if k == "stgp_pipeline" or k.startswith("stgp_pipeline.")
        ]:
            del sys.modules[name]
    yield


# --------------------------------------------------------------------------- #
# Synthetic builders (no GP/GIA required)
# --------------------------------------------------------------------------- #
def _make_region_result(region_num: int = 14, region_name: str = "Test", T: int = 5):
    import torch

    from stgp_pipeline.types import RegionResult

    t = np.linspace(0.0, 8000.0, T)
    comp = {
        "global": {
            "mean": np.zeros(T),
            "std": np.ones(T) * 0.5,
            "cov": np.eye(T) * 0.25,
            "rate": np.zeros(T - 1),
            "rate_sd": np.ones(T - 1) * 0.1,
        }
    }
    return RegionResult(
        region_num=region_num,
        region_name=region_name,
        test_age=t,
        y_total=np.linspace(-50.0, 0.0, T),
        y_std=np.ones(T),
        y_res_mean=np.zeros(T),
        gia_ts=np.linspace(-55.0, 0.0, T),
        gia_var=np.ones(T),
        gia_cov_test=np.eye(T),
        rsl_time=0.5 * (t[1:] + t[:-1]),
        rsl_rate=np.zeros(T - 1),
        rsl_rate_sd=np.ones(T - 1) * 0.1,
        gia_rate=np.zeros(T - 1),
        gia_rate_sd=np.ones(T - 1) * 0.1,
        pred_matrix=torch.zeros((1, 3)),
        y_var_res=torch.zeros((T, T)),
        components=comp,
    )


def _make_rsl_data(lats, lons, regions):
    import torch

    from stgp_pipeline.types import RSLData

    n = len(lats)
    ages = np.full(n, 4000.0)
    X = torch.tensor(np.column_stack([ages, lats, lons])).double()
    empty = np.array([])
    return RSLData(
        X=X,
        y=torch.tensor(np.linspace(-40.0, -1.0, n)).double(),
        y_sigma=torch.ones(n).double(),
        age_sigma=torch.ones(n).double() * 50.0,
        rsl_lat=np.asarray(lats, float),
        rsl_lon=np.asarray(lons, float),
        rsl_region=np.asarray(regions, int),
        rsl_region_name=np.array(["Test"] * n, dtype=object),
        lim_age=empty,
        lim_y=empty,
        lim_y_sigma=empty,
        lim_age_sigma=empty,
        lim_lat=empty,
        lim_lon=empty,
        lim_flags=empty,
    )


def _suptitle_text(fig) -> str:
    st = getattr(fig, "_suptitle", None)
    if st is not None:
        return st.get_text()
    return " ".join(t.get_text() for t in fig.texts)


# --------------------------------------------------------------------------- #
# Projection math (pure, no model)
# --------------------------------------------------------------------------- #
def test_apply_projection_zero_correction_at_reference():
    pytest.importorskip("torch")
    from stgp_pipeline.predict import _apply_projection

    y = np.array([10.0, 8.0, 5.0])
    total = np.array([3.0, 2.0, 1.0])
    # site == ref → correction is zero → projected equals observed.
    out = _apply_projection(y, total, total)
    np.testing.assert_allclose(out, y)


def test_apply_projection_linear_case():
    pytest.importorskip("torch")
    from stgp_pipeline.predict import _apply_projection

    y = np.array([10.0, 8.0])
    total_site = np.array([3.0, 2.0])
    total_ref = np.array([1.0, 1.0])
    # correction = [2, 1] → projected = [8, 7]
    out = _apply_projection(y, total_site, total_ref)
    np.testing.assert_allclose(out, [8.0, 7.0])


# --------------------------------------------------------------------------- #
# Radius-based overlay selection
# --------------------------------------------------------------------------- #
def test_haversine_radius_selection():
    pytest.importorskip("torch")
    from stgp_pipeline.visualize import _haversine_km

    site_lat, site_lon = 22.2459, 114.2470
    # point 0: exact site; point 1: Lantau (~37 km); point 2: ~6 km north
    lats = np.array([22.2459, 22.2050, 22.30])
    lons = np.array([114.2470, 113.8930, 114.25])
    d = _haversine_km(lats, lons, site_lat, site_lon)
    assert d[0] == pytest.approx(0.0, abs=1e-6)
    assert int(np.sum(d <= 2.0)) == 1  # only the exact point is within 2 km


# --------------------------------------------------------------------------- #
# Plot refactor: backward compatibility + new overlay/title paths
# --------------------------------------------------------------------------- #
def test_plot_temporal_decomposition_default_overlay_backward_compatible():
    pytest.importorskip("torch")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stgp_pipeline import visualize

    result = _make_region_result(region_num=14)
    rsl = _make_rsl_data(
        lats=[22.0, 22.1, 30.0], lons=[114.0, 114.1, 120.0], regions=[14, 14, 9]
    )
    # Defaults: region-keyed overlay + "R{rn}" title (unchanged behavior).
    fig = visualize.plot_temporal_decomposition(14, result, rsl, comp_labels=["global"])
    assert isinstance(fig, plt.Figure)
    assert "R14" in _suptitle_text(fig)
    plt.close(fig)


def test_plot_temporal_decomposition_overlay_mask_and_title():
    pytest.importorskip("torch")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stgp_pipeline import visualize

    result = _make_region_result(region_num=14)
    rsl = _make_rsl_data(
        lats=[22.0, 22.1, 30.0], lons=[114.0, 114.1, 120.0], regions=[14, 14, 9]
    )
    fig = visualize.plot_temporal_decomposition(
        14,
        result,
        rsl,
        comp_labels=["global"],
        overlay_mask=np.array([True, False, False]),
        title="Custom Title",
    )
    assert "Custom Title" in _suptitle_text(fig)
    plt.close(fig)


def test_plot_temporal_decomposition_at_site_counts_local_slip():
    pytest.importorskip("torch")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stgp_pipeline import visualize

    result = _make_region_result(region_num=-1, region_name="Big Wave Bay")
    # Two SLIP exactly at the site, one far away.
    rsl = _make_rsl_data(
        lats=[22.2459, 22.2459, 30.0],
        lons=[114.2470, 114.2470, 120.0],
        regions=[14, 14, 9],
    )
    fig = visualize.plot_temporal_decomposition_at_site(
        result,
        rsl,
        22.2459,
        114.2470,
        site_label="Big Wave Bay",
        comp_labels=["global"],
    )
    txt = _suptitle_text(fig)
    assert "Big Wave Bay" in txt
    assert "2 SLIP within 2 km" in txt
    plt.close(fig)


def test_plot_projected_region_returns_figure():
    pytest.importorskip("torch")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stgp_pipeline import visualize
    from stgp_pipeline.types import ProjectedRegionData

    ref_result = _make_region_result(region_num=-1, region_name="Big Wave Bay")
    projected = ProjectedRegionData(
        region_num=14,
        ref_name="Big Wave Bay",
        ref_lat=22.2459,
        ref_lon=114.2470,
        age=np.array([4000.0, 6000.0, 8000.0]),
        orig_y=np.array([-1.0, -5.0, -20.0]),
        projected_y=np.array([-2.0, -6.0, -18.0]),
        y_sigma=np.ones(3),
        age_sigma=np.ones(3) * 50.0,
        correction=np.array([1.0, 1.0, -2.0]),
    )
    fig = visualize.plot_projected_region(ref_result, projected)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
