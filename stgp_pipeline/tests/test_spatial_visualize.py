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
    pkg = sys.modules.get("stgp_pipeline")
    if pkg is not None and getattr(pkg, "__file__", None) is None:
        for name in [
            k
            for k in list(sys.modules)
            if k == "stgp_pipeline" or k.startswith("stgp_pipeline.")
        ]:
            del sys.modules[name]


def test_mask_by_std():
    from stgp_pipeline.visualize import _mask_by_std

    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    std = np.array([[0.5, 1.5], [2.5, 0.5]])
    masked = _mask_by_std(data, std, threshold=1.0)
    assert masked.mask[0, 0] is np.False_
    assert masked.mask[0, 1] is np.True_
    assert masked.mask[1, 0] is np.True_
    assert masked.mask[1, 1] is np.False_


def test_calibrate_threshold():
    from stgp_pipeline.visualize import _calibrate_threshold

    assert _calibrate_threshold(np.array([1.0, 2.0, 3.0]), factor=2.0) == 4.0


def test_spatial_plot_mask_settings_defaults():
    from stgp_pipeline.visualize import spatial_plot_mask_settings

    ms = spatial_plot_mask_settings({})
    assert ms.enabled is True
    assert ms.calibration_factor == 2.0
    assert ms.rsl_std_fallback_m == 2.0
    assert ms.rate_sd_fallback_mm_yr == 50.0


def test_spatial_plot_mask_settings_from_yaml():
    from stgp_pipeline.visualize import spatial_plot_mask_settings

    cfg = {
        "spatial_plot_mask": {
            "enabled": True,
            "calibration_factor": 1.5,
            "rsl_std_fallback_m": 3.0,
            "rate_sd_fallback_mm_yr": 40.0,
        }
    }
    ms = spatial_plot_mask_settings(cfg)
    assert ms.calibration_factor == 1.5
    assert ms.rsl_std_fallback_m == 3.0
    assert ms.rate_sd_fallback_mm_yr == 40.0


def test_mask_with_lower_factor():
    from stgp_pipeline.visualize import (
        SpatialPlotMaskSettings,
        _apply_std_mask,
        _calibrate_threshold,
    )

    data = np.ones((2, 3))
    std = np.array([[1.0, 1.5, 2.0], [1.0, 1.0, 1.8]])
    slip_sds = np.array([1.0, 1.0, 1.0, 1.0])
    threshold_20 = _calibrate_threshold(slip_sds, factor=2.0)
    threshold_15 = _calibrate_threshold(slip_sds, factor=1.5)
    masked_20 = _apply_std_mask(
        data, std, threshold_20, SpatialPlotMaskSettings(calibration_factor=2.0)
    )
    masked_15 = _apply_std_mask(
        data, std, threshold_15, SpatialPlotMaskSettings(calibration_factor=1.5)
    )
    assert masked_20.mask.sum() == 0
    assert masked_15.mask.sum() > 0


def test_apply_std_mask_disabled():
    from stgp_pipeline.visualize import SpatialPlotMaskSettings, _apply_std_mask

    data = np.array([[1.0, 2.0]])
    std = np.array([[5.0, 10.0]])
    masked = _apply_std_mask(
        data,
        std,
        threshold=1.0,
        mask_settings=SpatialPlotMaskSettings(enabled=False),
    )
    assert not masked.mask.any()


def test_slip_grid_sds_uses_nearest_cell():
    from stgp_pipeline.visualize import _slip_grid_sds

    lat_grid = np.array([20.0, 21.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    sd_2d = np.array([[10.0, 20.0], [30.0, 40.0]])
    slip_lat = np.array([20.0, 21.0])
    slip_lon = np.array([110.0, 111.0])
    sds = _slip_grid_sds(slip_lat, slip_lon, lat_mat, lon_mat, sd_2d)
    np.testing.assert_allclose(sds, [10.0, 40.0])


def test_spatial_total_trust_mask_matches_map_mask():
    from types import SimpleNamespace

    from stgp_pipeline.visualize import (
        _apply_std_mask,
        spatial_plot_mask_settings,
        spatial_total_trust_mask,
    )

    lat_grid = np.array([0.0, 60.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    spatial_slice = SimpleNamespace(
        age_bp=6000.0,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        total_mean=np.array([[0.0, 1.0], [2.0, 100.0]]),
        total_std=np.array([[1.0, 1.0], [1.0, 3.0]]),
    )
    rsl = SimpleNamespace(
        rsl_lat=np.array([0.0, 0.0]),
        rsl_lon=np.array([110.0, 111.0]),
    )
    cfg = {"spatial_plot_mask": {"enabled": True, "calibration_factor": 1.5}}

    trusted, threshold = spatial_total_trust_mask(
        spatial_slice, rsl_data=rsl, cfg=cfg
    )
    settings = spatial_plot_mask_settings(cfg)
    map_masked = _apply_std_mask(
        spatial_slice.total_mean,
        spatial_slice.total_std,
        threshold,
        settings,
    )

    assert threshold == pytest.approx(1.5)
    np.testing.assert_array_equal(trusted, ~np.ma.getmaskarray(map_masked))
    np.testing.assert_array_equal(
        trusted, np.array([[True, True], [True, False]])
    )


def test_plot_spatial_total_with_rsl_data_runs():
    pytest.importorskip("cartopy")
    from stgp_pipeline.types import RSLData, SpatialSlice
    from stgp_pipeline import visualize

    lat_grid = np.array([20.0, 21.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    s = SpatialSlice(
        age_bp=0.0,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        total_mean=np.zeros((2, 2)),
        total_std=np.full((2, 2), 0.5),
        gp_mean=np.zeros((2, 2)),
        gp_var=np.zeros((2, 2)),
        gia_mean=np.zeros((2, 2)),
        gia_var=np.zeros((2, 2)),
    )
    rsl = RSLData(
        X=__import__("torch").zeros(2, 3),
        y=__import__("torch").zeros(2),
        y_sigma=__import__("torch").ones(2),
        age_sigma=__import__("torch").ones(2),
        rsl_lat=np.array([20.0, 21.0]),
        rsl_lon=np.array([110.0, 111.0]),
        rsl_region=np.array([1, 1]),
        rsl_region_name=np.array(["A", "B"]),
        lim_age=np.array([]),
        lim_y=np.array([]),
        lim_y_sigma=np.array([]),
        lim_age_sigma=np.array([]),
        lim_lat=np.array([]),
        lim_lon=np.array([]),
        lim_flags=np.array([]),
    )
    fig = visualize.plot_spatial_total(s, rsl_data=rsl)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_spatial_decomposition_total_uses_rdbu():
    pytest.importorskip("cartopy")
    from stgp_pipeline.types import SpatialSlice
    from stgp_pipeline import visualize

    lat_grid = np.array([20.0, 21.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    grid = np.array([[1.0, -1.0], [0.5, -0.5]])
    s = SpatialSlice(
        age_bp=4000.0,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        total_mean=grid,
        total_std=np.full((2, 2), 0.5),
        gp_mean=np.zeros((2, 2)),
        gp_var=np.zeros((2, 2)),
        gia_mean=np.zeros((2, 2)),
        gia_var=np.zeros((2, 2)),
        components={
            "global": {"mean": grid * 0.1, "std": np.full((2, 2), 0.1)},
        },
    )
    fig = visualize.plot_spatial_decomposition(s, comp_labels=["global"])
    ax = fig.axes[0]
    coll = ax.collections[0]
    assert coll.get_cmap().name == "RdBu_r"
    assert coll.get_clim() == (-1.0, 1.0)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_spatial_total_vs_regional_linear_two_axes():
    pytest.importorskip("cartopy")
    from stgp_pipeline.types import SpatialSlice
    from stgp_pipeline import visualize

    lat_grid = np.array([20.0, 21.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    grid = np.array([[1.0, -1.0], [0.5, -0.5]])
    s = SpatialSlice(
        age_bp=2000.0,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        total_mean=grid,
        total_std=np.full((2, 2), 0.5),
        gp_mean=np.zeros((2, 2)),
        gp_var=np.zeros((2, 2)),
        gia_mean=np.zeros((2, 2)),
        gia_var=np.zeros((2, 2)),
        components={
            "regional_linear": {
                "mean": grid * 0.2,
                "std": np.full((2, 2), 0.1),
            },
        },
    )
    fig = visualize.plot_spatial_total_vs_regional_linear(s)
    import cartopy.mpl.geoaxes as geoaxes

    map_axes = [ax for ax in fig.axes if isinstance(ax, geoaxes.GeoAxes)]
    assert len(map_axes) == 2
    assert map_axes[0].collections[0].get_cmap().name == "RdBu_r"
    assert map_axes[1].collections[0].get_cmap().name == "PuOr_r"
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_spatial_total_vs_regional_linear_vabs_across_slices():
    from stgp_pipeline.types import SpatialSlice
    from stgp_pipeline import visualize

    lat_grid = np.array([20.0, 21.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)

    s_young = SpatialSlice(
        age_bp=0.0,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        total_mean=np.array([[1.0, -1.0], [0.5, -0.5]]),
        total_std=np.full((2, 2), 0.5),
        gp_mean=np.zeros((2, 2)),
        gp_var=np.zeros((2, 2)),
        gia_mean=np.zeros((2, 2)),
        gia_var=np.zeros((2, 2)),
        components={
            "regional_linear": {
                "mean": np.full((2, 2), 0.2),
                "std": np.full((2, 2), 0.1),
            },
        },
    )
    s_old = SpatialSlice(
        age_bp=4000.0,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        total_mean=np.array([[5.0, -5.0], [4.0, -4.0]]),
        total_std=np.full((2, 2), 0.5),
        gp_mean=np.zeros((2, 2)),
        gp_var=np.zeros((2, 2)),
        gia_mean=np.zeros((2, 2)),
        gia_var=np.zeros((2, 2)),
        components={
            "regional_linear": {
                "mean": np.full((2, 2), 0.5),
                "std": np.full((2, 2), 0.1),
            },
        },
    )

    vabs_total, vabs_rl = visualize.spatial_total_vs_regional_linear_vabs(
        [s_young, s_old]
    )
    assert vabs_total == pytest.approx(5.0)
    assert vabs_rl == pytest.approx(0.5)


def test_plot_spatial_total_vs_regional_linear_fixed_clim_across_ages():
    pytest.importorskip("cartopy")
    from stgp_pipeline.types import SpatialSlice
    from stgp_pipeline import visualize
    import cartopy.mpl.geoaxes as geoaxes
    import matplotlib.pyplot as plt

    lat_grid = np.array([20.0, 21.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)

    s_young = SpatialSlice(
        age_bp=0.0,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        total_mean=np.array([[1.0, -1.0], [0.5, -0.5]]),
        total_std=np.full((2, 2), 0.5),
        gp_mean=np.zeros((2, 2)),
        gp_var=np.zeros((2, 2)),
        gia_mean=np.zeros((2, 2)),
        gia_var=np.zeros((2, 2)),
        components={
            "regional_linear": {
                "mean": np.full((2, 2), 0.2),
                "std": np.full((2, 2), 0.1),
            },
        },
    )
    s_old = SpatialSlice(
        age_bp=4000.0,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        total_mean=np.array([[5.0, -5.0], [4.0, -4.0]]),
        total_std=np.full((2, 2), 0.5),
        gp_mean=np.zeros((2, 2)),
        gp_var=np.zeros((2, 2)),
        gia_mean=np.zeros((2, 2)),
        gia_var=np.zeros((2, 2)),
        components={
            "regional_linear": {
                "mean": np.full((2, 2), 0.5),
                "std": np.full((2, 2), 0.1),
            },
        },
    )

    vabs_total, vabs_rl = visualize.spatial_total_vs_regional_linear_vabs(
        [s_young, s_old]
    )
    fig_young = visualize.plot_spatial_total_vs_regional_linear(
        s_young, vabs_total=vabs_total, vabs_rl=vabs_rl
    )
    fig_old = visualize.plot_spatial_total_vs_regional_linear(
        s_old, vabs_total=vabs_total, vabs_rl=vabs_rl
    )

    def _total_clim(fig):
        map_axes = [ax for ax in fig.axes if isinstance(ax, geoaxes.GeoAxes)]
        return map_axes[0].collections[0].get_clim()

    assert _total_clim(fig_young) == _total_clim(fig_old) == (-5.0, 5.0)
    plt.close(fig_young)
    plt.close(fig_old)


def test_plot_spatial_total_vs_regional_linear_missing_component():
    pytest.importorskip("cartopy")
    from stgp_pipeline.types import SpatialSlice
    from stgp_pipeline import visualize

    lat_grid = np.array([20.0])
    lon_grid = np.array([110.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    s = SpatialSlice(
        age_bp=0.0,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        total_mean=np.zeros((1, 1)),
        total_std=np.zeros((1, 1)),
        gp_mean=np.zeros((1, 1)),
        gp_var=np.zeros((1, 1)),
        gia_mean=np.zeros((1, 1)),
        gia_var=np.zeros((1, 1)),
    )
    with pytest.raises(KeyError, match="regional_linear"):
        visualize.plot_spatial_total_vs_regional_linear(s)


def _synthetic_spatial_rate_result(
    *,
    total_rate_m_yr: np.ndarray,
    rl_rate_m_yr: np.ndarray,
    global_rate_m_yr: np.ndarray | None = None,
    spatial_ages: np.ndarray | None = None,
) -> "SpatialRateResult":
    from stgp_pipeline.types import SpatialRateResult

    lat_grid = np.array([20.0, 21.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    n_grid = lat_mat.size
    if spatial_ages is None:
        spatial_ages = np.array([1000.0, 2000.0])
    n_intervals = len(spatial_ages) - 1
    rate_midpoints = (spatial_ages[:-1] + spatial_ages[1:]) / 2.0
    if n_intervals == 1 and total_rate_m_yr.size == n_grid:
        total_shape = total_rate_m_yr.reshape(1, n_grid)
        rl_shape = rl_rate_m_yr.reshape(1, n_grid)
        global_shape = (
            global_rate_m_yr.reshape(1, n_grid)
            if global_rate_m_yr is not None
            else None
        )
    else:
        total_shape = total_rate_m_yr.reshape(n_intervals, n_grid)
        rl_shape = rl_rate_m_yr.reshape(n_intervals, n_grid)
        global_shape = (
            global_rate_m_yr.reshape(n_intervals, n_grid)
            if global_rate_m_yr is not None
            else None
        )

    rate_comp = {
        "regional_linear": -rl_shape / 1000.0,
    }
    sd_comp = {"regional_linear": np.full((n_intervals, n_grid), 0.001)}
    if global_shape is not None:
        rate_comp["global"] = -global_shape / 1000.0
        sd_comp["global"] = np.full((n_intervals, n_grid), 0.001)

    return SpatialRateResult(
        spatial_ages=spatial_ages,
        rate_midpoints=rate_midpoints,
        lat_flat=lat_mat.flatten(),
        lon_flat=lon_mat.flatten(),
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        rate_total=-total_shape / 1000.0,
        sd_total=np.full((n_intervals, n_grid), 0.001),
        rate_gp=np.zeros((n_intervals, n_grid)),
        sd_gp=np.full((n_intervals, n_grid), 0.001),
        rate_gia=np.zeros((n_intervals, n_grid)),
        sd_gia=np.full((n_intervals, n_grid), 0.001),
        rate_comp=rate_comp,
        sd_comp=sd_comp,
    )


def test_spatial_total_vs_regional_linear_rate_vabs_shared_limit():
    from stgp_pipeline import visualize

    rates = _synthetic_spatial_rate_result(
        total_rate_m_yr=np.array([[5.0, -5.0], [4.0, -4.0]]),
        rl_rate_m_yr=np.array([[1.0, -1.0], [0.5, -0.5]]),
    )
    vabs = visualize.spatial_total_vs_regional_linear_rate_vabs(
        rates, interval_idx=0, sd_threshold_mm_yr=100.0
    )
    assert vabs == pytest.approx(5.0)


def test_plot_spatial_total_vs_regional_linear_rate_shared_clim():
    pytest.importorskip("cartopy")
    from stgp_pipeline import visualize
    import cartopy.mpl.geoaxes as geoaxes
    import matplotlib.pyplot as plt

    rates = _synthetic_spatial_rate_result(
        total_rate_m_yr=np.array([[3.0, -3.0], [2.0, -2.0]]),
        rl_rate_m_yr=np.array([[1.0, -1.0], [0.5, -0.5]]),
    )
    vabs = visualize.spatial_total_vs_regional_linear_rate_vabs(
        rates, interval_idx=0, sd_threshold_mm_yr=100.0
    )
    fig = visualize.plot_spatial_total_vs_regional_linear_rate(
        rates, interval_idx=0, sd_threshold_mm_yr=100.0, vabs=vabs
    )

    map_axes = [ax for ax in fig.axes if isinstance(ax, geoaxes.GeoAxes)]
    assert len(map_axes) == 2
    clim0 = map_axes[0].collections[0].get_clim()
    clim1 = map_axes[1].collections[0].get_clim()
    assert clim0 == clim1 == (-vabs, vabs)
    plt.close(fig)


def test_plot_spatial_total_vs_regional_linear_rate_missing_component():
    pytest.importorskip("cartopy")
    from stgp_pipeline.types import SpatialRateResult
    from stgp_pipeline import visualize

    lat_grid = np.array([20.0])
    lon_grid = np.array([110.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    rates = SpatialRateResult(
        spatial_ages=np.array([1000.0, 2000.0, 3000.0]),
        rate_midpoints=np.array([2000.0]),
        lat_flat=lat_mat.flatten(),
        lon_flat=lon_mat.flatten(),
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        rate_total=np.zeros((1, 1)),
        sd_total=np.zeros((1, 1)),
        rate_gp=np.zeros((1, 1)),
        sd_gp=np.zeros((1, 1)),
        rate_gia=np.zeros((1, 1)),
        sd_gia=np.zeros((1, 1)),
    )
    with pytest.raises(KeyError, match="regional_linear"):
        visualize.plot_spatial_total_vs_regional_linear_rate(
            rates, interval_idx=0)


def test_spatial_rate_vabs_by_panel_scans_all_intervals():
    from stgp_pipeline import visualize

    spatial_ages = np.array([0.0, 1000.0, 2000.0])
    rates = _synthetic_spatial_rate_result(
        spatial_ages=spatial_ages,
        total_rate_m_yr=np.array(
            [[1.0, -1.0, 1.0, -1.0], [5.0, -5.0, 5.0, -5.0]]
        ),
        rl_rate_m_yr=np.array(
            [[0.5, -0.5, 0.5, -0.5], [1.0, -1.0, 1.0, -1.0]]
        ),
        global_rate_m_yr=np.array(
            [[0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0]]
        ),
    )
    comp_labels = ["global", "regional_linear"]
    vabs = visualize.spatial_rate_vabs_by_panel(
        rates,
        comp_labels=comp_labels,
        sd_threshold_mm_yr=100.0,
    )
    assert vabs["total"] == pytest.approx(5.0)
    assert vabs["global"] == pytest.approx(2.0)
    assert vabs["regional_linear"] == pytest.approx(1.0)


def test_plot_spatial_rate_fixed_clim_across_intervals():
    pytest.importorskip("cartopy")
    from stgp_pipeline import visualize
    import cartopy.mpl.geoaxes as geoaxes
    import matplotlib.pyplot as plt

    spatial_ages = np.array([0.0, 1000.0, 2000.0])
    rates = _synthetic_spatial_rate_result(
        spatial_ages=spatial_ages,
        total_rate_m_yr=np.array(
            [[1.0, -1.0, 1.0, -1.0], [5.0, -5.0, 5.0, -5.0]]
        ),
        rl_rate_m_yr=np.array(
            [[0.5, -0.5, 0.5, -0.5], [1.0, -1.0, 1.0, -1.0]]
        ),
        global_rate_m_yr=np.array(
            [[0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0]]
        ),
    )
    comp_labels = ["global", "regional_linear"]
    rate_vabs = visualize.spatial_rate_vabs_by_panel(
        rates,
        comp_labels=comp_labels,
        sd_threshold_mm_yr=100.0,
    )
    fig_young = visualize.plot_spatial_rate(
        rates,
        0,
        comp_labels=comp_labels,
        sd_threshold_mm_yr=100.0,
        vabs_by_panel=rate_vabs,
    )
    fig_old = visualize.plot_spatial_rate(
        rates,
        1,
        comp_labels=comp_labels,
        sd_threshold_mm_yr=100.0,
        vabs_by_panel=rate_vabs,
    )

    def _rate_clim(fig, panel_idx):
        map_axes = [ax for ax in fig.axes if isinstance(ax, geoaxes.GeoAxes)]
        return map_axes[panel_idx].collections[0].get_clim()

    expected_total = (-rate_vabs["total"], rate_vabs["total"])
    expected_global = (-rate_vabs["global"], rate_vabs["global"])
    assert _rate_clim(fig_young, 0) == _rate_clim(fig_old, 0) == expected_total
    assert _rate_clim(fig_young, 2) == _rate_clim(
        fig_old, 2) == expected_global
    plt.close(fig_young)
    plt.close(fig_old)


def test_spatial_decomposition_vabs_by_panel_scans_all_ages():
    from stgp_pipeline.types import SpatialSlice
    from stgp_pipeline import visualize

    lat_grid = np.array([20.0, 21.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    comp_labels = ["global", "regional_linear"]

    slices = []
    for age_bp, scale in ((0.0, 1.0), (4000.0, 5.0)):
        grid = np.full((2, 2), scale)
        slices.append(
            SpatialSlice(
                age_bp=age_bp,
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                lat_mat=lat_mat,
                lon_mat=lon_mat,
                total_mean=grid,
                total_std=np.full((2, 2), 0.5),
                gp_mean=np.zeros((2, 2)),
                gp_var=np.zeros((2, 2)),
                gia_mean=np.zeros((2, 2)),
                gia_var=np.zeros((2, 2)),
                components={
                    "global": {
                        "mean": grid * 0.2,
                        "std": np.full((2, 2), 0.1),
                    },
                    "regional_linear": {
                        "mean": grid * 0.1,
                        "std": np.full((2, 2), 0.1),
                    },
                },
            )
        )

    vabs = visualize.spatial_decomposition_vabs_by_panel(
        slices, comp_labels=comp_labels, std_threshold_m=100.0
    )
    assert vabs["total"] == pytest.approx(5.0)
    assert vabs["global"] == pytest.approx(1.0)
    assert vabs["regional_linear"] == pytest.approx(0.5)


def test_spatial_slices_from_dataframe_round_trip():
    from stgp_pipeline import export
    from stgp_pipeline.types import SpatialSlice

    lat_grid = np.array([20.0, 21.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    comp_labels = ["global", "regional_linear"]

    slices = [
        SpatialSlice(
            age_bp=4000.0,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            lat_mat=lat_mat,
            lon_mat=lon_mat,
            total_mean=np.array([[1.0, 2.0], [3.0, 4.0]]),
            total_std=np.full((2, 2), 0.5),
            gp_mean=np.zeros((2, 2)),
            gp_var=np.full((2, 2), 0.25),
            gia_mean=np.ones((2, 2)),
            gia_var=np.full((2, 2), 0.09),
            components={
                "global": {
                    "mean": np.full((2, 2), 0.5),
                    "std": np.full((2, 2), 0.1),
                },
                "regional_linear": {
                    "mean": np.full((2, 2), 0.2),
                    "std": np.full((2, 2), 0.05),
                },
            },
        )
    ]

    df = export.spatial_to_dataframe(slices, comp_labels)
    rebuilt = export.spatial_slices_from_dataframe(df, [4000.0], comp_labels=comp_labels)
    assert len(rebuilt) == 1
    s = rebuilt[0]
    assert s.age_bp == pytest.approx(4000.0)
    np.testing.assert_allclose(s.total_mean, slices[0].total_mean)
    np.testing.assert_allclose(s.total_std, slices[0].total_std)
    np.testing.assert_allclose(s.gp_mean, slices[0].gp_mean)
    np.testing.assert_allclose(s.components["global"]["mean"], 0.5)


def test_spatial_rates_from_dataframe_round_trip():
    from stgp_pipeline import export
    from stgp_pipeline.types import SpatialRateResult

    lat_grid = np.array([20.0, 21.0])
    lon_grid = np.array([110.0, 111.0])
    lon_mat, lat_mat = np.meshgrid(lon_grid, lat_grid)
    n_grid = lat_mat.size
    comp_labels = ["global", "regional_linear"]

    rates = SpatialRateResult(
        spatial_ages=np.array([0.0, 1000.0, 2000.0]),
        rate_midpoints=np.array([500.0, 1500.0]),
        lat_flat=lat_mat.ravel(),
        lon_flat=lon_mat.ravel(),
        lat_mat=lat_mat,
        lon_mat=lon_mat,
        rate_total=np.array(
            [[-0.001, -0.002, -0.003, -0.004], [-0.005, -0.006, -0.007, -0.008]]
        ),
        sd_total=np.full((2, n_grid), 0.001),
        rate_gp=np.array(
            [[-0.0005, -0.001, -0.0015, -0.002], [-0.0025, -0.003, -0.0035, -0.004]]
        ),
        sd_gp=np.full((2, n_grid), 0.0005),
        rate_gia=np.array(
            [[-0.0005, -0.001, -0.0015, -0.002], [-0.0025, -0.003, -0.0035, -0.004]]
        ),
        sd_gia=np.full((2, n_grid), 0.0003),
        rate_comp={
            "global": np.array(
                [[-0.0002, -0.0003, -0.0004, -0.0005], [-0.0006, -0.0007, -0.0008, -0.0009]]
            ),
            "regional_linear": np.array(
                [[-0.0003, -0.0004, -0.0005, -0.0006], [-0.0007, -0.0008, -0.0009, -0.001]]
            ),
        },
        sd_comp={
            "global": np.full((2, n_grid), 0.0002),
            "regional_linear": np.full((2, n_grid), 0.0004),
        },
    )

    df = export.rates_to_dataframe(rates, comp_labels)
    rebuilt = export.spatial_rates_from_dataframe(df, comp_labels=comp_labels)
    np.testing.assert_allclose(rebuilt.spatial_ages, rates.spatial_ages)
    np.testing.assert_allclose(rebuilt.rate_midpoints, rates.rate_midpoints)
    np.testing.assert_allclose(rebuilt.rate_total, rates.rate_total)
    np.testing.assert_allclose(rebuilt.sd_total, rates.sd_total)
    np.testing.assert_allclose(rebuilt.rate_gia, rates.rate_gia)
    np.testing.assert_allclose(
        rebuilt.rate_comp["regional_linear"], rates.rate_comp["regional_linear"]
    )
