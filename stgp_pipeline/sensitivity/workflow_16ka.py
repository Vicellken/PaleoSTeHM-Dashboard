"""
Adapter for no-GIA spatiotemporal GP training and temporal NetCDF exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import netCDF4 as nc
import numpy as np

import PSTHM.post

from .. import data, training
from ..gm_rate import GmRateSeries, pointwise_peak_summary
from .workflow_common import (
    artifacts_newer_than_checkpoint,
    boundary_proximity_dataframe,
    emit_stage,
    save_trained_param_store,
    training_run_options,
)


def _get_temporal_test_age(cfg: Mapping[str, Any]) -> np.ndarray:
    tc = cfg.get("temporal_test_age", {}) or {}
    start = int(tc.get("start", 0))
    end = int(tc.get("end", 16000))
    interval = int(tc.get("interval", 200))
    if interval <= 0:
        raise ValueError(f"temporal_test_age.interval must be > 0, got {interval}")
    return np.arange(start, end + 1, interval, dtype=float)


def _get_max_age_bp(cfg: Mapping[str, Any]) -> int:
    analysis_cfg = cfg.get("analysis", {}) or {}
    if analysis_cfg.get("max_age_bp") is not None:
        return int(analysis_cfg["max_age_bp"])
    tc = cfg.get("temporal_test_age", {}) or {}
    return int(tc.get("end", 16000))


def _write_rsl_netcdf(
    path: Path,
    test_age: np.ndarray,
    region_names: Iterable[str],
    total_mean: np.ndarray,
    total_std: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    component_mean: Dict[str, np.ndarray],
    component_std: Dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_regions = int(total_mean.shape[0])
    num_times = int(total_mean.shape[1])
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("region", num_regions)
        ds.createDimension("time", num_times)
        ds.createVariable("time", "f4", ("time",))[:] = np.asarray(
            test_age, dtype=float
        )
        ds.createVariable("regions", str, ("region",))[:] = np.asarray(
            [str(x) for x in region_names], dtype=object
        )
        ds.createVariable("total_mean", "f4", ("region", "time"))[:] = total_mean
        ds.createVariable("total_std", "f4", ("region", "time"))[:] = total_std
        ds.createVariable("global_mean", "f4", ("time",))[:] = global_mean
        ds.createVariable("global_std", "f4", ("time",))[:] = global_std

        for key, values in component_mean.items():
            ds.createVariable(f"{key}_mean", "f4", ("region", "time"))[:] = values
            ds.createVariable(f"{key}_std", "f4", ("region", "time"))[:] = (
                component_std[key]
            )


def _write_rate_netcdf(
    path: Path,
    test_age: np.ndarray,
    rate_time: np.ndarray,
    region_names: Iterable[str],
    total_rate_mean: np.ndarray,
    total_rate_std: np.ndarray,
    global_rate_mean: np.ndarray,
    global_rate_std: np.ndarray,
    global_rate_cov: np.ndarray | None,
    component_rate_mean: Dict[str, np.ndarray],
    component_rate_std: Dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_regions = int(total_rate_mean.shape[0])
    num_times = int(test_age.shape[0])
    num_rate_times = int(rate_time.shape[0])

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("region", num_regions)
        ds.createDimension("time", num_times)
        ds.createDimension("rate_time", num_rate_times)
        ds.createVariable("time", "f4", ("time",))[:] = np.asarray(
            test_age, dtype=float
        )
        ds.createVariable("rate_time", "f4", ("rate_time",))[:] = np.asarray(
            rate_time, dtype=float
        )
        ds.createVariable("regions", str, ("region",))[:] = np.asarray(
            [str(x) for x in region_names], dtype=object
        )
        ds.createVariable("total_rate_mean", "f4", ("region", "rate_time"))[
            :
        ] = total_rate_mean
        ds.createVariable("total_rate_std", "f4", ("region", "rate_time"))[
            :
        ] = total_rate_std
        ds.createVariable("global_rate_mean", "f4", ("rate_time",))[
            :
        ] = global_rate_mean
        ds.createVariable("global_rate_std", "f4", ("rate_time",))[:] = global_rate_std
        if global_rate_cov is not None:
            ds.createVariable(
                "global_rate_cov", "f8", ("rate_time", "rate_time")
            )[:] = np.asarray(global_rate_cov, dtype=float)

        for key, values in component_rate_mean.items():
            ds.createVariable(f"{key}_rate_mean", "f4", ("region", "rate_time"))[
                :
            ] = values
            ds.createVariable(f"{key}_rate_std", "f4", ("region", "rate_time"))[:] = (
                component_rate_std[key]
            )


def _resolve_dataset_path(dataset: str | Path, base_dir: Optional[Path]) -> Path:
    p = Path(dataset).expanduser()
    if p.is_absolute():
        return p
    if base_dir is not None:
        candidate = (Path(base_dir) / p).resolve()
        if candidate.exists() or not p.exists():
            return candidate
    return p.resolve()


def run_workflow_16ka(
    cfg: Mapping[str, Any],
    scenario_dir: Path,
    *,
    base_dir: Optional[Path] = None,
    progress_callback=None,
    stage_callback=None,
    postprocess: bool = True,
    force_recompute: bool = True,
) -> Dict[str, Any]:
    """
    Run the no-GIA STGP workflow: training, parameter extraction, and optional exports.

    When ``postprocess`` is False, stop after NLL / boundary proximity.
    """
    paths = cfg.get("paths", {}) or {}
    dataset = paths.get("dataset") or paths.get("paleo_rsl")
    if not dataset:
        raise KeyError(
            "Missing cfg['paths']['dataset'] or cfg['paths']['paleo_rsl'] "
            "for no-GIA workflow"
        )
    dataset_path = _resolve_dataset_path(dataset, base_dir)

    max_age_bp = _get_max_age_bp(cfg)
    rsl = data.load_and_filter_rsl(
        str(dataset_path), max_age_bp=max_age_bp, config=dict(cfg)
    )

    run_opt, load_path, save_path = training_run_options(cfg, scenario_dir)

    if run_opt:
        emit_stage(stage_callback, "train", "Training GP model (SVI)")
    else:
        emit_stage(
            stage_callback,
            "load_model",
            "Loading trained parameters from checkpoint",
        )
    model = training.build_and_train(
        dict(cfg),
        rsl.X,
        rsl.y,
        rsl.y_sigma ** 2,
        rsl.age_sigma,
        run_key="stgp_no_gia",
        train=run_opt,
        param_store_path=str(load_path) if load_path else None,
        progress_callback=progress_callback,
    )
    if run_opt:
        emit_stage(stage_callback, "save_model", "Saving trained parameters")
        param_store_path = save_trained_param_store(save_path)
    else:
        param_store_path = str(load_path)

    gpr = model.gpr
    kernel_components = model.kernel_components

    emit_stage(stage_callback, "nll", "Computing negative log-likelihood")
    nll = float(training.negative_log_likelihhod(gpr).item())

    df_params = training.extract_hyperparams(model)
    df_params.to_csv(scenario_dir / "optimized_params.csv", index=False)

    emit_stage(stage_callback, "boundary", "Running prior range / boundary check")
    df_boundary = boundary_proximity_dataframe(df_params, cfg)
    df_boundary.to_csv(scenario_dir / "boundary_proximity.csv", index=False)

    if model.track_list is not None and not model.track_list.empty:
        model.track_list.to_csv(
            scenario_dir / "optimization_track.csv", index_label="iteration"
        )

    base_metrics: Dict[str, Any] = {
        "workflow": "16ka",
        "negative_log_likelihood": nll,
        "n_training_points": int(rsl.n_slip),
        "n_hyperparameters": int(len(df_params)),
        "regional_linear_enabled": bool(
            ((cfg.get("gp_model", {}) or {}).get("kernels", {}) or {})
            .get("regional_linear", {})
            .get("enabled", True)
        ),
        "param_store_path": param_store_path,
        "run_optimization": run_opt,
        "postprocess_complete": False,
    }

    if not postprocess:
        return base_metrics

    scenario_name = scenario_dir.name
    rsl_nc_path = (scenario_dir / f"rsl_{scenario_name}_16ka.nc").resolve()
    rate_nc_path = (scenario_dir / f"rate_{scenario_name}_16ka.nc").resolve()

    if not force_recompute and artifacts_newer_than_checkpoint(
        [rsl_nc_path, rate_nc_path], scenario_dir
    ):
        emit_stage(
            stage_callback,
            "temporal",
            "Using existing temporal NetCDF exports",
        )
        print(f"Reusing existing 16ka NetCDF exports in {scenario_dir}")
        return {
            **base_metrics,
            "peak_global_rate_mm_yr": float("nan"),
            "rsl_nc_path": str(rsl_nc_path),
            "rate_nc_path": str(rate_nc_path),
            "postprocess_complete": True,
        }

    test_age = _get_temporal_test_age(cfg)
    interval = int(abs(test_age[1] - test_age[0])) if len(test_age) > 1 else 200

    unique_regions = rsl.unique_regions
    unique_region_names = np.asarray(
        [rsl.region_num_to_name[int(r)] for r in unique_regions], dtype=object
    )
    region_slip = rsl.rsl_region
    rsl_lat_slip = rsl.rsl_lat
    rsl_lon_slip = rsl.rsl_lon
    num_regions = len(unique_regions)
    num_times = len(test_age)
    num_rate_times = max(num_times - 1, 0)

    test_total_mean = np.zeros((num_regions, num_times), dtype=float)
    test_total_std = np.zeros((num_regions, num_times), dtype=float)
    test_global_mean = np.zeros((num_times,), dtype=float)
    test_global_std = np.zeros((num_times,), dtype=float)
    test_total_rate_mean = np.zeros((num_regions, num_rate_times), dtype=float)
    test_total_rate_std = np.zeros((num_regions, num_rate_times), dtype=float)
    test_global_rate_mean = np.zeros((num_rate_times,), dtype=float)
    test_global_rate_std = np.zeros((num_rate_times,), dtype=float)
    test_global_rate_cov = np.zeros(
        (num_rate_times, num_rate_times), dtype=float
    )

    component_name_pairs = [
        ("regional_linear", "regional_linear"),
        ("regional_nl", "regional_nl"),
        ("local_nl", "local_nl"),
        ("merged_nl", "merged_nl"),
    ]
    component_mean = {
        out_name: np.zeros((num_regions, num_times), dtype=float)
        for _, out_name in component_name_pairs
    }
    component_std = {
        out_name: np.zeros((num_regions, num_times), dtype=float)
        for _, out_name in component_name_pairs
    }
    component_rate_mean = {
        out_name: np.zeros((num_regions, num_rate_times), dtype=float)
        for _, out_name in component_name_pairs
    }
    component_rate_std = {
        out_name: np.zeros((num_regions, num_rate_times), dtype=float)
        for _, out_name in component_name_pairs
    }

    decomp_pairs = [("global_temporal", "global"), *component_name_pairs]
    available_pairs = [(k, out) for k, out in decomp_pairs if k in kernel_components]
    decomp_kernels = [kernel_components[k] for k, _ in available_pairs]

    emit_stage(stage_callback, "temporal", "Temporal predictions per region")
    rate_time = np.array([], dtype=float)
    for i, region_val in enumerate(unique_regions):
        region_mask = np.asarray(region_slip == region_val)
        ave_coord = np.mean(
            np.column_stack([rsl_lat_slip[region_mask], rsl_lon_slip[region_mask]]),
            axis=0,
        )

        pred_matrix = PSTHM.post.gen_pred_matrix(test_age, *ave_coord)
        y_mean, y_var = gpr(pred_matrix.double(), full_cov=True)
        y_mean_np = y_mean.detach().cpu().numpy().reshape(-1)
        y_var_np = y_var.detach().cpu().numpy()
        y_std_np = np.sqrt(np.diag(y_var_np))

        test_total_mean[i] = y_mean_np
        test_total_std[i] = y_std_np

        rt, r_total, rsd_total = PSTHM.post.cal_rate_var(
            test_age,
            y_var_np,
            y_mean_np,
            difftimestep=interval,
        )
        if i == 0:
            rate_time = np.asarray(rt, dtype=float)
        test_total_rate_mean[i] = np.asarray(r_total, dtype=float)
        test_total_rate_std[i] = np.asarray(rsd_total, dtype=float)

        if decomp_kernels:
            decomp = PSTHM.post.decompose_kernels(gpr, pred_matrix, decomp_kernels)
            for (kernel_name, output_name), (d_mean, d_cov) in zip(
                available_pairs, decomp
            ):
                d_mean_np = d_mean.detach().cpu().numpy().reshape(-1)
                d_cov_np = d_cov.detach().cpu().numpy()
                d_std_np = np.sqrt(np.diag(d_cov_np))

                if output_name == "global":
                    _, d_rate, d_rate_cov, d_rate_sd = (
                        PSTHM.post.cal_spatiotemporal_rate_var(
                            test_age,
                            d_cov_np,
                            d_mean_np,
                            difftimestep=interval,
                        )
                    )
                else:
                    _, d_rate, d_rate_sd = PSTHM.post.cal_rate_var(
                        test_age,
                        d_cov_np,
                        d_mean_np,
                        difftimestep=interval,
                    )
                d_rate = np.asarray(d_rate, dtype=float)
                d_rate_sd = np.asarray(d_rate_sd, dtype=float)

                if output_name == "global":
                    if i == 0:
                        test_global_mean[:] = d_mean_np
                        test_global_std[:] = d_std_np
                        test_global_rate_mean[:] = d_rate
                        test_global_rate_std[:] = d_rate_sd
                        test_global_rate_cov[:, :] = np.asarray(
                            d_rate_cov, dtype=float
                        )
                    continue

                component_mean[output_name][i] = d_mean_np
                component_std[output_name][i] = d_std_np
                component_rate_mean[output_name][i] = d_rate
                component_rate_std[output_name][i] = d_rate_sd

    rsl_nc_path = (scenario_dir / f"rsl_{scenario_name}_16ka.nc").resolve()
    rate_nc_path = (scenario_dir / f"rate_{scenario_name}_16ka.nc").resolve()
    _write_rsl_netcdf(
        rsl_nc_path,
        test_age=test_age,
        region_names=unique_region_names,
        total_mean=test_total_mean,
        total_std=test_total_std,
        global_mean=test_global_mean,
        global_std=test_global_std,
        component_mean=component_mean,
        component_std=component_std,
    )
    _write_rate_netcdf(
        rate_nc_path,
        test_age=test_age,
        rate_time=rate_time,
        region_names=unique_region_names,
        total_rate_mean=test_total_rate_mean,
        total_rate_std=test_total_rate_std,
        global_rate_mean=test_global_rate_mean,
        global_rate_std=test_global_rate_std,
        global_rate_cov=test_global_rate_cov,
        component_rate_mean=component_rate_mean,
        component_rate_std=component_rate_std,
    )

    peak_global_rate = float("nan")
    if rate_time.size > 0:
        mask_9_13ka = (rate_time >= 9000) & (rate_time <= 13000)
        if np.any(mask_9_13ka):
            peak_global_rate = float(
                np.max(-test_global_rate_mean[mask_9_13ka] * 1000.0)
            )
    gm_series = GmRateSeries(
        age_bp=rate_time,
        mean_m_per_yr=test_global_rate_mean,
        std_m_per_yr=test_global_rate_std,
        covariance_m2_per_yr2=test_global_rate_cov,
        source_path=rate_nc_path,
    )
    has_mwp1b = rate_time.size > 0 and bool(
        np.any((rate_time >= 11000.0) & (rate_time <= 11500.0))
    )
    if has_mwp1b:
        mwp1b = pointwise_peak_summary(
            gm_series,
            age_lo_bp=11000.0,
            age_hi_bp=11500.0,
            interval_sigma=2.0,
        )
    else:
        mwp1b = {
            "mean_mm_per_yr": float("nan"),
            "age_bp": float("nan"),
            "interval_half_width_mm_per_yr": float("nan"),
            "interval_lower_mm_per_yr": float("nan"),
            "interval_upper_mm_per_yr": float("nan"),
        }

    return {
        **base_metrics,
        "peak_global_rate_mm_yr": peak_global_rate,
        "peak_global_rate_window_bp": [9000.0, 13000.0],
        "mwp1b_global_rate_mm_yr": mwp1b["mean_mm_per_yr"],
        "mwp1b_global_rate_age_bp": mwp1b["age_bp"],
        "mwp1b_global_rate_pointwise_2sigma_mm_yr": mwp1b[
            "interval_half_width_mm_per_yr"
        ],
        "mwp1b_global_rate_pointwise_95_lower_mm_yr": mwp1b[
            "interval_lower_mm_per_yr"
        ],
        "mwp1b_global_rate_pointwise_95_upper_mm_yr": mwp1b[
            "interval_upper_mm_per_yr"
        ],
        "rsl_nc_path": str(rsl_nc_path),
        "rate_nc_path": str(rate_nc_path),
        "postprocess_complete": True,
    }
