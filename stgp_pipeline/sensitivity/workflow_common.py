"""
Shared helpers for workflow adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

StageCallback = Optional[Callable[[str, str], None]]

import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml

from .config_mutations import collect_kernel_priors


EARTH_RADIUS_KM = 6371.0
LOG_SCALE_RATIO = 100.0

# Default 6-region SLIP subset (``Region.1`` codes) for summary rate panels.
DEFAULT_SLIP_REGION_IDS = (3, 4, 6, 7, 8, 9)


def uses_combined_3d_gia(paths: Mapping[str, Any]) -> bool:
    """True when cfg defines a 3D GIA directory (combined 1D + 3D ensemble)."""
    gia_3d_dir = paths.get("gia_3d_dir")
    return isinstance(gia_3d_dir, str) and bool(gia_3d_dir.strip())


def training_run_options(
    cfg: Mapping[str, Any],
    scenario_dir: Path,
) -> Tuple[bool, Optional[Path], Path]:
    """
    Resolve whether to run SVI or load a checkpoint.

    Returns ``(run_optimization, load_path, save_path)``.
    """
    gp_cfg = cfg.get("gp_model") or {}
    paths = cfg.get("paths") or {}
    run_opt = bool(gp_cfg.get("run_optimization", True))
    save_path = Path(scenario_dir) / "trained_param_store.pt"
    if run_opt:
        return True, None, save_path

    load_raw = paths.get("trained_param_store")
    if not load_raw:
        raise ValueError(
            "gp_model.run_optimization is false but paths.trained_param_store is not set"
        )
    load_path = Path(str(load_raw)).expanduser().resolve()
    if not load_path.is_file():
        raise FileNotFoundError(f"Trained param store not found: {load_path}")
    return False, load_path, save_path


def emit_stage(
    stage_callback: StageCallback, stage_id: str, message: str
) -> None:
    if stage_callback is not None:
        stage_callback(stage_id, message)


def _resolve_checkpoint_path(scenario_dir: Path) -> Optional[Path]:
    ckpt = Path(scenario_dir) / "trained_param_store.pt"
    return ckpt.resolve() if ckpt.is_file() else None


def artifacts_newer_than_checkpoint(
    artifact_paths: Sequence[Path],
    scenario_dir: Path,
) -> bool:
    """
    True when all ``artifact_paths`` exist and none are older than the checkpoint.

    If no checkpoint exists, existence of all artifacts is enough.
    """
    paths = [Path(p) for p in artifact_paths]
    if not paths or not all(p.is_file() for p in paths):
        return False
    ckpt = _resolve_checkpoint_path(scenario_dir)
    if ckpt is None:
        return True
    ckpt_mtime = ckpt.stat().st_mtime
    return all(p.stat().st_mtime >= ckpt_mtime for p in paths)


def _load_param_store(path: Path | str, map_location: str = "cpu") -> None:
    import pyro
    import torch

    with open(path, "rb") as f:
        try:
            state = torch.load(f, map_location=map_location, weights_only=False)
        except TypeError:
            f.seek(0)
            state = torch.load(f, map_location=map_location)
    pyro.clear_param_store()
    pyro.get_param_store().set_state(state)


def save_trained_param_store(path: Path) -> str:
    import pyro

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pyro.get_param_store().save(str(path))
    return str(path.resolve())


def load_trained_param_store(path: Path, gpr) -> None:
    _load_param_store(path)
    gpr.set_mode("guide")


_PARAM_TO_BOUND = {
    "K_G  lengthscale (yr)": ("global_temporal.lengthscale_prior", "identity"),
    "K_G  variance (m)": ("global_temporal.variance_prior", "sqrt"),
    "K_RL temporal variance (m/yr)": (
        "regional_linear.temporal_variance_prior",
        "sqrt",
    ),
    "K_RL spatial s_lengthscale (km)": (
        "regional_linear.spatial_lengthscale_prior",
        "km",
    ),
    "K_RNL temporal variance (m)": (
        "regional_nonlinear.temporal_variance_prior",
        "sqrt",
    ),
    "K_RNL temporal lengthscale (yr)": (
        "regional_nonlinear.temporal_lengthscale_prior",
        "identity",
    ),
    "K_RNL spatial s_lengthscale (km)": (
        "regional_nonlinear.spatial_lengthscale_prior",
        "km",
    ),
    "K_MNL temporal variance (m)": ("merged_nonlinear.temporal_variance_prior", "sqrt"),
    "K_MNL temporal lengthscale (yr)": (
        "merged_nonlinear.temporal_lengthscale_prior",
        "identity",
    ),
    "K_MNL spatial s_lengthscale (km)": (
        "merged_nonlinear.spatial_lengthscale_prior",
        "km",
    ),
    "K_LNL temporal variance (m)": ("local_nonlinear.temporal_variance_prior", "sqrt"),
    "K_LNL temporal lengthscale (yr)": (
        "local_nonlinear.temporal_lengthscale_prior",
        "identity",
    ),
    "K_LNL spatial s_lengthscale (km)": (
        "local_nonlinear.spatial_lengthscale_prior",
        "km",
    ),
    "K_G_FAST lengthscale (yr)": ("global_temporal_fast.lengthscale_prior", "identity"),
    "K_G_FAST variance (m)": ("global_temporal_fast.variance_prior", "sqrt"),
    "K_RNL_FAST temporal variance (m)": (
        "regional_nonlinear_fast.temporal_variance_prior",
        "sqrt",
    ),
    "K_RNL_FAST temporal lengthscale (yr)": (
        "regional_nonlinear_fast.temporal_lengthscale_prior",
        "identity",
    ),
    "K_RNL_FAST spatial s_lengthscale (km)": (
        "regional_nonlinear_fast.spatial_lengthscale_prior",
        "km",
    ),
    "K_LNL_FAST temporal variance (m)": (
        "local_nonlinear_fast.temporal_variance_prior",
        "sqrt",
    ),
    "K_LNL_FAST temporal lengthscale (yr)": (
        "local_nonlinear_fast.temporal_lengthscale_prior",
        "identity",
    ),
    "K_LNL_FAST spatial s_lengthscale (km)": (
        "local_nonlinear_fast.spatial_lengthscale_prior",
        "km",
    ),
    "K_SWN variance (m)": ("sp_whitenoise.variance_prior", "sqrt"),
    "K_WN  variance (m)": ("whitenoise.variance_prior", "sqrt"),
}

# Legacy parameter display names used by notebook CSV exports.
_NOTEBOOK_PARAM_TO_BOUND = {
    "global_kernel_lengthscale (yr)": ("global_temporal.lengthscale_prior", "identity"),
    "global_kernel_variance (m)": ("global_temporal.variance_prior", "sqrt"),
    "global_fast_kernel_lengthscale (yr)": (
        "global_temporal_fast.lengthscale_prior",
        "identity",
    ),
    "global_fast_kernel_variance (m)": (
        "global_temporal_fast.variance_prior",
        "sqrt",
    ),
    "regional_linear_kernel_variance (m/yr)": (
        "regional_linear.temporal_variance_prior",
        "sqrt",
    ),
    "regional_linearl_kernel_spatil_lengthscale (km)": (
        "regional_linear.spatial_lengthscale_prior",
        "km",
    ),
    "regional_nl_kernel_variance (m)": (
        "regional_nonlinear.temporal_variance_prior",
        "sqrt",
    ),
    "regional_nl_kernel_lengthscale (yr)": (
        "regional_nonlinear.temporal_lengthscale_prior",
        "identity",
    ),
    "regional_nl_kernel_spatial_lengthscale (km)": (
        "regional_nonlinear.spatial_lengthscale_prior",
        "km",
    ),
    "local_nl_kernel_variance (m)": (
        "local_nonlinear.temporal_variance_prior",
        "sqrt",
    ),
    "local_nl_kernel_lengthscale (yr)": (
        "local_nonlinear.temporal_lengthscale_prior",
        "identity",
    ),
    "local_nl_spatial_spatial_lengthscale (km)": (
        "local_nonlinear.spatial_lengthscale_prior",
        "km",
    ),
    "local_nl_fast_kernel_variance (m)": (
        "local_nonlinear_fast.temporal_variance_prior",
        "sqrt",
    ),
    "local_nl_fast_kernel_lengthscale (yr)": (
        "local_nonlinear_fast.temporal_lengthscale_prior",
        "identity",
    ),
    "local_nl_fast_spatial_spatial_lengthscale (km)": (
        "local_nonlinear_fast.spatial_lengthscale_prior",
        "km",
    ),
    "sp_whitenoise_kernel_variance (m)": ("sp_whitenoise.variance_prior", "sqrt"),
    "whitenoise_kernel_variance (m)": ("whitenoise.variance_prior", "sqrt"),
}


def save_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, sort_keys=False)


def _convert_bounds(bounds: Tuple[float, float], mode: str) -> Tuple[float, float]:
    lo, hi = bounds
    if mode == "identity":
        return float(lo), float(hi)
    if mode == "sqrt":
        return float(np.sqrt(lo)), float(np.sqrt(hi))
    if mode == "km":
        return float(lo * EARTH_RADIUS_KM), float(hi * EARTH_RADIUS_KM)
    raise ValueError(f"Unknown conversion mode: {mode!r}")


def compute_boundary_proximity(value: float, lo: float, hi: float) -> Tuple[float, str]:
    """Compute % location inside [lo, hi] in linear/log space."""
    if hi <= lo:
        return float("nan"), "invalid"
    if lo > 0 and hi / lo > LOG_SCALE_RATIO and value > 0:
        pct = (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo)) * 100.0
        return float(pct), "log"
    pct = (value - lo) / (hi - lo) * 100.0
    return float(pct), "linear"


def classify_boundary_hit(
    proximity_pct: float, *, lower_pct: float = 5.0, upper_pct: float = 95.0
) -> Tuple[bool, str]:
    """Classify whether proximity indicates a boundary hit."""
    if not np.isfinite(proximity_pct):
        return False, ""
    if proximity_pct <= lower_pct:
        return True, "lower"
    if proximity_pct >= upper_pct:
        return True, "upper"
    return False, ""


def _boundary_proximity_rows(
    params: Mapping[str, float],
    cfg: Mapping[str, Any],
    *,
    param_map: Mapping[str, Tuple[str, str]],
) -> List[Dict[str, Any]]:
    priors = collect_kernel_priors(cfg, include_disabled=True)
    rows: List[Dict[str, Any]] = []
    for param_name, value in params.items():
        spec = param_map.get(str(param_name))
        if spec is None:
            continue
        prior_key, mode = spec
        if prior_key not in priors:
            continue
        lo, hi = _convert_bounds(priors[prior_key], mode)
        pct, scale = compute_boundary_proximity(float(value), lo, hi)
        boundary_hit, boundary_side = classify_boundary_hit(pct)
        rows.append(
            {
                "parameter": str(param_name),
                "value": float(value),
                "bound_low": lo,
                "bound_high": hi,
                "proximity_pct": pct,
                "boundary_hit": boundary_hit,
                "boundary_side": boundary_side,
                "scale": scale,
                "prior_key": prior_key,
                "conversion": mode,
            }
        )
    return rows


def boundary_proximity_from_notebook_params(
    optimized_params: Mapping[str, float],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Build boundary proximity table from notebook-style optimized param names."""
    rows = _boundary_proximity_rows(
        optimized_params,
        cfg,
        param_map=_NOTEBOOK_PARAM_TO_BOUND,
    )
    return pd.DataFrame(rows)


def format_boundary_proximity_report(df: pd.DataFrame) -> str:
    """Return a human-readable bound-proximity summary for stdout."""
    if df.empty:
        return "Bound proximity: no parameters matched config priors."

    lines = ["Bound proximity (0%=lower bound, 100%=upper bound):"]
    for _, row in df.sort_values("proximity_pct", ascending=False).iterrows():
        side = row.get("boundary_side", "")
        side_txt = f" ({side})" if pd.notna(side) and str(side).strip() else ""
        lines.append(
            f"  {row['parameter']}: {row['proximity_pct']:.1f}%{side_txt}"
        )

    hits = df[df["boundary_hit"].astype(bool)]
    lines.append("")
    if hits.empty:
        lines.append("Boundary hits (<=5% or >=95%): none")
    else:
        lines.append("Boundary hits (<=5% or >=95%):")
        for _, row in hits.iterrows():
            side = row.get("boundary_side", "")
            lines.append(
                f"  {row['parameter']}: {row['proximity_pct']:.1f}% ({side})"
            )
    return "\n".join(lines)


def boundary_proximity_dataframe(
    df_params: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Build a boundary-proximity table from hyperparameters + cfg priors.
    """
    params = {
        str(row["Parameter"]): float(row["Value"])
        for _, row in df_params.iterrows()
    }
    rows = _boundary_proximity_rows(params, cfg, param_map=_PARAM_TO_BOUND)
    return pd.DataFrame(rows)


def extract_kernel_components(combined_kernel: Any) -> Dict[str, Any]:
    """
    Extract name->kernel mapping from factory-built kernel trees.
    """
    explicit = getattr(combined_kernel, "component_kernels", None)
    if isinstance(explicit, dict):
        return dict(explicit)

    # Legacy fallback (assumes full canonical tree).
    names = [
        "whitenoise",
        "sp_whitenoise",
        "local_nl",
        "regional_nl",
        "regional_linear",
        "global_temporal",
    ]
    components: Dict[str, Any] = {}
    node = combined_kernel
    for name in names[:-1]:
        if not hasattr(node, "kern1") or not hasattr(node, "kern0"):
            return {}
        components[name] = node.kern1
        node = node.kern0
    components[names[-1]] = node
    return components


def workflow_nc_suffix(workflow: str) -> str:
    """NetCDF filename suffix for a sensitivity workflow."""
    if workflow == "4ka":
        return "4ka"
    if workflow == "regional-gia":
        return "regional_gia"
    if workflow in {"16ka", "3d-gia"}:
        return "16ka" if workflow == "16ka" else "3d_gia"
    return workflow.replace("-", "_")


def peak_rate_age_window(workflow: str) -> Tuple[float, float]:
    """Inclusive BP age window for peak global/total rate summaries."""
    if workflow == "4ka":
        return 0.0, 4000.0
    if workflow == "regional-gia":
        return 0.0, 5000.0
    return 9000.0, 13000.0


def peak_global_rate_mm_yr(
    rate_time: np.ndarray,
    global_rate_mean: np.ndarray,
    *,
    workflow: str = "16ka",
) -> float:
    """Peak global kernel rate (mm/yr) inside the workflow-specific age window."""
    lo, hi = peak_rate_age_window(workflow)
    mask = (rate_time >= lo) & (rate_time <= hi)
    if not np.any(mask):
        return float("nan")
    return float(np.max(-np.asarray(global_rate_mean, dtype=float)[mask] * 1000.0))


def _region_codes_from_paleo_csv(csv_path: Path) -> np.ndarray:
    """
    Unique ``Region.1`` codes in the same order as training (``np.unique``).

    When the CSV has ``Limiting``, use SLIP rows only (``Limiting == 0``) so
    limiting-only region codes are not mistaken for trained regions.
    """
    df = pd.read_csv(csv_path)
    if "Region.1" not in df.columns:
        return np.array([], dtype=int)
    if "Limiting" in df.columns:
        df = df[df["Limiting"] == 0]
    return np.sort(df["Region.1"].dropna().unique().astype(int))


def compute_nc_row_to_region_id(
    num_regions: int,
    region_source_csv: str | Path,
) -> Tuple[np.ndarray | None, List[int]]:
    """
    Map each NetCDF ``region`` row index to integer ``Region.1`` from the paleo CSV.

    Uses the same ordering convention as ``np.unique(rsl_region)`` on SLIP training data.
    """
    csv_path = Path(region_source_csv)
    if not csv_path.is_file():
        return None, list(range(num_regions))

    if "Region.1" not in pd.read_csv(csv_path, nrows=0).columns:
        return None, list(range(num_regions))

    codes_sorted = _region_codes_from_paleo_csv(csv_path)
    if codes_sorted.size == 0:
        return None, list(range(num_regions))

    if len(codes_sorted) == num_regions:
        nc_row_to_region_id = codes_sorted.astype(int)
    elif num_regions == len(codes_sorted) - 1 and np.any(codes_sorted == 0):
        nc_row_to_region_id = codes_sorted[codes_sorted != 0].astype(int)
    else:
        if len(codes_sorted) >= num_regions:
            nc_row_to_region_id = codes_sorted[:num_regions].astype(int)
        else:
            nc_row_to_region_id = np.arange(1, num_regions + 1, dtype=int)

    non_zero_rows = [i for i in range(
        num_regions) if int(nc_row_to_region_id[i]) != 0]
    summary_region_rows = non_zero_rows or list(range(num_regions))
    return nc_row_to_region_id, summary_region_rows


def filter_nc_rows_by_region_ids(
    nc_row_to_region_id: np.ndarray | None,
    row_indices: List[int],
    region_ids: Sequence[int],
) -> List[int]:
    """
    Keep NetCDF row indices whose ``Region.1`` code is in ``region_ids``.

    Panel order follows ``region_ids`` (not NetCDF row order). If mapping is
    unavailable, returns ``row_indices`` unchanged.
    """
    if nc_row_to_region_id is None:
        return list(row_indices)

    code_to_row = {
        int(nc_row_to_region_id[i]): i
        for i in row_indices
        if 0 <= i < nc_row_to_region_id.shape[0]
    }
    return [code_to_row[code] for code in region_ids if code in code_to_row]


def _component_field(comp: Any, key: str, default: Any = None) -> Any:
    if isinstance(comp, Mapping):
        return comp.get(key, default)
    return getattr(comp, key, default)


def region_display_name(
    row_idx: int,
    nc_row_to_region_id: np.ndarray | None,
    *,
    fallback_prefix: str = "R",
) -> str:
    if nc_row_to_region_id is not None and 0 <= row_idx < nc_row_to_region_id.shape[0]:
        return f"Region {int(nc_row_to_region_id[row_idx])}"
    return f"{fallback_prefix}{row_idx + 1}"


def build_temporal_netcdf_arrays(
    temporal_results: Mapping[int, Any],
) -> Dict[str, Any]:
    """
    Pack ``predict.temporal_all_regions`` results into arrays for RSL/rate NetCDF.
    """
    region_ids = sorted(int(k) for k in temporal_results.keys())
    empty: Dict[str, Any] = {
        "test_age": np.array([], dtype=float),
        "rate_time": np.array([], dtype=float),
        "region_names": [],
        "total_mean": np.zeros((0, 0), dtype=float),
        "total_std": np.zeros((0, 0), dtype=float),
        "global_mean": np.array([], dtype=float),
        "global_std": np.array([], dtype=float),
        "total_rate_mean": np.zeros((0, 0), dtype=float),
        "total_rate_std": np.zeros((0, 0), dtype=float),
        "global_rate_mean": np.array([], dtype=float),
        "global_rate_std": np.array([], dtype=float),
        "component_mean": {},
        "component_std": {},
        "component_rate_mean": {},
        "component_rate_std": {},
    }
    if not region_ids:
        return empty

    first = temporal_results[region_ids[0]]
    test_age = np.asarray(first.test_age, dtype=float)
    rate_time = np.asarray(first.rsl_time, dtype=float)
    n_regions = len(region_ids)
    n_times = int(test_age.shape[0])
    n_rate = int(rate_time.shape[0])

    component_labels: List[str] = []
    for rid in region_ids:
        comps = getattr(temporal_results[rid], "components", None) or {}
        for label in comps:
            if label not in component_labels:
                component_labels.append(label)

    total_mean = np.full((n_regions, n_times), np.nan, dtype=float)
    total_std = np.full((n_regions, n_times), np.nan, dtype=float)
    total_rate_mean = np.full((n_regions, n_rate), np.nan, dtype=float)
    total_rate_std = np.full((n_regions, n_rate), np.nan, dtype=float)
    global_mean = np.full((n_times,), np.nan, dtype=float)
    global_std = np.full((n_times,), np.nan, dtype=float)
    global_rate_mean = np.full((n_rate,), np.nan, dtype=float)
    global_rate_std = np.full((n_rate,), np.nan, dtype=float)
    region_names: List[str] = []

    component_mean = {
        label: np.full((n_regions, n_times), np.nan, dtype=float)
        for label in component_labels
    }
    component_std = {
        label: np.full((n_regions, n_times), np.nan, dtype=float)
        for label in component_labels
    }
    component_rate_mean = {
        label: np.full((n_regions, n_rate), np.nan, dtype=float)
        for label in component_labels
    }
    component_rate_std = {
        label: np.full((n_regions, n_rate), np.nan, dtype=float)
        for label in component_labels
    }

    for idx, region_id in enumerate(region_ids):
        res = temporal_results[region_id]
        region_names.append(str(res.region_name))

        y_total = np.asarray(res.y_total, dtype=float)
        y_std = np.asarray(res.y_std, dtype=float)
        t_n = min(n_times, y_total.shape[0], y_std.shape[0])
        if t_n > 0:
            total_mean[idx, :t_n] = y_total[:t_n]
            total_std[idx, :t_n] = y_std[:t_n]

        rate = np.asarray(res.rsl_rate, dtype=float)
        rate_sd = np.asarray(res.rsl_rate_sd, dtype=float)
        r_n = min(n_rate, rate.shape[0], rate_sd.shape[0])
        if r_n > 0:
            total_rate_mean[idx, :r_n] = rate[:r_n]
            total_rate_std[idx, :r_n] = rate_sd[:r_n]

        for label in component_labels:
            comp = (res.components or {}).get(label)
            if not comp:
                continue
            c_mean = np.asarray(_component_field(
                comp, "mean", []), dtype=float)
            c_std = np.asarray(_component_field(comp, "std", []), dtype=float)
            c_rate = np.asarray(_component_field(
                comp, "rate", []), dtype=float)
            c_rate_sd = np.asarray(_component_field(
                comp, "rate_sd", []), dtype=float)
            c_t = min(n_times, c_mean.shape[0], c_std.shape[0])
            if c_t > 0:
                component_mean[label][idx, :c_t] = c_mean[:c_t]
                component_std[label][idx, :c_t] = c_std[:c_t]
            c_r = min(n_rate, c_rate.shape[0], c_rate_sd.shape[0])
            if c_r > 0:
                component_rate_mean[label][idx, :c_r] = c_rate[:c_r]
                component_rate_std[label][idx, :c_r] = c_rate_sd[:c_r]

            if label == "global":
                if t_n > 0 and np.all(np.isnan(global_mean)):
                    g_t = min(n_times, c_mean.shape[0], c_std.shape[0])
                    if g_t > 0:
                        global_mean[:g_t] = c_mean[:g_t]
                        global_std[:g_t] = c_std[:g_t]
                if r_n > 0 and np.all(np.isnan(global_rate_mean)):
                    g_r = min(n_rate, c_rate.shape[0], c_rate_sd.shape[0])
                    if g_r > 0:
                        global_rate_mean[:g_r] = c_rate[:g_r]
                        global_rate_std[:g_r] = c_rate_sd[:g_r]

    return {
        "test_age": test_age,
        "rate_time": rate_time,
        "region_names": region_names,
        "total_mean": total_mean,
        "total_std": total_std,
        "global_mean": global_mean,
        "global_std": global_std,
        "total_rate_mean": total_rate_mean,
        "total_rate_std": total_rate_std,
        "global_rate_mean": global_rate_mean,
        "global_rate_std": global_rate_std,
        "component_mean": component_mean,
        "component_std": component_std,
        "component_rate_mean": component_rate_mean,
        "component_rate_std": component_rate_std,
    }


def write_rsl_netcdf(
    path: Path,
    *,
    test_age: np.ndarray,
    region_names: Iterable[str],
    total_mean: np.ndarray,
    total_std: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    component_mean: Mapping[str, np.ndarray],
    component_std: Mapping[str, np.ndarray],
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
        ds.createVariable("total_mean", "f4", ("region", "time"))[
            :] = total_mean
        ds.createVariable("total_std", "f4", ("region", "time"))[:] = total_std
        ds.createVariable("global_mean", "f4", ("time",))[:] = global_mean
        ds.createVariable("global_std", "f4", ("time",))[:] = global_std
        for key, values in component_mean.items():
            if key == "global":
                continue
            ds.createVariable(f"{key}_mean", "f4", ("region", "time"))[
                :] = values
            ds.createVariable(f"{key}_std", "f4", ("region", "time"))[:] = (
                component_std[key]
            )


def write_rate_netcdf(
    path: Path,
    *,
    test_age: np.ndarray,
    rate_time: np.ndarray,
    region_names: Iterable[str],
    total_rate_mean: np.ndarray,
    total_rate_std: np.ndarray,
    global_rate_mean: np.ndarray,
    global_rate_std: np.ndarray,
    component_rate_mean: Mapping[str, np.ndarray] | None = None,
    component_rate_std: Mapping[str, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_regions = int(total_rate_mean.shape[0])
    num_times = int(test_age.shape[0])
    num_rate_times = int(rate_time.shape[0])
    comp_rate_mean = component_rate_mean or {}
    comp_rate_std = component_rate_std or {}

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
        ds.createVariable("global_rate_std", "f4", ("rate_time",))[
            :] = global_rate_std
        for key, values in comp_rate_mean.items():
            if key == "global":
                continue
            ds.createVariable(f"{key}_rate_mean", "f4", ("region", "rate_time"))[
                :
            ] = values
            std_values = comp_rate_std.get(key)
            if std_values is not None:
                ds.createVariable(f"{key}_rate_std", "f4", ("region", "rate_time"))[
                    :
                ] = std_values


def export_spatial_artifacts(
    model: Any,
    rsl: Any,
    gia: Any | None,
    cfg: Mapping[str, Any],
    scenario_dir: Path,
    *,
    max_age_bp: int,
    bp_min: float = 0.0,
    bp_max: float | None = None,
    stage_callback: StageCallback = None,
    force_recompute: bool = True,
) -> Dict[str, str]:
    """
    Export spatial_predictions.csv and spatial_rates.csv for dashboard figures.

    Skips gracefully when GIA is unavailable (e.g. 16 ka no-GIA workflow) or when
    the trained model has no decomposition labels.

    When ``force_recompute`` is False and both CSVs exist (and are not older than
    the checkpoint), returns existing paths without recomputing.
    """
    from .. import export, predict

    if gia is None:
        print("Skipping spatial export: no GIA result.")
        return {}

    comp_labels = list(getattr(model, "comp_labels", None) or [])
    if not comp_labels:
        print("Skipping spatial export: trained model has no component labels.")
        return {}

    cfg_dict = dict(cfg)
    scenario_dir = Path(scenario_dir)
    bp_max_val = float(max_age_bp if bp_max is None else bp_max)
    spatial_csv = scenario_dir / "spatial_predictions.csv"
    rates_csv = scenario_dir / "spatial_rates.csv"

    if not force_recompute and artifacts_newer_than_checkpoint(
        [spatial_csv, rates_csv], scenario_dir
    ):
        emit_stage(
            stage_callback,
            "spatial_rates",
            "Using existing spatial_predictions.csv and spatial_rates.csv",
        )
        print(f"Reusing existing spatial exports in {scenario_dir}")
        return {
            "spatial_csv_path": str(spatial_csv.resolve()),
            "spatial_rates_csv_path": str(rates_csv.resolve()),
        }

    lat_grid, lon_grid, _, _ = predict.build_spatial_grid(rsl, cfg_dict)
    spatial_slices = predict.spatial_all_ages(
        model,
        rsl,
        gia,
        cfg_dict,
        lat_grid,
        lon_grid,
        max_age_bp=max_age_bp,
        bp_min=bp_min,
        bp_max=bp_max_val,
    )
    df_spatial = export.spatial_to_dataframe(spatial_slices, comp_labels)
    df_spatial.to_csv(spatial_csv, index=False)
    print(f"Wrote {len(df_spatial)} rows to {spatial_csv}")

    emit_stage(
        stage_callback,
        "spatial_rates",
        "Computing spatial rates (this may take a while)",
    )
    spatial_rates = predict.spatial_rates(
        model,
        rsl,
        gia,
        cfg_dict,
        lat_grid,
        lon_grid,
        max_age_bp=max_age_bp,
        bp_min=bp_min,
        bp_max=bp_max_val,
    )
    df_rates = export.rates_to_dataframe(spatial_rates, comp_labels)
    df_rates.to_csv(rates_csv, index=False)
    print(f"Wrote {len(df_rates)} rows to {rates_csv}")

    return {
        "spatial_csv_path": str(spatial_csv.resolve()),
        "spatial_rates_csv_path": str(rates_csv.resolve()),
    }


def _run_sea_level_budget_stage(
    scenario_dir: Path,
    temporal_rates_csv: Path,
    cfg: Mapping[str, Any],
    *,
    stage_callback: StageCallback = None,
) -> None:
    """Notebook §5b: percent-contribution budget after temporal rates export."""
    from .. import budget

    emit_stage(stage_callback, "budget", "Sea-level budget contributions")
    budget_cfg = (cfg or {}).get("budget") or {}
    budget_df = budget.export_scenario_sea_level_budget(
        scenario_dir,
        budget_cfg,
        rates_csv=temporal_rates_csv,
    )
    if budget_df is not None and not budget_df.empty:
        print(
            f"Wrote sea-level budget "
            f"({budget_df['region'].nunique()} regions, "
            f"{budget_df[['window_start_bp', 'window_end_bp']].drop_duplicates().shape[0]} windows) "
            f"to {scenario_dir / 'sea_level_budget.csv'}"
        )


def export_gia_temporal_and_spatial(
    model: Any,
    rsl: Any,
    gia: Any,
    cfg: Mapping[str, Any],
    scenario_dir: Path,
    *,
    workflow: str,
    max_age_bp: int,
    bp_min: float = 0.0,
    bp_max: float | None = None,
    nc_suffix: str | None = None,
    stage_callback: StageCallback = None,
    force_recompute: bool = True,
) -> Dict[str, Any]:
    """
    Shared post-processing for GIA workflows: temporal NetCDF/CSV + spatial CSVs.

    Returns paths and peak global rate metrics suitable for merging into workflow
    return dicts.

    When ``force_recompute`` is False and required exports already exist (and are
    not older than the checkpoint), reuses them without recomputing.
    """
    from .. import export, predict

    scenario_dir = Path(scenario_dir)
    cfg_dict = dict(cfg)
    bp_max_val = float(max_age_bp if bp_max is None else bp_max)
    suffix = nc_suffix or workflow.replace("-", "_")
    scenario_name = scenario_dir.name

    rsl_nc_path = (scenario_dir / f"rsl_{scenario_name}_{suffix}.nc").resolve()
    rate_nc_path = (scenario_dir / f"rate_{scenario_name}_{suffix}.nc").resolve()
    spatial_csv = scenario_dir / "spatial_predictions.csv"
    spatial_rates_csv = scenario_dir / "spatial_rates.csv"
    temporal_csv = scenario_dir / "temporal_decomposition.csv"
    temporal_rates_csv = scenario_dir / "temporal_decomposition_rates.csv"

    reuse_core = [
        rsl_nc_path,
        rate_nc_path,
        spatial_csv,
        spatial_rates_csv,
    ]
    if not force_recompute and artifacts_newer_than_checkpoint(
        reuse_core, scenario_dir
    ):
        emit_stage(
            stage_callback,
            "temporal",
            "Using existing temporal NetCDF and spatial CSV exports",
        )
        print(f"Reusing existing post-process exports in {scenario_dir}")
        out: Dict[str, Any] = {
            "peak_global_rate_mm_yr": float("nan"),
            "rsl_nc_path": str(rsl_nc_path),
            "rate_nc_path": str(rate_nc_path),
            "spatial_csv_path": str(spatial_csv.resolve()),
            "spatial_rates_csv_path": str(spatial_rates_csv.resolve()),
            "postprocess_complete": True,
        }
        if temporal_csv.is_file():
            out["temporal_csv_path"] = str(temporal_csv.resolve())
        if temporal_rates_csv.is_file():
            out["temporal_rates_csv_path"] = str(temporal_rates_csv.resolve())
        # Budget is cheap vs spatial — refresh from rates when reusing exports.
        if temporal_rates_csv.is_file():
            _run_sea_level_budget_stage(
                scenario_dir,
                temporal_rates_csv,
                cfg,
                stage_callback=stage_callback,
            )
        return out

    emit_stage(stage_callback, "temporal", "Temporal predictions per region")
    temporal_results = predict.temporal_all_regions(
        model,
        rsl,
        gia,
        cfg_dict,
        max_age_bp=max_age_bp,
        bp_min=bp_min,
        bp_max=bp_max_val,
    )
    arrays = build_temporal_netcdf_arrays(temporal_results)

    write_rsl_netcdf(
        rsl_nc_path,
        test_age=arrays["test_age"],
        region_names=arrays["region_names"],
        total_mean=arrays["total_mean"],
        total_std=arrays["total_std"],
        global_mean=arrays["global_mean"],
        global_std=arrays["global_std"],
        component_mean=arrays["component_mean"],
        component_std=arrays["component_std"],
    )
    write_rate_netcdf(
        rate_nc_path,
        test_age=arrays["test_age"],
        rate_time=arrays["rate_time"],
        region_names=arrays["region_names"],
        total_rate_mean=arrays["total_rate_mean"],
        total_rate_std=arrays["total_rate_std"],
        global_rate_mean=arrays["global_rate_mean"],
        global_rate_std=arrays["global_rate_std"],
        component_rate_mean=arrays["component_rate_mean"],
        component_rate_std=arrays["component_rate_std"],
    )

    comp_labels = list(getattr(model, "comp_labels", None) or [])
    if not comp_labels and temporal_results:
        first = temporal_results[sorted(temporal_results.keys())[0]]
        comp_labels = list((first.components or {}).keys())
    if temporal_results and comp_labels:
        df_temporal, df_rates = export.temporal_to_dataframes(
            temporal_results, comp_labels
        )
        df_temporal.to_csv(temporal_csv, index=False)
        df_rates.to_csv(temporal_rates_csv, index=False)
        print(f"Wrote {len(df_temporal)} rows to {temporal_csv}")
        print(f"Wrote {len(df_rates)} rows to {temporal_rates_csv}")

    # Notebook §5b: sea-level budget after temporal rates, before spatial.
    if temporal_rates_csv.is_file():
        _run_sea_level_budget_stage(
            scenario_dir,
            temporal_rates_csv,
            cfg,
            stage_callback=stage_callback,
        )

    peak_global_rate = peak_global_rate_mm_yr(
        arrays["rate_time"],
        arrays["global_rate_mean"],
        workflow=workflow,
    )

    emit_stage(stage_callback, "spatial", "Spatial predictions and rates")
    spatial_paths = export_spatial_artifacts(
        model,
        rsl,
        gia,
        cfg,
        scenario_dir,
        max_age_bp=max_age_bp,
        bp_min=bp_min,
        bp_max=bp_max_val,
        stage_callback=stage_callback,
        force_recompute=force_recompute,
    )

    return {
        "peak_global_rate_mm_yr": peak_global_rate,
        "rsl_nc_path": str(rsl_nc_path),
        "rate_nc_path": str(rate_nc_path),
        "postprocess_complete": True,
        **spatial_paths,
    }


def cfg_for_checkpoint_postprocess(
    cfg: Mapping[str, Any],
    param_store_path: str | Path,
) -> Dict[str, Any]:
    """Return a config that loads ``param_store_path`` and skips SVI."""
    cfg_dict = dict(cfg)
    gp_cfg = dict(cfg_dict.get("gp_model") or {})
    gp_cfg["run_optimization"] = False
    cfg_dict["gp_model"] = gp_cfg
    paths = dict(cfg_dict.get("paths") or {})
    paths["trained_param_store"] = str(Path(param_store_path).resolve())
    cfg_dict["paths"] = paths
    return cfg_dict
