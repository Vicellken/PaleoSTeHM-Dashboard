"""
GIA ensemble loading for:
  1) combined 3D (text/CSV) + 1D (MAT or ANU-ICE txt) model ensembles, and
  2) five-model regional + global .npy ensembles.

Splits the GIA computation:
  - **GIA mean** (for residuals and predictions): configurable via
    ``gia.mean_weighting`` (``equal``, ``wrss_3d``, or ``wrss_all``).
  - **GIA covariance** (for noise and uncertainty): WRSS-weighted Bessel-
    corrected covariance from *all* models (3D + 1D), unchanged by mean_weighting.

Usage:
    funcs, names, idx_3d = load_ensemble_from_cfg(cfg, mat_dir, txt_dir)
    gia = compute_weights_and_covariance_3d(funcs, names, idx_3d, rsl, cfg)
    res = gia_setup.compute_residuals_and_noise(rsl, gia)  # reuse original
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from scipy.interpolate import RegularGridInterpolator

from gia_covariance_tools import (
    compute_wrss_weights_continuous,
    extract_gia_covariance,
)

from .types import RSLData, GIAResult
from . import gia_setup

_REGIONAL_3D_NAMES = ("ice6G", "iceANU")
_NPY_FIVE_MODEL_ORDER = ("ice6G", "iceANU", "ice7G", "iceGOWAN", "iceANU1d")

# ANU-ICE 1D regional grids (Data/ANU_1D): 34 non-uniform time columns (ka BP).
ANU_1D_TIME_KA_BP = np.array(
    [
        -21,
        -20,
        -19,
        -18,
        -17,
        -16,
        -15,
        -14,
        -13,
        -12,
        -11.5,
        -11,
        -10.5,
        -10,
        -9.5,
        -9,
        -8.5,
        -8,
        -7.5,
        -7,
        -6.5,
        -6,
        -5.5,
        -5,
        -4.5,
        -4,
        -3.5,
        -3,
        -2.5,
        -2,
        -1.5,
        -1,
        -0.5,
        0,
    ],
    dtype=np.float64,
)
# Columns are oldest-first; negative ka labels → calendar years BP.
ANU_1D_TIME_BP = -ANU_1D_TIME_KA_BP * 1000.0

_ANU_1D_MODEL_STEM_RE = re.compile(r"^RSL_ANU-ICE_EM(.+?)_21ka$", re.IGNORECASE)


# ── text/CSV grid parsing ────────────────────────────────────────────────────
# Self-contained parsing helpers for 3D GIA text and CSV grids.


def _infer_time_step(time_start: float, n_time: int, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    if n_time < 2:
        raise ValueError("Need at least 2 time columns")
    step = time_start / (n_time - 1)
    if not np.isclose(step, round(step), rtol=0, atol=1e-6):
        raise ValueError(
            f"Cannot infer integer time step from time_start={time_start} and "
            f"n_time={n_time}; pass time_step explicitly."
        )
    return float(round(step))


def _time_bp_descending(time_start: float, n_time: int, step: float) -> np.ndarray:
    expected = time_start - step * (n_time - 1)
    if not np.isclose(expected, 0.0, atol=1e-3):
        raise ValueError(
            f"time_start - (n_time-1)*step should be ~0, got {expected}. "
            "Check time_start / time_step / file column count."
        )
    return time_start - step * np.arange(n_time, dtype=np.float64)


def _parse_bp_from_header_token(token: str) -> float:
    """Parse BP time from CSV header token (e.g., ``BP_16000``)."""
    s = token.strip()
    # Common pattern in converted files: BP_16000 / BP-16000 / BP 16000
    m = re.match(r"^BP[\s_-]*([0-9]+(?:\.[0-9]+)?)$", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Fallback: allow bare numeric token
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"Cannot parse BP time from header token: {token!r}") from exc


def _load_csv_grid_with_optional_header(
    input_path: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load CSV numeric matrix and optional time axis from header."""
    first_line = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    header_tokens = [tok.strip() for tok in first_line.split(",")]

    has_header = False
    if len(header_tokens) >= 3:
        try:
            # If first row is fully numeric, this is data (no header)
            [float(tok) for tok in header_tokens]
        except ValueError:
            has_header = True

    if has_header:
        data = np.loadtxt(input_path, dtype=np.float64, delimiter=",", skiprows=1)
        time_tokens = header_tokens[2:]
        time_bp = np.array(
            [_parse_bp_from_header_token(t) for t in time_tokens], dtype=np.float64
        )
    else:
        data = np.loadtxt(input_path, dtype=np.float64, delimiter=",")
        time_bp = None

    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data, time_bp


def _convert_txt_to_volume(
    input_path: Path,
    time_start: float = 16000.0,
    time_step: float | None = None,
    time_bp: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load text grid → (rsl, lon_axis, lat_axis, time_bp).

    rsl shape: (n_lon, n_lat, n_time); time_bp descending (yr BP).
    """
    if input_path.suffix.lower() == ".csv":
        data, time_bp_from_header = _load_csv_grid_with_optional_header(input_path)
    else:
        data = np.loadtxt(input_path, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        time_bp_from_header = None
    ncols = data.shape[1]
    if ncols < 3:
        raise ValueError(f"Expected at least 3 columns, got {ncols}")

    lon = data[:, 0]
    lat = data[:, 1]
    rsl_rows = data[:, 2:]
    n_time = rsl_rows.shape[1]

    if time_bp is not None:
        if len(time_bp) != n_time:
            raise ValueError(
                f"Explicit time_bp has {len(time_bp)} values, "
                f"but data has {n_time} time columns."
            )
    elif time_bp_from_header is not None:
        if len(time_bp_from_header) != n_time:
            raise ValueError(
                f"CSV header has {len(time_bp_from_header)} time columns, "
                f"but data has {n_time} columns."
            )
        time_bp = time_bp_from_header
    else:
        step = _infer_time_step(time_start, n_time, time_step)
        time_bp = _time_bp_descending(time_start, n_time, step)

    lon_axis = np.unique(lon)
    lat_axis = np.unique(lat)
    lon_axis.sort()
    lat_axis.sort()

    n_lon, n_lat = len(lon_axis), len(lat_axis)
    if lon.shape[0] != n_lon * n_lat:
        raise ValueError(
            f"Row count {lon.shape[0]} != n_lon*n_lat "
            f"({n_lon}*{n_lat}={n_lon * n_lat}); grid may be irregular."
        )

    lon_idx = {float(v): i for i, v in enumerate(lon_axis)}
    lat_idx = {float(v): i for i, v in enumerate(lat_axis)}
    rsl = np.full((n_lon, n_lat, n_time), np.nan, dtype=np.float64)

    for row in range(lon.shape[0]):
        lo, la = float(lon[row]), float(lat[row])
        i, j = lon_idx[lo], lat_idx[la]
        rsl[i, j, :] = rsl_rows[row]

    if np.any(np.isnan(rsl)):
        raise ValueError("Missing (lon, lat) cells when filling regular grid")

    return rsl, lon_axis, lat_axis, time_bp


# ── Loading text/CSV GIA models as interpolation callables ───────────────────


def load_txt_gia_model(
    txt_path: Path,
    time_start: float = 16000.0,
    time_step: float | None = None,
    time_bp: Optional[np.ndarray] = None,
) -> Tuple[Callable, str]:
    """Load a 3D GIA text/CSV file and return an interpolator callable.

    Returns
    -------
    predict_fn : callable
        ``predict_fn(X) -> ndarray`` where *X* is (N, 3) as
        ``[time_BP, lat, lon]`` (lon in [0, 360)).
    model_name : str
        Stem of the source file.
    """
    rsl, lon_axis, lat_axis, time_bp = _convert_txt_to_volume(
        txt_path,
        time_start=time_start,
        time_step=time_step,
        time_bp=time_bp,
    )

    # rsl shape: (n_lon, n_lat, n_time) → (n_time, n_lat, n_lon)
    rsl_cube = rsl.transpose(2, 1, 0)  # (n_time, n_lat, n_lon)

    # Ensure ascending time axis
    if time_bp[0] > time_bp[-1]:
        time_bp_asc = time_bp[::-1].copy()
        rsl_cube = rsl_cube[::-1, :, :]
    else:
        time_bp_asc = time_bp.copy()

    # Ensure ascending lat / lon
    if lat_axis[0] > lat_axis[-1]:
        lat_axis = lat_axis[::-1].copy()
        rsl_cube = rsl_cube[:, ::-1, :]
    if lon_axis[0] > lon_axis[-1]:
        lon_axis = lon_axis[::-1].copy()
        rsl_cube = rsl_cube[:, :, ::-1]

    interp = RegularGridInterpolator(
        (time_bp_asc, lat_axis, lon_axis),
        rsl_cube,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    def _normalize_lon_for_grid(lon_values: np.ndarray) -> np.ndarray:
        """Match query longitudes to the loaded grid longitude convention."""
        lon_values = np.asarray(lon_values, dtype=float)
        # 3D txt grids may be in [-180, 180) or [0, 360).
        if np.nanmin(lon_axis) < 0.0:
            return ((lon_values + 180.0) % 360.0) - 180.0
        return lon_values % 360.0

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        pts = X.copy()
        pts[:, 2] = _normalize_lon_for_grid(pts[:, 2])
        return interp(pts).flatten()

    return predict_fn, txt_path.stem


def load_3d_gia_ensemble(
    gia_3d_dir: str,
    time_start: float = 16000.0,
    time_step: float | None = None,
) -> Tuple[List[Callable], List[str]]:
    """Load all 3D GIA models from a directory.

    Supports both ``*.csv`` and ``*.txt`` files. Files are loaded with a
    deterministic priority: all CSV files first (sorted), then TXT files
    (sorted).

    Returns
    -------
    interp_funcs : list of callable
    model_names : list of str
    """
    d = Path(gia_3d_dir)
    csv_files = sorted(d.glob("*.csv"))
    txt_files = sorted(d.glob("*.txt"))
    gia_files = csv_files + txt_files
    if not gia_files:
        raise FileNotFoundError(f"No 3D GIA files (.csv or .txt) found in {d}")

    interp_funcs: List[Callable] = []
    model_names: List[str] = []
    for f in gia_files:
        fn, name = load_txt_gia_model(f, time_start=time_start, time_step=time_step)
        interp_funcs.append(fn)
        model_names.append(name)

    return interp_funcs, model_names


def _anu_1d_model_stem(path: Path) -> str:
    """Map ``RSL_ANU-ICE_EM48p210_21ka`` → ``48p210`` (MAT-compatible stem)."""
    match = _ANU_1D_MODEL_STEM_RE.match(path.stem)
    if match:
        return match.group(1)
    return path.stem


def _resolve_cfg_path(
    value: str,
    base_dir: Path | None = None,
) -> Path:
    """Resolve a config path against an optional base directory."""
    p = Path(value).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = (base_dir / p).expanduser()
    return p.resolve()


GIA_1D_SOURCE_CHOICES = ("mat", "anu", "both")


def resolve_1d_source(
    gia_cfg: Dict[str, Any],
    paths: Dict[str, Any],
) -> str:
    """Return validated ``gia.1d_source`` (``mat`` | ``anu`` | ``both``).

    When ``1d_source`` is omitted, infer from ``paths``:
    only ``gia_ANU_dir`` → ``anu``; only ``gia_mat_dir`` → ``mat``;
    both present → ``anu`` (backward compatible default).
    """
    if "1d_source" in gia_cfg and gia_cfg["1d_source"] is not None:
        value = str(gia_cfg["1d_source"]).strip().lower()
        if value not in GIA_1D_SOURCE_CHOICES:
            raise ValueError(
                f"gia.1d_source must be one of {GIA_1D_SOURCE_CHOICES}, got {value!r}"
            )
        return value

    has_anu = bool(paths.get("gia_ANU_dir"))
    has_mat = bool(paths.get("gia_mat_dir"))
    if has_anu and not has_mat:
        return "anu"
    if has_mat and not has_anu:
        return "mat"
    if has_anu:
        return "anu"
    raise KeyError(
        "Cannot infer gia.1d_source: set gia.1d_source or paths.gia_mat_dir / "
        "paths.gia_ANU_dir"
    )


def resolve_gia_1d_dirs(
    cfg: Dict[str, Any],
    base_dir: Path | None = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """Resolve absolute 1D GIA directories and ``gia.1d_source`` from *cfg*."""
    paths = cfg.get("paths", {}) or {}
    gia_cfg = cfg.get("gia", {}) or {}
    one_d_source = resolve_1d_source(gia_cfg, paths)

    mat_raw = paths.get("gia_mat_dir")
    anu_raw = paths.get("gia_ANU_dir")
    mat_dir = (
        str(_resolve_cfg_path(str(mat_raw), base_dir=base_dir)) if mat_raw else None
    )
    anu_dir = (
        str(_resolve_cfg_path(str(anu_raw), base_dir=base_dir)) if anu_raw else None
    )

    if one_d_source == "mat" and not mat_dir:
        raise KeyError("gia.1d_source='mat' requires paths.gia_mat_dir")
    if one_d_source == "anu" and not anu_dir:
        raise KeyError("gia.1d_source='anu' requires paths.gia_ANU_dir")
    if one_d_source == "both" and (not mat_dir or not anu_dir):
        raise KeyError(
            "gia.1d_source='both' requires paths.gia_mat_dir and paths.gia_ANU_dir"
        )
    return mat_dir, anu_dir, one_d_source


def load_mat_1d_gia_ensemble(
    gia_mat_dir: str,
    exclude_models: Optional[List[str]] = None,
    name_suffix: str = "",
) -> Tuple[List[Callable], List[str]]:
    """Load ICE6G 1D MAT models, optionally suffixing names (e.g. ``_mat``)."""
    funcs, names = gia_setup.load_gia_ensemble(
        str(gia_mat_dir), exclude_models=exclude_models
    )
    if name_suffix:
        names = [f"{name}{name_suffix}" for name in names]
    return funcs, names


def load_anu_1d_gia_ensemble(
    gia_anu_dir: str,
    exclude_models: Optional[List[str]] = None,
    name_suffix: str = "",
) -> Tuple[List[Callable], List[str]]:
    """Load ANU-ICE 1D regional GIA models from whitespace txt/csv grids.

    Expects files such as ``RSL_ANU-ICE_EM48p210_21ka.txt`` with 36 columns
    (lon, lat, RSL at 34 non-uniform ka-BP ages). Model names are normalized
    to MAT-compatible stems (e.g. ``48p210``) for ``exclude_models`` parity.
    """
    if exclude_models is None:
        exclude_models = []
    exclude_set = set(exclude_models)

    d = Path(gia_anu_dir)
    csv_files = sorted(d.glob("*.csv"))
    txt_files = sorted(d.glob("*.txt"))
    gia_files = csv_files + txt_files
    if not gia_files:
        raise FileNotFoundError(f"No ANU 1D GIA files (.csv or .txt) found in {d}")

    interp_funcs: List[Callable] = []
    model_names: List[str] = []
    for f in gia_files:
        name = _anu_1d_model_stem(f)
        if name in exclude_set:
            continue
        fn, _ = load_txt_gia_model(f, time_bp=ANU_1D_TIME_BP.copy())
        interp_funcs.append(fn)
        model_names.append(f"{name}{name_suffix}")

    print(f"Loaded {len(interp_funcs)} ANU-ICE 1D GIA models:")
    for name in model_names:
        print(f"  {name}")

    return interp_funcs, model_names


def load_combined_ensemble(
    gia_3d_dir: str,
    *,
    gia_mat_dir: Optional[str] = None,
    gia_anu_dir: Optional[str] = None,
    exclude_models: Optional[List[str]] = None,
    time_start: float = 16000.0,
    one_d_source: Optional[str] = None,
) -> Tuple[List[Callable], List[str], np.ndarray]:
    """Load 3D (text/CSV) + 1D (MAT and/or ANU-ICE txt) GIA models.

    3D models are placed first in the returned lists so that
    ``indices_3d`` is always ``[0, 1, ..., n_3d-1]``.

    *one_d_source* selects the 1D backend: ``mat``, ``anu``, or ``both``.
    When ``both``, MAT and ANU models are loaded with ``_mat`` / ``_anu`` name
    suffixes. If *one_d_source* is omitted, infer from which path(s) are set
    (``both`` when both paths provided, else the single source).

    Returns
    -------
    interp_funcs : list of callable
        All model interpolators (3D first, then 1D).
    model_names : list of str
    indices_3d : ndarray
        Integer indices into the combined list identifying 3D models.
    """
    if one_d_source is None:
        if gia_mat_dir and gia_anu_dir:
            one_d_source = "both"
        elif gia_anu_dir:
            one_d_source = "anu"
        elif gia_mat_dir:
            one_d_source = "mat"
        else:
            raise ValueError(
                "load_combined_ensemble requires gia_mat_dir and/or gia_anu_dir"
            )
    else:
        one_d_source = str(one_d_source).strip().lower()
        if one_d_source not in GIA_1D_SOURCE_CHOICES:
            raise ValueError(
                f"one_d_source must be one of {GIA_1D_SOURCE_CHOICES}, "
                f"got {one_d_source!r}"
            )

    if one_d_source == "mat" and not gia_mat_dir:
        raise ValueError("one_d_source='mat' requires gia_mat_dir")
    if one_d_source == "anu" and not gia_anu_dir:
        raise ValueError("one_d_source='anu' requires gia_anu_dir")
    if one_d_source == "both" and (not gia_mat_dir or not gia_anu_dir):
        raise ValueError(
            "one_d_source='both' requires both gia_mat_dir and gia_anu_dir"
        )

    funcs_3d, names_3d = load_3d_gia_ensemble(gia_3d_dir, time_start=time_start)
    funcs_1d: List[Callable] = []
    names_1d: List[str] = []
    labels: List[str] = []

    if one_d_source in ("mat", "both"):
        suffix = "_mat" if one_d_source == "both" else ""
        funcs_mat, names_mat = load_mat_1d_gia_ensemble(
            str(gia_mat_dir),
            exclude_models=exclude_models,
            name_suffix=suffix,
        )
        funcs_1d.extend(funcs_mat)
        names_1d.extend(names_mat)
        labels.append("MAT")

    if one_d_source in ("anu", "both"):
        suffix = "_anu" if one_d_source == "both" else ""
        funcs_anu, names_anu = load_anu_1d_gia_ensemble(
            str(gia_anu_dir),
            exclude_models=exclude_models,
            name_suffix=suffix,
        )
        funcs_1d.extend(funcs_anu)
        names_1d.extend(names_anu)
        labels.append("ANU-ICE txt")

    label_1d = " + ".join(labels)

    all_funcs = funcs_3d + funcs_1d
    all_names = names_3d + names_1d
    indices_3d = np.arange(len(funcs_3d))

    print(
        f"\nCombined ensemble: {len(funcs_3d)} 3D + {len(funcs_1d)} 1D ({label_1d}) "
        f"= {len(all_funcs)} models"
    )
    print(f"  3D models (indices {list(indices_3d)}): {names_3d}")

    return all_funcs, all_names, indices_3d


def _resolve_ensemble_paths_from_cfg(
    cfg: Dict[str, Any],
    gia_mat_dir: Optional[str],
    gia_3d_dir: Optional[str],
    gia_anu_dir: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Fill missing ensemble directory paths from ``cfg['paths']``."""
    paths = cfg.get("paths", {}) or {}
    if gia_3d_dir is None:
        gia_3d_dir = paths.get("gia_3d_dir")
    if gia_anu_dir is None:
        gia_anu_dir = paths.get("gia_ANU_dir")
    if gia_mat_dir is None:
        gia_mat_dir = paths.get("gia_mat_dir")
    return gia_mat_dir, gia_3d_dir, gia_anu_dir


def resolved_1d_dir_from_paths(
    paths: Dict[str, Any],
    gia_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(gia_mat_dir, gia_anu_dir)`` paths for the resolved 1D source."""
    gia_cfg = gia_cfg or {}
    one_d_source = resolve_1d_source(gia_cfg, paths)
    mat = paths.get("gia_mat_dir")
    anu = paths.get("gia_ANU_dir")
    if one_d_source == "mat":
        return (str(mat), None) if mat else (None, None)
    if one_d_source == "anu":
        return (None, str(anu)) if anu else (None, None)
    return (str(mat) if mat else None, str(anu) if anu else None)


def load_ensemble_from_cfg(
    cfg: Dict[str, Any],
    gia_mat_dir: Optional[str] = None,
    gia_3d_dir: Optional[str] = None,
    exclude_models: Optional[List[str]] = None,
    time_start: float = 16000.0,
    gia_anu_dir: Optional[str] = None,
    base_dir: Path | None = None,
) -> Tuple[List[Callable], List[str], np.ndarray]:
    """Load a GIA ensemble according to ``cfg['gia']['ensemble']``.

    Supported values:
    - ``"all"`` (default): 3D + 1D combined ensemble via :func:`load_combined_ensemble`.
    - ``"3d"``: all 3D models in ``gia_3d_dir`` only.
    - list of model stems: subset of 3D models (e.g. ANU-only).

    1D models are selected by ``gia.1d_source`` (``mat`` | ``anu`` | ``both``).
    Paths may be omitted and are read from ``cfg['paths']``.
    """
    override_mat_dir = gia_mat_dir
    override_anu_dir = gia_anu_dir
    gia_mat_dir, gia_3d_dir, gia_anu_dir = _resolve_ensemble_paths_from_cfg(
        cfg, gia_mat_dir, gia_3d_dir, gia_anu_dir
    )
    if gia_3d_dir is not None:
        gia_3d_dir = str(_resolve_cfg_path(str(gia_3d_dir), base_dir=base_dir))

    gia_cfg = cfg.get("gia", {}) or {}
    ensemble = gia_cfg.get("ensemble", "all")

    if ensemble == "all":
        if gia_3d_dir is None:
            raise KeyError("gia.ensemble='all' requires paths.gia_3d_dir")
        resolved_mat, resolved_anu, one_d_source = resolve_gia_1d_dirs(
            cfg, base_dir=base_dir
        )
        if override_mat_dir is not None:
            resolved_mat = str(
                _resolve_cfg_path(str(override_mat_dir), base_dir=base_dir)
            )
        if override_anu_dir is not None:
            resolved_anu = str(
                _resolve_cfg_path(str(override_anu_dir), base_dir=base_dir)
            )
        return load_combined_ensemble(
            str(gia_3d_dir),
            gia_mat_dir=resolved_mat,
            gia_anu_dir=resolved_anu,
            exclude_models=exclude_models,
            time_start=time_start,
            one_d_source=one_d_source,
        )

    if ensemble == "3d":
        if gia_3d_dir is None:
            raise KeyError("gia.ensemble='3d' requires paths.gia_3d_dir")
        funcs, names = load_3d_gia_ensemble(str(gia_3d_dir), time_start=time_start)
        indices_3d = np.arange(len(funcs))
        print(f"\n3D-only ensemble: {len(funcs)} models")
        print(f"  models (indices {list(indices_3d)}): {names}")
        return funcs, names, indices_3d

    if isinstance(ensemble, (list, tuple)):
        if gia_3d_dir is None:
            raise KeyError("gia.ensemble list requires paths.gia_3d_dir")
        funcs_all, names_all = load_3d_gia_ensemble(
            str(gia_3d_dir), time_start=time_start
        )
        name_to_idx = {name: idx for idx, name in enumerate(names_all)}
        missing = [name for name in ensemble if name not in name_to_idx]
        if missing:
            raise KeyError(
                f"gia.ensemble lists unknown 3D models: {missing}; "
                f"available={names_all}"
            )
        indices = [name_to_idx[name] for name in ensemble]
        funcs = [funcs_all[i] for i in indices]
        names = [names_all[i] for i in indices]
        indices_3d = np.arange(len(funcs))
        print(f"\nSubset 3D ensemble: {len(funcs)} models")
        print(f"  models (indices {list(indices_3d)}): {names}")
        return funcs, names, indices_3d

    raise ValueError(
        f"gia.ensemble must be 'all', '3d', or a list of model stems; got {ensemble!r}"
    )


def _validate_covariance_matrix(
    cov: np.ndarray, n_obs: int, source: Path
) -> np.ndarray:
    """Validate basic covariance matrix invariants and return float64 matrix."""
    if not isinstance(cov, np.ndarray):
        raise TypeError(f"Expected ndarray from {source}, got {type(cov)}")
    if cov.ndim != 2:
        raise ValueError(
            f"Expected a 2-D covariance matrix in {source}, got ndim={cov.ndim}"
        )
    if cov.shape != (n_obs, n_obs):
        raise ValueError(
            f"Covariance shape mismatch from {source}: got {cov.shape}, expected ({n_obs}, {n_obs})"
        )
    cov = np.asarray(cov, dtype=np.float64)
    if not np.isfinite(cov).all():
        raise ValueError(f"Covariance from {source} contains non-finite values")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError(f"Covariance from {source} is not symmetric")
    return cov


def _build_npy_predict_fn(
    interp: RegularGridInterpolator,
    model_name: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap a raw interpolator to the shared call contract:
    ``f(X)`` with ``X[:, :] = [age_bp, lat, lon]``.
    """

    def _predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(
                f"{model_name} interpolator expects X as (N, 3) [age_bp, lat, lon], got {X.shape}"
            )

        if model_name in _REGIONAL_3D_NAMES:
            # Regional cubes follow (lon, lat, age_bp) axis ordering.
            pts = np.column_stack([X[:, 2], X[:, 1], X[:, 0]])
        else:
            # Global cubes are (age_kyr, lat, lon_0_360).
            pts = np.column_stack([X[:, 0] / 1000.0, X[:, 1], X[:, 2] % 360.0])
        return interp(pts).reshape(-1)

    return _predict


def load_npy_5model_ensemble(
    cfg: Dict[str, Any],
    base_dir: Path | None = None,
) -> Tuple[List[Callable], List[str], np.ndarray]:
    """
    Load a five-model regional + global ``.npy`` GIA ensemble.

    Optional regional grid bounds can be supplied via
    ``cfg['gia']['npy_regional_grid']`` (``lon_min``, ``lon_max``, ``lat_min``,
    ``lat_max``, ``step``).

    Returns
    -------
    interp_funcs : list of callables
    model_names : list of str
    indices_3d : ndarray
        Indices for the regional pair used as the GIA mean baseline.
    """
    paths_cfg = cfg.get("paths", {}) or {}
    gia_cfg = cfg.get("gia", {}) or {}
    reg_grid = gia_cfg.get("npy_regional_grid") or {}

    required_path_keys = {
        "gia_ice6g_npy": "ice6G",
        "gia_anu_npy": "iceANU",
        "gia_ice7g_npy": "ice7G",
        "gia_gowan_npy": "iceGOWAN",
        "gia_anu1d_npy": "iceANU1d",
    }
    missing_keys = [k for k in required_path_keys if not paths_cfg.get(k)]
    if missing_keys:
        raise KeyError(
            f"Missing five-model .npy path keys in cfg['paths']: {sorted(missing_keys)}"
        )

    resolved_files = {
        model_name: _resolve_cfg_path(str(paths_cfg[cfg_key]), base_dir=base_dir)
        for cfg_key, model_name in required_path_keys.items()
    }
    for model_name, file_path in resolved_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing {model_name} file: {file_path}")

    # Regional 0.5-deg grids (ice6G + iceANU); configurable via gia.npy_regional_grid
    step = float(reg_grid.get("step", 0.5))
    lon_min = float(reg_grid.get("lon_min", 115.0))
    lon_max = float(reg_grid.get("lon_max", 130.0))
    lat_min = float(reg_grid.get("lat_min", 30.0))
    lat_max = float(reg_grid.get("lat_max", 43.0))
    lon_reg = np.arange(lon_min, lon_max + step * 0.5, step)
    lat_reg = np.arange(lat_min, lat_max + step * 0.5, step)
    times_6g_bp = np.arange(12000, -500, -500)
    times_anu_bp = np.arange(12000, -1000, -1000)

    # Global 1-deg grids (ice7G + gowan + ANU1d)
    lat_global = np.arange(-89.5, 90, 1)
    lon_global = np.arange(0.5, 360, 1)
    times_7g_kyr = np.arange(0, 13)
    times_anu1d_kyr = np.arange(0, 18)

    model_arrays = {
        name: np.load(path, allow_pickle=True) for name, path in resolved_files.items()
    }

    raw_interps = {
        "ice6G": RegularGridInterpolator(
            (lon_reg, lat_reg, times_6g_bp),
            model_arrays["ice6G"],
            method="linear",
            bounds_error=False,
            fill_value=None,
        ),
        "iceANU": RegularGridInterpolator(
            (lon_reg, lat_reg, times_anu_bp),
            model_arrays["iceANU"],
            method="linear",
            bounds_error=False,
            fill_value=None,
        ),
        "ice7G": RegularGridInterpolator(
            (times_7g_kyr, lat_global, lon_global),
            model_arrays["ice7G"],
            method="linear",
            bounds_error=False,
            fill_value=None,
        ),
        "iceGOWAN": RegularGridInterpolator(
            (times_7g_kyr, lat_global, lon_global),
            model_arrays["iceGOWAN"],
            method="linear",
            bounds_error=False,
            fill_value=None,
        ),
        "iceANU1d": RegularGridInterpolator(
            (times_anu1d_kyr, lat_global, lon_global),
            model_arrays["iceANU1d"],
            method="linear",
            bounds_error=False,
            fill_value=None,
        ),
    }

    model_names = list(_NPY_FIVE_MODEL_ORDER)
    interp_funcs = [
        _build_npy_predict_fn(raw_interps[name], model_name=name)
        for name in model_names
    ]
    indices_3d = np.array([0, 1], dtype=int)

    print(f"\nFive-model .npy GIA ensemble loaded: {len(model_names)} models")
    print(
        f"  regional-mean models (indices {list(indices_3d)}): {list(_REGIONAL_3D_NAMES)}"
    )
    return interp_funcs, model_names, indices_3d


# ── Weighting and covariance (split mean / covariance) ───────────────────────

MEAN_WEIGHTING_CHOICES = ("equal", "wrss_3d", "wrss_all")


def _parse_mean_weighting(gia_cfg: Dict[str, Any]) -> str:
    value = str(gia_cfg.get("mean_weighting") or "equal").strip().lower()
    if value not in MEAN_WEIGHTING_CHOICES:
        raise ValueError(
            f"gia.mean_weighting must be one of {MEAN_WEIGHTING_CHOICES}, got {value!r}"
        )
    return value


def _resolve_mean_model_indices(
    model_names: List[str],
    indices_3d: np.ndarray,
    mean_model_names: Optional[List[str]],
) -> np.ndarray:
    if mean_model_names:
        name_to_idx = {name: idx for idx, name in enumerate(model_names)}
        missing = [name for name in mean_model_names if name not in name_to_idx]
        if missing:
            raise KeyError(
                f"mean_model_names contains unknown models: {missing}; "
                f"available={model_names}"
            )
        return np.array([name_to_idx[name] for name in mean_model_names], dtype=int)
    return np.asarray(indices_3d, dtype=int)


def _compute_gia_mean_from_preds(
    preds: np.ndarray,
    weights: np.ndarray,
    mean_indices: np.ndarray,
    mean_weighting: str,
) -> np.ndarray:
    """Combine per-model predictions into GIA mean at observation/test points."""
    if mean_weighting == "wrss_all":
        return np.average(preds, axis=0, weights=weights)
    pool = preds[mean_indices]
    if mean_weighting == "equal":
        return np.mean(pool, axis=0)
    if mean_weighting == "wrss_3d":
        w_pool = weights[mean_indices]
        w_pool = w_pool / w_pool.sum()
        return np.average(pool, axis=0, weights=w_pool)
    raise ValueError(
        f"mean_weighting must be one of {MEAN_WEIGHTING_CHOICES}, got {mean_weighting!r}"
    )


def _weighted_bessel_covariance_from_predictions(
    preds: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Apply the pipeline's established all-model WRSS covariance convention."""
    preds = np.asarray(preds, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if preds.ndim != 2:
        raise ValueError(f"preds must be 2-D (models, points), got {preds.shape}")
    if preds.shape[0] != weights.size:
        raise ValueError(
            f"preds has {preds.shape[0]} models but weights has {weights.size}"
        )
    n_models, n_points = preds.shape
    if n_models < 2:
        return np.zeros((n_points, n_points), dtype=np.float64)

    weighted_mean_all = np.average(preds, axis=0, weights=weights)
    centered = preds - weighted_mean_all
    covariance = np.sum(
        weights[:, None, None] * (centered[:, :, None] * centered[:, None, :]),
        axis=0,
    )
    denominator = weights.sum() - weights.sum() / n_models
    if denominator <= 0:
        raise ValueError("GIA covariance correction denominator must be positive")
    covariance /= denominator
    return 0.5 * (covariance + covariance.T)


def _effective_mean_indices(
    mean_model_indices: Optional[np.ndarray],
    indices_3d: np.ndarray,
) -> np.ndarray:
    if mean_model_indices is not None:
        return np.asarray(mean_model_indices, dtype=int)
    return np.asarray(indices_3d, dtype=int)


def compute_weights_and_covariance_3d(
    interp_funcs: List[Any],
    model_names: List[str],
    indices_3d: np.ndarray,
    rsl_data: RSLData,
    cfg: Dict[str, Any],
    bp_min: float = 0.0,
    bp_max: float = 16000.0,
    base_dir: Path | None = None,
) -> GIAResult:
    """Compute WRSS weights (all 14) and split GIA mean (3D) / covariance (all).

    Parameters
    ----------
    interp_funcs : list
        Combined 3D + 1D interpolators.
    model_names : list of str
    indices_3d : ndarray
        Indices of 3D models in ``interp_funcs``.
    rsl_data : RSLData
    cfg : dict
        YAML config (reads ``gia.n_search``).
    bp_min, bp_max : float
        GIA time bounds for WRSS search window.

    Returns
    -------
    GIAResult
        With ``gia_mean_at_obs`` per ``gia.mean_weighting`` and
        ``gia_cov_at_obs`` from all models WRSS-weighted.
    """
    n_search = int(cfg.get("gia", {}).get("n_search", 50))

    # WRSS weights from all models
    weights, misfits = compute_wrss_weights_continuous(
        interp_funcs,
        rsl_data.lim_age,
        rsl_data.lim_lat,
        rsl_data.lim_lon,
        rsl_data.lim_y,
        rsl_data.lim_y_sigma,
        rsl_data.lim_age_sigma,
        rsl_data.lim_flags,
        bp_min=bp_min,
        bp_max=bp_max,
        n_search=n_search,
    )

    print(
        f"\nModel weights (WRSS/erf, {len(rsl_data.lim_age)} limiting points, "
        f"n_search={n_search}):"
    )
    for m, name in enumerate(model_names):
        tag = " [3D]" if m in indices_3d else ""
        print(f"  {name}: weight={weights[m]:.4f}  misfit={misfits[m]:.2f}{tag}")

    # Observation coordinates
    ages_bp = rsl_data.X[:, 0].numpy()
    lats = rsl_data.X[:, 1].numpy()
    lons = rsl_data.X[:, 2].numpy()
    X_gia = np.column_stack([ages_bp, lats, lons % 360])

    gia_cfg = cfg.get("gia", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}

    use_precomputed_cov = bool(gia_cfg.get("use_precomputed_covariance", False))
    precomputed_cov_path = gia_cfg.get("precomputed_covariance_path")
    if precomputed_cov_path is None:
        precomputed_cov_path = paths_cfg.get("gia_5model_covariance_npy")

    if use_precomputed_cov and precomputed_cov_path:
        cov_path = _resolve_cfg_path(str(precomputed_cov_path), base_dir=base_dir)
        if not cov_path.exists():
            raise FileNotFoundError(
                f"Configured precomputed covariance file does not exist: {cov_path}"
            )
        gia_cov_loaded = np.load(cov_path, allow_pickle=True)
        gia_cov = _validate_covariance_matrix(
            gia_cov_loaded, n_obs=len(ages_bp), source=cov_path
        )
        print(f"\nUsing precomputed GIA covariance from: {cov_path}")
    elif len(interp_funcs) < 2:
        n_obs = len(ages_bp)
        gia_cov = np.zeros((n_obs, n_obs), dtype=np.float64)
        print(
            f"\nGIA covariance dropped: ensemble has {len(interp_funcs)} model(s); "
            "need >= 2 for Bessel-corrected covariance."
        )
    else:
        # GIA covariance from ALL models (WRSS-weighted, Bessel-corrected).
        # Weighted mean from all models is used internally for centering only;
        # the *returned* mean for residuals comes from selected mean models below.
        gia_cov = extract_gia_covariance(
            interp_funcs,
            X_gia,
            weights=weights,
            regularize=True,
            return_mean=False,
        )

    mean_weighting = _parse_mean_weighting(gia_cfg)
    mean_indices = _resolve_mean_model_indices(
        model_names, indices_3d, gia_cfg.get("mean_model_names")
    )

    preds_all = np.array([interp_funcs[i](X_gia) for i in range(len(interp_funcs))])
    gia_mean_at_obs = _compute_gia_mean_from_preds(
        preds_all, weights, mean_indices, mean_weighting
    )

    eigvals = np.linalg.eigvalsh(gia_cov)
    print(f"\nGIA covariance (all {len(interp_funcs)} models): shape={gia_cov.shape}")
    print(
        f"  Diagonal (std) range: {np.sqrt(np.diag(gia_cov)).min():.4f} "
        f"– {np.sqrt(np.diag(gia_cov)).max():.4f} m"
    )
    print(f"  Eigenvalue range: [{eigvals.min():.2e}, {eigvals.max():.2e}]")
    print(f"  Positive definite: {eigvals.min() > 0}")

    if mean_weighting == "wrss_all":
        mean_log_models = model_names
        mean_log_weights = weights
    else:
        mean_log_models = [model_names[int(i)] for i in mean_indices]
        if mean_weighting == "wrss_3d":
            w_pool = weights[mean_indices]
            mean_log_weights = w_pool / w_pool.sum()
        else:
            mean_log_weights = np.full(len(mean_indices), 1.0 / len(mean_indices))
    weight_parts = ", ".join(
        f"{name}={w:.3f}" for name, w in zip(mean_log_models, mean_log_weights)
    )
    print(
        f"\nGIA mean at obs: {mean_weighting} over {len(mean_log_models)} model(s); "
        f"effective weights: {weight_parts}"
    )

    return GIAResult(
        interp_funcs=interp_funcs,
        model_names=model_names,
        weights=weights,
        misfits=misfits,
        gia_mean_at_obs=gia_mean_at_obs,
        gia_cov_at_obs=gia_cov,
        indices_3d=indices_3d,
        mean_weighting=mean_weighting,
        mean_model_indices=mean_indices,
    )


# ── Test-point GIA evaluation (split mean / variance) ───────────────────────


def gia_spatial_mean_covariance_at_ages_3d(
    interp_funcs: List[Any],
    weights: np.ndarray,
    indices_3d: np.ndarray,
    test_ages: np.ndarray,
    spatial_lat: np.ndarray,
    spatial_lon: np.ndarray,
    *,
    spatial_weights: Optional[np.ndarray] = None,
    mean_weighting: str = "equal",
    mean_model_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """GIA mean/covariance after averaging each ensemble member spatially.

    This is algebraically equivalent to averaging the full test-point
    covariance, without allocating that potentially large matrix.
    """
    ages = np.asarray(test_ages, dtype=np.float64).reshape(-1)
    lat = np.asarray(spatial_lat, dtype=np.float64).reshape(-1)
    lon = np.asarray(spatial_lon, dtype=np.float64).reshape(-1) % 360.0
    if lat.size == 0 or lat.size != lon.size:
        raise ValueError("spatial_lat and spatial_lon must have equal non-zero size")

    if spatial_weights is None:
        point_weights = np.full(lat.size, 1.0 / lat.size, dtype=np.float64)
    else:
        point_weights = np.asarray(spatial_weights, dtype=np.float64).reshape(-1)
        if point_weights.size != lat.size:
            raise ValueError(
                "spatial_weights must have one value per spatial coordinate"
            )
        if not np.isfinite(point_weights).all() or np.any(point_weights < 0):
            raise ValueError("spatial_weights must be finite and non-negative")
        weight_sum = float(point_weights.sum())
        if weight_sum <= 0:
            raise ValueError("spatial_weights must sum to a positive value")
        point_weights = point_weights / weight_sum

    model_age_means = np.empty(
        (len(interp_funcs), ages.size), dtype=np.float64
    )
    for model_index, interp in enumerate(interp_funcs):
        for age_index, age_bp in enumerate(ages):
            X_test = np.column_stack([np.full(lat.size, age_bp), lat, lon])
            values = np.asarray(interp(X_test), dtype=np.float64).reshape(-1)
            if values.size != lat.size:
                raise ValueError(
                    "GIA interpolator returned an unexpected prediction size"
                )
            model_age_means[model_index, age_index] = float(
                point_weights @ values
            )

    mean_indices = _effective_mean_indices(mean_model_indices, indices_3d)
    model_weights = np.asarray(weights, dtype=np.float64)
    gia_mean = _compute_gia_mean_from_preds(
        model_age_means,
        model_weights,
        mean_indices,
        mean_weighting,
    )
    gia_cov = _weighted_bessel_covariance_from_predictions(
        model_age_means,
        model_weights,
    )
    return gia_mean, gia_cov


def gia_at_test_points_3d(
    interp_funcs: List[Any],
    weights: np.ndarray,
    indices_3d: np.ndarray,
    test_ages: np.ndarray,
    test_lat: np.ndarray,
    test_lon: np.ndarray,
    *,
    mean_weighting: str = "equal",
    mean_model_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """GIA mean (per mean_weighting) and Bessel-corrected variance (all, WRSS-weighted).

    No regularization is applied here (unlike obs-point covariance) because
    test-point variance feeds uncertainty propagation, not matrix inversion.

    Returns
    -------
    gia_mean : ndarray — per ``mean_weighting`` (default: equal 3D average)
    gia_var : ndarray — WRSS-weighted Bessel-corrected variance (all models)
    """
    M = len(interp_funcs)
    test_lon_360 = np.asarray(test_lon) % 360
    X_test = np.column_stack([test_ages, test_lat, test_lon_360])

    preds = np.array([f(X_test) for f in interp_funcs])  # (M, n_pts)

    mean_indices = _effective_mean_indices(mean_model_indices, indices_3d)
    gia_mean = _compute_gia_mean_from_preds(
        preds, weights, mean_indices, mean_weighting
    )

    if M < 2:
        gia_var = np.zeros(len(test_ages), dtype=np.float64)
        return gia_mean, gia_var

    # Variance: WRSS-weighted Bessel-corrected from all models
    weighted_mean_all = np.average(preds, axis=0, weights=weights)
    centered = preds - weighted_mean_all

    gia_var = np.zeros(len(test_ages))
    for m in range(M):
        gia_var += weights[m] * centered[m] ** 2
    gia_var /= weights.sum() - weights.sum() / M

    return gia_mean, gia_var


def gia_covariance_at_test_points_3d(
    interp_funcs: List[Any],
    weights: np.ndarray,
    indices_3d: np.ndarray,
    test_ages: np.ndarray,
    test_lat: np.ndarray,
    test_lon: np.ndarray,
    *,
    mean_weighting: str = "equal",
    mean_model_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """GIA mean (per mean_weighting) and full covariance (all, WRSS-weighted).

    No regularization is applied here (unlike obs-point covariance) because
    test-point covariance feeds uncertainty propagation, not matrix inversion.

    Returns
    -------
    gia_mean : ndarray (n,) — per ``mean_weighting``
    gia_cov : ndarray (n, n) — WRSS-weighted Bessel-corrected covariance
    """
    M = len(interp_funcs)
    test_lon_360 = np.asarray(test_lon) % 360
    X_test = np.column_stack([test_ages, test_lat, test_lon_360])

    preds = np.array([f(X_test) for f in interp_funcs])  # (M, n_pts)

    mean_indices = _effective_mean_indices(mean_model_indices, indices_3d)
    gia_mean = _compute_gia_mean_from_preds(
        preds, weights, mean_indices, mean_weighting
    )

    n_pts = len(test_ages)
    if M < 2:
        gia_cov = np.zeros((n_pts, n_pts), dtype=np.float64)
        return gia_mean, gia_cov

    # Covariance: WRSS-weighted Bessel-corrected from all models
    weighted_mean_all = np.average(preds, axis=0, weights=weights)
    centered = preds - weighted_mean_all

    gia_cov = np.sum(
        weights[:, None, None] * (centered[:, :, None] * centered[:, None, :]),
        axis=0,
    )
    gia_cov /= weights.sum() - weights.sum() / M

    return gia_mean, gia_cov
