"""
Lightweight data-level adjustments for CSV inputs.

This module intentionally avoids torch/pyro imports so it can be reused in
preprocessing and tests without heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import pandas as pd

_RSL_SIGMA_COLS: Tuple[str, ...] = ("RSLer_up_2sd", "RSLer_low_2sd")
_AGE_SIGMA_COLS: Tuple[str, ...] = ("Age_up_er_2sd", "Age_low_er_2sd")


def _as_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"config must be a mapping/dict, got {type(value)}")
    return dict(value)


def _validate_targets(scale_targets: Sequence[str]) -> set[str]:
    normalized = {str(t).strip().lower() for t in scale_targets}
    if not normalized:
        raise ValueError("scale_targets must be non-empty")
    if not normalized <= {"rsl", "age"}:
        raise ValueError(
            "scale_targets must be a subset of {'rsl', 'age'}, "
            f"got {sorted(normalized)}"
        )
    return normalized


def _resolve_level_filter_levels(
    filter_cfg: Mapping[str, Any],
    *,
    purpose: str = "train",
) -> Any:
    """
    Resolve which level allow-list to apply.

    Precedence:
        1. ``data_filter.levels`` (explicit, any purpose)
        2. ``data_filter.train_levels`` when ``purpose='train'``
        3. ``data_filter.prediction_levels`` when ``purpose='predict'``
        4. ``data_filter.train_levels`` as predict fallback when prediction_levels absent
    """
    if filter_cfg.get("levels") is not None:
        return filter_cfg.get("levels")
    if purpose == "train":
        return filter_cfg.get("train_levels")
    if purpose == "predict":
        return filter_cfg.get("prediction_levels") or filter_cfg.get("train_levels")
    raise ValueError(f"purpose must be 'train' or 'predict', got {purpose!r}")


def apply_level_filter(
    df: pd.DataFrame,
    cfg: Mapping[str, Any] | None,
    *,
    purpose: str = "train",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Keep only rows whose ``Level`` column is in the configured allow-list.

    Expected config paths (first match wins):
        data_filter.levels
        data_filter.train_levels       (purpose='train', or predict fallback)
        data_filter.prediction_levels  (purpose='predict')
    """
    cfg_map = _as_mapping(cfg)
    filter_cfg = _as_mapping(cfg_map.get("data_filter"))
    raw_levels = _resolve_level_filter_levels(filter_cfg, purpose=purpose)
    if raw_levels is None:
        return df.copy(), {
            "enabled": False,
            "levels": [],
            "rows_before": len(df),
            "rows_after": len(df),
        }

    if isinstance(raw_levels, (int, float, str)):
        raw_levels = [raw_levels]
    levels = [int(x) for x in raw_levels]
    if not levels:
        return df.copy(), {
            "enabled": False,
            "levels": [],
            "rows_before": len(df),
            "rows_after": len(df),
        }

    out = df.copy()
    if "Level" not in out.columns:
        raise KeyError(
            "Missing required column for data_filter.levels: Level. "
            f"Columns found: {', '.join(out.columns)}"
        )

    rows_before = len(out)
    keep = out["Level"].isin(levels)
    out = out.loc[keep].reset_index(drop=True)
    rows_after = len(out)
    if rows_after == 0:
        raise ValueError(
            f"No rows remain after data_filter.levels={levels}; "
            f"started with {rows_before} rows."
        )

    stats: Dict[str, Any] = {
        "enabled": True,
        "levels": levels,
        "rows_before": rows_before,
        "rows_after": rows_after,
    }
    return out, stats


def apply_level2_downweighting(
    df: pd.DataFrame, cfg: Mapping[str, Any] | None
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Inflate uncertainty columns for Level-2 rows according to config.

    Expected config path:
        data_adjustments.level2_downweighting
    """
    cfg_map = _as_mapping(cfg)
    da_cfg = _as_mapping(cfg_map.get("data_adjustments"))
    l2_cfg = _as_mapping(da_cfg.get("level2_downweighting"))

    enabled = bool(l2_cfg.get("enabled", False))
    base_stats: Dict[str, Any] = {
        "enabled": enabled,
        "selected_rows": 0,
        "scaled_columns": [],
    }
    out = df.copy()
    if not enabled:
        return out, base_stats

    multiplier = float(l2_cfg.get("multiplier", 1.0))
    if multiplier < 1.0:
        raise ValueError(
            f"level2_downweighting.multiplier must be >= 1.0, got {multiplier}"
        )

    scope = str(l2_cfg.get("scope", "slip_only")).strip().lower()
    if scope not in {"slip_only", "all"}:
        raise ValueError(
            "level2_downweighting.scope must be 'slip_only' or 'all', " f"got {scope!r}"
        )

    raw_targets = l2_cfg.get("scale_targets", ["rsl", "age"])
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    targets = _validate_targets(raw_targets)

    required = ["Level"]
    if scope == "slip_only":
        required.append("Limiting")
    missing_required = [c for c in required if c not in out.columns]
    if missing_required:
        raise KeyError(
            "Missing required columns for level2_downweighting: "
            + ", ".join(missing_required)
        )

    if scope == "slip_only":
        mask = (out["Level"] == 2) & (out["Limiting"] == 0)
    else:
        mask = out["Level"] == 2

    columns_to_scale: list[str] = []
    if "rsl" in targets:
        columns_to_scale.extend(_RSL_SIGMA_COLS)
    if "age" in targets:
        columns_to_scale.extend(_AGE_SIGMA_COLS)

    missing_scale_cols = [c for c in columns_to_scale if c not in out.columns]
    if missing_scale_cols:
        raise KeyError(
            "Missing uncertainty columns for level2_downweighting: "
            + ", ".join(missing_scale_cols)
        )

    if mask.any():
        out.loc[mask, columns_to_scale] = out.loc[mask, columns_to_scale] * multiplier

    stats = {
        "enabled": True,
        "multiplier": multiplier,
        "scope": scope,
        "scale_targets": sorted(targets),
        "selected_rows": int(mask.sum()),
        "scaled_columns": list(columns_to_scale),
    }
    return out, stats


def read_adjusted_rsl_dataframe(
    csv_path: str,
    config: Union[str, Path, Mapping[str, Any], None] = None,
    *,
    purpose: str = "train",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read a regional RSL CSV and apply ``data_filter`` / ``data_adjustments``.

    Pipeline order:
        ``read_csv`` → ``apply_level_filter`` → ``apply_level2_downweighting``.

    Parameters
    ----------
    csv_path :
        Path to the CSV file.
    config :
        None, path to a YAML config, or an in-memory config dict (root keys
        include ``data_filter`` and ``data_adjustments`` when enabled).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Adjusted dataframe and combined stats (level filter + L2 downweighting).
    """
    from PSTHM.config.load import coerce_config_to_dict

    df = pd.read_csv(csv_path)
    cfg_dict = coerce_config_to_dict(config)
    df, filter_stats = apply_level_filter(df, cfg_dict, purpose=purpose)
    df, l2_stats = apply_level2_downweighting(df, cfg_dict)
    return df, {**l2_stats, "level_filter": filter_stats}
