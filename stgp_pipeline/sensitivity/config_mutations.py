"""
Config mutation helpers for STGP sensitivity scenarios.

These utilities keep scenario generation deterministic and side-effect free by
always mutating deep-copied config dictionaries.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple


DEFAULT_MERGED_NONLINEAR_PRIORS = {
    "temporal_variance_prior": [1.0e-6, 1000.0],
    "temporal_lengthscale_prior": [1.0, 5000.0],
    "spatial_lengthscale_prior": [0.001, 0.1],
}


def clone_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied config dict."""
    return copy.deepcopy(dict(cfg))


def _kernel_blocks(cfg: Mapping[str, Any]) -> MutableMapping[str, Any]:
    gp_model = cfg.get("gp_model", {}) or {}
    kernels = gp_model.get("kernels", {}) or {}
    if not isinstance(kernels, MutableMapping):
        raise TypeError("cfg['gp_model']['kernels'] must be a mapping")
    return kernels


def set_kernel_enabled(
    cfg: Mapping[str, Any],
    kernel_name: str,
    enabled: bool,
) -> Dict[str, Any]:
    """
    Return a copied config with `gp_model.kernels.<kernel_name>.enabled` set.
    """
    out = clone_config(cfg)
    kernels = _kernel_blocks(out)
    if kernel_name not in kernels or not isinstance(
        kernels[kernel_name], MutableMapping
    ):
        raise KeyError(f"Kernel block not found or not a mapping: {kernel_name!r}")
    kernels[kernel_name]["enabled"] = bool(enabled)
    return out


def set_kernel_enabled_map(
    cfg: Mapping[str, Any],
    enabled_by_kernel: Mapping[str, bool],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Return a copied config with multiple kernel enabled flags updated.
    """
    out = clone_config(cfg)
    records: List[Dict[str, Any]] = []
    for kernel_name, enabled in enabled_by_kernel.items():
        out = set_kernel_enabled(out, kernel_name, bool(enabled))
        records.append(
            {
                "type": "toggle",
                "kernel": str(kernel_name),
                "enabled": bool(enabled),
            }
        )
    return out, records


def configure_merged_nonlinear(
    cfg: Mapping[str, Any],
    *,
    enabled: bool,
    prior_ranges: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Enable/disable the merged regional+local nonlinear kernel block.

    When enabled, this helper:
    - creates/updates ``gp_model.kernels.merged_nonlinear``,
    - disables ``regional_nonlinear`` and ``local_nonlinear``.
    """
    out = clone_config(cfg)
    kernels = _kernel_blocks(out)
    records: List[Dict[str, Any]] = []

    merged_block = kernels.get("merged_nonlinear", {})
    if not isinstance(merged_block, MutableMapping):
        raise TypeError("Kernel block 'merged_nonlinear' must be a mapping")

    if enabled:
        existing = dict(merged_block)
        # Prefer priors already in the YAML block; fall back to coded defaults only
        # when the block carries no prior ranges at all.
        if any(str(k).endswith("_prior") for k in existing):
            merged_cfg = existing
        else:
            merged_cfg = dict(DEFAULT_MERGED_NONLINEAR_PRIORS)
        # Explicit prior_ranges kwarg wins over both YAML and defaults.
        if prior_ranges is not None:
            merged_cfg.update(dict(prior_ranges))
        merged_cfg["enabled"] = True
        merged_cfg.setdefault("temporal_type", "Matern32")
        merged_cfg.setdefault("spatial_type", "Matern21")
        kernels["merged_nonlinear"] = merged_cfg
        records.append(
            {
                "type": "toggle",
                "kernel": "merged_nonlinear",
                "enabled": True,
            }
        )

        out, toggle_records = set_kernel_enabled_map(
            out,
            {
                "regional_nonlinear": False,
                "local_nonlinear": False,
            },
        )
        records.extend(toggle_records)
    else:
        merged_block["enabled"] = False
        kernels["merged_nonlinear"] = merged_block
        records.append(
            {
                "type": "toggle",
                "kernel": "merged_nonlinear",
                "enabled": False,
            }
        )

    return out, records


def _is_prior_pair(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except (TypeError, ValueError):
        return False
    return True


def collect_kernel_priors(
    cfg: Mapping[str, Any],
    *,
    include_disabled: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """
    Collect all `[lo, hi]` prior ranges under `gp_model.kernels`.

    Returns a flat mapping where keys are:
        "<kernel_name>.<prior_key>"
    """
    priors: Dict[str, Tuple[float, float]] = {}
    for kernel_name, block in _kernel_blocks(cfg).items():
        if not isinstance(block, Mapping):
            continue
        if not include_disabled and block.get("enabled", True) is False:
            continue
        for key, value in block.items():
            if not key.endswith("_prior"):
                continue
            if not _is_prior_pair(value):
                continue
            lo, hi = float(value[0]), float(value[1])
            priors[f"{kernel_name}.{key}"] = (lo, hi)
    return priors


def _scale_bounds_linear(
    lo: float,
    hi: float,
    multiplier: float,
) -> Tuple[float, float]:
    """
    Scale bounds symmetrically in linear space: lo/m, hi*m.
    """
    if lo <= 0 or hi <= 0:
        raise ValueError(f"Bound scaling requires positive bounds, got ({lo}, {hi})")
    if hi <= lo:
        raise ValueError(f"Bounds must satisfy hi > lo, got ({lo}, {hi})")
    if multiplier <= 0:
        raise ValueError(f"Multiplier must be > 0, got {multiplier}")

    return lo / multiplier, hi * multiplier


def apply_bound_multiplier(
    cfg: Mapping[str, Any],
    *,
    multiplier: float,
    kernel_names: Iterable[str] | None = None,
    include_disabled: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Return a copied config with prior bounds scaled by `multiplier`.

    Each ``[lo, hi]`` prior range is mapped to ``[lo / multiplier, hi * multiplier]``.

    Parameters
    ----------
    multiplier
        Linear scaling factor (> 0). Values > 1 widen bounds; < 1 narrows them.
    kernel_names
        Optional subset of kernels to mutate. Default: all eligible kernels.

    Returns
    -------
    (mutated_cfg, mutation_records)
    """
    out = clone_config(cfg)
    kernels = _kernel_blocks(out)
    selected = set(kernel_names) if kernel_names is not None else None

    records: List[Dict[str, Any]] = []
    for kernel_name, block in kernels.items():
        if not isinstance(block, MutableMapping):
            continue
        if selected is not None and kernel_name not in selected:
            continue
        if not include_disabled and block.get("enabled", True) is False:
            continue

        for key, value in list(block.items()):
            if not key.endswith("_prior") or not _is_prior_pair(value):
                continue
            lo, hi = float(value[0]), float(value[1])
            new_lo, new_hi = _scale_bounds_linear(lo, hi, multiplier)
            block[key] = [new_lo, new_hi]
            records.append(
                {
                    "kernel": kernel_name,
                    "prior_key": key,
                    "old_bounds": [lo, hi],
                    "new_bounds": [new_lo, new_hi],
                    "multiplier": float(multiplier),
                }
            )
    return out, records


def multiplier_tag(multiplier: float) -> str:
    """Return a filesystem-safe tag for a multiplier."""
    return f"{multiplier:.3f}".rstrip("0").rstrip(".").replace(".", "p")
