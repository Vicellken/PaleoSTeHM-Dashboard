"""
RSL data loading, filtering, and tensor preparation.

Wraps ``PSTHM.load.load_regional_rsl_data()`` with validity masking
(age <= max, finite, non-negative sigma) and combines ML+TL into
limiting arrays with direction flags.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import PSTHM.load
from PSTHM.data_adjustments import read_adjusted_rsl_dataframe

from .types import RSLData


def merge_regions(
    rsl: RSLData,
    source: int,
    target: int,
    *,
    quiet: bool = False,
) -> RSLData:
    """Relabel *source* → *target* in-memory (SLIP + limiting). Does not modify the CSV."""
    if source == target:
        return rsl
    if target not in rsl.region_num_to_name:
        raise ValueError(f"target region {target} not in loaded data")
    target_name = rsl.region_num_to_name[target]

    rsl_region = rsl.rsl_region.copy()
    slip_mask = rsl_region == source
    rsl_region[slip_mask] = target

    rsl_region_name = rsl.rsl_region_name.copy()
    rsl_region_name[slip_mask] = target_name

    lim_region = rsl.lim_region.copy()
    lim_mask = lim_region == source
    lim_region[lim_mask] = target

    lim_region_name = rsl.lim_region_name.copy()
    lim_region_name[lim_mask] = target_name

    region_num_to_name = dict(rsl.region_num_to_name)
    region_num_to_name.pop(source, None)

    if not quiet:
        n_slip = int(slip_mask.sum())
        n_lim = int(lim_mask.sum())
        print(
            f"Merged region {source} → {target} ({target_name}): "
            f"{n_slip} SLIP, {n_lim} limiting"
        )
    return replace(
        rsl,
        rsl_region=rsl_region,
        rsl_region_name=rsl_region_name,
        lim_region=lim_region,
        lim_region_name=lim_region_name,
        region_num_to_name=region_num_to_name,
    )


def _parse_region_merge_pairs(
    raw: Any,
) -> List[Tuple[int, int]]:
    """Parse ``data_filter.region_merges`` as list of [source, target] pairs."""
    if not raw:
        return []
    pairs: List[Tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"region_merges entries must be [source, target] pairs; got {item!r}"
            )
        pairs.append((int(item[0]), int(item[1])))
    return pairs


def _apply_region_merges(
    rsl: RSLData,
    pairs: Sequence[Tuple[int, int]],
    *,
    quiet: bool = False,
) -> RSLData:
    for source, target in pairs:
        rsl = merge_regions(rsl, source=source, target=target, quiet=quiet)
    return rsl


def load_and_filter_rsl(
    data_csv: str,
    max_age_bp: int = 14000,
    *,
    config: Optional[Mapping[str, Any]] = None,
    purpose: str = "train",
) -> RSLData:
    """
    Load RSL data from CSV, apply validity filters, and return an RSLData container.

    Parameters
    ----------
    data_csv : str
        Path to the paleo RSL CSV file.
    max_age_bp : int
        Maximum age in cal yr BP to retain.
    config : mapping, optional
        Full run config (e.g. ``NotebookContext.cfg``). When provided, CSV rows are
        passed through ``data_adjustments`` (e.g. level-2 downweighting) before
        parsing.
    purpose : str
        ``'train'`` or ``'predict'``. Selects ``data_filter.train_levels`` vs
        ``data_filter.prediction_levels`` when ``data_filter.levels`` is absent.

    Returns
    -------
    RSLData
        Filtered SLIP training data and combined limiting data.
    """
    if config is not None:
        df, _l2_stats = read_adjusted_rsl_dataframe(
            str(data_csv), config, purpose=purpose
        )
        ml, SLIP, tl = PSTHM.load.load_regional_rsl_dataframe(df, CE=False)
    else:
        ml, SLIP, tl = PSTHM.load.load_regional_rsl_data(
            str(data_csv), CE=False)

    age, y, y_sigma, age_sigma, rsl_lon, rsl_lat, rsl_region, rsl_region_name = SLIP
    (
        ml_age,
        ml_y,
        ml_y_sigma,
        ml_age_sigma,
        ml_lon,
        ml_lat,
        ml_region,
        ml_region_name,
    ) = ml
    (
        tl_age,
        tl_y,
        tl_y_sigma,
        tl_age_sigma,
        tl_lon,
        tl_lat,
        tl_region,
        tl_region_name,
    ) = tl

    # --- Region IDs for limiting data (used for per-region limiting overlay) ---
    # ml_region and tl_region are numpy arrays of int region IDs (Region.1 column).
    lim_region_combined = np.concatenate(
        [np.asarray(ml_region, dtype=int), np.asarray(tl_region, dtype=int)]
    )
    lim_region_name_combined = np.concatenate(
        [
            np.asarray(ml_region_name, dtype=object),
            np.asarray(tl_region_name, dtype=object),
        ]
    )

    print(
        f"Raw counts — ML: {len(ml_age)}, SLIP: {len(age)}, TL: {len(tl_age)}")

    # Filter SLIP points
    slip_mask = age <= max_age_bp
    slip_mask = slip_mask & torch.isfinite(age)
    slip_mask = slip_mask & torch.isfinite(y)
    slip_mask = slip_mask & torch.isfinite(y_sigma)
    slip_mask = slip_mask & torch.isfinite(age_sigma)
    slip_mask = slip_mask & (y_sigma >= 0)
    slip_mask = slip_mask & (age_sigma >= 0)
    coord_mask = np.isfinite(rsl_lon) & np.isfinite(rsl_lat)
    slip_mask = slip_mask & torch.tensor(coord_mask)
    slip_mask_np = slip_mask.detach().cpu().numpy().astype(bool)

    age = age[slip_mask]
    y = y[slip_mask]
    y_sigma = y_sigma[slip_mask]
    age_sigma = age_sigma[slip_mask]
    rsl_lon = np.asarray(rsl_lon)[slip_mask_np]
    rsl_lat = np.asarray(rsl_lat)[slip_mask_np]
    rsl_region = np.asarray(rsl_region)[slip_mask_np]
    rsl_region_name = np.asarray(rsl_region_name)[slip_mask_np]

    if len(age) == 0:
        raise ValueError(
            f"No SLIP points after filtering (max_age_bp={max_age_bp}). "
            f"Check CSV Limiting==0 rows and age units (CE vs BP)."
        )

    rsl_coord = torch.tensor(np.vstack((rsl_lat, rsl_lon)).T)
    X = torch.hstack([age[:, None], rsl_coord])

    print(f"Filtered SLIP training points: {len(age)}")
    print(f"Unique regions: {np.unique(rsl_region)}")
    print(f"Age range: {age.min().item():.0f} – {age.max().item():.0f} BP")

    # Combine and filter limiting data
    lim_age = torch.cat([ml_age, tl_age])
    lim_y = torch.cat([ml_y, tl_y])
    lim_y_sigma = torch.cat([ml_y_sigma, tl_y_sigma])
    lim_age_sigma = torch.cat([ml_age_sigma, tl_age_sigma])
    lim_lat = np.concatenate([ml_lat, tl_lat])
    lim_lon = np.concatenate([ml_lon, tl_lon])
    lim_flags = np.concatenate(
        [
            np.full(len(ml_age), -1.0),
            np.full(len(tl_age), 1.0),
        ]
    )

    lim_mask = lim_age <= max_age_bp
    lim_mask = lim_mask & torch.isfinite(lim_age)
    lim_mask = lim_mask & torch.isfinite(lim_y)
    lim_mask = lim_mask & torch.isfinite(lim_y_sigma)
    lim_mask = lim_mask & torch.isfinite(lim_age_sigma)
    lim_mask = lim_mask & (lim_y_sigma >= 0)
    lim_mask = lim_mask & (lim_age_sigma >= 0)
    lim_coord_mask = np.isfinite(lim_lat) & np.isfinite(lim_lon)
    lim_mask = lim_mask & torch.tensor(lim_coord_mask)

    lim_mask_np = lim_mask.numpy().astype(bool)
    lim_age_np = lim_age[lim_mask].numpy()
    lim_y_np = lim_y[lim_mask].numpy()
    lim_y_sigma_np = lim_y_sigma[lim_mask].numpy()
    lim_age_sigma_np = lim_age_sigma[lim_mask].numpy()
    lim_lat_np = lim_lat[lim_mask_np]
    lim_lon_np = lim_lon[lim_mask_np]
    lim_flags_np = lim_flags[lim_mask_np]
    lim_region_np = lim_region_combined[lim_mask_np]
    lim_region_name_np = lim_region_name_combined[lim_mask_np]

    n_ml = int((lim_flags_np == -1).sum())
    n_tl = int((lim_flags_np == 1).sum())
    print(f"Filtered limiting data: {len(lim_age_np)} (ML={n_ml}, TL={n_tl})")

    # Build region_num_to_name mapping
    region_num_to_name: Dict[int, str] = {}
    for rn, rname in zip(rsl_region, rsl_region_name):
        region_num_to_name[int(rn)] = str(rname)

    rsl = RSLData(
        X=X,
        y=y,
        y_sigma=y_sigma,
        age_sigma=age_sigma,
        rsl_lat=np.asarray(rsl_lat),
        rsl_lon=np.asarray(rsl_lon),
        rsl_region=np.asarray(rsl_region),
        rsl_region_name=np.asarray(rsl_region_name),
        lim_age=lim_age_np,
        lim_y=lim_y_np,
        lim_y_sigma=lim_y_sigma_np,
        lim_age_sigma=lim_age_sigma_np,
        lim_lat=np.asarray(lim_lat_np, dtype=float),
        lim_lon=np.asarray(lim_lon_np, dtype=float),
        lim_flags=np.asarray(lim_flags_np, dtype=float),
        lim_region=lim_region_np,
        lim_region_name=lim_region_name_np,
        region_num_to_name=region_num_to_name,
    )

    if config is not None:
        data_filter = config.get("data_filter", {}) or {}
        merge_pairs = _parse_region_merge_pairs(
            data_filter.get("region_merges"))
        if merge_pairs:
            rsl = _apply_region_merges(rsl, merge_pairs)
            print(f"Unique regions after merge: {rsl.unique_regions}")

    return rsl
