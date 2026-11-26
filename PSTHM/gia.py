"""
Utilities for loading and querying gridded GIA model outputs.

Supports:
- NumPy cubes (time, lat, lon), e.g. ICE7G `.npy`
- NumPy cubes (lon, lat, time_bp), e.g. ICE6G / ANU 3-D arrays
- CSV tables with `Lat`, `Long`, and `{N}ka` columns
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as rgi

import PSTHM.time


def _to_numpy_2d(x) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Input must be shape (N, 3) as [time, lat, lon].")
    return arr


def _parse_ka_value(col_name: str, time_suffix: str) -> float:
    col = str(col_name).strip()
    if not col.endswith(time_suffix):
        raise ValueError(f"Column {col_name!r} does not end with {time_suffix!r}.")
    return float(col[: -len(time_suffix)])


@dataclass
class GIAGrid:
    """
    Gridded GIA model with interpolation and BP/CE conversion helpers.

    Notes:
    - Time axis is represented as CE years in this class.
    - `present_ce` controls the ka->CE map: CE = present_ce - ka*1000.
    - `reference_year` controls BP<->CE conversion of user inputs (usually 1950).
    """

    interp: rgi
    time_ce: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    reference_year: int = 1950
    present_ce: int = 2017
    lon_wrap: Optional[float] = 360.0

    @property
    def ce_offset(self) -> float:
        return float(self.present_ce - self.reference_year)

    @classmethod
    def from_npy(
        cls,
        path: str | Path,
        *,
        lat: Sequence[float],
        lon: Sequence[float],
        times_ka: Sequence[float],
        reference_year: int = 1950,
        present_ce: int = 2017,
        lon_wrap: Optional[float] = 360.0,
        **interp_kwargs,
    ) -> "GIAGrid":
        path = Path(path)
        grid = np.load(str(path))
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        times_ka = np.asarray(times_ka, dtype=float)

        expected_shape = (len(times_ka), len(lat), len(lon))
        if grid.shape != expected_shape:
            raise ValueError(
                f"GIA grid shape {grid.shape} does not match expected {expected_shape}."
            )

        # Match notebook behavior: CE axis runs present -> past, interpolator needs ascending.
        time_ce_desc = float(present_ce) - times_ka * 1000.0
        interp_obj = rgi(
            (time_ce_desc[::-1], lat, lon),
            grid[::-1],
            **interp_kwargs,
        )
        return cls(
            interp=interp_obj,
            time_ce=np.asarray(time_ce_desc, dtype=float),
            lat=lat,
            lon=lon,
            reference_year=int(reference_year),
            present_ce=int(present_ce),
            lon_wrap=lon_wrap,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        lat_col: str = "Lat",
        lon_col: str = "Long",
        time_suffix: str = "ka",
        reference_year: int = 1950,
        present_ce: int = 2017,
        lon_wrap: Optional[float] = 360.0,
        **interp_kwargs,
    ) -> "GIAGrid":
        path = Path(path)
        df = pd.read_csv(path)

        time_pairs: list[tuple[float, str]] = []
        for col in df.columns:
            col_str = str(col)
            if col_str.endswith(time_suffix):
                try:
                    ka = _parse_ka_value(col_str, time_suffix=time_suffix)
                except ValueError:
                    continue
                time_pairs.append((ka, col_str))

        if not time_pairs:
            raise ValueError(
                f"No time columns ending with {time_suffix!r} found in {path}."
            )

        # Legacy behavior: process from largest ka (past) to smallest ka (present).
        time_pairs = sorted(time_pairs, key=lambda kv: kv[0], reverse=True)
        lat = np.sort(df[lat_col].unique().astype(float))
        lon = np.sort(df[lon_col].unique().astype(float))
        grid = np.zeros((len(time_pairs), len(lat), len(lon)), dtype=float)

        for i, (_, col_name) in enumerate(time_pairs):
            pivot = df.pivot_table(index=lat_col, columns=lon_col, values=col_name)
            pivot = pivot.sort_index(axis=0, ascending=True).sort_index(
                axis=1, ascending=True
            )
            pivot = pivot.reindex(index=lat, columns=lon)
            if pivot.isna().values.any():
                raise ValueError(
                    f"Missing values encountered after gridding CSV column {col_name!r}."
                )
            grid[i, :, :] = pivot.values

        # Reverse to present->past ka ordering to match notebook convention.
        times_ka = np.asarray([t for t, _ in time_pairs], dtype=float)[::-1]
        grid = grid[::-1, :, :]
        time_ce_desc = float(present_ce) - times_ka * 1000.0

        interp_obj = rgi(
            (time_ce_desc[::-1], lat, lon),
            grid[::-1, :, :],
            **interp_kwargs,
        )
        return cls(
            interp=interp_obj,
            time_ce=np.asarray(time_ce_desc, dtype=float),
            lat=lat,
            lon=lon,
            reference_year=int(reference_year),
            present_ce=int(present_ce),
            lon_wrap=lon_wrap,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        **kwargs,
    ) -> "GIAGrid":
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return cls.from_npy(path, **kwargs)
        if suffix == ".csv":
            return cls.from_csv(path, **kwargs)
        raise ValueError(f"Unsupported GIA file extension: {path.suffix!r}")

    def _wrap_lon(self, lon_values: np.ndarray) -> np.ndarray:
        if self.lon_wrap is None:
            return lon_values
        return np.mod(lon_values, float(self.lon_wrap))

    def x_from_bp(self, x_bp_lat_lon) -> np.ndarray:
        x_np = _to_numpy_2d(x_bp_lat_lon)
        age_ce = PSTHM.time.bp_to_ce(x_np[:, 0], reference_year=self.reference_year)
        age_ce = age_ce + self.ce_offset
        lon = self._wrap_lon(x_np[:, 2])
        return np.column_stack([age_ce, x_np[:, 1], lon])

    def x_from_ce(self, x_ce_lat_lon) -> np.ndarray:
        x_np = _to_numpy_2d(x_ce_lat_lon)
        lon = self._wrap_lon(x_np[:, 2])
        return np.column_stack([x_np[:, 0] + self.ce_offset, x_np[:, 1], lon])

    def predict_bp(self, x_bp_lat_lon) -> np.ndarray:
        return self.interp(self.x_from_bp(x_bp_lat_lon))

    def predict_ce(self, x_ce_lat_lon) -> np.ndarray:
        return self.interp(self.x_from_ce(x_ce_lat_lon))

    def coverage_bp_min_max(self) -> tuple[float, float]:
        bp_min = float(PSTHM.time.ce_to_bp(np.max(self.time_ce), self.reference_year))
        bp_max = float(PSTHM.time.ce_to_bp(np.min(self.time_ce), self.reference_year))
        return bp_min, bp_max

    @classmethod
    def from_npy_cube(
        cls,
        path: Union[str, Path],
        *,
        lon_range: Tuple[float, float, float],
        lat_range: Tuple[float, float, float],
        time_bp_range: Tuple[float, float, float],
        reference_year: int = 1950,
        present_ce: int = 1950,
        lon_wrap: Optional[float] = None,
        **interp_kwargs,
    ) -> "GIAGrid":
        """
        Load a 3-D GIA cube stored with shape ``(n_lon, n_lat, n_time)``.

        Interpolator axes are ``(lon, lat, time_bp)`` and the data
        is queried with coordinates ``(lon, lat, time_bp)``.

        Parameters
        ----------
        path : str or Path
            Path to the ``.npy`` file.
        lon_range : (start, stop, step)
            Passed to ``np.arange`` to produce the longitude axis.
        lat_range : (start, stop, step)
            Passed to ``np.arange`` to produce the latitude axis.
        time_bp_range : (start, stop, step)
            Passed to ``np.arange`` to produce the time axis in years BP.
            Example: ``(12000, -500, -500)`` gives [12000, 11500, …, 0].
        reference_year, present_ce, lon_wrap
            Forwarded to the ``GIAGrid`` dataclass.
        **interp_kwargs
            Extra keyword arguments for ``RegularGridInterpolator``.
        """
        path = Path(path)
        cube = np.load(str(path), allow_pickle=True)
        if cube.ndim == 0:
            cube = cube.item()
            if not isinstance(cube, np.ndarray):
                raise ValueError(f"Cannot interpret {path} as a 3-D array.")

        lon = np.arange(*lon_range)
        lat = np.arange(*lat_range)
        time_bp = np.arange(*time_bp_range)

        expected = (len(lon), len(lat), len(time_bp))
        if cube.shape != expected:
            raise ValueError(
                f"Cube shape {cube.shape} does not match expected "
                f"(n_lon, n_lat, n_time) = {expected} from the given axis ranges."
            )

        grid = cube.transpose(2, 1, 0)  # (time_bp, lat, lon)

        time_internal = float(present_ce) - time_bp.astype(float)
        sort_idx = np.argsort(time_internal)
        time_sorted = time_internal[sort_idx]
        grid_sorted = grid[sort_idx, :, :]

        interp_obj = rgi(
            (time_sorted, lat, lon),
            grid_sorted,
            **interp_kwargs,
        )
        return cls(
            interp=interp_obj,
            time_ce=time_internal,
            lat=lat,
            lon=lon,
            reference_year=int(reference_year),
            present_ce=int(present_ce),
            lon_wrap=lon_wrap,
        )


class MeanGIAGrids:
    """Proxy that returns the arithmetic mean of several GIAGrid predictions."""

    def __init__(self, grids: Sequence[GIAGrid]):
        self.grids: List[GIAGrid] = list(grids)

    def predict_bp(self, x_bp_lat_lon) -> np.ndarray:
        preds = np.stack([g.predict_bp(x_bp_lat_lon) for g in self.grids])
        return preds.mean(axis=0)

    def predict_ce(self, x_ce_lat_lon) -> np.ndarray:
        preds = np.stack([g.predict_ce(x_ce_lat_lon) for g in self.grids])
        return preds.mean(axis=0)

    def coverage_bp_min_max(self) -> tuple[float, float]:
        bp_mins, bp_maxs = zip(*(g.coverage_bp_min_max() for g in self.grids))
        return max(bp_mins), min(bp_maxs)


def mean_gia_grids(*grids: GIAGrid) -> MeanGIAGrids:
    """Return a proxy whose ``predict_bp`` / ``predict_ce`` average *grids*."""
    return MeanGIAGrids(grids)
