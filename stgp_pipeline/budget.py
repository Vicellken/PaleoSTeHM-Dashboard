"""
Sea-level budget pie/donut charts from exported rate CSVs.

Reads the ``temporal_decomposition_rates.csv`` produced by
:func:`stgp_pipeline.export.temporal_to_dataframes` instead of NetCDF, so no
retraining is needed and future kernel changes are picked up automatically
(components are discovered from CSV columns ending in ``_rate``), and
supports an arbitrary list of age windows rendered side-by-side per region.

Budget quantity per (region, window, component):
    abs_contribution_m = |mean(rate_within_window)| * (window_end - window_start)

When GMSL and GRD are split from GIA (``gmsl_xlsx``), their budget wedges sum to
the combined GIA contribution ``|mean(gmsl_rate + grd_rate)| * window_length``,
allocated by relative ``|mean(gmsl_rate)|`` and ``|mean(grd_rate)|``. This avoids
inflating the denominator when the two means have opposite signs.

Without ``gmsl_xlsx``, ``gia_rate`` and ``global_rate`` are merged into
``gia_gmsl_rate`` before budgeting by default (one eustatic wedge: GIA + global
kernel Gm). Pass ``merge_gia_global=False`` to keep **GIA prior** and **Global
(Gm)** as separate wedges with independent ``|mean(rate)|`` contributions.
Percentages partition ``Σ |mean(component_rate)| × window_length`` within each
(region, window); they are not shares of ``|mean(total_rate)|``.

Rates in ``temporal_decomposition_rates.csv`` are in m/year (see
``export.py:60-67``), so the result is in meters of integrated displacement.

:func:`export_sea_level_budget` also writes companion window-rate CSVs
(``*_window_rates.csv`` and ``*_window_rates_field.csv``) with signed
window-mean ``total_rate`` and ``regional_linear_rate`` in mm/yr
(display convention: ``mm/yr = -rate_m_yr × 1000``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Patch


DEFAULT_AGE_WINDOWS: Tuple[Tuple[float, float], ...] = (
    (0.0, 4000.0),
    (4000.0, 8000.0),
    (8000.0, 12000.0),
)

# In-slice % labels on tiny wedges overlap; hide below this threshold and list below.
SMALL_PCT_ANNOTATION_THRESHOLD = 5.0


# Paul Tol "bright" qualitative — colorblind-safe; no red/green pairing, Lm vs NL
# distinguished by yellow (#CCBB44) vs purple (#AA3377), not orange vs red.
COMPONENT_COLORS: Dict[str, str] = {
    "gmsl": "#4477AA",
    "grd": "#66CCEE",
    "gia_gmsl": "#4477AA",
    "gia": "#4477AA",
    "global": "#332288",
    "regional_linear": "#CCBB44",
    "regional_nl": "#AA3377",
    "local_nl": "#EE6677",
    "merged_nl": "#AA3377",
}

# Stable legend order when GMSL/GRD split is active.
GMSL_SPLIT_COMPONENT_ORDER: Tuple[str, ...] = (
    "gmsl",
    "grd",
    "global",
    "regional_linear",
    "local_nl",
)

# Stable legend order when GIA prior and Global (Gm) are budgeted separately.
GIA_GLOBAL_SPLIT_COMPONENT_ORDER: Tuple[str, ...] = (
    "gia",
    "global",
    "regional_linear",
    "local_nl",
    "regional_nl",
    "merged_nl",
)


COMPONENT_LABELS: Dict[str, str] = {
    "gmsl": "GMSL",
    "grd": "GRD",
    "gia_gmsl": "GIA + Global (Gm)",
    "gia": "GIA prior",
    "global": "Global (Gm)",
    "regional_linear": "Regional Linear (Lm)",
    "regional_nl": "Regional NL",
    "local_nl": "Local NL (Ln)",
    "merged_nl": "Merged NL",
}


def _discover_components(columns: Sequence[str], *, include_gia: bool) -> List[str]:
    comps: List[str] = []
    for col in columns:
        if not col.endswith("_rate"):
            continue
        if col == "total_rate":
            continue
        name = col[: -len("_rate")]
        if name == "gia" and not include_gia:
            continue
        comps.append(name)
    return comps


def load_mean_gmsl_curve(
    anu_xlsx: str | Path,
    sixg_xlsx: str | Path,
) -> pd.DataFrame:
    """Load mean GMSL RSL curve (ANU + 6G) with ``age_bp`` and ``rsl_m`` columns.

    Excel ``Year`` is ka CE (e.g. -4 → 4000 cal yr BP); ``RSL`` is in metres.
    """
    anu = pd.read_excel(anu_xlsx)
    g6 = pd.read_excel(sixg_xlsx)
    for path, frame in ((anu_xlsx, anu), (sixg_xlsx, g6)):
        if not {"Year", "RSL"}.issubset(frame.columns):
            raise ValueError(f"{path}: expected columns Year and RSL")

    merged = anu[["Year", "RSL"]].merge(
        g6[["Year", "RSL"]], on="Year", suffixes=("_anu", "_6g")
    )
    merged["age_bp"] = -merged["Year"].astype(float) * 1000.0
    merged["rsl_m"] = (
        merged["RSL_anu"].astype(float) + merged["RSL_6g"].astype(float)
    ) / 2.0
    return (
        merged[["age_bp", "rsl_m"]]
        .sort_values("age_bp", ascending=False)
        .reset_index(drop=True)
    )


def _infer_difftimestep(age_bp: pd.Series) -> float:
    ages = np.sort(np.unique(age_bp.to_numpy(dtype=float)))
    if ages.size < 2:
        return 200.0
    diffs = np.diff(ages)
    positive = diffs[diffs > 0]
    return float(np.median(positive)) if positive.size else 200.0


def _reconstruct_test_age_grid(
    rate_ages: np.ndarray, difftimestep: float
) -> np.ndarray:
    """Rebuild the ``test_age`` grid that ``cal_rate_var`` expects from rate midpoints."""
    rate_ages = np.sort(np.unique(rate_ages.astype(float)))
    lo = float(rate_ages.min()) - 0.5 * difftimestep
    hi = float(rate_ages.max()) + 0.5 * difftimestep
    n_steps = int(round((hi - lo) / difftimestep)) + 1
    return lo + np.arange(n_steps, dtype=float) * difftimestep


def _cal_rate_from_rsl(
    test_age: np.ndarray,
    mean_rsl: np.ndarray,
    difftimestep: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy-only copy of ``PSTHM.post.cal_rate_var`` (rate series only)."""
    test_X = np.asarray(test_age, dtype=float)
    mean_rsl = np.asarray(mean_rsl, dtype=float)
    Mdiff = np.equal.outer(test_X, test_X).astype(int) - np.equal.outer(
        test_X, test_X + difftimestep
    ).astype(int)
    Mdiff = Mdiff * np.equal.outer(np.ones(len(test_X)), np.ones(len(test_X)))
    sub = np.where(np.sum(Mdiff, axis=1) == 0)[0]
    Mdiff = Mdiff[sub, :]
    difftimes = np.abs(Mdiff) @ test_X / np.sum(np.abs(Mdiff), axis=1)
    Mdiff = Mdiff / (Mdiff @ test_X.T)[:, None]
    rate = Mdiff @ mean_rsl
    return difftimes, rate


def attach_gmsl_grd_rates(
    df: pd.DataFrame,
    gmsl_curve: pd.DataFrame,
    *,
    difftimestep: Optional[float] = None,
) -> pd.DataFrame:
    """Split ``gia_rate`` into external ``gmsl_rate`` and residual ``grd_rate``.

    GMSL rates use the same finite-difference definition as
    :func:`PSTHM.post.cal_rate_var` on the mean ANU+6G curve.
    """
    if "gia_rate" not in df.columns:
        raise ValueError("rates CSV must contain gia_rate for GMSL/GRD split")

    if difftimestep is None:
        difftimestep = _infer_difftimestep(df["age_bp"])

    rate_ages = df["age_bp"].to_numpy(dtype=float)
    test_age = _reconstruct_test_age_grid(rate_ages, difftimestep)

    curve_ages = gmsl_curve["age_bp"].to_numpy(dtype=float)
    curve_rsl = gmsl_curve["rsl_m"].to_numpy(dtype=float)
    rsl_on_test = np.interp(test_age, curve_ages[::-1], curve_rsl[::-1])

    rate_times, gmsl_rates = _cal_rate_from_rsl(
        test_age, rsl_on_test, difftimestep)
    gmsl_rate_map = dict(
        zip(rate_times.astype(float), gmsl_rates.astype(float)))

    out = df.copy()
    out["gmsl_rate"] = out["age_bp"].astype(float).map(gmsl_rate_map)
    if out["gmsl_rate"].isna().any():
        missing = out.loc[out["gmsl_rate"].isna(), "age_bp"].unique()
        raise ValueError(
            "GMSL rate interpolation failed for age_bp values: "
            f"{sorted(missing.tolist())}"
        )
    out["grd_rate"] = out["gia_rate"] - out["gmsl_rate"]
    drop_cols = [c for c in ("gia_rate", "gia_rate_sd") if c in out.columns]
    return out.drop(columns=drop_cols)


def _prepare_rates_df(
    df: pd.DataFrame,
    *,
    include_gia: bool,
    gmsl_xlsx: Optional[Tuple[str | Path, str | Path]] = None,
    merge_gia_global: bool = True,
) -> pd.DataFrame:
    if gmsl_xlsx is not None:
        if not include_gia:
            raise ValueError("gmsl_xlsx requires include_gia=True")
        gmsl_curve = load_mean_gmsl_curve(gmsl_xlsx[0], gmsl_xlsx[1])
        return attach_gmsl_grd_rates(df, gmsl_curve)
    if merge_gia_global:
        return _merge_gia_gmsl(df, include_gia=include_gia)
    if not include_gia:
        drops = [c for c in ("gia_rate", "gia_rate_sd") if c in df.columns]
        return df.drop(columns=drops) if drops else df
    return df


def ordered_components(components: List[str]) -> List[str]:
    if "gia_gmsl" in components:
        preferred: Tuple[str, ...] = (
            "gia_gmsl",
            "regional_linear",
            "local_nl",
            "regional_nl",
            "merged_nl",
            "global",
        )
    elif "gia" in components and "global" in components:
        preferred = GIA_GLOBAL_SPLIT_COMPONENT_ORDER
    else:
        preferred = GMSL_SPLIT_COMPONENT_ORDER
    ordered = [c for c in preferred if c in components]
    ordered.extend(c for c in components if c not in ordered)
    return ordered


def _merge_gia_gmsl(df: pd.DataFrame, *, include_gia: bool) -> pd.DataFrame:
    """Sum GIA and Global rate columns into a single ``gia_gmsl_rate`` column.

    Both are near-eustatic signals physically, so we combine the underlying rate
    time series before computing the budget (more faithful than summing
    post-abs contributions). Standard deviations are dropped — uncertainty
    propagation for a sum needs covariance which isn't stored in the CSV.
    """
    has_gia = "gia_rate" in df.columns
    has_global = "global_rate" in df.columns
    if not include_gia:
        drops = [c for c in ("gia_rate", "gia_rate_sd") if c in df.columns]
        return df.drop(columns=drops) if drops else df
    if not (has_gia and has_global):
        return df

    df = df.copy()
    df["gia_gmsl_rate"] = df["gia_rate"] + df["global_rate"]
    drop_cols = [
        c
        for c in ("gia_rate", "gia_rate_sd", "global_rate", "global_rate_sd")
        if c in df.columns
    ]
    return df.drop(columns=drop_cols)


def _component_label(name: str) -> str:
    return COMPONENT_LABELS.get(name, name.replace("_", " ").title())


def _component_color(name: str, overrides: Optional[Dict[str, str]] = None) -> str:
    if overrides and name in overrides:
        return overrides[name]
    return COMPONENT_COLORS.get(name, "gray")


def _contrast_text_color(facecolor: str) -> str:
    """Return black or white label color for readable autopct on wedge fills."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(facecolor)
    # WCAG relative luminance (sRGB)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.45 else "black"


def _style_budget_autopct(text, facecolor: str, *, fontsize: float = 9) -> None:
    """Style in-wedge percentage label with contrast fill and stroke halo."""
    fill = _contrast_text_color(facecolor)
    stroke = "black" if fill == "white" else "white"
    text.set_fontsize(fontsize)
    text.set_color(fill)
    text.set_path_effects([pe.withStroke(linewidth=1.5, foreground=stroke)])


def _validate_age_windows_coverage(
    age_bp: pd.Series, age_windows: Sequence[Tuple[float, float]]
) -> None:
    """Fail fast when requested windows are outside available age coverage.

    Notes
    -----
    Rates are often reported at interval midpoints (e.g., 100..5900 BP for a
    0..6000 BP axis with 200-yr spacing). In that case we allow each edge to
    extend by half the inferred age spacing so users can request natural
    endpoint windows such as 0-4000 or 4000-6000.
    """
    if len(age_windows) == 0:
        raise ValueError(
            "age_windows must contain at least one (start_bp, end_bp) pair"
        )

    age_vals = age_bp.to_numpy(dtype=float)
    finite_age_vals = age_vals[np.isfinite(age_vals)]
    if finite_age_vals.size == 0:
        raise ValueError("rates CSV has no finite age_bp values")

    age_min = float(np.min(finite_age_vals))
    age_max = float(np.max(finite_age_vals))
    tol = 0.0
    unique_ages = np.unique(np.sort(finite_age_vals))
    if unique_ages.size >= 2:
        diffs = np.diff(unique_ages)
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size > 0:
            # Half-step edge tolerance for midpoint-sampled rate axes.
            tol = 0.5 * float(np.median(positive_diffs))

    coverage_min = age_min - tol
    coverage_max = age_max + tol

    invalid_windows: List[Tuple[float, float]] = []
    for w0, w1 in age_windows:
        lo, hi = (float(w0), float(w1)) if w0 <= w1 else (float(w1), float(w0))
        if hi <= lo:
            raise ValueError(
                f"age window ({w0}, {w1}) is invalid: end must be greater than start"
            )
        if lo < coverage_min or hi > coverage_max:
            invalid_windows.append((lo, hi))

    if invalid_windows:
        pretty = ", ".join(
            f"({int(lo)}, {int(hi)})" for lo, hi in invalid_windows)
        raise ValueError(
            "Requested age_windows extend beyond available rate ages "
            f"[{age_min:.1f}, {age_max:.1f}] BP"
            f" (effective coverage [{coverage_min:.1f}, {coverage_max:.1f}] with edge tolerance {tol:.1f}): {pretty}"
        )


def _abs_contributions_in_window(
    g: pd.DataFrame,
    mask: pd.Series,
    components: List[str],
    window_length: float,
) -> Dict[str, float]:
    """Per-component absolute contributions (m) for one region and age window."""
    abs_vals: Dict[str, float] = {}
    split_gmsl_grd = "gmsl" in components and "grd" in components

    if split_gmsl_grd:
        gmsl_rates = g.loc[mask, "gmsl_rate"].to_numpy(dtype=float)
        grd_rates = g.loc[mask, "grd_rate"].to_numpy(dtype=float)
        combined_abs = abs(
            float(np.mean(gmsl_rates + grd_rates))) * window_length
        mean_gmsl_abs = abs(float(np.mean(gmsl_rates)))
        mean_grd_abs = abs(float(np.mean(grd_rates)))
        weight_sum = mean_gmsl_abs + mean_grd_abs
        if weight_sum > 0:
            abs_vals["gmsl"] = combined_abs * (mean_gmsl_abs / weight_sum)
            abs_vals["grd"] = combined_abs * (mean_grd_abs / weight_sum)
        else:
            abs_vals["gmsl"] = combined_abs if mean_gmsl_abs > 0 else 0.0
            abs_vals["grd"] = combined_abs if mean_grd_abs > 0 else 0.0

    for comp in components:
        if split_gmsl_grd and comp in ("gmsl", "grd"):
            continue
        col = f"{comp}_rate"
        mean_rate = float(np.mean(g.loc[mask, col].to_numpy(dtype=float)))
        abs_vals[comp] = abs(mean_rate) * window_length

    return abs_vals


def compute_rate_contributions(
    rates_csv: str | Path | pd.DataFrame,
    age_windows: Sequence[Tuple[float, float]] = DEFAULT_AGE_WINDOWS,
    *,
    include_gia: bool = True,
    gmsl_xlsx: Optional[Tuple[str | Path, str | Path]] = None,
    merge_gia_global: bool = True,
) -> pd.DataFrame:
    """Aggregate per-region per-window rate contributions into a tidy table.

    Returns long-format columns:
        region, region_name, window_start_bp, window_end_bp,
        component, abs_contribution_m, pct

    `pct` sums to 100 within each (region, window) group (except when every
    component is exactly zero, in which case `pct` is NaN).

    With ``gmsl_xlsx``, GMSL and GRD contributions partition the combined GIA
    budget (see module docstring) rather than using independent ``|mean|`` terms.
    """
    if isinstance(rates_csv, pd.DataFrame):
        df = rates_csv.copy()
        source_label = "rates DataFrame"
    else:
        df = pd.read_csv(rates_csv)
        source_label = str(rates_csv)

    required = {"region", "region_name", "age_bp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_label}: missing required columns {sorted(missing)}")
    _validate_age_windows_coverage(df["age_bp"], age_windows)

    df = _prepare_rates_df(
        df,
        include_gia=include_gia,
        gmsl_xlsx=gmsl_xlsx,
        merge_gia_global=merge_gia_global,
    )
    components = _discover_components(df.columns, include_gia=include_gia)
    if not components:
        raise ValueError(
            f"{source_label}: no *_rate columns found to build a budget")
    components = ordered_components(components)

    rows: List[Dict[str, object]] = []
    for (region, region_name), g in df.groupby(["region", "region_name"], sort=True):
        for w0, w1 in age_windows:
            lo, hi = (w0, w1) if w0 <= w1 else (w1, w0)
            mask = (g["age_bp"] >= lo) & (g["age_bp"] <= hi)
            window_length = hi - lo
            if not mask.any() or window_length <= 0:
                for comp in components:
                    rows.append(
                        {
                            "region": region,
                            "region_name": region_name,
                            "window_start_bp": lo,
                            "window_end_bp": hi,
                            "component": comp,
                            "abs_contribution_m": 0.0,
                            "pct": float("nan"),
                        }
                    )
                continue

            abs_vals = _abs_contributions_in_window(
                g, mask, components, window_length)

            total = sum(abs_vals.values())
            for comp, val in abs_vals.items():
                rows.append(
                    {
                        "region": region,
                        "region_name": region_name,
                        "window_start_bp": lo,
                        "window_end_bp": hi,
                        "component": comp,
                        "abs_contribution_m": val,
                        "pct": (val / total * 100.0) if total > 0 else float("nan"),
                    }
                )

    return pd.DataFrame(rows)


def _mean_rate_m_yr_to_mm_yr(rate_m_yr: float) -> float:
    """Convert internal m/yr rate to display mm/yr (RSL rise positive)."""
    return -rate_m_yr * 1000.0


def _window_mean_rate_mm_yr(g: pd.DataFrame, mask: pd.Series, col: str) -> float:
    if col not in g.columns:
        return float("nan")
    vals = g.loc[mask, col].to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    return _mean_rate_m_yr_to_mm_yr(float(np.mean(vals)))


def compute_window_mean_rates(
    rates_csv: str | Path | pd.DataFrame,
    age_windows: Sequence[Tuple[float, float]] = DEFAULT_AGE_WINDOWS,
    *,
    include_gia: bool = True,
    gmsl_xlsx: Optional[Tuple[str | Path, str | Path]] = None,
    merge_gia_global: bool = True,
) -> pd.DataFrame:
    """Per-region window-mean total and regional-linear rates in mm/yr.

    Returns columns:
        region, region_name, window_start_bp, window_end_bp,
        total_mean_rate_mm_yr, regional_linear_mean_rate_mm_yr

    Uses signed arithmetic mean of rate samples within each window (not
    ``|mean|``). Missing rate columns yield NaN for that field.
    """
    if isinstance(rates_csv, pd.DataFrame):
        df = rates_csv.copy()
        source_label = "rates DataFrame"
    else:
        df = pd.read_csv(rates_csv)
        source_label = str(rates_csv)

    required = {"region", "region_name", "age_bp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_label}: missing required columns {sorted(missing)}")
    _validate_age_windows_coverage(df["age_bp"], age_windows)

    df = _prepare_rates_df(
        df,
        include_gia=include_gia,
        gmsl_xlsx=gmsl_xlsx,
        merge_gia_global=merge_gia_global,
    )

    rows: List[Dict[str, object]] = []
    for (region, region_name), g in df.groupby(["region", "region_name"], sort=True):
        for w0, w1 in age_windows:
            lo, hi = (w0, w1) if w0 <= w1 else (w1, w0)
            mask = (g["age_bp"] >= lo) & (g["age_bp"] <= hi)
            rows.append(
                {
                    "region": region,
                    "region_name": region_name,
                    "window_start_bp": lo,
                    "window_end_bp": hi,
                    "total_mean_rate_mm_yr": _window_mean_rate_mm_yr(
                        g, mask, "total_rate"
                    ),
                    "regional_linear_mean_rate_mm_yr": _window_mean_rate_mm_yr(
                        g, mask, "regional_linear_rate"
                    ),
                }
            )

    return pd.DataFrame(rows)


def compute_field_window_mean_rates(
    window_rates_df: pd.DataFrame,
    *,
    aggregator: str = "mean",
) -> pd.DataFrame:
    """Aggregate regional window-mean rates into a field-wide summary."""
    if aggregator not in {"mean", "sum"}:
        raise ValueError(
            f"aggregator must be 'mean' or 'sum', got {aggregator!r}")

    rate_cols = ["total_mean_rate_mm_yr", "regional_linear_mean_rate_mm_yr"]
    missing = [c for c in rate_cols if c not in window_rates_df.columns]
    if missing:
        raise ValueError(
            f"window_rates_df missing required columns: {sorted(missing)}")

    agg = (
        window_rates_df.groupby(["window_start_bp", "window_end_bp"], sort=True)[
            rate_cols
        ]
        .agg(aggregator)
        .reset_index()
    )
    return agg


def _sort_budget_subframe(sub: pd.DataFrame) -> pd.DataFrame:
    sort_order = ordered_components(sub["component"].unique().tolist())
    comp_order = {c: i for i, c in enumerate(sort_order)}
    return (
        sub.assign(_sort_key=sub["component"].map(
            lambda c: comp_order.get(c, 999)))
        .sort_values("_sort_key")
        .drop(columns="_sort_key")
    )


def _draw_budget_donut_on_ax(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    title: Optional[str] = None,
    component_colors: Optional[Dict[str, str]] = None,
    title_fontsize: float = 14,
    note_y: float = -0.06,
    label_fontsize: float = 9,
    pctdistance: float = 0.75,
    min_wedge_deg_for_label: Optional[float] = None,
) -> Tuple[List[str], int]:
    """Draw one budget donut on ``ax``.

    Returns component order for legend and the number of small-component note lines
    (including the ``Components <5%:`` header when present).
    """
    if sub.empty or not np.isfinite(sub["pct"]).any():
        ax.set_visible(False)
        return [], 0

    sub = _sort_budget_subframe(sub)
    sizes = sub["pct"].to_numpy()
    colors = [_component_color(c, component_colors) for c in sub["component"]]

    wedges, _, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=pctdistance,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=1.2),
        textprops={"fontsize": label_fontsize, "fontweight": "bold"},
    )

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    small_lines: List[str] = []
    for i, (_, row) in enumerate(sub.iterrows()):
        pct_val = float(row["pct"])
        if i >= len(autotexts) or i >= len(wedges):
            continue
        t = autotexts[i]
        wedge = wedges[i]
        if not np.isfinite(pct_val):
            t.set_text("")
            continue

        wedge_deg = float(wedge.theta2 - wedge.theta1)
        too_small_pct = pct_val < SMALL_PCT_ANNOTATION_THRESHOLD
        too_narrow = (
            min_wedge_deg_for_label is not None
            and wedge_deg < min_wedge_deg_for_label
        )
        if too_small_pct or too_narrow:
            t.set_text("")
            small_lines.append(
                f"{_component_label(str(row['component']))}: {pct_val:.1f}%"
            )
        else:
            _style_budget_autopct(t, wedge.get_facecolor(),
                                  fontsize=label_fontsize)

    n_small_note_lines = 0
    if small_lines:
        n_small_note_lines = len(small_lines) + 1  # +1 header line
        note = "Components <5%:\n" + "\n".join(small_lines)
        ax.text(
            0.5,
            note_y - 0.04 * (len(small_lines) - 1),
            note,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            linespacing=1.25,
        )

    return sub["component"].astype(str).tolist(), n_small_note_lines


def _plot_window_panel(
    panel_df: pd.DataFrame,
    age_windows: Sequence[Tuple[float, float]],
    *,
    suptitle: str,
    figsize_per_pie: Tuple[float, float],
    component_colors: Optional[Dict[str, str]],
) -> Figure:
    n_win = len(age_windows)
    fig, axes = plt.subplots(
        1,
        n_win,
        figsize=(figsize_per_pie[0] * n_win, figsize_per_pie[1]),
        subplot_kw={"aspect": "equal"},
    )
    axes = np.atleast_1d(axes)

    # Preserve first-seen component order across windows for a stable legend.
    legend_components: List[str] = []
    for comp in panel_df["component"].tolist():
        if comp not in legend_components:
            legend_components.append(comp)

    max_small_note_lines = 0
    for ax, (w0, w1) in zip(axes, age_windows):
        lo, hi = (w0, w1) if w0 <= w1 else (w1, w0)
        sub = panel_df[
            (panel_df["window_start_bp"] == lo) & (
                panel_df["window_end_bp"] == hi)
        ]
        _, n_small = _draw_budget_donut_on_ax(
            ax,
            sub,
            title=f"{int(lo)}–{int(hi)} BP",
            component_colors=component_colors,
        )
        max_small_note_lines = max(max_small_note_lines, n_small)

    handles = [
        Patch(
            facecolor=_component_color(c, component_colors),
            edgecolor="white",
            label=_component_label(c),
        )
        for c in ordered_components(legend_components)
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 5),
            bbox_to_anchor=(0.5, 0.0),
            frameon=False,
            fontsize=12,
        )

    fig.suptitle(suptitle, fontsize=18, y=1.02)
    # Extra bottom margin when per-panel small-component notes are shown.
    bottom = 0.08 + min(0.22, 0.028 * float(max_small_note_lines))
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.98))
    return fig


def _plot_single_region(
    region: int,
    region_name: str,
    region_df: pd.DataFrame,
    age_windows: Sequence[Tuple[float, float]],
    *,
    figsize_per_pie: Tuple[float, float],
    component_colors: Optional[Dict[str, str]],
) -> Figure:
    return _plot_window_panel(
        region_df,
        age_windows,
        suptitle=f"R{region} — {region_name}",
        figsize_per_pie=figsize_per_pie,
        component_colors=component_colors,
    )


def compute_field_contributions(
    budget_df: pd.DataFrame,
    *,
    aggregator: str = "mean",
) -> pd.DataFrame:
    """Aggregate regional budgets into a single field-wide budget.

    Parameters
    ----------
    budget_df : DataFrame
        Output of :func:`compute_rate_contributions`.
    aggregator : {"mean", "sum"}
        How to combine ``abs_contribution_m`` across regions before
        re-normalizing to percent. ``pct`` is identical under either choice
        (scale-invariant); ``abs_contribution_m`` differs.

    Returns
    -------
    DataFrame with columns: window_start_bp, window_end_bp, component,
    abs_contribution_m, pct.
    """
    if aggregator not in {"mean", "sum"}:
        raise ValueError(
            f"aggregator must be 'mean' or 'sum', got {aggregator!r}")

    agg = (
        budget_df.groupby(
            ["window_start_bp", "window_end_bp", "component"], sort=False
        )["abs_contribution_m"]
        .agg(aggregator)
        .reset_index()
    )

    rows: List[Dict[str, object]] = []
    for (w0, w1), g in agg.groupby(["window_start_bp", "window_end_bp"], sort=True):
        total = float(g["abs_contribution_m"].sum())
        for _, r in g.iterrows():
            val = float(r["abs_contribution_m"])
            rows.append(
                {
                    "window_start_bp": float(w0),
                    "window_end_bp": float(w1),
                    "component": r["component"],
                    "abs_contribution_m": val,
                    "pct": (val / total * 100.0) if total > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def plot_field_contributions_pie(
    field_df: pd.DataFrame,
    *,
    figsize_per_pie: Tuple[float, float] = (5.0, 5.0),
    component_colors: Optional[Dict[str, str]] = None,
    suptitle: str = "Field-wide sea-level budget (all regions)",
) -> Figure:
    """Single Figure with one donut per age window, aggregated across regions."""
    age_windows = [
        (float(a), float(b))
        for a, b in (
            field_df[["window_start_bp", "window_end_bp"]]
            .drop_duplicates()
            .sort_values("window_start_bp")
            .to_numpy()
        )
    ]
    return _plot_window_panel(
        field_df,
        age_windows,
        suptitle=suptitle,
        figsize_per_pie=figsize_per_pie,
        component_colors=component_colors,
    )


def plot_budget_regions_grid(
    budget_df: pd.DataFrame,
    *,
    window: Tuple[float, float] = (0.0, 4000.0),
    region_ids: Optional[Sequence[int]] = None,
    n_cols: int = 4,
    figsize_per_panel: Tuple[float, float] = (3.2, 3.6),
    component_colors: Optional[Dict[str, str]] = None,
    suptitle: Optional[str] = None,
) -> Figure:
    """One donut per region in a grid, with shared legend and small-slice notes."""
    if budget_df.empty:
        raise ValueError("budget_df is empty")

    lo, hi = (float(window[0]), float(window[1]))
    if lo > hi:
        lo, hi = hi, lo

    sub = budget_df[
        (budget_df["window_start_bp"] == lo) & (
            budget_df["window_end_bp"] == hi)
    ]
    if sub.empty:
        raise ValueError(f"No budget rows for window {lo}–{hi} BP")

    if region_ids is None:
        region_ids = sorted(sub["region"].unique())
    else:
        region_ids = [int(r) for r in region_ids]

    region_meta = (
        sub[["region", "region_name"]]
        .drop_duplicates()
        .set_index("region")["region_name"]
        .to_dict()
    )

    n_regions = len(region_ids)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_regions / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        subplot_kw={"aspect": "equal"},
    )
    axes = np.atleast_1d(axes).ravel()

    legend_components: List[str] = []
    max_small_note_lines = 0
    for ax, rn in zip(axes, region_ids):
        region_sub = sub[sub["region"] == rn]
        region_name = str(region_meta.get(rn, f"Region {rn}"))
        comps, n_small = _draw_budget_donut_on_ax(
            ax,
            region_sub,
            title=f"R{rn} — {region_name}",
            component_colors=component_colors,
            title_fontsize=10,
            note_y=-0.08,
            label_fontsize=7,
            pctdistance=0.68,
            min_wedge_deg_for_label=22,
        )
        max_small_note_lines = max(max_small_note_lines, n_small)
        for c in comps:
            if c not in legend_components:
                legend_components.append(c)

    for ax in axes[n_regions:]:
        ax.set_visible(False)

    ordered_legend = ordered_components(legend_components)
    handles = [
        Patch(
            facecolor=_component_color(c, component_colors),
            edgecolor="white",
            label=_component_label(c),
        )
        for c in ordered_legend
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 5),
            bbox_to_anchor=(0.5, 0.0),
            frameon=False,
            fontsize=11,
        )

    if suptitle is None:
        suptitle = f"Sea-level budget {int(lo)}–{int(hi)} BP — all regions"
    fig.suptitle(suptitle, fontsize=16, y=1.0)

    bottom = 0.06 + min(0.24, 0.028 * float(max_small_note_lines))
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.97))
    return fig


def plot_rate_contributions_pie(
    budget_df: pd.DataFrame,
    *,
    figsize_per_pie: Tuple[float, float] = (5.0, 5.0),
    component_colors: Optional[Dict[str, str]] = None,
) -> List[Figure]:
    """One donut-panel Figure per region (windows across columns)."""
    if budget_df.empty:
        return []

    age_windows = [
        (float(a), float(b))
        for a, b in (
            budget_df[["window_start_bp", "window_end_bp"]]
            .drop_duplicates()
            .sort_values("window_start_bp")
            .to_numpy()
        )
    ]

    figs: List[Figure] = []
    for (region, region_name), region_df in budget_df.groupby(
        ["region", "region_name"], sort=True
    ):
        fig = _plot_single_region(
            int(region),
            str(region_name),
            region_df,
            age_windows,
            figsize_per_pie=figsize_per_pie,
            component_colors=component_colors,
        )
        figs.append(fig)
    return figs


def export_sea_level_budget(
    rates_csv: str | Path,
    pdf_path: str | Path,
    csv_path: str | Path,
    age_windows: Sequence[Tuple[float, float]] = DEFAULT_AGE_WINDOWS,
    *,
    include_gia: bool = True,
    gmsl_xlsx: Optional[Tuple[str | Path, str | Path]] = None,
    merge_gia_global: bool = True,
    figsize_per_pie: Tuple[float, float] = (5.0, 5.0),
    component_colors: Optional[Dict[str, str]] = None,
    field_aggregator: str = "mean",
    field_csv_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Compute budget, write tidy CSV, and write a multi-page donut PDF.

    The PDF has one page per region plus a final **field-wide** page that
    aggregates all regions into a single overall budget. A companion
    ``*_field.csv`` (or ``field_csv_path`` if given) stores the aggregated
    table alongside the per-region ``csv_path``.

    Also writes ``{csv_path.stem}_window_rates.csv`` and a field-wide
    ``*_window_rates_field.csv`` with signed window-mean total and
    regional-linear rates in mm/yr.
    """
    budget_df = compute_rate_contributions(
        rates_csv,
        age_windows=age_windows,
        include_gia=include_gia,
        gmsl_xlsx=gmsl_xlsx,
        merge_gia_global=merge_gia_global,
    )
    field_df = compute_field_contributions(
        budget_df, aggregator=field_aggregator)
    window_rates_df = compute_window_mean_rates(
        rates_csv,
        age_windows=age_windows,
        include_gia=include_gia,
        gmsl_xlsx=gmsl_xlsx,
        merge_gia_global=merge_gia_global,
    )
    field_window_rates_df = compute_field_window_mean_rates(
        window_rates_df, aggregator=field_aggregator
    )

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    budget_df.to_csv(csv_path, index=False)

    window_rates_path = csv_path.with_name(
        csv_path.stem + "_window_rates" + csv_path.suffix
    )
    window_rates_path.parent.mkdir(parents=True, exist_ok=True)
    window_rates_df.to_csv(window_rates_path, index=False)

    if field_csv_path is None:
        field_csv_path = csv_path.with_name(
            csv_path.stem + "_field" + csv_path.suffix)
        field_window_rates_path = csv_path.with_name(
            csv_path.stem + "_window_rates_field" + csv_path.suffix
        )
    else:
        field_csv_path = Path(field_csv_path)
        field_window_rates_path = field_csv_path.with_name(
            field_csv_path.stem + "_window_rates_field" + field_csv_path.suffix
        )
    field_csv_path = Path(field_csv_path)
    field_csv_path.parent.mkdir(parents=True, exist_ok=True)
    field_df.to_csv(field_csv_path, index=False)

    field_window_rates_path = Path(field_window_rates_path)
    field_window_rates_path.parent.mkdir(parents=True, exist_ok=True)
    field_window_rates_df.to_csv(field_window_rates_path, index=False)

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    region_figs = plot_rate_contributions_pie(
        budget_df,
        figsize_per_pie=figsize_per_pie,
        component_colors=component_colors,
    )
    field_fig = plot_field_contributions_pie(
        field_df,
        figsize_per_pie=figsize_per_pie,
        component_colors=component_colors,
    )
    with PdfPages(pdf_path) as pdf:
        for fig in region_figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.savefig(field_fig, bbox_inches="tight")
        plt.close(field_fig)

    return budget_df


def age_windows_from_config(
    budget_cfg: Optional[Mapping] = None,
    *,
    default: Sequence[Tuple[float, float]] = ((0.0, 4000.0),),
) -> List[Tuple[float, float]]:
    """Parse ``budget.age_windows_bp`` from a config mapping."""
    raw = None
    if budget_cfg is not None:
        raw = budget_cfg.get("age_windows_bp")
    windows: List[Tuple[float, float]] = []
    if raw:
        for item in raw:
            if item is None:
                continue
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    lo, hi = float(item[0]), float(item[1])
                else:
                    continue
            except (TypeError, ValueError):
                continue
            if lo > hi:
                lo, hi = hi, lo
            if lo == hi:
                continue
            windows.append((lo, hi))
    if not windows:
        windows = [(float(a), float(b)) for a, b in default]
    return windows


def gmsl_xlsx_from_config(
    budget_cfg: Optional[Mapping] = None,
    *,
    base_dir: Optional[str | Path] = None,
) -> Optional[Tuple[Path, Path]]:
    """Parse optional ``budget.gmsl_xlsx`` into ``(anu_path, ice6g_path)``.

    Accepts a mapping with ``anu`` / ``ice6g`` keys (aliases ``ANU``, ``6g``,
    ``ice_6g``) or a two-element list/tuple. Relative paths resolve against
    ``base_dir`` when given; absolute paths are used as-is. Returns ``None``
    when the key is absent or empty.
    """
    if not budget_cfg:
        return None
    raw = budget_cfg.get("gmsl_xlsx")
    if raw is None or raw is False or raw == "":
        return None

    anu_raw: Optional[object] = None
    ice6g_raw: Optional[object] = None
    if isinstance(raw, Mapping):
        anu_raw = raw.get("anu") or raw.get("ANU")
        ice6g_raw = (
            raw.get("ice6g")
            or raw.get("6g")
            or raw.get("ice_6g")
            or raw.get("ICE6G")
        )
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        anu_raw, ice6g_raw = raw[0], raw[1]
    else:
        raise ValueError(
            "budget.gmsl_xlsx must be {anu, ice6g} or a two-element [anu, ice6g] list"
        )

    anu_text = str(anu_raw).strip() if anu_raw is not None else ""
    ice6g_text = str(ice6g_raw).strip() if ice6g_raw is not None else ""
    if not anu_text and not ice6g_text:
        return None
    if not anu_text or not ice6g_text:
        raise ValueError(
            "budget.gmsl_xlsx requires both anu and ice6g Excel paths"
        )

    def _resolve(text: str) -> Path:
        path = Path(text).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        return path.expanduser().resolve()

    return _resolve(anu_text), _resolve(ice6g_text)


def export_scenario_sea_level_budget(
    scenario_dir: str | Path,
    budget_cfg: Optional[Mapping] = None,
    *,
    rates_csv: Optional[str | Path] = None,
    include_gia: bool = True,
    merge_gia_global: Optional[bool] = None,
    base_dir: Optional[str | Path] = None,
    gmsl_xlsx: Optional[Tuple[str | Path, str | Path]] = None,
) -> Optional[pd.DataFrame]:
    """Export sea-level budget CSVs/PDF for a scenario output directory.

    Looks for ``temporal_decomposition_rates.csv`` (or ``rates_csv``) and writes
    ``sea_level_budget.csv`` (+ companions) and ``sea_level_budget.pdf`` beside it.
    Returns the budget DataFrame, or ``None`` when rates are missing / export fails.

    When ``budget.gmsl_xlsx`` (or ``gmsl_xlsx``) is set, splits the GIA prior into
    GMSL + GRD via the mean ANU + ICE-6G ice-equivalent curves; Global stays a
    separate wedge. Relative GMSL paths resolve against ``base_dir``, else the
    parent of ``scenario_dir`` (typical location of ``dashboard_config.yaml``).
    """
    scenario_dir = Path(scenario_dir)
    rates_path = Path(rates_csv) if rates_csv is not None else (
        scenario_dir / "temporal_decomposition_rates.csv"
    )
    if not rates_path.is_file():
        return None

    cfg = dict(budget_cfg or {})
    age_windows = age_windows_from_config(cfg)
    if merge_gia_global is None:
        merge_gia_global = bool(cfg.get("merge_gia_global", True))

    resolve_base = Path(base_dir) if base_dir is not None else scenario_dir.parent
    if gmsl_xlsx is None:
        gmsl_xlsx = gmsl_xlsx_from_config(cfg, base_dir=resolve_base)

    try:
        return export_sea_level_budget(
            rates_csv=rates_path,
            pdf_path=scenario_dir / "sea_level_budget.pdf",
            csv_path=scenario_dir / "sea_level_budget.csv",
            age_windows=age_windows,
            include_gia=include_gia,
            gmsl_xlsx=gmsl_xlsx,
            merge_gia_global=merge_gia_global,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Sea-level budget export skipped: {exc}")
        return None
