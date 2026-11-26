"""Shared age-range bounds and data-quality warnings for dashboard widgets."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

SUSPICIOUS_AGE_BP = 25_000
DEFAULT_WIDGET_MAX_BP = 25_000
DEFAULT_WIDGET_MIN_BP = 100
WIDGET_HEADROOM_BP = 1_000


def _preview_max_bp(preview: Optional[Mapping[str, Any]]) -> int:
    if not preview or "age_max_bp" not in preview:
        return 0
    return int(preview["age_max_bp"])


def resolve_age_widget_bounds(
    preview: Optional[Mapping[str, Any]],
    stored_max_age_bp: int,
    *,
    min_bp: int = DEFAULT_WIDGET_MIN_BP,
) -> Tuple[int, int, int]:
    """Return (min_bp, max_bp, clamped_value) safe for Streamlit number_input."""
    preview_max = _preview_max_bp(preview)
    max_bp = max(
        DEFAULT_WIDGET_MAX_BP,
        preview_max,
        int(stored_max_age_bp),
    ) + WIDGET_HEADROOM_BP
    clamped = max(min_bp, min(int(stored_max_age_bp), max_bp))
    return min_bp, max_bp, clamped


def age_quality_warnings(preview: Optional[Mapping[str, Any]]) -> List[str]:
    """Warnings for suspicious Age values in the dataset preview."""
    if not preview:
        return []

    warnings: List[str] = []
    age_max = preview.get("age_max_bp")
    age_min = preview.get("age_min_bp")

    if age_max is not None and float(age_max) > SUSPICIOUS_AGE_BP:
        warnings.append(
            f"Dataset maximum age is **{float(age_max):.0f} cal yr BP**, which exceeds "
            f"{SUSPICIOUS_AGE_BP:,} BP. Verify the Age column uses **cal yr BP** "
            "(not CE years, radiocarbon years, or typos)."
        )
    if age_min is not None and float(age_min) < 0:
        warnings.append(
            f"Dataset contains negative ages (minimum **{float(age_min):.0f}**). "
            "Check Age units and filtering."
        )
    return warnings


def default_max_age_from_preview(
    preview: Optional[Mapping[str, Any]],
    current_max_age_bp: int,
) -> int:
    """Pick a widget-safe default max age when auto-syncing from preview."""
    if not preview or "age_max_bp" not in preview:
        return current_max_age_bp
    preview_max = int(preview["age_max_bp"])
    _, _, clamped = resolve_age_widget_bounds(preview, preview_max)
    return clamped
