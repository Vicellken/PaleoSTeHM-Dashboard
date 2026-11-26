"""Visualization scope helpers (Full analysis vs RSL curve only)."""

from __future__ import annotations

from typing import Mapping, MutableMapping

SCOPE_OPTIONS = [
    ("full", "Full analysis"),
    ("rsl_curve", "RSL curve only"),
]
SCOPE_LABELS = [label for _, label in SCOPE_OPTIONS]
SCOPE_LABEL_TO_VALUE = {label: value for value, label in SCOPE_OPTIONS}
SCOPE_VALUE_TO_LABEL = {value: label for value, label in SCOPE_OPTIONS}

# Streamlit widget keys for scope radios (must match step UIs).
PERIOD_SCOPE_RADIO_KEY = "period_viz_scope_radio"
RESULTS_SCOPE_RADIO_KEY = "viz_scope_radio"

# Wizard step indices (see stgp_dashboard.state.STEP_LABELS).
STEP_ANALYSIS_PERIOD = 2
STEP_PRIORS = 6
STEP_GRIDS = 7
STEP_REVIEW = 8


def normalize_viz_scope(value: object) -> str:
    """Return a valid scope; unknown/empty values become ``full``."""
    text = str(value or "full").strip()
    return text if text in SCOPE_VALUE_TO_LABEL else "full"


def is_rsl_curve_only(viz_scope: object) -> bool:
    return normalize_viz_scope(viz_scope) == "rsl_curve"


def requires_budget_windows(viz_scope: object) -> bool:
    """Budget windows are only required for the full analysis path."""
    return not is_rsl_curve_only(viz_scope)


def step_after_priors(viz_scope: object) -> int:
    """Priors → Next: skip Spatial grid when RSL curve only."""
    return STEP_REVIEW if is_rsl_curve_only(viz_scope) else STEP_GRIDS


def step_before_review(viz_scope: object) -> int:
    """Review ← Back: return to Priors when Spatial grid was skipped."""
    return STEP_PRIORS if is_rsl_curve_only(viz_scope) else STEP_GRIDS


def should_skip_grids_step(viz_scope: object) -> bool:
    return is_rsl_curve_only(viz_scope)


def scope_from_state(state: Mapping[str, object] | None) -> str:
    if not state:
        return "full"
    return normalize_viz_scope(state.get("viz_scope", "full"))


def _scope_from_radio_label(label: object) -> str | None:
    """Map a radio label to scope, or None if not a known label."""
    text = str(label or "").strip()
    return SCOPE_LABEL_TO_VALUE.get(text)


def sync_viz_scope_from_widgets(
    state: MutableMapping[str, object],
    *,
    mode: str,
    step: int,
) -> str:
    """Copy the active page's scope radio into ``viz_scope`` before sidebar/UI.

    Streamlit updates widget keys before the script runs, but step renderers
    (which call ``update_state(viz_scope=...)``) run *after* the sidebar. Without
    this sync, the sidebar can show Spatial grid as skipped for one frame after
    switching back to Full analysis on Analysis period.
    """
    synced: str | None = None
    if mode == "wizard" and int(step) == STEP_ANALYSIS_PERIOD:
        synced = _scope_from_radio_label(state.get(PERIOD_SCOPE_RADIO_KEY))
    elif mode == "results":
        synced = _scope_from_radio_label(state.get(RESULTS_SCOPE_RADIO_KEY))

    if synced is not None:
        state["viz_scope"] = synced
        return synced
    return normalize_viz_scope(state.get("viz_scope", "full"))
