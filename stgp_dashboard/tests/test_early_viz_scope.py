"""Tests for early Visualization scope (Analysis period light shortcut)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stgp_dashboard.services.viz_scope import (  # noqa: E402
    PERIOD_SCOPE_RADIO_KEY,
    RESULTS_SCOPE_RADIO_KEY,
    STEP_ANALYSIS_PERIOD,
    STEP_GRIDS,
    STEP_PRIORS,
    STEP_REVIEW,
    is_rsl_curve_only,
    normalize_viz_scope,
    requires_budget_windows,
    should_skip_grids_step,
    step_after_priors,
    step_before_review,
    sync_viz_scope_from_widgets,
)
from stgp_dashboard.state import DEFAULT_STATE  # noqa: E402


def test_default_scope_still_full():
    assert DEFAULT_STATE["viz_scope"] == "full"
    assert normalize_viz_scope(None) == "full"
    assert normalize_viz_scope("bogus") == "full"


def test_rsl_curve_does_not_require_budget_windows():
    assert requires_budget_windows("full") is True
    assert requires_budget_windows("rsl_curve") is False
    assert requires_budget_windows(None) is True


def test_priors_next_skips_grids_for_rsl_curve():
    assert step_after_priors("full") == STEP_GRIDS
    assert step_after_priors("rsl_curve") == STEP_REVIEW


def test_review_back_skips_grids_for_rsl_curve():
    assert step_before_review("full") == STEP_GRIDS
    assert step_before_review("rsl_curve") == STEP_PRIORS


def test_should_skip_grids_step():
    assert should_skip_grids_step("full") is False
    assert should_skip_grids_step("rsl_curve") is True
    assert is_rsl_curve_only("rsl_curve") is True
    assert is_rsl_curve_only("full") is False


def test_sync_viz_scope_from_period_radio_before_sidebar():
    """Clicking Full analysis must clear skip even if viz_scope was rsl_curve."""
    state = {
        "viz_scope": "rsl_curve",
        PERIOD_SCOPE_RADIO_KEY: "Full analysis",
    }
    scope = sync_viz_scope_from_widgets(
        state, mode="wizard", step=STEP_ANALYSIS_PERIOD
    )
    assert scope == "full"
    assert state["viz_scope"] == "full"
    assert should_skip_grids_step(scope) is False


def test_sync_viz_scope_from_period_radio_sets_rsl_curve():
    state = {
        "viz_scope": "full",
        PERIOD_SCOPE_RADIO_KEY: "RSL curve only",
    }
    scope = sync_viz_scope_from_widgets(
        state, mode="wizard", step=STEP_ANALYSIS_PERIOD
    )
    assert scope == "rsl_curve"
    assert should_skip_grids_step(scope) is True


def test_sync_ignores_stale_period_radio_on_later_steps():
    """After leaving Analysis period, Next-saved viz_scope wins over old radio."""
    state = {
        "viz_scope": "full",
        PERIOD_SCOPE_RADIO_KEY: "RSL curve only",  # stale widget key
    }
    scope = sync_viz_scope_from_widgets(state, mode="wizard", step=STEP_PRIORS)
    assert scope == "full"
    assert state["viz_scope"] == "full"


def test_sync_viz_scope_from_results_radio():
    state = {
        "viz_scope": "rsl_curve",
        RESULTS_SCOPE_RADIO_KEY: "Full analysis",
    }
    scope = sync_viz_scope_from_widgets(state, mode="results", step=9)
    assert scope == "full"
    assert state["viz_scope"] == "full"
