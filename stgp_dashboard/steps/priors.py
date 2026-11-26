"""Step 6 — Kernel prior ranges."""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from stgp_dashboard.services.config_builder import load_yaml_preset
from stgp_dashboard.services.viz_scope import step_after_priors
from stgp_dashboard.services.workflow_labels import EARTH_RADIUS_KM
from stgp_dashboard.state import go_to_step, update_state

_FAST_KERNELS = (
    "global_temporal_fast",
    "regional_nonlinear_fast",
    "local_nonlinear_fast",
)

_REGIONAL_MAIN_KERNELS = (
    "regional_linear",
    "regional_nonlinear",
    "local_nonlinear",
    "merged_nonlinear",
    "sp_whitenoise",
    "whitenoise",
)

_ALL_KERNELS = ("global_temporal",) + _REGIONAL_MAIN_KERNELS + _FAST_KERNELS

_PRIOR_KEYS = (
    "lengthscale_prior",
    "variance_prior",
    "temporal_lengthscale_prior",
    "temporal_variance_prior",
    "spatial_lengthscale_prior",
)


def _prior_unit(key: str, kernel_name: str) -> str:
    if key == "lengthscale_prior" or key == "temporal_lengthscale_prior":
        return "cal yr BP"
    if key == "spatial_lengthscale_prior":
        return "rad"
    if key == "temporal_variance_prior" and kernel_name == "regional_linear":
        return "m²/yr²"
    if key in {"variance_prior", "temporal_variance_prior"}:
        return "m²"
    return ""


def _prior_label(key: str, kernel_name: str) -> str:
    labels = {
        "lengthscale_prior": "Temporal lengthscale prior",
        "variance_prior": "Variance prior",
        "temporal_lengthscale_prior": "Temporal lengthscale prior",
        "temporal_variance_prior": "Temporal variance prior",
        "spatial_lengthscale_prior": "Spatial lengthscale prior",
    }
    unit = _prior_unit(key, kernel_name)
    base = labels.get(key, key)
    return f"{base} [{unit}]" if unit else base


def _widget_key(preset: str, kernel_name: str, prior_key: str, bound: str) -> str:
    return f"prior_{preset}_{kernel_name}_{prior_key}_{bound}"


def _render_kernel_block(
    kernel_name: str,
    base: dict,
    overrides: dict,
    preset: str,
) -> None:
    default_enabled = bool(base.get("enabled", True))
    enabled = st.checkbox(
        f"Enable {kernel_name}",
        value=bool(overrides.get(kernel_name, {}).get("enabled", default_enabled)),
        key=f"prior_enable_{preset}_{kernel_name}",
    )
    block = overrides.setdefault(kernel_name, {})
    block["enabled"] = enabled
    if not enabled:
        return

    for key in _PRIOR_KEYS:
        if key not in base:
            continue
        label = _prior_label(key, kernel_name)
        lo, hi = base[key]
        c1, c2 = st.columns(2)
        with c1:
            new_lo = st.number_input(
                f"{label} — lower",
                value=float(lo),
                format="%.6g",
                key=_widget_key(preset, kernel_name, key, "lo"),
            )
        with c2:
            new_hi = st.number_input(
                f"{label} — upper",
                value=float(hi),
                format="%.6g",
                key=_widget_key(preset, kernel_name, key, "hi"),
            )
        block[key] = [float(new_lo), float(new_hi)]
        if key == "spatial_lengthscale_prior":
            st.caption(
                f"≈ {float(new_lo) * EARTH_RADIUS_KM:.2f} – "
                f"{float(new_hi) * EARTH_RADIUS_KM:.2f} km"
            )


def _kernel_base(
    kernel_name: str,
    kernels_base: dict,
    preset_block: dict,
    overrides: dict,
) -> dict:
    base = dict(kernels_base.get(kernel_name, {}))
    base.update(preset_block.get(kernel_name, {}))
    base.update(overrides.get(kernel_name, {}))
    return base


def collect_live_kernel_priors(
    preset: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Rebuild ``kernel_priors`` from live Streamlit widgets when present.

    Returns ``None`` if no prior widgets have been rendered yet (user never
    visited the priors step).
    """
    preset = str(preset or st.session_state.get("prior_preset", "regional"))
    if "prior_preset_select" in st.session_state:
        preset = str(st.session_state["prior_preset_select"])

    overrides: Dict[str, Any] = {}
    found_any = False
    for kernel_name in _ALL_KERNELS:
        enable_key = f"prior_enable_{preset}_{kernel_name}"
        block: Dict[str, Any] = {}
        if enable_key in st.session_state:
            found_any = True
            block["enabled"] = bool(st.session_state[enable_key])
        for key in _PRIOR_KEYS:
            lo_key = _widget_key(preset, kernel_name, key, "lo")
            hi_key = _widget_key(preset, kernel_name, key, "hi")
            if lo_key in st.session_state and hi_key in st.session_state:
                found_any = True
                block[key] = [
                    float(st.session_state[lo_key]),
                    float(st.session_state[hi_key]),
                ]
        if block:
            overrides[kernel_name] = block
    if not found_any:
        return None
    return overrides


def flush_live_priors_to_state() -> None:
    """Commit live prior widgets into session state before building config."""
    preset = str(st.session_state.get("prior_preset", "regional"))
    if "prior_preset_select" in st.session_state:
        preset = str(st.session_state["prior_preset_select"])
    live = collect_live_kernel_priors(preset)
    if live is not None:
        update_state(prior_preset=preset, kernel_priors=live)


def render() -> None:
    st.header("Kernel priors")
    st.caption(
        "Start with the global trend (Gc), then pick a spatial domain preset for "
        "regional/local kernels. Enable or disable components and adjust prior ranges."
    )

    presets = load_yaml_preset("priors_regional.yaml")
    preset_keys = ["small", "regional", "large"]
    current = st.session_state.get("prior_preset", "regional")
    if current not in preset_keys:
        current = "regional"
    # Initialize / keep selectbox key aligned with persisted prior_preset (e.g. after
    # load_config). Do not pass index= when key= is set — Streamlit ignores it.
    if "prior_preset_select" not in st.session_state:
        st.session_state["prior_preset_select"] = current
    overrides = dict(st.session_state.get("kernel_priors") or {})
    kernels_base = presets.get("kernels_base", {})

    st.markdown("### 1. Spatial domain preset")
    preset = st.selectbox(
        "Spatial domain preset",
        options=preset_keys,
        format_func=lambda k: presets[k].get("label", k),
        help="Adjusts spatial lengthscale upper bounds for regional/local kernels only.",
        key="prior_preset_select",
    )
    st.caption(
        "Does not change global temporal priors — only spatial bounds for "
        "regional_linear, regional_nonlinear, and local_nonlinear. "
        "Changing the preset refreshes spatial prior widgets."
    )
    preset_block = presets.get(preset, {})

    st.markdown("### 2. Global trend (Gc)")
    st.caption(
        "Review the global temporal kernel first — it sets the Holocene-scale smooth "
        "trend. When Gc is enabled, regional/local nonlinear kernels inherit its "
        "temporal lengthscale."
    )
    global_base = _kernel_base("global_temporal", kernels_base, {}, overrides)
    if global_base:
        with st.expander("global_temporal", expanded=True):
            _render_kernel_block("global_temporal", global_base, overrides, preset)

    st.markdown("### 3. Regional / local / noise kernels")
    for kernel_name in _REGIONAL_MAIN_KERNELS:
        base = _kernel_base(kernel_name, kernels_base, preset_block, overrides)
        if not base:
            continue
        with st.expander(f"{kernel_name}", expanded=kernel_name in preset_block):
            _render_kernel_block(kernel_name, base, overrides, preset)

    with st.expander("Advanced — fast kernels"):
        st.caption(
            "Optional shorter-timescale kernels. Disable unless you need "
            "finer temporal structure."
        )
        for kernel_name in _FAST_KERNELS:
            base = _kernel_base(kernel_name, kernels_base, {}, overrides)
            if not base:
                continue
            st.markdown(f"**{kernel_name}**")
            _render_kernel_block(kernel_name, base, overrides, preset)

    boundary_df = st.session_state.get("boundary_df")
    if boundary_df is not None and not getattr(boundary_df, "empty", True):
        hits = boundary_df[boundary_df.get("boundary_hit", False) == True]  # noqa: E712
        if len(hits):
            st.warning(
                f"{len(hits)} parameter(s) hit prior bounds in the last run. "
                "Consider widening priors above."
            )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("← Back", key="prior_back"):
            go_to_step(5)
            st.rerun()
    with col2:
        if st.button("Next →", type="primary", key="prior_next"):
            update_state(prior_preset=preset, kernel_priors=overrides)
            go_to_step(step_after_priors(st.session_state.get("viz_scope", "full")))
            st.rerun()
