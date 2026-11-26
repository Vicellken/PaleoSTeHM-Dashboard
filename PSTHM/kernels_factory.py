"""
Canonical kernel factory for PSTHM spatio-temporal GP models.

This is the single source of truth for constructing the multi-kernel STGP stack
used across:
- Holocene UI scripts
- Research pipelines (e.g. exceedance timing)
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any, Dict

import torch
import pyro.distributions as dist

from . import kernels


def _is_enabled(cfg_block: Any) -> bool:
    """Return True unless the config block explicitly sets ``enabled: false``."""
    if not isinstance(cfg_block, dict):
        return True
    return bool(cfg_block.get("enabled", True))


_KERNEL_TYPE_ALIASES = {
    "matern32": "Matern32",
    "matern21": "Matern21",
    "matern52": "Matern52",
    "linear": "Linear",
    "rbf": "RBF",
    "whitenoise": "WhiteNoise",
    "whitenoise_sp": "WhiteNoise_SP",
}


def _normalize_kernel_type(kernel_type: str) -> str:
    if not isinstance(kernel_type, str) or not kernel_type.strip():
        raise ValueError(f"Kernel type must be a non-empty string, got {kernel_type!r}")
    normalized = _KERNEL_TYPE_ALIASES.get(kernel_type.strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported kernel type {kernel_type!r}. "
            f"Supported aliases: {sorted(_KERNEL_TYPE_ALIASES.keys())}"
        )
    return normalized


def _resolve_kernel_class(kernel_type: str):
    class_name = _normalize_kernel_type(kernel_type)
    kernel_cls = getattr(kernels, class_name, None)
    if kernel_cls is None:
        raise ValueError(
            f"Kernel class {class_name!r} is not available in PSTHM.kernels"
        )
    return kernel_cls


def _build_kernel(kernel_type: str, **kwargs):
    kernel_cls = _resolve_kernel_class(kernel_type)
    try:
        return kernel_cls(input_dim=1, **kwargs)
    except TypeError:
        # Some kernels (e.g. Linear, WhiteNoise) do not accept geo/sp/lengthscale kwargs.
        return kernel_cls(input_dim=1)


def build_spatiotemporal_kernel(gp_model_cfg: Dict) -> Any:
    """
    Build the spatio-temporal kernel stack following the Holocene UI pattern.

    Units conventions (assuming RSL observations in meters and ages in years):
      - ``variance_prior`` on a Matern or RBF kernel: m^2 (output variance of RSL).
      - ``variance_prior`` on the regional-linear Linear temporal kernel: m^2/yr^2
        (Linear kernel contributes sigma^2 * (t - ref_year)^2 to the diagonal).
      - ``lengthscale_prior`` / ``temporal_lengthscale_prior``: yr.
      - ``spatial_lengthscale_prior`` (``s_lengthscale``): dimensionless radians
        on the unit Earth sphere; multiply by R=6371 km to convert to km.
      - WhiteNoise / WhiteNoise_SP ``variance_prior``: m^2.

    Expected gp_model_cfg shape (subset):
      gp_model:
        kernels:
          global_temporal: {type, lengthscale_prior: [lo,hi] (yr), variance_prior: [lo,hi] (m^2)}
          regional_linear: {temporal_variance_prior: [lo,hi] (m^2/yr^2), spatial_lengthscale_prior: [lo,hi] (rad)}
          regional_nonlinear: {temporal_variance_prior (m^2), temporal_lengthscale_prior (yr), spatial_lengthscale_prior (rad)}
          merged_nonlinear: {enabled, temporal_variance_prior (m^2), temporal_lengthscale_prior (yr), spatial_lengthscale_prior (rad)}
          local_nonlinear: {temporal_variance_prior (m^2), temporal_lengthscale_prior (yr), spatial_lengthscale_prior (rad)}
          sp_whitenoise: {variance_prior: [lo,hi] (m^2)}
          whitenoise: {variance_prior: [lo,hi] (m^2)}

    Each kernel block may also carry an optional ``enabled: false`` flag to
    remove it from the Sum. Default is enabled. At least one kernel must remain
    enabled.

    Returns
    -------
    Kernel
        The combined (left-nested) Sum of enabled kernels. Also carries a
        ``component_kernels`` attribute: an ordered dict mapping canonical
        names (``global_temporal``, ``regional_linear``, ``regional_nl``,
        ``local_nl``, ``sp_whitenoise``, ``whitenoise``) to the constructed
        kernel for each enabled component.
    """
    kernel_cfg = gp_model_cfg.get("kernels", {})

    components: "OrderedDict[str, Any]" = OrderedDict()

    # ------------------------------------------------------------------
    # Global temporal kernel (m^2 variance, yr lengthscale)
    # ------------------------------------------------------------------
    global_cfg = kernel_cfg.get("global_temporal", {}) or {}
    global_kernel = None
    if _is_enabled(global_cfg):
        global_type = global_cfg.get("type", "Matern32")
        global_kernel = _build_kernel(global_type, geo=False)
        ls_prior = global_cfg.get("lengthscale_prior", [100.0, 20000.0])
        var_prior = global_cfg.get("variance_prior", [1e-5, 1000.0])
        if hasattr(global_kernel, "lengthscale"):
            global_kernel.set_prior(
                "lengthscale",
                dist.Uniform(torch.tensor(ls_prior[0]), torch.tensor(ls_prior[1])),
            )
        if hasattr(global_kernel, "variance"):
            global_kernel.set_prior(
                "variance",
                dist.Uniform(torch.tensor(var_prior[0]), torch.tensor(var_prior[1])),
            )
        components["global_temporal"] = global_kernel

    # ------------------------------------------------------------------
    # Global temporal FAST kernel (optional; shorter lengthscale companion).
    # Independent lengthscale from global_temporal. Absent or enabled=false
    # -> skipped.
    # ------------------------------------------------------------------
    global_fast_cfg = kernel_cfg.get("global_temporal_fast")
    if isinstance(global_fast_cfg, dict) and _is_enabled(global_fast_cfg):
        global_fast_type = global_fast_cfg.get("type", "Matern32")
        global_fast_kernel = _build_kernel(global_fast_type, geo=False)
        ls_prior = global_fast_cfg.get("lengthscale_prior", [50.0, 300.0])
        var_prior = global_fast_cfg.get("variance_prior", [1e-6, 5.0])
        if hasattr(global_fast_kernel, "lengthscale"):
            global_fast_kernel.set_prior(
                "lengthscale",
                dist.Uniform(torch.tensor(ls_prior[0]), torch.tensor(ls_prior[1])),
            )
        if hasattr(global_fast_kernel, "variance"):
            global_fast_kernel.set_prior(
                "variance",
                dist.Uniform(torch.tensor(var_prior[0]), torch.tensor(var_prior[1])),
            )
        components["global_temporal_fast"] = global_fast_kernel

    # The regional/local non-linear temporal kernels traditionally share their
    # lengthscale parameter with the global temporal kernel. If the global
    # kernel is disabled, they must fall back to independent lengthscales.
    shared_lengthscale = getattr(global_kernel, "lengthscale", None)
    if global_kernel is None:
        warnings.warn(
            "global_temporal is disabled; regional_nl and local_nl will use "
            "independent lengthscales instead of sharing with the global kernel.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Regional linear kernel (Linear[m^2/yr^2] x Matern21[rad])
    # ------------------------------------------------------------------
    rlin_cfg = kernel_cfg.get("regional_linear", {}) or {}
    if _is_enabled(rlin_cfg):
        rlin_tem_type = rlin_cfg.get("temporal_type", "Linear")
        rlin_tem = _build_kernel(rlin_tem_type, geo=False)
        rlin_tem.set_prior(
            "variance",
            dist.Uniform(
                torch.tensor(rlin_cfg.get("temporal_variance_prior", [1e-12, 1e-6])[0]),
                torch.tensor(rlin_cfg.get("temporal_variance_prior", [1e-12, 1e-6])[1]),
            ),
        )
        if hasattr(rlin_tem, "ref_year"):
            rlin_tem.ref_year = torch.tensor(0.0)
        rlin_sp_type = rlin_cfg.get("spatial_type", "Matern21")
        rlin_sp = _build_kernel(rlin_sp_type, geo=True)
        rlin_sls = rlin_cfg.get("spatial_lengthscale_prior", [0.01, 0.5])
        if hasattr(rlin_sp, "s_lengthscale"):
            rlin_sp.set_prior(
                "s_lengthscale",
                dist.Uniform(torch.tensor(rlin_sls[0]), torch.tensor(rlin_sls[1])),
            )
        components["regional_linear"] = kernels.Product(rlin_tem, rlin_sp)

    # ------------------------------------------------------------------
    # Regional non-linear kernel (Matern32[m^2, yr] x Matern21[rad])
    # ------------------------------------------------------------------
    rnl_cfg = kernel_cfg.get("regional_nonlinear", {}) or {}
    if _is_enabled(rnl_cfg):
        rnl_tem_type = rnl_cfg.get("temporal_type", "Matern32")
        rnl_tem = _build_kernel(
            rnl_tem_type,
            lengthscale=shared_lengthscale,
            geo=False,
        )
        rnl_tem.set_prior(
            "variance",
            dist.Uniform(
                torch.tensor(rnl_cfg.get("temporal_variance_prior", [1e-6, 100.0])[0]),
                torch.tensor(rnl_cfg.get("temporal_variance_prior", [1e-6, 100.0])[1]),
            ),
        )
        if hasattr(rnl_tem, "lengthscale"):
            rnl_tem.set_prior(
                "lengthscale",
                dist.Uniform(
                    torch.tensor(
                        rnl_cfg.get("temporal_lengthscale_prior", [100.0, 20000.0])[0]
                    ),
                    torch.tensor(
                        rnl_cfg.get("temporal_lengthscale_prior", [100.0, 20000.0])[1]
                    ),
                ),
            )
        rnl_sp_type = rnl_cfg.get("spatial_type", "Matern21")
        rnl_sp = _build_kernel(rnl_sp_type, geo=True)
        rnl_sls = rnl_cfg.get("spatial_lengthscale_prior", [0.01, 0.5])
        if hasattr(rnl_sp, "s_lengthscale"):
            rnl_sp.set_prior(
                "s_lengthscale",
                dist.Uniform(torch.tensor(rnl_sls[0]), torch.tensor(rnl_sls[1])),
            )
        components["regional_nl"] = kernels.Product(rnl_tem, rnl_sp)

    # ------------------------------------------------------------------
    # Regional non-linear FAST kernel (optional; for sensitivity tests).
    # Same Matern32 x Matern21 structure as regional_nonlinear, intended
    # for shorter temporal lengthscales to probe identifiability of
    # high-frequency ocean-dynamic signals. Absent or enabled=false -> skipped.
    # ------------------------------------------------------------------
    rnl_fast_cfg = kernel_cfg.get("regional_nonlinear_fast")
    if isinstance(rnl_fast_cfg, dict) and _is_enabled(rnl_fast_cfg):
        rnl_fast_tem_type = rnl_fast_cfg.get("temporal_type", "Matern32")
        rnl_fast_tem = _build_kernel(
            rnl_fast_tem_type,
            lengthscale=shared_lengthscale,
            geo=False,
        )
        rnl_fast_tem.set_prior(
            "variance",
            dist.Uniform(
                torch.tensor(
                    rnl_fast_cfg.get("temporal_variance_prior", [1e-5, 1.0])[0]
                ),
                torch.tensor(
                    rnl_fast_cfg.get("temporal_variance_prior", [1e-5, 1.0])[1]
                ),
            ),
        )
        if hasattr(rnl_fast_tem, "lengthscale"):
            rnl_fast_tem.set_prior(
                "lengthscale",
                dist.Uniform(
                    torch.tensor(
                        rnl_fast_cfg.get("temporal_lengthscale_prior", [50.0, 300.0])[0]
                    ),
                    torch.tensor(
                        rnl_fast_cfg.get("temporal_lengthscale_prior", [50.0, 300.0])[1]
                    ),
                ),
            )
        rnl_fast_sp_type = rnl_fast_cfg.get("spatial_type", "Matern21")
        rnl_fast_sp = _build_kernel(rnl_fast_sp_type, geo=True)
        rnl_fast_sls = rnl_fast_cfg.get("spatial_lengthscale_prior", [0.01, 0.2])
        if hasattr(rnl_fast_sp, "s_lengthscale"):
            rnl_fast_sp.set_prior(
                "s_lengthscale",
                dist.Uniform(
                    torch.tensor(rnl_fast_sls[0]), torch.tensor(rnl_fast_sls[1])
                ),
            )
        components["regional_nl_fast"] = kernels.Product(rnl_fast_tem, rnl_fast_sp)

    # ------------------------------------------------------------------
    # Merged regional+local non-linear kernel (optional).
    # ------------------------------------------------------------------
    mnl_cfg = kernel_cfg.get("merged_nonlinear")
    if isinstance(mnl_cfg, dict) and _is_enabled(mnl_cfg):
        mnl_tem_type = mnl_cfg.get("temporal_type", "Matern32")
        mnl_tem = _build_kernel(
            mnl_tem_type,
            lengthscale=shared_lengthscale,
            geo=False,
        )
        mnl_tem.set_prior(
            "variance",
            dist.Uniform(
                torch.tensor(mnl_cfg.get("temporal_variance_prior", [1e-6, 1000.0])[0]),
                torch.tensor(mnl_cfg.get("temporal_variance_prior", [1e-6, 1000.0])[1]),
            ),
        )
        if hasattr(mnl_tem, "lengthscale"):
            mnl_tem.set_prior(
                "lengthscale",
                dist.Uniform(
                    torch.tensor(
                        mnl_cfg.get("temporal_lengthscale_prior", [1.0, 5000.0])[0]
                    ),
                    torch.tensor(
                        mnl_cfg.get("temporal_lengthscale_prior", [1.0, 5000.0])[1]
                    ),
                ),
            )
        mnl_sp_type = mnl_cfg.get("spatial_type", "Matern21")
        mnl_sp = _build_kernel(mnl_sp_type, geo=True)
        mnl_sls = mnl_cfg.get("spatial_lengthscale_prior", [0.001, 0.1])
        if hasattr(mnl_sp, "s_lengthscale"):
            mnl_sp.set_prior(
                "s_lengthscale",
                dist.Uniform(torch.tensor(mnl_sls[0]), torch.tensor(mnl_sls[1])),
            )
        components["merged_nl"] = kernels.Product(mnl_tem, mnl_sp)

    # ------------------------------------------------------------------
    # Local non-linear kernel (Matern32[m^2, yr] x Matern21[rad])
    # ------------------------------------------------------------------
    lnl_cfg = kernel_cfg.get("local_nonlinear", {}) or {}
    if _is_enabled(lnl_cfg):
        lnl_tem_type = lnl_cfg.get("temporal_type", "Matern32")
        lnl_tem = _build_kernel(
            lnl_tem_type,
            lengthscale=shared_lengthscale,
            geo=False,
        )
        lnl_tem.set_prior(
            "variance",
            dist.Uniform(
                torch.tensor(lnl_cfg.get("temporal_variance_prior", [1e-6, 1.0])[0]),
                torch.tensor(lnl_cfg.get("temporal_variance_prior", [1e-6, 1.0])[1]),
            ),
        )
        if hasattr(lnl_tem, "lengthscale"):
            lnl_tem.set_prior(
                "lengthscale",
                dist.Uniform(
                    torch.tensor(
                        lnl_cfg.get("temporal_lengthscale_prior", [100.0, 20000.0])[0]
                    ),
                    torch.tensor(
                        lnl_cfg.get("temporal_lengthscale_prior", [100.0, 20000.0])[1]
                    ),
                ),
            )
        lnl_sp_type = lnl_cfg.get("spatial_type", "Matern21")
        lnl_sp = _build_kernel(lnl_sp_type, geo=True)
        lnl_sls = lnl_cfg.get("spatial_lengthscale_prior", [0.001, 0.01])
        if hasattr(lnl_sp, "s_lengthscale"):
            lnl_sp.set_prior(
                "s_lengthscale",
                dist.Uniform(torch.tensor(lnl_sls[0]), torch.tensor(lnl_sls[1])),
            )
        components["local_nl"] = kernels.Product(lnl_tem, lnl_sp)

    # ------------------------------------------------------------------
    # Local non-linear FAST kernel (optional; shorter temporal lengthscale).
    # Same Matern32 x Matern21 structure as local_nonlinear. Absent or
    # enabled=false -> skipped.
    # ------------------------------------------------------------------
    lnl_fast_cfg = kernel_cfg.get("local_nonlinear_fast")
    if isinstance(lnl_fast_cfg, dict) and _is_enabled(lnl_fast_cfg):
        lnl_fast_tem_type = lnl_fast_cfg.get("temporal_type", "Matern32")
        lnl_fast_tem = _build_kernel(lnl_fast_tem_type, geo=False)
        lnl_fast_tem.set_prior(
            "variance",
            dist.Uniform(
                torch.tensor(
                    lnl_fast_cfg.get("temporal_variance_prior", [1e-5, 0.5])[0]
                ),
                torch.tensor(
                    lnl_fast_cfg.get("temporal_variance_prior", [1e-5, 0.5])[1]
                ),
            ),
        )
        if hasattr(lnl_fast_tem, "lengthscale"):
            lnl_fast_tem.set_prior(
                "lengthscale",
                dist.Uniform(
                    torch.tensor(
                        lnl_fast_cfg.get("temporal_lengthscale_prior", [50.0, 300.0])[0]
                    ),
                    torch.tensor(
                        lnl_fast_cfg.get("temporal_lengthscale_prior", [50.0, 300.0])[1]
                    ),
                ),
            )
        lnl_fast_sp_type = lnl_fast_cfg.get("spatial_type", "Matern21")
        lnl_fast_sp = _build_kernel(lnl_fast_sp_type, geo=True)
        lnl_fast_sls = lnl_fast_cfg.get("spatial_lengthscale_prior", [0.0005, 0.01])
        if hasattr(lnl_fast_sp, "s_lengthscale"):
            lnl_fast_sp.set_prior(
                "s_lengthscale",
                dist.Uniform(
                    torch.tensor(lnl_fast_sls[0]), torch.tensor(lnl_fast_sls[1])
                ),
            )
        components["local_nl_fast"] = kernels.Product(lnl_fast_tem, lnl_fast_sp)

    # ------------------------------------------------------------------
    # Site-specific datum correction (WhiteNoise_SP, m^2)
    # ------------------------------------------------------------------
    spwn_cfg = kernel_cfg.get("sp_whitenoise", {}) or {}
    if _is_enabled(spwn_cfg):
        spwn_type = spwn_cfg.get("type", "WhiteNoise_SP")
        sp_whitenoise_kernel = _build_kernel(spwn_type, sp=False, geo=True)
        spwn_var = spwn_cfg.get("variance_prior", [1e-7, 1.0])
        if hasattr(sp_whitenoise_kernel, "variance"):
            sp_whitenoise_kernel.set_prior(
                "variance",
                dist.Uniform(torch.tensor(spwn_var[0]), torch.tensor(spwn_var[1])),
            )
        components["sp_whitenoise"] = sp_whitenoise_kernel

    # ------------------------------------------------------------------
    # Observational white noise (m^2)
    # ------------------------------------------------------------------
    wn_cfg = kernel_cfg.get("whitenoise", {}) or {}
    if _is_enabled(wn_cfg):
        wn_type = wn_cfg.get("type", "WhiteNoise")
        whitenoise_kernel = _build_kernel(wn_type)
        wn_var = wn_cfg.get("variance_prior", [1e-7, 1.0])
        if hasattr(whitenoise_kernel, "variance"):
            whitenoise_kernel.set_prior(
                "variance",
                dist.Uniform(torch.tensor(wn_var[0]), torch.tensor(wn_var[1])),
            )
        components["whitenoise"] = whitenoise_kernel

    if not components:
        raise ValueError(
            "build_spatiotemporal_kernel: no kernel components are enabled. "
            "At least one of gp_model.kernels.* must have enabled=true."
        )

    # Fold the enabled components into a left-nested Sum tree, matching the
    # canonical order. The tree shape is not load-bearing anymore because we
    # also expose ``component_kernels`` as a direct lookup on the returned
    # object, but preserving left-nesting keeps legacy consumers working for
    # the common case where every kernel is enabled.
    ordered_kernels = list(components.values())
    combined = ordered_kernels[0]
    for kern in ordered_kernels[1:]:
        combined = kernels.Sum(combined, kern)

    combined.component_kernels = components
    return combined
