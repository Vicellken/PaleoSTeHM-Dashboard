"""
GP model setup, optimization, and hyperparameter extraction.

Includes systematic kernel tree traversal that replaces fragile
hardcoded ``.kern0/.kern1`` chains.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyro
import torch

import PSTHM
from PSTHM import kernels_factory

from .sensitivity.workflow_common import _load_param_store
from .types import TrainedModel


# Canonical kernel names in the order produced by build_spatiotemporal_kernel.
# When every component is enabled, the factory builds a left-nested Sum:
#   Sum(Sum(Sum(Sum(Sum(global, regional_linear), regional_nl),
#                     local_nl), sp_whitenoise), whitenoise)
# Peeling kern1 off iteratively walks the tree in reverse order. With the
# ``enabled`` flag, the tree shape becomes variable, so we prefer the
# ``component_kernels`` attribute attached by the factory and only fall back
# to tree traversal for legacy kernels built without that attribute.
KERNEL_NAMES = [
    "whitenoise",
    "sp_whitenoise",
    "local_nl",
    "regional_nl",
    "regional_linear",
    "global_temporal",
]

# Signal (non-noise) components used for GP decomposition plots.
_SIGNAL_COMPONENT_ORDER = [
    ("global_temporal", "global"),
    ("global_temporal_fast", "global_fast"),
    ("regional_linear", "regional_linear"),
    ("regional_nl", "regional_nl"),
    ("regional_nl_fast", "regional_nl_fast"),
    ("merged_nl", "merged_nl"),
    ("local_nl", "local_nl"),
    ("local_nl_fast", "local_nl_fast"),
]


def _extract_kernel_components(combined_kernel: Any) -> Dict[str, Any]:
    """
    Return the ordered name -> kernel map for ``combined_kernel``.

    If the factory attached a ``component_kernels`` dict (the new path), use
    it directly. Otherwise fall back to the legacy left-nested traversal,
    which assumes every one of the six canonical components is present.
    """
    explicit = getattr(combined_kernel, "component_kernels", None)
    if isinstance(explicit, dict):
        return dict(explicit)

    components: Dict[str, Any] = {}
    node = combined_kernel
    for name in KERNEL_NAMES[:-1]:
        components[name] = node.kern1
        node = node.kern0
    components[KERNEL_NAMES[-1]] = node
    return components


def build_and_train(
    cfg: Dict[str, Any],
    X: torch.Tensor,
    y_residual: torch.Tensor,
    noise: torch.Tensor,
    age_sigma: torch.Tensor,
    run_key: str = "stgp_with_gia",
    train: bool = True,
    param_store_path: Optional[str] = None,
    progress_callback=None,
) -> TrainedModel:
    """
    Build the 6-kernel GP, optionally train via SVI, and return a TrainedModel.

    Parameters
    ----------
    cfg : dict
        Full YAML config.
    X : Tensor (N, 3)
        Training inputs [age_bp, lat, lon].
    y_residual : Tensor (N,)
        Residual observations (data - GIA mean).
    noise : Tensor (N, N)
        Total noise matrix.
    age_sigma : Tensor (N,)
        Age uncertainties.
    run_key : str
        Key into cfg["runs"] for hyperparameters.
    train : bool
        If True, run SVI optimization. If False, load from param_store_path.
    param_store_path : str, optional
        Path to Pyro param store checkpoint (for load mode).

    Returns
    -------
    TrainedModel
    """
    gp_cfg = cfg.get("gp_model", {}) or {}
    run_cfg = (cfg.get("runs", {}) or {}).get(run_key, {}) or {}

    def _hp(k, default=None):
        if k in run_cfg:
            return run_cfg[k]
        if k in gp_cfg:
            return gp_cfg[k]
        return default

    # Validate kernel config. Sections may carry ``enabled: false`` to remove
    # a kernel from the stack, but they must still be present so that defaults
    # and priors are explicit in the config.
    required_keys = {
        "global_temporal",
        "regional_linear",
        "regional_nonlinear",
        "local_nonlinear",
        "sp_whitenoise",
        "whitenoise",
    }
    kernel_cfg = gp_cfg.get("kernels", {}) or {}
    missing = required_keys - set(kernel_cfg.keys())
    if missing:
        raise KeyError(f"Missing gp_model.kernels entries: {sorted(missing)}")

    disabled = sorted(
        name
        for name, block in kernel_cfg.items()
        if isinstance(block, dict) and block.get("enabled") is False
    )
    if disabled:
        print(f"Disabled kernel components (via enabled: false): {disabled}")

    # Seed before kernel/GPR construction so that scenario order does not
    # affect initialisation state (seed → build → SVI).
    if train and gp_cfg.get("fixed_seed", False):
        seed = int(gp_cfg.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pyro.set_rng_seed(seed)
        print(f"Random seed fixed to {seed}")

    # Build kernel
    combined_kernel = kernels_factory.build_spatiotemporal_kernel(gp_cfg)
    print("Kernel structure:")
    print(combined_kernel)

    # Build GP model
    pyro.clear_param_store()
    gpr = PSTHM.model.GPRegression_V(
        X,
        y_residual,
        combined_kernel,
        noise=noise,
        jitter=float(_hp("jitter", 1e-5)),
    )
    print(f"Training points: {X.shape[0]}")

    track_list = None
    if train:
        st = time.time()
        gpr, track_list = PSTHM.opti.SVI_NI_optm(
            gpr,
            x_sigma=age_sigma,
            num_iteration=int(_hp("num_iterations", 1000)),
            lr=float(_hp("learning_rate", 0.3)),
            decay_r=float(_hp("decay_r", 0.8)),
            step_size=int(_hp("step_size", 25)),
            gpu=bool(_hp("gpu", False)),
            progress_callback=progress_callback,
            show_progress=progress_callback is None,
        )
        gpr.set_mode("guide")
        print(f"Training time: {time.time() - st:.1f} s")
    elif param_store_path is not None:
        _load_param_store(param_store_path)
        gpr.set_mode("guide")
        print(f"Loaded param store from {param_store_path}")

    # Extract named kernel components
    kernel_components = _extract_kernel_components(gpr.kernel)

    # Decomposition kernels (the signal kernels, excluding noise). Only include
    # components that are actually present after the ``enabled`` filtering.
    decomp_kernels = []
    comp_labels: List[str] = []
    for name, label in _SIGNAL_COMPONENT_ORDER:
        if name in kernel_components:
            decomp_kernels.append(kernel_components[name])
            comp_labels.append(label)

    return TrainedModel(
        gpr=gpr,
        kernel_components=kernel_components,
        decomp_kernels=decomp_kernels,
        comp_labels=comp_labels,
        track_list=track_list,
    )


def extract_hyperparams(trained: TrainedModel) -> pd.DataFrame:
    """
    Extract optimized hyperparameters as a readable DataFrame.

    Only rows for components that are present in ``trained.kernel_components``
    are emitted, so kernels disabled via ``enabled: false`` in the YAML simply
    do not appear in the output.

    Returns
    -------
    pd.DataFrame with columns ["Parameter", "Value"]
    """
    kc = trained.kernel_components
    EARTH_R = 6371  # km

    rows: List[Tuple[str, float]] = []

    if "global_temporal" in kc:
        rows.append(("K_G  lengthscale (yr)", kc["global_temporal"].lengthscale.item()))
        rows.append(
            ("K_G  variance (m)", np.sqrt(kc["global_temporal"].variance.item()))
        )

    if "global_temporal_fast" in kc:
        rows.append(
            (
                "K_G_FAST lengthscale (yr)",
                kc["global_temporal_fast"].lengthscale.item(),
            )
        )
        rows.append(
            (
                "K_G_FAST variance (m)",
                np.sqrt(kc["global_temporal_fast"].variance.item()),
            )
        )

    if "regional_linear" in kc:
        rows.append(
            (
                "K_RL temporal variance (m/yr)",
                np.sqrt(kc["regional_linear"].kern0.variance.item()),
            )
        )
        rows.append(
            (
                "K_RL spatial s_lengthscale (km)",
                kc["regional_linear"].kern1.s_lengthscale.item() * EARTH_R,
            )
        )

    if "regional_nl" in kc:
        rows.append(
            (
                "K_RNL temporal variance (m)",
                np.sqrt(kc["regional_nl"].kern0.variance.item()),
            )
        )
        rows.append(
            (
                "K_RNL temporal lengthscale (yr)",
                kc["regional_nl"].kern0.lengthscale.item(),
            )
        )
        rows.append(
            (
                "K_RNL spatial s_lengthscale (km)",
                kc["regional_nl"].kern1.s_lengthscale.item() * EARTH_R,
            )
        )

    if "regional_nl_fast" in kc:
        rows.append(
            (
                "K_RNL_FAST temporal variance (m)",
                np.sqrt(kc["regional_nl_fast"].kern0.variance.item()),
            )
        )
        rows.append(
            (
                "K_RNL_FAST temporal lengthscale (yr)",
                kc["regional_nl_fast"].kern0.lengthscale.item(),
            )
        )
        rows.append(
            (
                "K_RNL_FAST spatial s_lengthscale (km)",
                kc["regional_nl_fast"].kern1.s_lengthscale.item() * EARTH_R,
            )
        )

    if "merged_nl" in kc:
        rows.append(
            (
                "K_MNL temporal variance (m)",
                np.sqrt(kc["merged_nl"].kern0.variance.item()),
            )
        )
        rows.append(
            (
                "K_MNL temporal lengthscale (yr)",
                kc["merged_nl"].kern0.lengthscale.item(),
            )
        )
        rows.append(
            (
                "K_MNL spatial s_lengthscale (km)",
                kc["merged_nl"].kern1.s_lengthscale.item() * EARTH_R,
            )
        )

    if "local_nl" in kc:
        rows.append(
            (
                "K_LNL temporal variance (m)",
                np.sqrt(kc["local_nl"].kern0.variance.item()),
            )
        )
        rows.append(
            (
                "K_LNL temporal lengthscale (yr)",
                kc["local_nl"].kern0.lengthscale.item(),
            )
        )
        rows.append(
            (
                "K_LNL spatial s_lengthscale (km)",
                kc["local_nl"].kern1.s_lengthscale.item() * EARTH_R,
            )
        )

    if "local_nl_fast" in kc:
        rows.append(
            (
                "K_LNL_FAST temporal variance (m)",
                np.sqrt(kc["local_nl_fast"].kern0.variance.item()),
            )
        )
        rows.append(
            (
                "K_LNL_FAST temporal lengthscale (yr)",
                kc["local_nl_fast"].kern0.lengthscale.item(),
            )
        )
        rows.append(
            (
                "K_LNL_FAST spatial s_lengthscale (km)",
                kc["local_nl_fast"].kern1.s_lengthscale.item() * EARTH_R,
            )
        )

    if "sp_whitenoise" in kc:
        rows.append(
            ("K_SWN variance (m)", np.sqrt(kc["sp_whitenoise"].variance.item()))
        )

    if "whitenoise" in kc:
        rows.append(("K_WN  variance (m)", np.sqrt(kc["whitenoise"].variance.item())))

    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def negative_log_likelihhod(gpr: Any) -> torch.Tensor:
    """
    Compute the negative log-likelihood for a trained GP regression model.

    Notes
    -----
    Function name keeps historical spelling for notebook compatibility with
    existing workflows.
    """
    n = len(gpr.X)
    eye = torch.eye(n, dtype=torch.double, device=gpr.X.device)
    k_noisy = gpr.kernel(gpr.X).double() + eye * gpr.jitter + gpr.noise
    chol = torch.linalg.cholesky(k_noisy)
    chol_inv = torch.linalg.inv(chol)
    k_inv = chol_inv.T @ chol_inv

    term1 = -0.5 * (gpr.y @ k_inv @ gpr.y)
    term2 = -torch.sum(torch.log(torch.diag(chol)))
    term3 = -0.5 * n * torch.log(torch.tensor(2 * torch.pi, dtype=torch.double))

    log_likelihood = term1 + term2 + term3
    return -log_likelihood
