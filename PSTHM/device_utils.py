"""Device utilities for CUDA-aware GP training."""

from __future__ import annotations

import torch


def move_gpr_to_device(gpr, device: torch.device):
    """Move a GPRegression_V model and its training data tensors to *device*.

    ``GPRegression_V`` inherits from ``torch.nn.Module`` via Pyro's GP stack.
    Calling ``.to(device)`` moves registered parameters (kernel hyperparameters,
    etc.) but not the plain tensor attributes ``gpr.X``, ``gpr.y``, and
    ``gpr.noise``, which must be relocated manually.
    """
    gpr = gpr.to(device)

    for attr in ("X", "y", "noise"):
        t = getattr(gpr, attr, None)
        if t is not None and isinstance(t, torch.Tensor):
            setattr(gpr, attr, t.to(device))

    return gpr


def move_gpr_to_cpu(gpr):
    """Move *gpr* back to CPU after GPU training."""
    return move_gpr_to_device(gpr, torch.device("cpu"))
