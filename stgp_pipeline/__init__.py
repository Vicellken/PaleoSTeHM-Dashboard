"""
stgp_pipeline — Reusable modules for the STGP + GIA workflow.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

_MODULE_EXPORTS = {
    "data",
    "gia_setup",
    "gia_setup_3d",
    "training",
    "predict",
    "export",
    "visualize",
    "budget",
    "sensitivity",
    "spatial_summaries",
}
_TYPE_EXPORTS = {
    "RSLData",
    "GIAResult",
    "ResidualData",
    "TrainedModel",
    "RegionResult",
    "SpatialSlice",
    "SpatialRateResult",
    "RegionalMeanRateResult",
}

__all__ = [
    "data",
    "gia_setup",
    "gia_setup_3d",
    "training",
    "predict",
    "export",
    "visualize",
    "budget",
    "sensitivity",
    "spatial_summaries",
    "RSLData",
    "GIAResult",
    "ResidualData",
    "TrainedModel",
    "RegionResult",
    "SpatialSlice",
    "SpatialRateResult",
    "RegionalMeanRateResult",
]


def __getattr__(name: str):
    if name in _MODULE_EXPORTS:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _TYPE_EXPORTS:
        types_mod = importlib.import_module(".types", __name__)
        value = getattr(types_mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
