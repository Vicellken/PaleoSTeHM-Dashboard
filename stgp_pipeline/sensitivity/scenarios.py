"""
Scenario definitions for sensitivity workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .config_mutations import multiplier_tag


BOUNDS_MULTIPLIERS = (1.5, 2.0, 3.0)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    workflow: str
    category: str
    description: str
    kernel_enabled: Dict[str, bool] | None = None
    merged_nonlinear_enabled: bool | None = None
    merged_nonlinear_prior_ranges: Mapping[str, Any] | None = None
    bounds_multiplier: float | None = None

    def to_manifest_dict(self) -> dict:
        return {
            "name": self.name,
            "workflow": self.workflow,
            "category": self.category,
            "description": self.description,
            "kernel_enabled": dict(self.kernel_enabled or {}),
            "merged_nonlinear_enabled": self.merged_nonlinear_enabled,
            "merged_nonlinear_prior_ranges": dict(
                self.merged_nonlinear_prior_ranges or {}
            ),
            "bounds_multiplier": self.bounds_multiplier,
        }


def _kernel_boundary_test_scenarios(workflow: str) -> List[ScenarioSpec]:
    """
    Kernel-structure scenarios for the boundary-test phase.

    Base config is test_16ka_without_gia.yaml or test_4ka_with_gia.yaml (all
    kernels enabled). The adopted scenario (defined separately) disables
    regional_nonlinear; all_kernels_on is an unmodified copy of the base YAML.
    """
    return [
        ScenarioSpec(
            name="all_kernels_on",
            workflow=workflow,
            category="kernel",
            description=(
                "Unmodified full kernel stack (equivalent to "
                "test_16ka_without_gia.yaml)."
            ),
        ),
        ScenarioSpec(
            name="no_local_nonlinear",
            workflow=workflow,
            category="kernel",
            description="Disable local nonlinear kernel component.",
            kernel_enabled={"local_nonlinear": False},
        ),
        ScenarioSpec(
            name="no_regional_local_nonlinear",
            workflow=workflow,
            category="kernel",
            description="Disable both regional and local nonlinear kernels.",
            kernel_enabled={"regional_nonlinear": False, "local_nonlinear": False},
        ),
        ScenarioSpec(
            name="regional_linear_off",
            workflow=workflow,
            category="regional-linear",
            description="Disable regional linear kernel component.",
            kernel_enabled={"regional_linear": False},
        ),
    ]


def _kernel_toggle_scenarios(workflow: str) -> List[ScenarioSpec]:
    """Single-kernel disable scenarios for boundary-hit sensitivity (no merged kernels)."""
    return [
        ScenarioSpec(
            name="no_regional_nonlinear",
            workflow=workflow,
            category="kernel",
            description="Disable regional nonlinear kernel component.",
            kernel_enabled={"regional_nonlinear": False},
        ),
        ScenarioSpec(
            name="no_local_nonlinear",
            workflow=workflow,
            category="kernel",
            description="Disable local nonlinear kernel component.",
            kernel_enabled={"local_nonlinear": False},
        ),
        ScenarioSpec(
            name="no_regional_local_nonlinear",
            workflow=workflow,
            category="kernel",
            description="Disable both regional and local nonlinear kernels.",
            kernel_enabled={"regional_nonlinear": False, "local_nonlinear": False},
        ),
        ScenarioSpec(
            name="regional_linear_off",
            workflow=workflow,
            category="regional-linear",
            description="Disable regional linear kernel component.",
            kernel_enabled={"regional_linear": False},
        ),
    ]


def _bounds_scenarios(
    workflow: str, multipliers: Iterable[float]
) -> List[ScenarioSpec]:
    return [
        ScenarioSpec(
            name=f"bounds_m{multiplier_tag(m)}",
            workflow=workflow,
            category="bounds",
            description=f"Scale all enabled kernel bounds by multiplier={m}",
            bounds_multiplier=float(m),
        )
        for m in multipliers
    ]


def build_workflow_scenarios(
    workflow: str,
    scenario_selector: str = "all",
    *,
    multipliers: Iterable[float] = BOUNDS_MULTIPLIERS,
) -> List[ScenarioSpec]:
    """
    Build the scenario list for `workflow`.

    Parameters
    ----------
    workflow
        One of: "16ka", "4ka", "3d-gia", "regional-gia".
    scenario_selector
        16ka/4ka: one of "adopted", "regional-linear", "kernel-toggles", "bounds", "all".
        3d-gia / regional-gia: one of "adopted", "merged-nonlinear",
        "kernel-toggles", "bounds", "all".
    """
    selector = scenario_selector.strip().lower()
    if workflow not in {"16ka", "4ka", "3d-gia", "regional-gia"}:
        raise ValueError(f"Unsupported workflow: {workflow!r}")

    if workflow in {"3d-gia", "regional-gia"}:
        if selector not in {
            "adopted",
            "merged-nonlinear",
            "kernel-toggles",
            "bounds",
            "all",
        }:
            raise ValueError(f"Unsupported scenario selector: {scenario_selector!r}")
    elif selector not in {
        "adopted",
        "regional-linear",
        "kernel-toggles",
        "bounds",
        "all",
    }:
        raise ValueError(f"Unsupported scenario selector: {scenario_selector!r}")

    adopted_spec = ScenarioSpec(
        name="adopted",
        workflow=workflow,
        category="adopted",
        description="Unmodified adopted configuration.",
    )

    bounds = _bounds_scenarios(workflow, multipliers)

    if workflow in {"3d-gia", "regional-gia"}:
        kernel_toggles = _kernel_toggle_scenarios(workflow)
        merged_regional_local_nonlinear = ScenarioSpec(
            name="merged_regional_local_nonlinear",
            workflow=workflow,
            category="kernel",
            description=(
                "Disable regional_nonlinear_fast and use one merged "
                "regional+local nonlinear kernel."
            ),
            kernel_enabled={"regional_nonlinear_fast": False},
            merged_nonlinear_enabled=True,
            # Prior ranges come from the YAML's merged_nonlinear block; no override here.
            merged_nonlinear_prior_ranges=None,
        )
        kernel_suite = [*kernel_toggles, merged_regional_local_nonlinear]
        if selector == "adopted":
            return [adopted_spec]
        if selector == "merged-nonlinear":
            return [adopted_spec, merged_regional_local_nonlinear]
        if selector == "kernel-toggles":
            return [adopted_spec, *kernel_toggles]
        if selector == "bounds":
            return [adopted_spec, *bounds]
        return [adopted_spec, *kernel_suite, *bounds]

    if workflow == "16ka":
        kernel_toggles = _kernel_toggle_scenarios(workflow)
        regional_linear_off = next(
            s for s in kernel_toggles if s.name == "regional_linear_off"
        )
        merged_regional_local_nonlinear = ScenarioSpec(
            name="merged_regional_local_nonlinear",
            workflow=workflow,
            category="kernel",
            description=(
                "Use one merged regional+local nonlinear kernel, keep regional linear."
            ),
            merged_nonlinear_enabled=True,
        )
        merged_regional_local_nonlinear_regional_linear_off = ScenarioSpec(
            name="merged_regional_local_nonlinear_regional_linear_off",
            workflow=workflow,
            category="kernel",
            description=(
                "Use merged regional+local nonlinear kernel and disable regional linear."
            ),
            merged_nonlinear_enabled=True,
            kernel_enabled={"regional_linear": False},
        )
        kernel_suite = [
            *kernel_toggles,
            merged_regional_local_nonlinear,
            merged_regional_local_nonlinear_regional_linear_off,
        ]

        if selector == "adopted":
            return [adopted_spec]
        if selector == "regional-linear":
            return [adopted_spec, regional_linear_off]
        if selector == "kernel-toggles":
            adopted_boundary = ScenarioSpec(
                name="adopted",
                workflow=workflow,
                category="adopted",
                description="Adopted structure: regional_nonlinear disabled.",
                kernel_enabled={"regional_nonlinear": False},
            )
            return [adopted_boundary, *_kernel_boundary_test_scenarios(workflow)]
        if selector == "bounds":
            return [adopted_spec, *bounds]
        return [adopted_spec, *kernel_suite, *bounds]

    # 4ka
    if selector == "regional-linear":
        raise ValueError(
            f"Scenario selector {scenario_selector!r} is only valid for workflow=16ka"
        )
    if selector == "adopted":
        return [adopted_spec]
    if selector == "kernel-toggles":
        adopted_boundary = ScenarioSpec(
            name="adopted",
            workflow=workflow,
            category="adopted",
            description="Adopted structure: regional_nonlinear disabled.",
            kernel_enabled={"regional_nonlinear": False},
        )
        return [adopted_boundary, *_kernel_boundary_test_scenarios(workflow)]
    if selector == "bounds":
        return [adopted_spec, *bounds]
    return [adopted_spec, *bounds]


def filter_scenarios_for_cfg(
    scenarios: Sequence[ScenarioSpec],
    cfg: Mapping[str, Any],
) -> List[ScenarioSpec]:
    """
    Drop scenarios that are incompatible with the given config dict.

    Rules:
    - A scenario that disables ``regional_linear`` is dropped when
      ``gp_model.kernels.regional_linear.enabled`` is already ``False`` in the
      YAML (the toggle would be a no-op and the scenario is misleading).
    - A scenario that disables ``regional_nonlinear`` is dropped when it is
      already ``False`` in the YAML.
    - A scenario that disables ``local_nonlinear`` is dropped when it is
      already ``False`` in the YAML.
    - A scenario that enables ``merged_nonlinear`` is dropped when the YAML has
      no ``gp_model.kernels.merged_nonlinear`` block with prior ranges defined
      (there are no priors to use, so the scenario cannot be meaningful).
    """
    kernels = (cfg.get("gp_model") or {}).get("kernels") or {}

    regional_linear_enabled = bool(
        (kernels.get("regional_linear") or {}).get("enabled", True)
    )
    regional_nonlinear_enabled = bool(
        (kernels.get("regional_nonlinear") or {}).get("enabled", True)
    )
    local_nonlinear_enabled = bool(
        (kernels.get("local_nonlinear") or {}).get("enabled", True)
    )

    merged_block = kernels.get("merged_nonlinear") or {}
    has_merged_priors = isinstance(merged_block, Mapping) and any(
        str(k).endswith("_prior") for k in merged_block
    )

    out: List[ScenarioSpec] = []
    for s in scenarios:
        toggles = s.kernel_enabled or {}
        if (toggles.get("regional_linear") is False) and (not regional_linear_enabled):
            continue
        if (toggles.get("regional_nonlinear") is False) and (
            not regional_nonlinear_enabled
        ):
            continue
        if (toggles.get("local_nonlinear") is False) and (not local_nonlinear_enabled):
            # Keep combined toggles that also disable regional_nonlinear.
            if toggles.get("regional_nonlinear") is not False:
                continue
        if s.merged_nonlinear_enabled and not has_merged_priors:
            continue
        out.append(s)
    return out
