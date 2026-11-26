"""Tests for YAML-relative input path resolution on config load."""

from __future__ import annotations

from pathlib import Path

import yaml

from stgp_dashboard.services.config_builder import build_config
from stgp_dashboard.services.config_loader import (
    config_dict_to_state,
    load_config_file,
    summarize_resolved_inputs,
)
from stgp_dashboard.state import DEFAULT_STATE


def _write_cfg(path: Path, paths: dict, extra: dict | None = None) -> Path:
    payload = {"paths": paths, "gia": {}}
    if extra:
        payload.update(extra)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_relative_inputs_resolve_against_yaml_dir(tmp_path: Path):
    # Mirror nested study layout: study/configs/../Data/paleo.csv and ../../Data/GIA
    configs = tmp_path / "studies" / "region_a" / "configs"
    configs.mkdir(parents=True)
    paleo = tmp_path / "studies" / "region_a" / "Data" / "combined.csv"
    paleo.parent.mkdir(parents=True)
    paleo.write_text("x\n", encoding="utf-8")
    gia_mat = tmp_path / "Data" / "GIA"
    gia_anu = tmp_path / "Data" / "ANU_1D"
    gia_3d = tmp_path / "studies" / "Data" / "3D_GIA"
    gia_mat.mkdir(parents=True)
    gia_anu.mkdir(parents=True)
    gia_3d.mkdir(parents=True)

    cfg_path = _write_cfg(
        configs / "run.yaml",
        {
            "paleo_rsl": "../Data/combined.csv",
            "gia_mat_dir": "../../../Data/GIA",
            "gia_ANU_dir": "../../../Data/ANU_1D",
            "gia_3d_dir": "../../Data/3D_GIA",
            "output_dir": "../../Outputs/my_run",
        },
    )

    state = load_config_file(cfg_path)

    assert Path(state["paleo_rsl"]) == paleo.resolve()
    assert Path(state["gia_mat_dir"]) == gia_mat.resolve()
    assert Path(state["gia_anu_dir"]) == gia_anu.resolve()
    assert Path(state["gia_3d_dir"]) == gia_3d.resolve()
    assert Path(state["paleo_rsl"]).is_file()
    assert Path(state["gia_mat_dir"]).is_dir()

    # output_dir must stay as the raw relative string from YAML
    assert state["output_dir"] == "../../Outputs/my_run"

    summary = summarize_resolved_inputs(state)
    assert "Paleo RSL" in summary["found"]
    assert "GIA MAT dir" in summary["found"]
    assert summary["missing"] == []


def test_absolute_input_paths_unchanged(tmp_path: Path):
    paleo = tmp_path / "data.csv"
    paleo.write_text("x\n", encoding="utf-8")
    gia = tmp_path / "gia"
    gia.mkdir()
    cfg_path = _write_cfg(
        tmp_path / "cfg.yaml",
        {
            "paleo_rsl": str(paleo),
            "gia_mat_dir": str(gia),
            "output_dir": "relative_out",
        },
    )

    state = load_config_file(cfg_path)
    assert Path(state["paleo_rsl"]) == paleo.resolve()
    assert Path(state["gia_mat_dir"]) == gia.resolve()
    assert state["output_dir"] == "relative_out"


def test_missing_relative_input_still_absolute_candidate(tmp_path: Path):
    configs = tmp_path / "configs"
    configs.mkdir()
    cfg_path = _write_cfg(
        configs / "missing.yaml",
        {"paleo_rsl": "../Data/nope.csv", "output_dir": "../out"},
    )

    state = load_config_file(cfg_path)
    resolved = Path(state["paleo_rsl"])
    assert resolved.is_absolute()
    assert resolved == (configs / "../Data/nope.csv").resolve()
    assert not resolved.exists()
    assert state["output_dir"] == "../out"

    summary = summarize_resolved_inputs(state)
    assert "Paleo RSL" in summary["missing"]


def test_config_dict_to_state_without_base_dir_keeps_relative():
    state = config_dict_to_state(
        {"paths": {"paleo_rsl": "../Data/x.csv", "output_dir": "../out"}}
    )
    assert state["paleo_rsl"] == "../Data/x.csv"
    assert state["output_dir"] == "../out"


def test_existing_checkpoint_does_not_force_load_mode(tmp_path: Path):
    """run_optimization:true + on-disk .pt must keep training_mode=train."""
    paleo = tmp_path / "data.csv"
    paleo.write_text("x\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    run_dir = output_dir / "run"
    run_dir.mkdir(parents=True)
    ckpt = run_dir / "trained_param_store.pt"
    ckpt.write_bytes(b"fake")

    cfg_path = _write_cfg(
        tmp_path / "train.yaml",
        {
            "paleo_rsl": str(paleo),
            "output_dir": str(output_dir),
        },
        extra={"gp_model": {"run_optimization": True, "kernels": {}}},
    )

    state = load_config_file(cfg_path)
    assert state["training_mode"] == "train"
    assert state["checkpoint_available"] is True
    assert Path(state["param_store_path"]) == ckpt.resolve()

    merged = {**DEFAULT_STATE, **state}
    cfg = build_config(merged)
    assert cfg["gp_model"]["run_optimization"] is True
    assert not cfg["paths"].get("trained_param_store")


def test_explicit_run_optimization_false_loads_checkpoint(tmp_path: Path):
    """YAML run_optimization:false must select load_checkpoint mode."""
    paleo = tmp_path / "data.csv"
    paleo.write_text("x\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    run_dir = output_dir / "run"
    run_dir.mkdir(parents=True)
    ckpt = run_dir / "trained_param_store.pt"
    ckpt.write_bytes(b"fake")

    cfg_path = _write_cfg(
        tmp_path / "load.yaml",
        {
            "paleo_rsl": str(paleo),
            "output_dir": str(output_dir),
            "trained_param_store": str(ckpt),
        },
        extra={"gp_model": {"run_optimization": False, "kernels": {}}},
    )

    state = load_config_file(cfg_path)
    assert state["training_mode"] == "load_checkpoint"
    assert Path(state["param_store_path"]) == ckpt.resolve()

    merged = {**DEFAULT_STATE, **state}
    cfg = build_config(merged)
    assert cfg["gp_model"]["run_optimization"] is False
    assert Path(cfg["paths"]["trained_param_store"]) == ckpt.resolve()
