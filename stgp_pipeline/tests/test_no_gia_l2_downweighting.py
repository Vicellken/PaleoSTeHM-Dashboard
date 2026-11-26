from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from PSTHM.data_adjustments import (
    apply_level2_downweighting,
    apply_level_filter,
    read_adjusted_rsl_dataframe,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Level": [1, 2, 2, 2],
            "Limiting": [0, 0, -1, 1],
            "RSLer_up_2sd": [1.0, 2.0, 3.0, 4.0],
            "RSLer_low_2sd": [1.1, 2.1, 3.1, 4.1],
            "Age_up_er_2sd": [10.0, 20.0, 30.0, 40.0],
            "Age_low_er_2sd": [11.0, 21.0, 31.0, 41.0],
        }
    )


def test_read_adjusted_rsl_dataframe_matches_apply_on_disk(tmp_path):
    df = _sample_df()
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    cfg = {
        "data_adjustments": {
            "level2_downweighting": {
                "enabled": True,
                "multiplier": 3.0,
                "scope": "slip_only",
                "scale_targets": ["rsl", "age"],
            }
        }
    }
    out_disk, stats_disk = read_adjusted_rsl_dataframe(str(path), cfg)
    out_mem, stats_mem = apply_level2_downweighting(df, cfg)
    pd.testing.assert_frame_equal(
        out_disk.reset_index(drop=True), out_mem.reset_index(drop=True)
    )
    assert stats_disk["selected_rows"] == stats_mem["selected_rows"]


def test_slip_only_scales_only_level2_slip_rows():
    df = _sample_df()
    cfg = {
        "data_adjustments": {
            "level2_downweighting": {
                "enabled": True,
                "multiplier": 3.0,
                "scope": "slip_only",
                "scale_targets": ["rsl", "age"],
            }
        }
    }

    out, stats = apply_level2_downweighting(df, cfg)

    # Only index 1 (Level=2, Limiting=0) should be scaled.
    assert out.loc[1, "RSLer_up_2sd"] == pytest.approx(6.0)
    assert out.loc[1, "RSLer_low_2sd"] == pytest.approx(6.3)
    assert out.loc[1, "Age_up_er_2sd"] == pytest.approx(60.0)
    assert out.loc[1, "Age_low_er_2sd"] == pytest.approx(63.0)

    # Limiting rows (indices 2 and 3) remain unchanged under slip_only.
    assert out.loc[2, "RSLer_up_2sd"] == pytest.approx(3.0)
    assert out.loc[3, "Age_up_er_2sd"] == pytest.approx(40.0)
    assert stats["selected_rows"] == 1


def test_all_scope_scales_level2_limiting_rows_too():
    df = _sample_df()
    cfg = {
        "data_adjustments": {
            "level2_downweighting": {
                "enabled": True,
                "multiplier": 2.0,
                "scope": "all",
                "scale_targets": ["rsl"],
            }
        }
    }

    out, stats = apply_level2_downweighting(df, cfg)

    # Level=2 rows across all limiting types are scaled for RSL columns.
    assert out.loc[1, "RSLer_up_2sd"] == pytest.approx(4.0)
    assert out.loc[2, "RSLer_low_2sd"] == pytest.approx(6.2)
    assert out.loc[3, "RSLer_up_2sd"] == pytest.approx(8.0)
    # Age columns untouched since scale_targets=["rsl"].
    assert out.loc[2, "Age_up_er_2sd"] == pytest.approx(30.0)
    assert stats["selected_rows"] == 3


def test_level_filter_drops_unlisted_levels():
    df = pd.DataFrame({"Level": [1, 2, 3, 2], "RSL": [0.0, 1.0, 2.0, 3.0]})
    cfg = {"data_filter": {"levels": [1, 2]}}

    out, stats = apply_level_filter(df, cfg)

    assert list(out["Level"]) == [1, 2, 2]
    assert stats["enabled"] is True
    assert stats["levels"] == [1, 2]
    assert stats["rows_before"] == 4
    assert stats["rows_after"] == 3


def test_level_filter_missing_config_is_noop():
    df = pd.DataFrame({"Level": [1, 2, 3], "RSL": [0.0, 1.0, 2.0]})

    out, stats = apply_level_filter(df, {})

    pd.testing.assert_frame_equal(out, df)
    assert stats["enabled"] is False


def test_level_filter_train_levels_when_levels_absent():
    df = pd.DataFrame({"Level": [1, 2, 3], "RSL": [0.0, 1.0, 2.0]})
    cfg = {"data_filter": {"train_levels": [1], "prediction_levels": [1, 2]}}

    out_train, stats_train = apply_level_filter(df, cfg, purpose="train")
    out_pred, stats_pred = apply_level_filter(df, cfg, purpose="predict")

    assert list(out_train["Level"]) == [1]
    assert stats_train["enabled"] is True
    assert stats_train["levels"] == [1]

    assert list(out_pred["Level"]) == [1, 2]
    assert stats_pred["enabled"] is True
    assert stats_pred["levels"] == [1, 2]


def test_level_filter_explicit_levels_overrides_train_levels():
    df = pd.DataFrame({"Level": [1, 2, 3], "RSL": [0.0, 1.0, 2.0]})
    cfg = {"data_filter": {"levels": [2], "train_levels": [1]}}

    out, stats = apply_level_filter(df, cfg, purpose="train")

    assert list(out["Level"]) == [2]
    assert stats["levels"] == [2]


def test_level_filter_empty_result_raises():
    df = pd.DataFrame({"Level": [3, 3], "RSL": [0.0, 1.0]})
    cfg = {"data_filter": {"levels": [1, 2]}}

    with pytest.raises(ValueError, match="No rows remain"):
        apply_level_filter(df, cfg)


def test_level_filter_missing_level_column_raises():
    df = pd.DataFrame({"RSL": [0.0, 1.0]})
    cfg = {"data_filter": {"levels": [1, 2]}}

    with pytest.raises(KeyError, match="Level"):
        apply_level_filter(df, cfg)


def test_read_adjusted_rsl_dataframe_applies_level_filter_before_l2(tmp_path):
    df = pd.DataFrame(
        {
            "Level": [1, 2, 3],
            "Limiting": [0, 0, 0],
            "RSLer_up_2sd": [2.0, 4.0, 6.0],
            "RSLer_low_2sd": [2.0, 4.0, 6.0],
            "Age_up_er_2sd": [10.0, 20.0, 30.0],
            "Age_low_er_2sd": [10.0, 20.0, 30.0],
        }
    )
    path = tmp_path / "levels.csv"
    df.to_csv(path, index=False)
    cfg = {
        "data_filter": {"levels": [1, 2]},
        "data_adjustments": {
            "level2_downweighting": {
                "enabled": True,
                "multiplier": 2.0,
                "scope": "slip_only",
                "scale_targets": ["rsl"],
            }
        },
    }

    out, stats = read_adjusted_rsl_dataframe(str(path), cfg)

    assert list(out["Level"]) == [1, 2]
    assert out.loc[1, "RSLer_up_2sd"] == pytest.approx(8.0)
    assert stats["level_filter"]["rows_after"] == 2
    assert stats["selected_rows"] == 1


def test_disabled_or_missing_block_returns_unchanged_dataframe():
    df = _sample_df()

    out1, stats1 = apply_level2_downweighting(df, {})
    pd.testing.assert_frame_equal(out1, df)
    assert stats1["enabled"] is False

    cfg_disabled = {"data_adjustments": {"level2_downweighting": {"enabled": False}}}
    out2, stats2 = apply_level2_downweighting(df, cfg_disabled)
    pd.testing.assert_frame_equal(out2, df)
    assert stats2["enabled"] is False


def test_invalid_inputs_raise_clear_errors():
    df = _sample_df()

    bad_multiplier = {
        "data_adjustments": {
            "level2_downweighting": {
                "enabled": True,
                "multiplier": 0.5,
            }
        }
    }
    with pytest.raises(ValueError, match="multiplier"):
        apply_level2_downweighting(df, bad_multiplier)

    bad_scope = {
        "data_adjustments": {
            "level2_downweighting": {
                "enabled": True,
                "multiplier": 2.0,
                "scope": "unknown",
            }
        }
    }
    with pytest.raises(ValueError, match="scope"):
        apply_level2_downweighting(df, bad_scope)

    missing_level = df.drop(columns=["Level"])
    cfg = {
        "data_adjustments": {
            "level2_downweighting": {
                "enabled": True,
                "multiplier": 2.0,
            }
        }
    }
    with pytest.raises(KeyError, match="Level"):
        apply_level2_downweighting(missing_level, cfg)


WORKFLOW_16KA_PATH = (
    Path(__file__).resolve().parents[1] / "sensitivity" / "workflow_16ka.py"
)


def test_workflow_16ka_passes_config_to_load_and_filter_rsl():
    src = WORKFLOW_16KA_PATH.read_text(encoding="utf-8")
    assert "data.load_and_filter_rsl(" in src
    assert "config=dict(cfg)" in src


def test_workflow_16ka_resolves_paleo_rsl_when_dataset_absent():
    src = WORKFLOW_16KA_PATH.read_text(encoding="utf-8")
    assert 'paths.get("dataset") or paths.get("paleo_rsl")' in src
    assert "Missing cfg['paths']['dataset'] or cfg['paths']['paleo_rsl']" in src
