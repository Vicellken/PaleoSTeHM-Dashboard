from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _minimal_regional_csv(tmp_path) -> str:
    """One SLIP row (L2) + one ML row; columns match PSTHM.load.load_regional_rsl_dataframe."""
    rows = [
        {
            "Region": "Test",
            "Latitude": 20.0,
            "Longitude": 110.0,
            "RSL": 1.0,
            "RSLer_up_2sd": 4.0,
            "RSLer_low_2sd": 4.0,
            "Age": 1000.0,
            "Age_up_er_2sd": 20.0,
            "Age_low_er_2sd": 20.0,
            "Limiting": 0,
            "Level": 2,
            "Region.1": 1,
        },
        {
            "Region": "Test",
            "Latitude": 20.0,
            "Longitude": 110.0,
            "RSL": -2.0,
            "RSLer_up_2sd": 1.0,
            "RSLer_low_2sd": 1.0,
            "Age": 2000.0,
            "Age_up_er_2sd": 50.0,
            "Age_low_er_2sd": 50.0,
            "Limiting": -1,
            "Level": 1,
            "Region.1": 1,
        },
    ]
    path = tmp_path / "rsl.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_load_and_filter_rsl_applies_l2_downweighting_when_config_passed(tmp_path):
    pytest.importorskip("torch")
    from stgp_pipeline import data as stgp_data

    csv_path = _minimal_regional_csv(tmp_path)
    cfg = {
        "data_adjustments": {
            "level2_downweighting": {
                "enabled": True,
                "multiplier": 2.0,
                "scope": "slip_only",
                "scale_targets": ["rsl"],
            }
        }
    }

    rsl_on = stgp_data.load_and_filter_rsl(csv_path, max_age_bp=5000, config=cfg)
    rsl_off = stgp_data.load_and_filter_rsl(csv_path, max_age_bp=5000, config=None)

    assert rsl_on.n_slip == 1
    assert rsl_off.n_slip == 1
    # PSTHM.load: y_sigma = (RSLer_up_2sd + RSLer_low_2sd) / 4
    # Base (4+4)/4=2; after x2 on 2sd cols -> (8+8)/4=4
    assert float(rsl_off.y_sigma.item()) == pytest.approx(2.0)
    assert float(rsl_on.y_sigma.item()) == pytest.approx(4.0)


def _level_split_regional_csv(tmp_path) -> str:
    rows = [
        {
            "Region": "Test",
            "Latitude": 20.0,
            "Longitude": 110.0,
            "RSL": 1.0,
            "RSLer_up_2sd": 2.0,
            "RSLer_low_2sd": 2.0,
            "Age": 1000.0,
            "Age_up_er_2sd": 20.0,
            "Age_low_er_2sd": 20.0,
            "Limiting": 0,
            "Level": 1,
            "Region.1": 1,
        },
        {
            "Region": "Test",
            "Latitude": 20.5,
            "Longitude": 110.5,
            "RSL": 1.5,
            "RSLer_up_2sd": 2.0,
            "RSLer_low_2sd": 2.0,
            "Age": 1200.0,
            "Age_up_er_2sd": 20.0,
            "Age_low_er_2sd": 20.0,
            "Limiting": 0,
            "Level": 1,
            "Region.1": 1,
        },
        {
            "Region": "Test",
            "Latitude": 21.0,
            "Longitude": 111.0,
            "RSL": 2.0,
            "RSLer_up_2sd": 4.0,
            "RSLer_low_2sd": 4.0,
            "Age": 1500.0,
            "Age_up_er_2sd": 30.0,
            "Age_low_er_2sd": 30.0,
            "Limiting": 0,
            "Level": 2,
            "Region.1": 1,
        },
    ]
    path = tmp_path / "levels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_load_and_filter_rsl_honors_train_levels_when_levels_absent(tmp_path):
    pytest.importorskip("torch")
    from stgp_pipeline import data as stgp_data

    csv_path = _level_split_regional_csv(tmp_path)
    cfg = {"data_filter": {"train_levels": [1], "prediction_levels": [1, 2]}}

    rsl_train = stgp_data.load_and_filter_rsl(
        csv_path, max_age_bp=5000, config=cfg, purpose="train"
    )
    rsl_pred = stgp_data.load_and_filter_rsl(
        csv_path, max_age_bp=5000, config=cfg, purpose="predict"
    )

    assert rsl_train.n_slip == 2
    assert rsl_pred.n_slip == 3
