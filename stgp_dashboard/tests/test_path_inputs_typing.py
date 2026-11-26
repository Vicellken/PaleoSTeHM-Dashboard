"""Regression tests for path_inputs typing persistence and Welcome config loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _app_test_available() -> bool:
    try:
        from streamlit.testing.v1 import AppTest  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _app_test_available(), reason="streamlit AppTest unavailable")
def test_welcome_typing_example_config_persists_and_loads():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_REPO_ROOT / "stgp_dashboard" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception

    example_cfg = "configs/example_us_atlantic.yaml"
    at.text_input[0].input(example_cfg).run()

    assert at.session_state["loaded_config_path"] == example_cfg
    assert at.text_input[0].value == example_cfg

    load_btn = next(b for b in at.button if b.label == "Load config")
    load_btn.click().run()

    assert not at.exception
    assert not at.error
    assert [h.value for h in at.header] == ["Data & output paths"]
    assert at.session_state["paleo_rsl"].endswith("US_Atlantic_Coast_for_ESTGP.csv")


@pytest.mark.skipif(not _app_test_available(), reason="streamlit AppTest unavailable")
def test_dir_path_input_typing_and_external_sync():
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from stgp_dashboard.components.path_inputs import dir_path_input

        st.session_state.setdefault("test_dir", "")
        if st.button("Set external"):
            st.session_state["test_dir"] = "/tmp/external_dir"
        dir_path_input("Output dir", state_key="test_dir")

    at = AppTest.from_function(_app, default_timeout=30)
    at.run()
    assert at.text_input[0].value == ""

    at.text_input[0].input("Outputs/typed_run").run()
    assert at.session_state["test_dir"] == "Outputs/typed_run"
    assert at.text_input[0].value == "Outputs/typed_run"

    at.button[0].click().run()
    assert at.session_state["test_dir"] == "/tmp/external_dir"
    assert at.text_input[0].value == "/tmp/external_dir"
