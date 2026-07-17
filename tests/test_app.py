"""Smoke test: the Streamlit UI must render without exceptions."""

import pytest

streamlit = pytest.importorskip("streamlit")


def test_streamlit_app_renders():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
