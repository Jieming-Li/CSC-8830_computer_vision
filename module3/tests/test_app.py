"""Streamlit state checks. Run: python -m pytest tests/test_app.py -v"""
from pathlib import Path

from streamlit.testing.v1 import AppTest
import filtering


def test_app_runs_only_on_button_and_invalidates_settings(monkeypatch):
    calls = []
    original = filtering.run_comparison

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(filtering, 'run_comparison', counted)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert not calls
    app.button[0].click().run()
    assert not app.exception
    assert len(calls) == 1 and len(app.metric) == 3
    assert 'Agreement: PASS' in app.success[0].value
    app.run()  # An unrelated rerun must not redo numerical filtering.
    assert len(calls) == 1
    next(control for control in app.selectbox if control.label == 'Filter type').select('Gaussian').run()
    assert not app.exception and len(app.metric) == 0
    assert len(calls) == 1
    app.button[0].click().run()
    assert not app.exception and len(calls) == 2
    assert 'Agreement: PASS' in app.success[0].value
