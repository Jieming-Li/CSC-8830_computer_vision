"""Integration/regression checks. From the repository root: python -m pytest -q.

These tests read saved trials but never rewrite calibration, images, or results.
The browser component itself is additionally checked with a local browser.
"""
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from module2 import app as book_app
from module2.measure_book import load_calibration, measure_at_depth
from module2.measurement_utils import load_gray
from module3 import filtering

BASE = Path(__file__).resolve().parents[1]
TRIALS = sorted((BASE / 'module2/results/measurements').glob('*.json'))


def shared_app():
    return AppTest.from_file(str(BASE / 'app.py'), default_timeout=30).run()


@pytest.mark.parametrize('path', TRIALS, ids=lambda p: p.stem)
def test_saved_book_measurements_preserved(path):
    saved = json.loads(path.read_text())
    matrix, distortion, size = load_calibration()
    assert size == (3024, 4032)
    result = measure_at_depth(saved['corners_px'], matrix, distortion, saved['distance_m'])
    for key in ['width_cm', 'height_cm', 'normalized_corners', 'planar_corners_cm']:
        np.testing.assert_allclose(result[key], saved[key], rtol=1e-12, atol=1e-12)


def test_upload_orientation_matches_cli_and_rejects_wrong_resolution():
    image = BASE / 'module2/book_measurement_images/book_01_2p134m.jpeg'
    np.testing.assert_array_equal(book_app.decode_upload(image.read_bytes(), (3024, 4032)), load_gray(image))
    buffer = BytesIO()
    Image.new('L', (100, 80)).save(buffer, format='PNG')
    with pytest.raises(ValueError, match='resolution mismatch'):
        book_app.decode_upload(buffer.getvalue(), (3024, 4032))


def test_clicks_scale_deduplicate_and_stop_after_four():
    state = book_app.new_selection()
    first = {'x': 100, 'y': 200, 'width': 756, 'height': 1008, 'unix_time': 1}
    assert book_app.accept_click(state, first, (4032, 3024))
    assert state['corners'] == [[400, 800]]
    assert not book_app.accept_click(state, first, (4032, 3024))
    for stamp in range(2, 5):
        assert book_app.accept_click(state, {**first, 'unix_time': stamp}, (4032, 3024))
    assert not book_app.accept_click(state, {**first, 'unix_time': 5}, (4032, 3024))
    assert len(state['corners']) == 4


def test_home_and_module2_use_local_assets_from_any_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    app = shared_app()
    assert not app.exception
    assert app.title[0].value == 'CSC 8830 Computer Vision'
    assert len(app.get('page_link')) == 2
    app.switch_page('module2/page.py').run()
    assert not app.exception and not app.error
    assert app.title[0].value == 'Single-Image Book Dimension Measurement'
    assert len(app.metric) == 4  # Existing evaluation summary.
    assert len(app.image) == 3
    assert app.number_input(key='module2.distance').value == 2.134


def test_navigation_clears_inactive_state_and_does_not_run_filters_automatically(monkeypatch):
    calls = []
    original = filtering.run_comparison

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(filtering, 'run_comparison', counted)
    app = shared_app().switch_page('module3/page.py').run()
    assert not app.exception and not calls
    app.selectbox(key='module3.filter').select('Gaussian').run()
    app.button(key='module3.run').click().run()
    assert not app.exception and len(calls) == 1
    assert 'Agreement: PASS' in app.success[0].value
    app.run()
    assert len(calls) == 1  # Ordinary reruns reuse the result.
    assert 'module3.comparison' in app.session_state
    app.switch_page('module2/page.py').run()
    assert not app.exception and not app.error
    assert not any(k.startswith('module3.') for k in app.session_state)
    assert not any(m.label == 'RMSE' for m in app.metric)
    app.number_input(key='module2.distance').set_value(3.048).run()
    assert app.number_input(key='module2.distance').value == 3.048
    app.switch_page('module3/page.py').run()
    assert not app.exception and not app.metric
    assert not any(k.startswith('module2.') for k in app.session_state)
    assert app.selectbox(key='module3.filter').value == 'Box'
    assert len(calls) == 1
    app.switch_page('home.py').run()
    assert not app.exception
    assert not any(k.startswith(('module2.', 'module3.')) for k in app.session_state)


def test_book_upload_calculation_reset_and_page_switch(monkeypatch):
    saved = json.loads(TRIALS[0].read_text())
    image = BASE / 'module2' / saved['image']
    events = iter([{'x': x, 'y': y, 'width': 3024, 'height': 4032, 'unix_time': i}
                   for i, (x, y) in enumerate(saved['corners_px'], 1)])
    monkeypatch.setattr(book_app, 'streamlit_image_coordinates', lambda *a, **kw: next(events, None))
    app = shared_app().switch_page('module2/page.py').run()
    app.file_uploader(key='module2.upload').set_value((image.name, image.read_bytes(), 'image/jpeg')).run()
    assert not app.exception
    assert len(app.session_state['module2.selection']['corners']) == 4
    app.button(key='module2.calculate').click().run()
    assert not app.exception and not app.error
    result = app.session_state['module2.selection']['result']
    np.testing.assert_allclose([result['width_cm'], result['height_cm']],
                               [saved['width_cm'], saved['height_cm']], rtol=1e-12)
    assert len(app.download_button) == 2
    png = app.session_state['module2.selection']['annotation_png']
    assert cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE).shape[1] == 3024
    app.number_input(key='module2.distance').set_value(3.048).run()
    assert app.session_state['module2.selection']['result'] is None
    assert len(app.session_state['module2.selection']['corners']) == 4
    app.button(key='module2.reset_corners').click().run()
    assert not app.session_state['module2.selection']['corners']
    app.switch_page('module3/page.py').run()
    assert not app.exception and not any(k.startswith('module2.') for k in app.session_state)
    app.switch_page('module2/page.py').run()
    assert not app.exception
    assert app.file_uploader(key='module2.upload').value is None
    assert app.number_input(key='module2.distance').value == 2.134
    assert not app.session_state['module2.selection']['corners']


def test_both_modules_retain_standalone_entry_points():
    for folder in ['module2', 'module3']:
        app = AppTest.from_file(str(BASE / folder / 'app.py'), default_timeout=30).run()
        assert not app.exception and not app.error
