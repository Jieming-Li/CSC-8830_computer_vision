"""Local Streamlit UI for the existing calibrated book-measurement function.

Run: streamlit run app.py
Measurements and downloads stay in the current browser session. The app does
not append trials, recalibrate, or load reference-photo data. Its only local
inputs are the final calibration, aggregate metrics, and three named figures.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from measure_book import CALIBRATION_PATH, load_calibration, measure_at_depth
from measurement_utils import BASE_DIR, CORNER_LABELS, validate_quad

STATE_KEY = 'book_measurement_state'
FIGURES = (
    ('predicted_vs_true_width.png', 'Predicted vs. true width'),
    ('predicted_vs_true_height.png', 'Predicted vs. true height'),
    ('absolute_error_vs_distance.png', 'Absolute error vs. distance'),
)


def new_selection(image_token=None, generation=0):
    return {'image_token': image_token, 'generation': generation,
            'corners': [], 'last_event': None, 'result': None,
            'result_fingerprint': None, 'annotation_png': None}


def decode_upload(data, expected_size):
    """Decode bytes using calibration's grayscale/default EXIF convention."""
    if not data:
        raise ValueError('The uploaded file is empty. Choose a JPG, JPEG, or PNG image.')
    # Same flags as cv2.imread(path, cv2.IMREAD_GRAYSCALE) in the CLI.
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError('OpenCV could not decode this file. Choose a valid JPG, JPEG, or PNG.')
    actual = (gray.shape[1], gray.shape[0])
    if actual != expected_size:
        raise ValueError(f'Image resolution mismatch: uploaded image loads as {actual[0]} × {actual[1]} pixels; '
                         f'calibration requires {expected_size[0]} × {expected_size[1]} pixels. '
                         'Upload the original portrait photo without resizing, cropping, or rotating it.')
    return gray


def accept_click(state, event, image_shape):
    """Consume each component event once, with a strict four-corner limit.

    The component returns rendered width/height and a timestamp. Its last value
    repeats on Streamlit reruns; comparing the complete event prevents number
    input and download interactions from silently adding another corner.
    """
    if not event:
        return False
    fields = ('x', 'y', 'width', 'height', 'unix_time')
    if not isinstance(event, dict) or any(k not in event for k in fields):
        raise ValueError('The image component returned an incomplete click. Reset Corners and try again.')
    signature = tuple(event[k] for k in fields)
    if signature == state['last_event']:
        return False
    state['last_event'] = signature
    if len(state['corners']) >= 4:
        return False
    x, y, displayed_width, displayed_height = (float(event[k]) for k in fields[:4])
    if (not np.isfinite([x, y, displayed_width, displayed_height]).all()
            or displayed_width <= 0 or displayed_height <= 0
            or not 0 <= x < displayed_width or not 0 <= y < displayed_height):
        raise ValueError('Click inside the image to select a corner.')
    height, width = image_shape[:2]
    # Browser offsets are in rendered CSS pixels. Geometry uses the full image.
    original_x = min(x * width / displayed_width, width - 1.0)
    original_y = min(y * height / displayed_height, height - 1.0)
    state['corners'].append([original_x, original_y])
    state['result'] = None
    state['result_fingerprint'] = None
    state['annotation_png'] = None
    return True


def draw_corners(gray, corners, preview_width=None):
    """Draw UI overlays only; this function never estimates physical dimensions."""
    if preview_width is None:
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        height = round(gray.shape[0] * preview_width / gray.shape[1])
        canvas = cv2.cvtColor(cv2.resize(gray, (preview_width, height),
                                        interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
    scale = np.array([canvas.shape[1] / gray.shape[1], canvas.shape[0] / gray.shape[0]])
    points = np.rint(np.asarray(corners, float).reshape(-1, 2) * scale).astype(np.int32)
    thickness = max(2, round(canvas.shape[1] / 500))
    if len(points) > 1:
        cv2.polylines(canvas, [points], len(points) == 4, (30, 220, 100), thickness, cv2.LINE_AA)
    for index, point in enumerate(points):
        location = tuple(point)
        cv2.circle(canvas, location, thickness * 3, (255, 190, 30), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index + 1), (location[0] + thickness * 3, location[1] - thickness * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, max(.65, canvas.shape[1] / 1400),
                    (255, 190, 30), thickness, cv2.LINE_AA)
    return canvas


def make_annotation(gray, corners, distance_m, dimensions):
    """Return PNG bytes with the original-size photo and a readable header."""
    canvas = draw_corners(gray, corners)
    header_height = max(110, round(gray.shape[1] * .08))
    header = np.full((header_height, gray.shape[1], 3), 245, np.uint8)
    font_scale = max(.65, gray.shape[1] / 1400)
    thickness = max(1, round(gray.shape[1] / 1000))
    lines = [f'Distance: {distance_m:.3f} m',
             f'Width: {dimensions["width_cm"]:.2f} cm    Height: {dimensions["height_cm"]:.2f} cm']
    for index, text in enumerate(lines):
        cv2.putText(header, text, (25, round(header_height * (.36 + .43 * index))),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (25, 25, 25), thickness, cv2.LINE_AA)
    image = cv2.cvtColor(np.vstack([header, canvas]), cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise RuntimeError('Could not encode the annotated image for download.')
    return encoded.tobytes()


def input_fingerprint(state, distance_m, calibration_hash):
    return (state['image_token'], float(distance_m), calibration_hash,
            tuple(tuple(point) for point in state['corners']))


def evaluation_summary():
    st.divider()
    st.header('Evaluation Summary')
    try:
        # Never render the raw JSON: show only the requested aggregate values.
        saved = json.loads((BASE_DIR / 'results' / 'evaluation_metrics.json').read_text())
        values = [('Width MAE', saved['width']['mae_cm'], 'cm'),
                  ('Width MAPE', saved['width']['mape_percent'], '%'),
                  ('Height MAE', saved['height']['mae_cm'], 'cm'),
                  ('Height MAPE', saved['height']['mape_percent'], '%')]
        if not all(np.isfinite(float(value)) and float(value) >= 0 for _, value, _ in values):
            raise ValueError('Evaluation metrics must be finite and nonnegative.')
        st.caption(f'Saved evaluation of {int(saved["trial_count"])} completed trials.')
        for column, (label, value, unit) in zip(st.columns(4), values):
            column.metric(label, f'{float(value):.2f}{" " if unit == "cm" else ""}{unit}')
        st.caption('MAE is mean absolute error. MAPE is mean absolute percentage error. '
                   'These summarize the completed trials, not a guarantee for a new photo.')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'Could not read the saved evaluation metrics: {exc}')
    for filename, caption in FIGURES:
        path = BASE_DIR / 'results' / 'figures' / filename
        if path.is_file():
            st.image(str(path), caption=caption, width='stretch')
        else:
            st.warning(f'Evaluation figure is missing: results/figures/{filename}')


def measurement_panel():
    try:
        matrix, distortion, expected_size = load_calibration()
        calibration_hash = hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Could not load the final calibration: {exc}')
        return
    uploaded = st.file_uploader('Upload a portrait phone image', type=['jpg', 'jpeg', 'png'], key='book_upload')
    distance_m = st.number_input('Camera-to-book distance (m)', min_value=0.001, value=2.134,
                                 step=0.001, format='%.3f', key='book_distance',
                                 help='Measure from the rear camera lens perpendicularly to the book-cover plane.')
    st.caption(f'Required image resolution: {expected_size[0]} × {expected_size[1]} pixels after loading.')
    data = uploaded.getvalue() if uploaded is not None else None
    token = (uploaded.name, hashlib.sha256(data).hexdigest()) if uploaded is not None else None
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = new_selection(token)
    state = st.session_state[STATE_KEY]
    if token != state['image_token']:
        state = new_selection(token, state['generation'] + 1)
        st.session_state[STATE_KEY] = state
    if uploaded is None:
        st.info('Upload an image to select its four book-cover corners.')
        return
    try:
        gray = decode_upload(data, expected_size)
    except (ValueError, cv2.error) as exc:
        st.error(str(exc))
        return
    if st.button('Reset Corners', key='reset_corners'):
        state = new_selection(token, state['generation'] + 1)
        st.session_state[STATE_KEY] = state
    fingerprint = input_fingerprint(state, distance_m, calibration_hash)
    if fingerprint != state['result_fingerprint']:
        state['result'] = None
        state['annotation_png'] = None
    st.subheader('Select the book-cover corners')
    st.write('**1. Top-left → 2. Top-right → 3. Bottom-right → 4. Bottom-left**')
    image_column, corner_column = st.columns([2, 1])
    with image_column:
        preview = draw_corners(gray, state['corners'], preview_width=900)
        event = streamlit_image_coordinates(
            preview, key=f'corners_{token[1]}_{state["generation"]}',
            width='stretch', cursor='crosshair',
            image_format='PNG', png_compression_level=3)
        try:
            if accept_click(state, event, gray.shape):
                st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
    with corner_column:
        count = len(state['corners'])
        st.metric('Selected corners', f'{count} / 4')
        if count < 4:
            st.info(f'Next: {count + 1}. {CORNER_LABELS[count].capitalize()}')
        else:
            st.success('Four corners selected. Check the outline, then press Calculate.')
            st.caption('Further clicks are ignored. Use Reset Corners to start again.')
        if count:
            st.table([{'Corner': f'{i + 1}. {CORNER_LABELS[i]}', 'X (px)': round(x, 2), 'Y (px)': round(y, 2)}
                      for i, (x, y) in enumerate(state['corners'])])
        if st.button('Calculate', type='primary', disabled=count != 4, key='calculate'):
            try:
                if (gray.shape[1], gray.shape[0]) != expected_size:
                    raise ValueError('Image resolution no longer matches calibration.')
                points = validate_quad(state['corners'], gray.shape)
                # One source of truth: exactly the CLI's undistortion/depth code.
                dimensions = measure_at_depth(points, matrix, distortion, distance_m)
                result = {'image_filename': Path(uploaded.name).name,
                          'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                          'distance_m': float(distance_m),
                          'distance_definition': 'perpendicular camera-to-cover depth Z',
                          'image_width': expected_size[0], 'image_height': expected_size[1],
                          'corner_order': list(CORNER_LABELS), 'corners_px': points.tolist(),
                          **dimensions, 'calibration_sha256': calibration_hash}
                # Validate serialization before exposing either download.
                json.dumps(result, allow_nan=False)
                state['annotation_png'] = make_annotation(gray, points, distance_m, dimensions)
                state['result'] = result
                state['result_fingerprint'] = input_fingerprint(state, distance_m, calibration_hash)
            except (ValueError, RuntimeError, cv2.error) as exc:
                st.error(f'Cannot calculate: {exc}')
    if state['result'] is not None:
        result = state['result']
        st.subheader('Estimated book dimensions')
        width_column, height_column = st.columns(2)
        width_column.metric('Width', f'{result["width_cm"]:.2f} cm')
        height_column.metric('Height', f'{result["height_cm"]:.2f} cm')
        st.image(state['annotation_png'], caption='Selected book-cover corners and estimated dimensions',
                 width='stretch')
        stem = Path(uploaded.name).stem
        image_download, json_download = st.columns(2)
        image_download.download_button('Download annotated image', state['annotation_png'],
                                       file_name=f'{stem}_annotated.png', mime='image/png', key='download_image')
        json_download.download_button('Download JSON result', json.dumps(result, indent=2, allow_nan=False),
                                      file_name=f'{stem}_measurement.json', mime='application/json', key='download_json')


def main():
    st.set_page_config(page_title='Single-Image Book Dimension Measurement', layout='wide')
    st.title('Single-Image Book Dimension Measurement')
    st.markdown('1. Upload a **portrait phone image** captured with the calibrated rear camera.\n'
                '2. Enter the **camera-to-book distance in meters**.\n'
                '3. Click **top-left, top-right, bottom-right, bottom-left** book-cover corners.\n'
                '4. Press **Calculate** to view and download the measurement.')
    with st.expander('Method assumptions', expanded=True):
        st.markdown('- The object surface is **planar**.\n'
                    '- The book cover is **approximately parallel to the camera image plane**.\n'
                    '- Measure distance **from the rear camera lens to the book-cover plane**, '
                    'perpendicularly rather than diagonally to a corner.\n'
                    '- Use the **same calibrated camera settings**: rear camera and lens, '
                    'portrait orientation, 3024 × 4032 resolution, and 1× zoom.')
    measurement_panel()
    evaluation_summary()


if __name__ == '__main__':
    main()
