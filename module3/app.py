"""Interactive Module 3 demonstration. Run locally from module3/:
    streamlit run app.py

The app performs one selected comparison only after Run both methods is pressed.
Uploaded bytes and raw outputs stay in session memory; images are not persisted.
Both numerical methods are imported from filtering.py, also used by the runner.
"""
import hashlib
from io import BytesIO
import json

import numpy as np
import streamlit as st
from PIL import UnidentifiedImageError

# Package imports avoid mixing helper modules across assignments. The local
# branch preserves `cd module3 && streamlit run app.py`.
if __package__:
    from .filtering import RTOL, ATOL, make_kernel, run_comparison
    from .image_utils import (BASE_DIR, SYNTHETIC_FILES, bundled_images, difference_panel,
                             load_grayscale, png_bytes, synthetic_examples, comparison_panel)
else:
    from filtering import RTOL, ATOL, make_kernel, run_comparison
    from image_utils import (BASE_DIR, SYNTHETIC_FILES, bundled_images, difference_panel,
                            load_grayscale, png_bytes, synthetic_examples, comparison_panel)

STATE_KEY = 'module3.comparison'


def example_options():
    """Only look inside this module's images/; generated fallbacks work offline."""
    files = {path.name: path for path in bundled_images(BASE_DIR / 'images')}
    for name in SYNTHETIC_FILES:
        files.setdefault(name, None)
    return files


def example_label(filename):
    if filename == 'checkerboard.png':
        return 'Generated checkerboard'
    if filename.startswith('synthetic_'):
        return f'Synthetic example (not a photograph): {filename}'
    return f'Photograph: {filename}'


def main(*, configure_page=True):
    if configure_page:
        st.set_page_config(page_title='Spatial and Fourier Image Blurring', layout='wide')
    st.title('Spatial and Fourier Image Blurring')
    st.caption('CSC 8830 · Module 3 · Explicit convolution and the convolution theorem')
    st.markdown('Choose an image and a blur kernel, then run the two methods on the **same grayscale input** '
                'with the **same discrete kernel**. Compare raw floating-point outputs.')
    with st.expander('How the methods match', expanded=True):
        st.markdown('**Spatial:** flip the kernel and sum weighted image neighborhoods. '
                    '**Fourier:** pad to the full linear-convolution size, transform both arrays, '
                    'multiply their transforms, invert, then crop consistently.\n\n'
                    'Both use **zero padding**, which can darken image boundaries. Inputs are resized '
                    'without changing aspect ratio to at most **512 pixels** on their longest side.')
    source = st.radio('Image source', ['Bundled example', 'Upload an image'], horizontal=True, key='module3.source')
    try:
        if source == 'Bundled example':
            files = example_options()
            name = st.selectbox('Example image', list(files), format_func=example_label, key='module3.example')
            raw = files[name].read_bytes() if files[name] is not None else png_bytes(synthetic_examples()[name])
            label = example_label(name)
        else:
            upload = st.file_uploader('Choose a photograph', type=['jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp'], key='module3.upload')
            if upload is None:
                st.info('Upload an image to continue. It stays in memory and is not saved permanently.')
                st.session_state.pop(STATE_KEY, None)
                return
            raw, name, label = upload.getvalue(), upload.name, upload.name
        image, metadata = load_grayscale(raw)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        st.error(f'Cannot load image: {exc}')
        st.session_state.pop(STATE_KEY, None)
        return
    preview, controls = st.columns([1, 1])
    with preview:
        st.subheader('Grayscale input')
        st.image(image, clamp=True, width='stretch', caption=label)
        st.caption(f'EXIF-oriented original: {metadata["original_width"]} × {metadata["original_height"]} px. '
                   f'Processed: {metadata["processed_width"]} × {metadata["processed_height"]} px. '
                   f'{"Resized with LANCZOS." if metadata["resized"] else "No resizing needed."} '
                   'Float64 intensity range: [0, 1].')
    with controls:
        st.subheader('Blur settings')
        choice = st.selectbox('Filter type', ['Box', 'Gaussian'], key='module3.filter')
        size = st.select_slider('Odd kernel size', options=list(range(1, 32, 2)), value=7, key='module3.size')
        sigma = None
        if choice == 'Gaussian':
            sigma = st.number_input('Gaussian sigma (pixels)', min_value=.001, value=float(size / 6),
                                    step=.1, format='%.4f', key=f'module3.sigma_{size}')
        st.caption('The assignment experiments use sizes 3, 7, 15 and Gaussian sigma = size / 6.')
        kernel = make_kernel(choice.lower(), size, sigma)
        st.write(f'Kernel sum: {kernel.sum():.15f}')
        fingerprint = (hashlib.sha256(raw).hexdigest(), name, choice, size, sigma)
        stored = st.session_state.get(STATE_KEY)
        if stored is not None and stored['fingerprint'] != fingerprint:
            st.session_state.pop(STATE_KEY, None)
        if st.button('Run both methods', type='primary', key='module3.run'):
            try:
                with st.spinner('Computing spatial and Fourier convolution…'):
                    result = run_comparison(image, kernel)
                st.session_state[STATE_KEY] = {'fingerprint': fingerprint, 'result': result}
            except (ValueError, FloatingPointError) as exc:
                st.error(str(exc))
    stored = st.session_state.get(STATE_KEY)
    if stored is None:
        st.info('Press Run both methods to calculate the selected comparison.')
        return
    result = stored['result']
    metrics = result['metrics']
    st.subheader('Comparison results')
    for column, key, label in zip(st.columns(3),
                                  ['mean_absolute_difference', 'max_absolute_difference', 'rmse'],
                                  ['Mean absolute difference', 'Maximum absolute difference', 'RMSE']):
        column.metric(label, f'{metrics[key]:.6e}')
    if metrics['agrees']:
        st.success(f'Agreement: PASS at rtol={RTOL:g}, atol={ATOL:g}.')
    else:
        st.error(f'Agreement: FAIL at rtol={RTOL:g}, atol={ATOL:g}.')
    st.caption('Metrics compare raw float64 arrays before clipping or PNG conversion. '
               'They measure agreement between implementations, not image quality.')
    spatial_column, fourier_column = st.columns(2)
    spatial_column.image(np.clip(result['spatial'], 0, 1), caption='Spatial result · display range [0, 1]',
                         width='stretch', clamp=True)
    fourier_column.image(np.clip(result['fourier'], 0, 1), caption='Fourier result · display range [0, 1]',
                         width='stretch', clamp=True)
    st.image(difference_panel(result['difference']), width='content')
    st.caption('The heatmap uses an amplified numerical scale with a labeled color bar. '
               'A colorful map at approximately 10⁻¹⁵ can be floating-point noise, not a visible mismatch.')
    st.caption(f'Maximum imaginary residual: {result["fft"]["max_imaginary_residual"]:.3e}; '
               f'checked limit: {result["fft"]["imaginary_residual_limit"]:.3e}.')
    payload = {'image_filename': name, **metadata, 'filter_type': choice.lower(),
               'kernel_size': size, 'sigma': sigma, 'kernel': kernel.tolist(),
               'boundary': 'zero', 'mode': 'same centered', **metrics, **result['fft']}
    downloads = st.columns(3)
    downloads[0].download_button('Download spatial PNG', png_bytes(result['spatial']),
                                 'spatial_result.png', 'image/png', key='module3.download_spatial')
    downloads[1].download_button('Download Fourier PNG', png_bytes(result['fourier']),
                                 'fourier_result.png', 'image/png', key='module3.download_fourier')
    downloads[2].download_button('Download metrics JSON', json.dumps(payload, indent=2, allow_nan=False),
                                 'comparison_metrics.json', 'application/json', key='module3.download_metrics')
    st.download_button('Download comparison panel', comparison_panel(image, result,
                        f'{name} | {choice}, k={size}, sigma={sigma}'), 'comparison_panel.png', 'image/png', key='module3.download_panel')
    raw_download = BytesIO()
    np.savez_compressed(raw_download, input=image, kernel=kernel, spatial=result['spatial'],
                        fourier=result['fourier'], absolute_difference=result['difference'])
    st.download_button('Download raw float64 arrays', raw_download.getvalue(),
                       'comparison_arrays.npz', 'application/octet-stream', key='module3.download_arrays')


if __name__ == '__main__':
    main()
