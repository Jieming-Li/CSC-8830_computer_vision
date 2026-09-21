"""Independent correctness checks. Run: python -m pytest tests/test_filtering.py -v

SciPy convolve2d is used only here as an independent reference, never in the
application's spatial or Fourier implementations. Tolerances are fixed.
"""
import numpy as np
import pytest
from scipy.signal import convolve2d

from filtering import (RTOL, ATOL, box_kernel, gaussian_kernel, make_kernel,
                       spatial_convolve, fourier_convolve, compare_outputs, run_comparison)

METHODS = [spatial_convolve, fourier_convolve]
ASYMMETRIC = np.array([[1, 2, 4], [8, 3, 5], [7, 6, 9]], dtype=np.float64) / 45


@pytest.mark.parametrize('method', METHODS)
def test_random_asymmetric_kernel(method):
    image = np.random.default_rng(8830).random((8, 11))
    expected = convolve2d(image, ASYMMETRIC, mode='same', boundary='fill', fillvalue=0)
    actual = method(image, ASYMMETRIC)
    assert actual.shape == image.shape and actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('method', METHODS)
def test_impulse_preserves_kernel_orientation(method):
    image = np.zeros((9, 9), dtype=np.float64)
    image[4, 4] = 1
    expected = convolve2d(image, ASYMMETRIC, mode='same', boundary='fill', fillvalue=0)
    actual = method(image, ASYMMETRIC)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(actual[3:6, 3:6], ASYMMETRIC, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('method', METHODS)
def test_constant_zero_padding_edges(method):
    image = np.ones((8, 10), dtype=np.float64)
    kernel = box_kernel(3)
    actual = method(image, kernel)
    np.testing.assert_allclose(actual, convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(actual[1:-1, 1:-1], 1, rtol=RTOL, atol=ATOL)
    assert actual[0, 0] == pytest.approx(4 / 9, rel=RTOL, abs=ATOL)
    assert actual[0, 3] == pytest.approx(6 / 9, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize('shape', [(3, 17), (17, 3), (1, 13), (13, 1), (2, 2)])
@pytest.mark.parametrize('method', METHODS)
def test_non_square_and_kernel_larger_than_image(method, shape):
    image = np.random.default_rng(42).random(shape)
    kernel = np.arange(1, 16, dtype=np.float64).reshape(3, 5) / 120
    expected = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    assert method(image, kernel).shape == shape
    np.testing.assert_allclose(method(image, kernel), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('method', METHODS)
def test_identity_kernel(method):
    image = np.random.default_rng(7).random((7, 12))
    np.testing.assert_allclose(method(image, np.ones((1, 1))), image, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('filter_type', ['box', 'gaussian'])
@pytest.mark.parametrize('size', [1, 3, 7, 15])
@pytest.mark.parametrize('method', METHODS)
def test_blur_kernels_against_scipy(method, size, filter_type):
    image = np.random.default_rng(24).random((19, 23))
    kernel = make_kernel(filter_type, size, size / 6)
    assert kernel.shape == (size, size) and kernel.dtype == np.float64
    assert kernel.sum() == pytest.approx(1, rel=0, abs=1e-14)
    assert np.all(kernel >= 0)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(method(image, kernel),
                               convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0),
                               rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('size', [0, -1, 2, 4, 1.5, True, '3'])
def test_invalid_kernel_size(size):
    with pytest.raises(ValueError, match='positive odd'):
        box_kernel(size)
    with pytest.raises(ValueError, match='positive odd'):
        gaussian_kernel(size, 1)


@pytest.mark.parametrize('sigma', [0, -1, np.inf, np.nan, None, True, '1'])
def test_invalid_sigma(sigma):
    with pytest.raises(ValueError, match='sigma'):
        gaussian_kernel(3, sigma)


def test_tiny_sigma_is_finite_and_normalized():
    kernel = gaussian_kernel(3, 1e-300)
    assert np.isfinite(kernel).all() and kernel.sum() == 1
    assert kernel[1, 1] == 1


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('image', [np.empty((0, 3)), np.ones((2, 2, 3)), [[np.nan]],
                                  [[np.inf]], [[-0.1]], [[1.1]], [[1 + 2j]]])
def test_invalid_images(method, image):
    with pytest.raises(ValueError):
        method(image, box_kernel(3))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('kernel', [np.ones((2, 3)), np.empty((0, 3)), [[np.nan]],
                                   np.ones((3, 3, 1)), [[1j]]])
def test_invalid_kernel_array(method, kernel):
    with pytest.raises(ValueError):
        method(np.ones((5, 5)), kernel)


def test_full_padding_crop_and_no_circular_wrap():
    image = np.zeros((9, 13), dtype=np.float64)
    image[0, 0] = 1
    result, info = fourier_convolve(image, box_kernel(3), return_diagnostics=True)
    assert info['fft_shape'] == [11, 15]
    assert info['crop_start'] == [1, 1]
    assert info['max_imaginary_residual'] <= info['imaginary_residual_limit']
    np.testing.assert_allclose(result[-1], 0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result[:, -1], 0, rtol=RTOL, atol=ATOL)


def test_imaginary_residual_guard(monkeypatch):
    original = np.fft.ifft2
    monkeypatch.setattr(np.fft, 'ifft2', lambda *a, **k: original(*a, **k) + 1e-4j)
    with pytest.raises(FloatingPointError, match='imaginary residual'):
        fourier_convolve(np.ones((3, 3)), box_kernel(3))


def test_raw_error_metrics_do_not_clip():
    # Clipping these would misleadingly erase all differences.
    metrics = compare_outputs(np.array([[1.1, -0.1]]), np.array([[1.2, -0.2]]))
    assert metrics['mean_absolute_difference'] == pytest.approx(.1)
    assert metrics['max_absolute_difference'] == pytest.approx(.1)
    assert metrics['rmse'] == pytest.approx(.1)
    assert not metrics['agrees']


def test_comparison_and_input_preservation():
    image = np.random.default_rng(2).random((6, 10))
    kernel = gaussian_kernel(7, 7 / 6)
    before_image, before_kernel = image.copy(), kernel.copy()
    result = run_comparison(image, kernel)
    assert result['metrics']['agrees']
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(kernel, before_kernel)


def test_invalid_filter_and_comparison_shape():
    with pytest.raises(ValueError, match='Filter type'):
        make_kernel('unsupported', 3)
    with pytest.raises(ValueError, match='identical shapes'):
        compare_outputs(np.ones((2, 3)), np.ones((3, 2)))
