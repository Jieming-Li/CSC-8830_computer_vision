"""Explicit zero-padded spatial and Fourier convolution for Module 3.

Library module; run its independent checks with:
    python -m pytest tests/test_filtering.py -v

Both methods consume the same finite, odd-sized discrete kernel. SciPy is
intentionally absent here: its convolution is used only by the test suite.
All geometry is (row, column); no FFT shifts occur in the filtering path.
"""
from numbers import Integral, Real

import numpy as np

RTOL = 1e-10
ATOL = 1e-12
IMAGINARY_TOLERANCE = 1e-12


def _positive_odd(size):
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 1 or size % 2 == 0:
        raise ValueError('Kernel size must be a positive odd integer.')
    return int(size)


def box_kernel(size):
    """Return a centered size x size averaging kernel, normalized to one."""
    size = _positive_odd(size)
    return np.full((size, size), 1.0 / (size * size), dtype=np.float64)


def gaussian_kernel(size, sigma):
    """Sample a spatial Gaussian on an integer grid and normalize its sum.

    The Fourier method transforms THIS sampled/truncated array. It never
    substitutes the transform of a continuous, untruncated Gaussian.
    """
    size = _positive_odd(size)
    if isinstance(sigma, bool) or not isinstance(sigma, Real) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Gaussian sigma must be finite and positive.')
    coordinates = np.arange(size, dtype=np.float64) - size // 2
    x, y = np.meshgrid(coordinates, coordinates)
    # Tiny positive sigma can underflow off-center weights to zero, correctly
    # approaching a unit impulse. The center remains exactly exp(0) = 1.
    with np.errstate(over='ignore', under='ignore'):
        kernel = np.exp(-0.5 * ((x / sigma) ** 2 + (y / sigma) ** 2))
    return kernel / kernel.sum(dtype=np.float64)


def make_kernel(filter_type, size, sigma=None):
    if filter_type == 'box':
        return box_kernel(size)
    if filter_type == 'gaussian':
        return gaussian_kernel(size, sigma)
    raise ValueError("Filter type must be 'box' or 'gaussian'.")


def _real_matrix(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real, not complex.')
    try:
        value = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a numeric 2D array.') from exc
    if value.ndim != 2 or 0 in value.shape or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a nonempty finite 2D array.')
    return value


def _inputs(image, kernel):
    image, kernel = _real_matrix(image, 'Image'), _real_matrix(kernel, 'Kernel')
    if np.any(image < 0) or np.any(image > 1):
        raise ValueError('Input image must be normalized to [0, 1].')
    for size in kernel.shape:
        _positive_odd(size)
    return image, kernel


def spatial_convolve(image, kernel):
    """Explicit centered 2D convolution with zero padding and same-size output.

    Loop over kernel coefficients, operating on all pixels in each NumPy slice.
    Flipping the kernel makes this convolution, not cross-correlation. General
    asymmetric and rectangular odd-sized kernels are supported.
    """
    image, kernel = _inputs(image, kernel)
    height, width = image.shape
    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode='constant')
    flipped = kernel[::-1, ::-1]
    output = np.zeros(image.shape, dtype=np.float64)
    for row in range(kh):
        for column in range(kw):
            output += flipped[row, column] * padded[row:row + height, column:column + width]
    return output


def fourier_convolve(image, kernel, *, return_diagnostics=False):
    """Compute linear convolution via padded FFTs and a matching centered crop.

    Zero padding to (H+Kh-1, W+Kw-1) prevents circular wrap-around. The ordinary
    unshifted kernel is transformed. Crop starts at ((Kh-1)//2, (Kw-1)//2).
    Raw real float64 values, including tiny roundoff, are returned without clips.
    """
    image, kernel = _inputs(image, kernel)
    height, width = image.shape
    kh, kw = kernel.shape
    fft_shape = (height + kh - 1, width + kw - 1)
    image_transform = np.fft.fft2(image, s=fft_shape, axes=(0, 1))
    kernel_transform = np.fft.fft2(kernel, s=fft_shape, axes=(0, 1))
    full = np.fft.ifft2(image_transform * kernel_transform, axes=(0, 1))
    imaginary_max = float(np.max(np.abs(full.imag)))
    imaginary_limit = IMAGINARY_TOLERANCE * max(1.0, float(np.max(np.abs(full.real))))
    if not np.isfinite(full).all() or imaginary_max > imaginary_limit:
        raise FloatingPointError(f'Unexpected FFT imaginary residual: {imaginary_max:.6g}; '
                                 f'allowed {imaginary_limit:.6g}.')
    start_row, start_column = (kh - 1) // 2, (kw - 1) // 2
    output = full.real[start_row:start_row + height, start_column:start_column + width].copy()
    diagnostics = {'fft_shape': list(fft_shape), 'crop_start': [start_row, start_column],
                   'max_imaginary_residual': imaginary_max, 'imaginary_residual_limit': imaginary_limit}
    return (output, diagnostics) if return_diagnostics else output


def compare_outputs(spatial, fourier):
    """Measure agreement BEFORE display clipping, quantization, or PNG saving."""
    spatial, fourier = _real_matrix(spatial, 'Spatial output'), _real_matrix(fourier, 'Fourier output')
    if spatial.shape != fourier.shape:
        raise ValueError('Compared outputs must have identical shapes.')
    difference = np.abs(spatial - fourier)
    return {'mean_absolute_difference': float(np.mean(difference)),
            'max_absolute_difference': float(np.max(difference)),
            'rmse': float(np.sqrt(np.mean(difference ** 2))),
            'agrees': bool(np.allclose(fourier, spatial, rtol=RTOL, atol=ATOL)),
            'rtol': RTOL, 'atol': ATOL}


def run_comparison(image, kernel):
    """One experiment shared by the CLI and app; the identical kernel is reused."""
    spatial = spatial_convolve(image, kernel)
    fourier, diagnostics = fourier_convolve(image, kernel, return_diagnostics=True)
    return {'spatial': spatial, 'fourier': fourier, 'difference': np.abs(spatial - fourier),
            'metrics': compare_outputs(spatial, fourier), 'fft': diagnostics}
