"""Image preparation checks. Run: python -m pytest tests/ -v"""
from io import BytesIO

import numpy as np
from PIL import Image

from image_utils import load_grayscale, png_bytes


def test_exif_orientation_and_aspect_ratio():
    original = Image.fromarray(np.zeros((400, 800, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[274] = 6
    stream = BytesIO()
    original.save(stream, format='JPEG', exif=exif)
    image, info = load_grayscale(stream.getvalue())
    assert (info['original_width'], info['original_height']) == (400, 800)
    assert image.shape == (512, 256)
    assert image.dtype == np.float64 and info['resized']


def test_float64_range_and_no_upscaling():
    image, info = load_grayscale(png_bytes(np.array([[0, .5, 1]], dtype=np.float64)))
    assert image.shape == (1, 3) and image.dtype == np.float64
    assert image.min() == 0 and image.max() == 1
    assert not info['resized']
