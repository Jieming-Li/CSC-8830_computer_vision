"""Image loading, clearly labeled synthetic inputs, and display-only rendering.

Import from app.py or run_experiments.py; this module has no executable action.
The originals and raw filtering outputs are never modified by display helpers.
"""
from io import BytesIO
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MAX_DIMENSION = 512
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SYNTHETIC_FILES = ('synthetic_edges.png', 'synthetic_texture.png', 'checkerboard.png')
# Keep any Matplotlib cache within this module's folder during local execution.
os.environ.setdefault('MPLCONFIGDIR', str(BASE_DIR / '.cache' / 'matplotlib'))


def load_grayscale(source, max_dimension=MAX_DIMENSION):
    """EXIF transpose -> 8-bit luminance -> aspect-preserving resize -> float64.

    source is a local path or uploaded bytes. Bytes are decoded in memory and
    never persisted. Original dimensions are recorded after EXIF correction.
    """
    if isinstance(max_dimension, bool) or not isinstance(max_dimension, int) or max_dimension < 1:
        raise ValueError('Maximum dimension must be a positive integer.')
    stream = BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(stream) as original:
        oriented = ImageOps.exif_transpose(original)
        before = oriented.size
        gray = oriented.convert('L')
        gray.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64) / 255.0
    return pixels, {'original_width': before[0], 'original_height': before[1],
                    'processed_width': pixels.shape[1], 'processed_height': pixels.shape[0],
                    'resized': before != (pixels.shape[1], pixels.shape[0]),
                    'max_dimension': max_dimension, 'dtype': 'float64', 'range': [0.0, 1.0],
                    'orientation': 'Pillow ImageOps.exif_transpose',
                    'grayscale_conversion': 'Pillow L luminance / 255', 'resize_filter': 'LANCZOS'}


def synthetic_examples():
    """Deterministic test patterns, explicitly NOT photographs."""
    height, width = 320, 480
    y, x = np.mgrid[:height, :width]
    edges = .12 + .55 * x / (width - 1)
    edges[(x > 50) & (x < 200) & (y > 45) & (y < 160)] = .95
    edges[(x - 345) ** 2 + (y - 200) ** 2 < 65 ** 2] = .03
    texture = .5 + .22 * np.sin(2 * np.pi * x / 13) + .18 * np.cos(2 * np.pi * y / 31)
    cy, cx = np.mgrid[:384, :512]
    checkerboard = ((cy // 32 + cx // 32) % 2).astype(np.float64)
    return {'synthetic_edges.png': edges, 'synthetic_texture.png': texture,
            'checkerboard.png': checkerboard}


def png_bytes(array):
    """Quantize a COPY for display/download only; metrics use the raw arrays."""
    pixels = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).save(output, format='PNG')
    return output.getvalue()


def ensure_synthetic_files(directory):
    """Add missing generated examples; never overwrite any existing image."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename, array in synthetic_examples().items():
        path = directory / filename
        if not path.exists():
            path.write_bytes(png_bytes(array))


def bundled_images(directory):
    if not directory.is_dir():
        return []
    paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda p: p.name.lower())
    for path in paths:
        if path.resolve().parent != directory.resolve():
            raise ValueError(f'Example image must be inside images/: {path.name}')
    return paths


def comparison_panel(original, comparison, title):
    """Return a PNG with fixed image ranges and an explicitly amplified error map."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import ScalarFormatter
    fig = Figure(figsize=(13, 3.8), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 4)
    for ax, array, name in zip(axes[:3], [original, comparison['spatial'], comparison['fourier']],
                               ['Original', 'Spatial convolution', 'Fourier convolution']):
        ax.imshow(array, cmap='gray', vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_axis_off()
    difference = comparison['difference']
    upper = float(np.max(difference))
    # Zero differences get a finite scale but remain visibly zero.
    limit = upper if upper > 0 else np.finfo(np.float64).eps
    heatmap = axes[3].imshow(difference, cmap='magma', vmin=0, vmax=limit)
    axes[3].set_title('Absolute difference\nScale amplified to show numerical noise')
    axes[3].set_axis_off()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    bar = fig.colorbar(heatmap, ax=axes[3], fraction=.05, pad=.03, format=formatter)
    bar.set_label('Absolute intensity difference')
    fig.suptitle(title, fontsize=10)
    output = BytesIO()
    fig.savefig(output, format='png', dpi=140)
    return output.getvalue()


def difference_panel(difference):
    """Standalone labeled difference heatmap used by the web app."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import ScalarFormatter
    fig = Figure(figsize=(6, 4), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    limit = max(float(np.max(difference)), np.finfo(np.float64).eps)
    displayed = ax.imshow(difference, cmap='magma', vmin=0, vmax=limit)
    ax.set_axis_off()
    ax.set_title('Absolute difference — amplified numerical scale')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    fig.colorbar(displayed, ax=ax, format=formatter, label='Absolute intensity difference')
    output = BytesIO()
    fig.savefig(output, format='png', dpi=140)
    return output.getvalue()
