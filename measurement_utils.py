"""Shared corner-selection, geometry, and output helpers for local measurement."""
from pathlib import Path
import json
import re

import cv2
import numpy as np

CORNER_LABELS = ('top-left', 'top-right', 'bottom-right', 'bottom-left')
BASE_DIR = Path(__file__).resolve().parent


def validate_id(value):
    """Keep IDs safe for output filenames without silently changing them."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError('--id must contain only letters, digits, underscores, or hyphens')
    return value


def load_gray(path):
    """Match calibration: direct grayscale decode with default EXIF handling.

    Do not use IMREAD_UNCHANGED or IMREAD_IGNORE_ORIENTATION, and do not apply
    another EXIF rotation afterward. Coordinates refer to these decoded pixels.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Image file does not exist: {path}')
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f'OpenCV could not decode image: {path}')
    return gray


def validate_quad(points, image_shape=None):
    """Reject incomplete, crossed, reversed, tiny, or nonfinite selections."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Exactly four finite corners are required: TL, TR, BR, BL')
    if image_shape is not None:
        height, width = image_shape[:2]
        if (np.any(points[:, 0] < 0) or np.any(points[:, 0] > width - 1)
                or np.any(points[:, 1] < 0) or np.any(points[:, 1] > height - 1)):
            raise ValueError('Every corner must be inside the displayed image')
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    # Image coordinates have Y downward, so TL -> TR -> BR -> BL is positive.
    if not np.all(turns > 0):
        raise ValueError('Corners must form a convex outline in TL, TR, BR, BL order')
    if np.min(np.linalg.norm(edges, axis=1)) < 1:
        raise ValueError('Selected corners are too close together; select distinct corners')
    return points


def pyplot_interactive():
    """Delay importing pyplot so non-GUI geometry checks can run independently."""
    import matplotlib.pyplot as plt
    return plt


def select_corners(gray, title):
    """Block for exactly four clicks, in original-image pixel coordinates.

    Matplotlib may scale the display window, but ginput returns image/data
    coordinates. Right-click undoes a point; closing early cancels the operation.
    """
    plt = pyplot_interactive()
    fig, ax = plt.subplots(figsize=(9, 10))
    try:
        if fig.canvas.required_interactive_framework is None:
            raise RuntimeError('An interactive Matplotlib backend is required. '
                               'Run this script in a local macOS terminal with GUI access.')
        ax.imshow(gray, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title + '\nClick 1: top-left, 2: top-right, 3: bottom-right, 4: bottom-left'
                     '\nLeft-click: select; right-click: undo; close window: cancel', fontsize=10)
        ax.set_axis_off()
        fig.tight_layout()
        print(title, flush=True)
        print('Click exactly four corners: top-left, top-right, bottom-right, bottom-left.', flush=True)
        clicks = fig.ginput(n=4, timeout=0, show_clicks=True)
    finally:
        plt.close(fig)
    return validate_quad(clicks, gray.shape)


def edge_dimensions(points):
    """Return averaged width/height and individual edges, in input units."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Physical corner coordinates must be four finite 2D points')
    top, right, bottom, left = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return {'width_cm': float((top + bottom) / 2),
            'height_cm': float((left + right) / 2),
            'edge_lengths_cm': {'top': float(top), 'right': float(right),
                                'bottom': float(bottom), 'left': float(left)}}


def annotated_figure(gray, corners, title, extra_outline=None):
    """Render numbered corners and outline without changing the source image."""
    plt = pyplot_interactive()
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(gray, cmap='gray', vmin=0, vmax=255)
    points = np.asarray(corners)
    closed = np.vstack([points, points[0]])
    ax.plot(closed[:, 0], closed[:, 1], '-o', color='lime', linewidth=2, markersize=5)
    for i, (x, y) in enumerate(points):
        ax.annotate(f'{i + 1}: {CORNER_LABELS[i]}', (x, y), xytext=(6, 6),
                    textcoords='offset points', color='lime', fontsize=9,
                    bbox={'facecolor': 'black', 'alpha': 0.65, 'edgecolor': 'none'})
    if extra_outline is not None:
        closed = np.vstack([extra_outline, extra_outline[0]])
        ax.plot(closed[:, 0], closed[:, 1], '--', color='cyan', linewidth=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def save_json(path, result):
    """Atomically save readable JSON, refusing NaN or infinity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
