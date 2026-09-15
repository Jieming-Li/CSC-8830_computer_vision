#!/usr/bin/env python3
"""Measure a book in a reference photo using a known rectangular plate.

Usage:
    python3 measure_ground_truth.py --image reference_images/book_01_reference.jpeg --id book_01

The plate's outer rectangle is 30.48 cm wide and 15.24 cm high. Select its
corners TL, TR, BR, BL, then select the book-cover corners in the same order.
The plate and book cover must be flat and coplanar. A single planar homography
corrects perspective, but cannot correct depth differences or lens distortion;
use a top-down reference photo with negligible lens distortion for this stage.
The plate may be horizontal or vertical. By default, its orientation is inferred
from the average lengths of the clicked image edges, appropriate for top-down
photos. For a strongly foreshortened plate, set --plate-orientation explicitly:
landscape means its 30.48 cm edge runs TL to TR; portrait means it runs TL to BL.
The program prints the chosen physical mapping before measuring the book.

Only plate corners and dimensions are used. No OCR or plate-number text is
collected. The selected plate region is masked before book selection and in
all saved annotations. No unmasked photo copies are saved. Outputs for a book
ID replace that ID's earlier ground-truth JSON and annotated PNG.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np

from measurement_utils import (BASE_DIR, annotated_figure, edge_dimensions,
                               load_gray, pyplot_interactive, save_json,
                               select_corners, validate_id, validate_quad)

PLATE_WIDTH_CM = 30.48
PLATE_HEIGHT_CM = 15.24


def resolve_plate_orientation(plate_corners, orientation='auto'):
    """Assign the known long/short physical sides to the clicked edge pairs.

    Auto uses image edge lengths and assumes a roughly top-down reference.
    A projective view can shorten either pair, so an explicit override is
    available. We never infer the book's size or force it to match another run.
    """
    points = validate_quad(plate_corners)
    if orientation not in ('auto', 'landscape', 'portrait'):
        raise ValueError('Plate orientation must be auto, landscape, or portrait')
    if orientation == 'auto':
        edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        horizontal = float((edges[0] + edges[2]) / 2)
        vertical = float((edges[1] + edges[3]) / 2)
        if max(horizontal, vertical) / min(horizontal, vertical) < 1.2:
            raise ValueError('Plate orientation is ambiguous in this view. Specify '
                             '--plate-orientation landscape or portrait based on '
                             'which clicked edges are the physical 30.48 cm sides.')
        orientation = 'landscape' if horizontal > vertical else 'portrait'
    return orientation


def plate_plane_dimensions(orientation):
    """Return centimeter X/Y extents; physical plate size stays 30.48 x 15.24."""
    if orientation == 'portrait':
        return PLATE_HEIGHT_CM, PLATE_WIDTH_CM
    if orientation == 'landscape':
        return PLATE_WIDTH_CM, PLATE_HEIGHT_CM
    raise ValueError('Resolve the plate orientation before assigning dimensions')


def plate_homography(plate_corners, orientation='auto'):
    """Map pixels to centimeters with the actual plate orientation and TL origin."""
    orientation = resolve_plate_orientation(plate_corners, orientation)
    width_cm, height_cm = plate_plane_dimensions(orientation)
    rectangle_cm = np.array([[0, 0], [width_cm, 0],
                             [width_cm, height_cm], [0, height_cm]], np.float32)
    matrix = cv2.getPerspectiveTransform(np.asarray(plate_corners, np.float32), rectangle_cm)
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError('Degenerate plate selection; choose four distinct outer corners')
    return matrix


def transform_book(corners, homography):
    """Transform the cover and average opposite edge lengths in centimeters."""
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    homogeneous = np.column_stack([points, np.ones(4)]) @ homography.T
    denominator = homogeneous[:, 2]
    # A cover crossing the homography's vanishing line is not a usable planar
    # selection. Reject it rather than silently returning enormous dimensions.
    if (np.any(np.abs(denominator) < 1e-10)
            or not (np.all(denominator > 0) or np.all(denominator < 0))):
        raise ValueError('Book selection crosses or touches the homography vanishing line')
    centimeters = cv2.perspectiveTransform(points.reshape(4, 1, 2), homography).reshape(4, 2)
    result = edge_dimensions(centimeters)
    result['book_corners_cm'] = centimeters.tolist()
    return result


def mask_plate(gray, corners):
    """Remove the entire plate region plus a small margin from output pixels."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 255)
    margin = max(5, round(max(gray.shape) * 0.003))
    mask = cv2.dilate(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    redacted = gray.copy()
    redacted[mask != 0] = 32
    return redacted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--id', required=True)
    parser.add_argument('--plate-orientation', choices=('auto', 'landscape', 'portrait'),
                        default='auto', help='Default: infer orientation for a top-down view; '
                        'portrait places the 30.48 cm side along TL to BL')
    args = parser.parse_args()
    try:
        book_id = validate_id(args.id)
        gray = load_gray(args.image)
        plate = select_corners(gray, 'Select the four OUTER PLATE corners (30.48 x 15.24 cm)')
        orientation = resolve_plate_orientation(plate, args.plate_orientation)
        plane_width, plane_height = plate_plane_dimensions(orientation)
        print(f'Plate orientation: {orientation}; TL to TR = {plane_width:.2f} cm; '
              f'TL to BL = {plane_height:.2f} cm.', flush=True)
        homography = plate_homography(plate, orientation)
        # All subsequent visualizations use masked pixels, never the original.
        redacted = mask_plate(gray, plate)
        book = select_corners(redacted, 'Select the four BOOK-COVER corners')
        dimensions = transform_book(book, homography)
        output = BASE_DIR / 'results' / 'ground_truth'
        json_path = output / (book_id + '.json')
        annotation_path = output / 'annotated' / (book_id + '.png')
        result = {
            'id': book_id, 'image': str(args.image),
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'image_width': gray.shape[1], 'image_height': gray.shape[0],
            'corner_order': ['top-left', 'top-right', 'bottom-right', 'bottom-left'],
            'plate_width_cm': PLATE_WIDTH_CM, 'plate_height_cm': PLATE_HEIGHT_CM,
            'plate_orientation': orientation, 'plate_orientation_requested': args.plate_orientation,
            'plate_plane_width_cm': plane_width, 'plate_plane_height_cm': plane_height,
            'plate_corners_px': plate.tolist(), 'book_corners_px': book.tolist(),
            'homography_pixels_to_cm': homography.tolist(), **dimensions,
            'method': 'coplanar reference rectangle homography',
            'annotated_image': str(annotation_path),
        }
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        title = (f'{book_id} | Reference measurement\n'
                 f'Width: {dimensions["width_cm"]:.2f} cm | Height: {dimensions["height_cm"]:.2f} cm\n'
                 f'Reference plate: {orientation}, {plane_width:.2f} x {plane_height:.2f} cm (masked)')
        fig = annotated_figure(redacted, book, title, extra_outline=plate)
        try:
            fig.savefig(annotation_path, dpi=160)
            save_json(json_path, result)
            print(f'Reference width: {dimensions["width_cm"]:.2f} cm; '
                  f'height: {dimensions["height_cm"]:.2f} cm')
            print(f'Saved: {json_path}\nSaved: {annotation_path}')
            pyplot_interactive().show()
        finally:
            pyplot_interactive().close(fig)
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nReference measurement cancelled.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
