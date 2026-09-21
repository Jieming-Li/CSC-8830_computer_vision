#!/usr/bin/env python3
"""Measure a planar, front-facing book cover using calibrated camera rays.

Usage:
    python3 measure_book.py --image book_measurement_images/book_01_2p134m.jpeg --distance-m 2.134 --id book_01

Assumptions: the book cover is flat and parallel to the camera image plane.
--distance-m is the perpendicular distance from the camera's optical center to
that cover plane (camera-space depth Z), in meters. It is not a slanted range
to a corner. Each undistorted normalized ray (x, y, 1) intersects Z=distance-m
at (x*Z, y*Z, Z). A tilted cover violates this constant-depth assumption.
Use the same rear camera, portrait orientation, resolution, and 1x zoom as
calibration. Lens distortion is corrected for geometry using undistortPoints;
annotations remain on the original decoded image at the clicked coordinates.

Select four corners: top-left, top-right, bottom-right, bottom-left.
JSON and annotated PNG files use the input image stem; repeated runs replace
those files and append a new timestamped record to results/measurements.csv.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

# Support both package imports from the shared app and direct CLI execution.
if __package__:
    from .measurement_utils import (BASE_DIR, annotated_figure, edge_dimensions,
                                    load_gray, pyplot_interactive, save_json,
                                    select_corners, validate_id)
else:
    from measurement_utils import (BASE_DIR, annotated_figure, edge_dimensions,
                                   load_gray, pyplot_interactive, save_json,
                                   select_corners, validate_id)

CALIBRATION_PATH = BASE_DIR / 'calibration_output' / 'calibration.npz'


def load_calibration(path=CALIBRATION_PATH):
    with np.load(path, allow_pickle=False) as data:
        required = {'camera_matrix', 'dist_coeffs', 'image_width', 'image_height'}
        if not required.issubset(data.files):
            raise ValueError(f'Calibration archive is missing: {sorted(required - set(data.files))}')
        matrix = np.asarray(data['camera_matrix'], dtype=np.float64)
        distortion = np.asarray(data['dist_coeffs'], dtype=np.float64).reshape(-1)
        width, height = int(data['image_width']), int(data['image_height'])
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or distortion.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all() or min(width, height) <= 0):
        raise ValueError('Invalid intrinsic matrix, distortion coefficients, or calibration resolution')
    return matrix, distortion, (width, height)


def measure_at_depth(corners, matrix, distortion, distance_m):
    """Convert pixels to undistorted rays, then to XY centimeters at fixed Z."""
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError('--distance-m must be a finite positive number')
    pixels = np.asarray(corners, dtype=np.float64).reshape(4, 1, 2)
    # No P matrix: output is normalized camera coordinates, NOT pixels.
    normalized = cv2.undistortPoints(pixels, matrix, distortion).reshape(4, 2)
    planar_cm = normalized * distance_m * 100.0
    result = edge_dimensions(planar_cm)
    result.update(normalized_corners=normalized.tolist(), planar_corners_cm=planar_cm.tolist())
    return result


def csv_row(result):
    """Store every JSON result field in CSV, encoding nested values as JSON."""
    return {key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
            for key, value in result.items()}


def check_csv_header(path, fields):
    if path.exists() and path.stat().st_size:
        with path.open(newline='', encoding='utf-8') as stream:
            if next(csv.reader(stream), None) != fields:
                raise ValueError(f'Existing CSV has a different column layout: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--distance-m', required=True, type=float)
    parser.add_argument('--id', required=True)
    args = parser.parse_args()
    try:
        book_id = validate_id(args.id)
        if not np.isfinite(args.distance_m) or args.distance_m <= 0:
            raise ValueError('--distance-m must be a finite positive number')
        matrix, distortion, size = load_calibration()
        gray = load_gray(args.image)
        actual = (gray.shape[1], gray.shape[0])
        if actual != size:
            raise ValueError(f'Image resolution mismatch: loaded {actual[0]} x {actual[1]}, '
                             f'calibration requires {size[0]} x {size[1]}. Use a matching original photo.')
        print('Assumption: flat, front-facing cover; entered distance is perpendicular camera-to-cover depth.')
        corners = select_corners(gray, 'Select the four BOOK-COVER corners')
        dimensions = measure_at_depth(corners, matrix, distortion, args.distance_m)
        output = BASE_DIR / 'results'
        json_path = output / 'measurements' / (args.image.stem + '.json')
        annotation_path = output / 'measurements' / 'annotated' / (args.image.stem + '.png')
        csv_path = output / 'measurements.csv'
        result = {
            'id': book_id, 'image': str(args.image),
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'distance_m': args.distance_m,
            'distance_definition': 'perpendicular optical-center-to-cover depth Z',
            'image_width': size[0], 'image_height': size[1],
            'corner_order': ['top-left', 'top-right', 'bottom-right', 'bottom-left'],
            'corners_px': corners.tolist(), **dimensions,
            'calibration_file': str(CALIBRATION_PATH),
            'calibration_sha256': hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest(),
            'annotated_image': str(annotation_path),
        }
        row = csv_row(result)
        check_csv_header(csv_path, list(row))
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        title = (f'{book_id} | Distance: {args.distance_m:g} m\n'
                 f'Width: {dimensions["width_cm"]:.2f} cm | Height: {dimensions["height_cm"]:.2f} cm')
        fig = annotated_figure(gray, corners, title)
        try:
            fig.savefig(annotation_path, dpi=160)
            save_json(json_path, result)
            new_csv = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open('a', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if new_csv:
                    writer.writeheader()
                writer.writerow(row)
            print(f'Width: {dimensions["width_cm"]:.2f} cm; height: {dimensions["height_cm"]:.2f} cm')
            print(f'Saved: {json_path}\nSaved: {annotation_path}\nAppended: {csv_path}')
            pyplot_interactive().show()
        finally:
            pyplot_interactive().close(fig)
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error, csv.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nMeasurement cancelled.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
