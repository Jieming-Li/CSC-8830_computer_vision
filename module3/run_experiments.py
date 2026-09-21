#!/usr/bin/env python3
"""Run Module 3 comparisons locally (no downloads, deployment, or image edits).

Usage from module3/:
    python run_experiments.py
    python run_experiments.py --photos photo_01.jpg photo_02.jpg

Two existing images/ photographs plus a generated checkerboard give the final
18 comparisons. Missing photos are replaced ONLY by labeled synthetic examples;
the report explicitly marks this as provisional. --photos requires both named
files and never falls back for a missing explicit input. Results replace only
this runner's generated output files inside results/.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

from filtering import RTOL, ATOL, make_kernel, run_comparison
from image_utils import (BASE_DIR, MAX_DIMENSION, SYNTHETIC_FILES, bundled_images,
                         comparison_panel, ensure_synthetic_files, load_grayscale)

KERNEL_SIZES = (3, 7, 15)


def select_inputs(photo_names=None):
    directory = BASE_DIR / 'images'
    ensure_synthetic_files(directory)
    available = bundled_images(directory)
    photos = [p for p in available if p.name not in SYNTHETIC_FILES]
    if photo_names:
        photos = []
        for filename in photo_names:
            path = directory / filename
            if Path(filename).name != filename or not path.is_file() or path not in available:
                raise ValueError(f'Missing/invalid photograph: images/{filename}; pass a filename in images/.')
            if filename in SYNTHETIC_FILES:
                raise ValueError(f'{filename} is a labeled generated pattern, not a photograph.')
            photos.append(path)
        if photos[0] == photos[1]:
            raise ValueError('Choose two distinct photographs.')
    chosen = [(p, 'photograph') for p in photos[:2]]
    for filename in SYNTHETIC_FILES[:2]:
        if len(chosen) == 2:
            break
        chosen.append((directory / filename, 'synthetic placeholder (not a photograph)'))
    chosen.append((directory / 'checkerboard.png', 'generated checkerboard'))
    return chosen, [p.name for p in photos[2:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--photos', nargs=2, metavar=('PHOTO_1', 'PHOTO_2'),
                        help='Choose exactly two existing image filenames within images/')
    args = parser.parse_args()
    try:
        chosen, unused = select_inputs(args.photos)
        output = BASE_DIR / 'results'
        output.mkdir(exist_ok=True)
        panels = output / 'panels'
        panels.mkdir(exist_ok=True)
        final_inputs = sum(kind == 'photograph' for _, kind in chosen) == 2
        settings = {'run_utc': datetime.now(timezone.utc).isoformat(),
                    'final_photographs_available': final_inputs,
                    'status': 'final photo inputs' if final_inputs else 'provisional: missing photographs',
                    'expected_comparisons': 18, 'kernel_sizes': list(KERNEL_SIZES),
                    'filter_types': ['box', 'gaussian'], 'gaussian_sigma_rule': 'kernel_size / 6',
                    'max_dimension': MAX_DIMENSION, 'dtype': 'float64',
                    'rtol': RTOL, 'atol': ATOL,
                    'agreement_rule': 'abs(Fourier - spatial) <= atol + rtol * abs(spatial), at every pixel',
                    'boundary': 'zero padding', 'output_mode': 'same, centered',
                    'fft_shape_rule': '(H + Kh - 1, W + Kw - 1)',
                    'kernel_fft_convention': 'ordinary unshifted discrete kernel',
                    'crop_start': '((Kh - 1) // 2, (Kw - 1) // 2)',
                    'image_display_range': [0, 1],
                    'difference_display': 'linear auto scale per panel, labeled; numerical noise is amplified',
                    'unused_photo_candidates': unused,
                    'software': {'Python': platform.python_version(),
                                 **{name: importlib.metadata.version(name)
                                    for name in ['numpy', 'scipy', 'Pillow', 'matplotlib', 'streamlit']}},
                    'inputs': []}
        rows = []
        for image_index, (path, kind) in enumerate(chosen, 1):
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            image, metadata = load_grayscale(path)
            settings['inputs'].append({'filename': path.name, 'kind': kind, 'sha256': source_hash, **metadata})
            for filter_type in ('box', 'gaussian'):
                for size in KERNEL_SIZES:
                    sigma = size / 6 if filter_type == 'gaussian' else None
                    kernel = make_kernel(filter_type, size, sigma)
                    comparison = run_comparison(image, kernel)
                    panel_name = f'{image_index:02d}_{filter_type}_{size:02d}.png'
                    title = (f'{path.name} [{kind}] | {image.shape[1]} x {image.shape[0]} | '
                             f'{filter_type}, k={size}, sigma={sigma if sigma is not None else "N/A"}')
                    (panels / panel_name).write_bytes(comparison_panel(image, comparison, title))
                    rows.append({'image_filename': path.name, 'input_kind': kind,
                                 'processed_width': image.shape[1], 'processed_height': image.shape[0],
                                 'filter_type': filter_type, 'kernel_size': size,
                                 'sigma': sigma if sigma is not None else '',
                                 **comparison['metrics'], **comparison['fft'],
                                 'kernel_sum': float(kernel.sum()),
                                 'panel_file': str(Path('panels') / panel_name)})
                    print(f'{len(rows):02d}/18 {path.name} {filter_type} {size}: '
                          f'max difference={comparison["metrics"]["max_absolute_difference"]:.3e}, '
                          f'agrees={comparison["metrics"]["agrees"]}', flush=True)
            if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
                raise RuntimeError(f'Input changed during experiments: {path.name}')
        with (output / 'comparison_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            # Python's float representation preserves round-trip float64 precision.
            writer.writerows(rows)
        settings['completed_comparisons'] = len(rows)
        settings['agreeing_comparisons'] = sum(row['agrees'] for row in rows)
        settings['largest_max_absolute_difference'] = max(row['max_absolute_difference'] for row in rows)
        settings['largest_mean_absolute_difference'] = max(row['mean_absolute_difference'] for row in rows)
        settings['largest_rmse'] = max(row['rmse'] for row in rows)
        (output / 'experiment_settings.json').write_text(json.dumps(settings, indent=2, allow_nan=False) + '\n')
        summary = [f'Status: {settings["status"]}', f'Completed: {len(rows)}/18 comparisons',
                   f'Agreed: {settings["agreeing_comparisons"]}/18; rtol={RTOL:g}, atol={ATOL:g}',
                   f'Largest maximum absolute difference: {settings["largest_max_absolute_difference"]:.17g}',
                   f'Largest MAE: {settings["largest_mean_absolute_difference"]:.17g}',
                   f'Largest RMSE: {settings["largest_rmse"]:.17g}']
        if not final_inputs:
            summary.append('Add two real photographs, e.g. images/photo_01.jpg and images/photo_02.jpg, '
                           'then run python run_experiments.py --photos photo_01.jpg photo_02.jpg.')
        (output / 'experiment_summary.txt').write_text('\n'.join(summary) + '\n')
        print('\n' + '\n'.join(summary))
        return 0 if settings['agreeing_comparisons'] == 18 else 1
    except (OSError, ValueError, FloatingPointError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
