#!/usr/bin/env python3
"""Evaluate the 20 completed book trials without changing measurements.

Run from the project: python3 evaluate_results.py
Reads results/measurements.csv, checks it against every corresponding JSON in
results/measurements/, and joins dimensions from results/ground_truth/*.json.
The CSV supplies trial order; its image filename stem supplies the trial ID.
Numeric book IDs differing only in zero padding (book_5 / book_05) may match;
all such matches are explicitly reported. Ambiguous matches are errors.

Signed error = predicted - true. Percentage error is signed; the table also
includes absolute percentage error. MAPE uses absolute percentage errors.
Sample standard deviation uses n - 1 (ddof=1). No rows are filtered or deduped.
All required data is validated before evaluation outputs are written. No image,
calibration, or measurement scripts are opened or executed.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

EXPECTED_TRIALS = 20
BASE_DIR = Path(__file__).resolve().parent
REQUIRED_TRIAL = ('id', 'image', 'distance_m', 'width_cm', 'height_cm')
METRIC_LABELS = (
    ('mean_signed_error_cm', 'Mean signed error', 'cm'),
    ('mae_cm', 'Mean absolute error (MAE)', 'cm'),
    ('rmse_cm', 'Root mean squared error (RMSE)', 'cm'),
    ('mape_percent', 'Mean absolute percentage error (MAPE)', '%'),
    ('sample_std_signed_error_cm', 'Sample SD of signed error (n-1)', 'cm'),
    ('max_absolute_error_cm', 'Maximum absolute error', 'cm'),
)


class DataError(ValueError):
    """Missing, conflicting, or invalid source data; never omit a trial."""


def required(record, field, location):
    if field not in record or record[field] is None or record[field] == '':
        raise DataError(f'{location}: missing required field {field!r}')
    return record[field]


def positive_number(record, field, location):
    raw = required(record, field, location)
    try:
        if isinstance(raw, bool):
            raise ValueError('boolean is not a measurement')
        value = float(raw)
    except (TypeError, ValueError):
        raise DataError(f'{location}: field {field!r} must be a number, got {raw!r}')
    if not np.isfinite(value) or value <= 0:
        raise DataError(f'{location}: field {field!r} must be finite and positive, got {raw!r}')
    return value


def nonempty_text(record, field, location):
    value = required(record, field, location)
    if not isinstance(value, str) or not value.strip():
        raise DataError(f'{location}: field {field!r} must be nonempty text')
    return value


def canonical_book_id(book_id):
    """Normalize only numeric zero padding for joining; keep original IDs."""
    match = re.fullmatch(r'book_(\d+)', book_id)
    return f'book_{int(match.group(1)):02d}' if match else book_id


def read_json(path, hashes):
    if not path.is_file():
        raise DataError(f'Missing required file: {path}')
    payload = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(payload).hexdigest()
    try:
        record = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise DataError(f'{path}: invalid JSON: {exc}') from exc
    if not isinstance(record, dict):
        raise DataError(f'{path}: expected a JSON object')
    return record


def load_trials(results):
    """Read all 20 CSV records and reconcile their existing JSON counterparts."""
    hashes = {}
    source = results / 'measurements.csv'
    if not source.is_file():
        raise DataError(f'Missing required file: {source}')
    hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        for field in REQUIRED_TRIAL:
            if reader.fieldnames is None or field not in reader.fieldnames:
                raise DataError(f'{source}: missing required CSV column {field!r}')
        raw_trials = list(reader)
    if len(raw_trials) != EXPECTED_TRIALS:
        raise DataError(f'{source}: expected exactly {EXPECTED_TRIALS} trial rows; '
                        f'found {len(raw_trials)}. No rows will be excluded.')
    gt_dir = results / 'ground_truth'
    if not gt_dir.is_dir():
        raise DataError(f'Missing required directory: {gt_dir}')
    truth = {}
    for path in sorted(gt_dir.glob('*.json')):
        record = read_json(path, hashes)
        book_id = nonempty_text(record, 'id', path)
        key = canonical_book_id(book_id)
        if key in truth:
            raise DataError(f'Ambiguous ground truth for {key!r}: {truth[key][0]} and {path}')
        truth[key] = (path, record)
    trials, aliases, seen, used_json = [], {}, set(), set()
    for line_number, raw in enumerate(raw_trials, 2):
        location = f'{source}: CSV row {line_number}'
        if None in raw:
            raise DataError(f'{location}: extra CSV cells without column names')
        book_id = nonempty_text(raw, 'id', location)
        image_path = nonempty_text(raw, 'image', location)
        image_name = Path(image_path).name
        trial_id = Path(image_name).stem
        if not trial_id or trial_id in seen:
            raise DataError(f'{location}: missing or duplicate trial ID/image stem {trial_id!r}')
        seen.add(trial_id)
        values = {field: positive_number(raw, field, location)
                  for field in ('distance_m', 'width_cm', 'height_cm')}
        json_path = results / 'measurements' / f'{trial_id}.json'
        counterpart = read_json(json_path, hashes)
        used_json.add(json_path)
        for field in ('id', 'image'):
            other = nonempty_text(counterpart, field, json_path)
            same = Path(other) == Path(raw[field]) if field == 'image' else other == raw[field]
            if not same:
                raise DataError(f'{json_path}: field {field!r} conflicts with {location}')
        for field, value in values.items():
            if positive_number(counterpart, field, json_path) != value:
                raise DataError(f'{json_path}: field {field!r} conflicts with {location}')
        key = canonical_book_id(book_id)
        if key not in truth:
            raise DataError(f'{location}: missing ground-truth file for book ID {book_id!r}; '
                            f'expected {gt_dir / (book_id + ".json")} or a unique equivalent numeric ID')
        gt_path, gt = truth[key]
        gt_id = nonempty_text(gt, 'id', gt_path)
        if gt_id != book_id:
            aliases[book_id] = {'ground_truth_id': gt_id, 'file': str(gt_path)}
        trial = {
            'trial_id': trial_id, 'book_id': book_id, 'image_filename': image_name,
            'distance_m': values['distance_m'],
            'true_width_cm': positive_number(gt, 'width_cm', gt_path),
            'true_height_cm': positive_number(gt, 'height_cm', gt_path),
            'predicted_width_cm': values['width_cm'],
            'predicted_height_cm': values['height_cm'],
        }
        for dimension in ('width', 'height'):
            actual = trial[f'true_{dimension}_cm']
            error = trial[f'predicted_{dimension}_cm'] - actual
            trial[f'{dimension}_signed_error_cm'] = error
            trial[f'{dimension}_absolute_error_cm'] = abs(error)
            trial[f'{dimension}_percentage_error'] = 100 * error / actual
            trial[f'{dimension}_absolute_percentage_error'] = 100 * abs(error) / actual
        trial.update(measurement_json=str(json_path), ground_truth_id=gt_id,
                     ground_truth_json=str(gt_path), source_csv_row=line_number)
        trials.append(trial)
    extra = set((results / 'measurements').glob('*.json')) - used_json
    if extra:
        raise DataError('Trial JSON files not represented in the 20-row CSV: '
                        + ', '.join(str(p) for p in sorted(extra)))
    assert len(trials) == EXPECTED_TRIALS
    return trials, aliases, hashes


def calculate_metrics(predicted, actual):
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    if predicted.shape != actual.shape or predicted.ndim != 1 or predicted.size < 2:
        raise DataError('Metrics require matching 1D arrays with at least two trials')
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all() or np.any(actual <= 0):
        raise DataError('Metric inputs must be finite; true dimensions must be positive')
    signed = predicted - actual
    absolute = np.abs(signed)
    return {
        'mean_signed_error_cm': float(np.mean(signed)),
        'mae_cm': float(np.mean(absolute)),
        'rmse_cm': float(np.sqrt(np.mean(signed ** 2))),
        'mape_percent': float(np.mean(100 * absolute / actual)),
        'sample_std_signed_error_cm': float(np.std(signed, ddof=1)),
        'max_absolute_error_cm': float(np.max(absolute)),
        'underestimated_trials': int(np.sum(signed < 0)),
        'overestimated_trials': int(np.sum(signed > 0)),
        'exact_trials': int(np.sum(signed == 0)),
    }


def make_summary(trials, metrics, aliases, results):
    lines = ['BOOK MEASUREMENT EVALUATION', '=' * 72,
             f'Trials evaluated: {len(trials)} of {EXPECTED_TRIALS}; none excluded.',
             f'Books: {len(set(t["book_id"] for t in trials))}',
             'Distances (m): ' + ', '.join(f'{d:g}' for d in sorted({t['distance_m'] for t in trials})),
             f'Trial source: {results / "measurements.csv"}',
             'Each CSV record was checked against its measurement JSON.',
             f'Ground truth: {results / "ground_truth"}/*.json (saved reference estimates).',
             'Inputs are read-only. No image analysis, recalibration, or measurement changes.',
             '\nBook-ID joins:']
    lines.extend([f'  {book} -> {value["ground_truth_id"]} ({value["file"]}); zero-padding only.'
                  for book, value in aliases.items()] or ['  All IDs matched exactly.'])
    lines.extend(['\nMETRIC DEFINITIONS',
                  'Signed error e = predicted - true; positive = overestimate, negative = underestimate.',
                  'Mean signed error: mean(e), describing average bias (cm).',
                  'MAE: mean(abs(e)), the typical absolute deviation (cm).',
                  'RMSE: sqrt(mean(e^2)); larger errors receive more weight (cm).',
                  'Signed percentage error: 100*e/true; absolute percentage error: 100*abs(e)/true.',
                  'MAPE: mean(100*abs(e)/true), giving equal weight to each trial (%).',
                  'Sample SD: sqrt(sum((e-mean(e))^2)/(n-1)), spread around the bias (cm).',
                  'Maximum absolute error: max(abs(e)) (cm).'])
    for dimension in ('width', 'height'):
        lines.extend(['', dimension.upper() + ' METRICS', f'{"Metric":42} {"Value":>12}  Unit', '-' * 64])
        lines.extend(f'{label:42} {metrics[dimension][key]:12.6f}  {unit}'
                     for key, label, unit in METRIC_LABELS)
    lines.append('\nINTERPRETATION')
    for dimension in ('width', 'height'):
        m = metrics[dimension]
        direction = ('underestimation' if m['mean_signed_error_cm'] < 0 else
                     'overestimation' if m['mean_signed_error_cm'] > 0 else 'no net signed bias')
        lines.append(f'{dimension.capitalize()}: {direction} on average '
                     f'(mean signed error {m["mean_signed_error_cm"]:+.6f} cm); '
                     f'{m["underestimated_trials"]}/{len(trials)} below ground truth, '
                     f'{m["overestimated_trials"]}/{len(trials)} above, '
                     f'{m["exact_trials"]}/{len(trials)} exact.')
    lines.extend(['This describes the completed trials; it does not isolate a cause or establish significance.',
                  'Errors can arise from camera calibration, distance measurement, manual corner selection,',
                  'lens distortion, and imperfect front-facing alignment. Reference dimensions also have',
                  'uncertainty from the plate homography, manual clicks, and any difference in object planes.',
                  '\nOUTPUTS', 'final_trial_results.csv: every trial and its signed/absolute/percentage errors.',
                  'evaluation_metrics.json: aggregate metrics and source-file hashes.',
                  'figures/predicted_vs_true_width.png', 'figures/predicted_vs_true_height.png',
                  'figures/absolute_error_vs_distance.png'])
    return '\n'.join(lines) + '\n'


def make_figures(trials, output):
    # Use a noninteractive backend: the evaluator runs without GUI clicks.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    distances = sorted({t['distance_m'] for t in trials})
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    colors = plt.get_cmap('tab10')
    for dimension in ('width', 'height'):
        fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
        try:
            actual = np.array([t[f'true_{dimension}_cm'] for t in trials])
            predicted = np.array([t[f'predicted_{dimension}_cm'] for t in trials])
            low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
            pad = max((high - low) * .08, .2)
            limits = (low - pad, high + pad)
            ax.plot(limits, limits, '--', color='black', linewidth=1.2, label='Perfect agreement: y = x')
            for i, distance in enumerate(distances):
                group = [t for t in trials if t['distance_m'] == distance]
                ax.scatter([t[f'true_{dimension}_cm'] for t in group],
                           [t[f'predicted_{dimension}_cm'] for t in group],
                           color=colors(i % 10), marker=markers[i % len(markers)],
                           s=65, alpha=.85, label=f'{distance:g} m')
            ax.set(xlabel=f'True {dimension} (cm)', ylabel=f'Predicted {dimension} (cm)',
                   title=f'Predicted vs. true {dimension} — all {len(trials)} trials',
                   xlim=limits, ylim=limits)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.25)
            ax.legend(fontsize=9)
            fig.savefig(figures / f'predicted_vs_true_{dimension}.png', dpi=180)
        finally:
            plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout='constrained')
    try:
        books = sorted({t['book_id'] for t in trials})
        error_ceiling = max(t[f'{dimension}_absolute_error_cm']
                            for t in trials for dimension in ('width', 'height'))
        error_ceiling = max(error_ceiling * 1.10, 0.1)
        for ax, dimension in zip(axes, ('width', 'height')):
            for i, book in enumerate(books):
                group = sorted((t for t in trials if t['book_id'] == book), key=lambda t: t['distance_m'])
                ax.plot([t['distance_m'] for t in group],
                        [t[f'{dimension}_absolute_error_cm'] for t in group],
                        marker=markers[i % len(markers)], color=colors(i % 10),
                        linewidth=1, markersize=6, label=book)
            ax.set(xlabel='Camera-to-book distance (m)', ylabel='Absolute error (cm)',
                   title=f'{dimension.capitalize()} absolute error')
            ax.set_xticks(distances)
            ax.set_ylim(0, error_ceiling)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(f'Absolute error vs. distance — all {len(trials)} trials')
        fig.savefig(figures / 'absolute_error_vs_distance.png', dpi=180)
    finally:
        plt.close(fig)


def evaluate(results, output):
    trials, aliases, hashes = load_trials(results)
    metrics = {dimension: calculate_metrics([t[f'predicted_{dimension}_cm'] for t in trials],
                                           [t[f'true_{dimension}_cm'] for t in trials])
               for dimension in ('width', 'height')}
    summary = make_summary(trials, metrics, aliases, results)
    payload = {'trial_count': len(trials), 'book_count': len({t['book_id'] for t in trials}),
               'generated_utc': datetime.now(timezone.utc).isoformat(),
               'error_definition': 'predicted minus true',
               'percentage_error_definition': '100 * signed error / true',
               'mape_definition': 'mean(100 * abs(signed error) / true)',
               'sample_standard_deviation_ddof': 1,
               'trial_weighting': 'equal weight for each of the 20 trials',
               'width': metrics['width'], 'height': metrics['height'],
               'book_id_aliases': aliases, 'input_sha256': hashes}
    # Prepare the entire deliverable before replacing any evaluation output.
    with tempfile.TemporaryDirectory(prefix='book-evaluation-') as directory:
        temporary = Path(directory)
        with (temporary / 'final_trial_results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
        (temporary / 'evaluation_summary.txt').write_text(summary, encoding='utf-8')
        (temporary / 'evaluation_metrics.json').write_text(
            json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        make_figures(trials, temporary)
        for filename, digest in hashes.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise DataError(f'Source changed during evaluation: {filename}; refusing to publish results')
        for staged in sorted(temporary.rglob('*')):
            if staged.is_file():
                target = output / staged.relative_to(temporary)
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_name(target.name + '.pending')
                pending.write_bytes(staged.read_bytes())
                pending.replace(target)
    print(summary)
    print(f'Figures saved under: {output / "figures"}')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', type=Path, default=BASE_DIR / 'results',
                        help='Directory containing the completed CSV and JSON sources')
    parser.add_argument('--output-dir', type=Path,
                        help='Evaluation output directory; defaults to --results-dir')
    args = parser.parse_args()
    try:
        results = args.results_dir.resolve()
        evaluate(results, (args.output_dir or results).resolve())
        return 0
    except (OSError, DataError, csv.Error) as exc:
        print(f'ERROR: {exc}\nEvaluation stopped; no trial was silently excluded.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
