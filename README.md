# Single-Image Book Dimension Measurement

Local Streamlit application for the completed Computer Vision assignment.
Requires Python 3.10 or newer and the existing project files.

## Run locally

Create the virtual environment once, if it does not already exist:

```bash
python3 -m venv .venv
```

Activate the environment, install dependencies, and start the webpage:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.
Stop the server with **Ctrl+C**. The application is local; deployment is not needed.

## Measure a book

1. Upload a JPG, JPEG, or PNG from the calibrated rear phone camera. The decoded
   image must be portrait **3024 × 4032 pixels**. Use the same camera/lens and 1× zoom.
2. Enter the perpendicular distance from the rear camera lens to the cover plane,
   in meters. Keep the cover flat and approximately parallel to the camera image plane.
3. Click exactly four cover corners: **top-left, top-right, bottom-right, bottom-left**.
   The page numbers each point and lists its original-image pixel coordinates.
4. Use **Reset Corners** to correct a selection, then press **Calculate**.
5. Download the annotated PNG and JSON result. These are session downloads;
   the app does not append to or overwrite the completed assignment measurements.

A new upload clears the selection. Changing distance invalidates the previous
result; press Calculate again. After four corners, further clicks are ignored.
An invalid corner order or mismatched image resolution produces an error.

## Measurement implementation

`app.py` imports `load_calibration()` and `measure_at_depth()` directly from
`measure_book.py`, and `validate_quad()` from `measurement_utils.py`. It uses
`calibration_output/calibration.npz` as the final calibration. There is no separate
web measurement formula: `cv2.undistortPoints()` and the constant-depth projection
are the same code used by the command-line script.

Uploaded bytes are decoded with OpenCV's `IMREAD_GRAYSCALE` and default EXIF
orientation handling, matching calibration and the CLI. Browser click coordinates
are scaled back from the displayed image to the original image before measurement.
The entered distance is treated as camera-space depth Z. Lens-to-cover distance
approximates this depth; cover tilt or a diagonal distance can introduce error.

The interactive component is
[`streamlit-image-coordinates`](https://github.com/blackary/streamlit-image-coordinates).
The app checks each event's timestamp to prevent repeated Streamlit reruns from
adding duplicate clicks.

## Evaluation Summary

The page reads aggregate values from `results/evaluation_metrics.json` and shows:

| Metric | Value |
|---|---:|
| Width MAE | 1.05 cm |
| Width MAPE | 7.29% |
| Height MAE | 2.13 cm |
| Height MAPE | 9.71% |

It displays only these existing evaluation figures:

- `results/figures/predicted_vs_true_width.png`
- `results/figures/predicted_vs_true_height.png`
- `results/figures/absolute_error_vs_distance.png`

No reference-photo files, reference annotations, or reference-object details are
loaded into the webpage. The raw evaluation JSON and internal source paths are
not displayed. The app does not change calibration or the completed 20-trial data.

## Command-line tools

Run these commands from the project directory with `.venv` activated. The
calibration and 20-trial evaluation are already complete; starting the webpage
only requires the setup/run commands above.

### Measure a book from a phone image

```bash
python3 measure_book.py --image book_measurement_images/book_01_2p134m.jpeg --distance-m 2.134 --id book_01
```

Replace the image, distance, and book ID for a new trial. Matplotlib asks for
four book-cover corners in top-left, top-right, bottom-right, bottom-left order.
Results go to `results/measurements/` and `results/measurements.csv`.
Rerunning a trial replaces its JSON/annotation and appends another CSV row.

### Measure ground truth from a reference photo

```bash
python3 measure_ground_truth.py --image reference_images/book_01_reference.jpeg --id book_01
```

Select the four outer reference-rectangle corners, then the four book-cover
corners, each in top-left, top-right, bottom-right, bottom-left order. Orientation
is detected automatically for a top-down image. To explicitly use the vertical
orientation present in this example:

```bash
python3 measure_ground_truth.py --image reference_images/book_01_reference.jpeg --id book_01 --plate-orientation portrait
```

`--plate-orientation` accepts `auto`, `portrait`, or `landscape`. Use `landscape`
only when the known long side runs from top-left to top-right. Results go to
`results/ground_truth/`; rerunning replaces that book's JSON and masked annotation.
This is a local CLI tool and is not part of the webpage.

### Evaluate the completed 20 trials

```bash
python3 evaluate_results.py
```

Reads the saved measurements and ground truth, checks all 20 trials, and writes
`results/final_trial_results.csv`, `results/evaluation_summary.txt`,
`results/evaluation_metrics.json`, and the three figures under `results/figures/`.
It does not change measurements. Missing fields, conflicting records, or a CSV
with other than 20 trial rows cause a clear error.

The equivalent command with explicit input and output directories is:

```bash
python3 evaluate_results.py --results-dir results --output-dir results
```

### Calibration diagnostics and rebuilding (optional)

These steps are already complete. Use them when investigating or rebuilding
calibration; they are not required to run the webpage.

Test detection on one image without recalibrating:

```bash
python3 calibrate_camera.py --single-image IMG_9050.jpeg
```

Compare candidate corner patterns on the diagnostic image:

```bash
python3 diagnose_checkerboard.py
```

Run detection and calibration on the calibration-image folder:

```bash
python3 calibrate_camera.py
```

A successful full run replaces `calibration_output/calibration.npz` with that
run's calibration. The selected final model for this assignment is the existing
**subset B / FIX_K3** candidate. To install or restore that saved final candidate:

```bash
cp calibration_output/calibration_B_fix_k3.npz calibration_output/calibration.npz
```

This copies the existing candidate; it does not recalibrate or modify the source
candidate. The calibration comparison reports and candidate archives are already
saved under `calibration_output/`.

### Syntax checks

```bash
python3 -m py_compile app.py calibrate_camera.py diagnose_checkerboard.py measure_book.py measure_ground_truth.py measurement_utils.py evaluate_results.py
```

### Command help

```bash
python3 calibrate_camera.py --help
python3 measure_book.py --help
python3 measure_ground_truth.py --help
python3 evaluate_results.py --help
streamlit run --help
```

`diagnose_checkerboard.py` runs directly and has no command-line options.

### Stop and exit

Press **Ctrl+C** in the terminal running Streamlit to stop the webpage. To leave
the virtual environment afterward:

```bash
deactivate
```
