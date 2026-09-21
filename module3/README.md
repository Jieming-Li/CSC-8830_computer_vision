# CSC 8830 — Module 3: Spatial and Fourier Image Blurring

A local Python/Streamlit demonstration that discrete spatial convolution agrees
with multiplication in the Fourier domain when the kernel, padding, and crop match.
All project paths are relative to this `module3/` folder. Nothing in Module 2 is used.

For the shared Home / Module 2 / Module 3 application, launch the repository-root
`app.py`; see the [shared README](../README.md). The commands below still run this
module independently.

## Setup and run locally

From the `CSC-8830_computer_vision` repository root:

```bash
cd module3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer is supported; Python 3.12 is a suitable choice for a new environment.
The required libraries are NumPy, SciPy, Pillow, Matplotlib, Streamlit, and pytest.
OpenCV, camera calibration, and a GPU are not used.

Run the independent correctness checks:

```bash
python -m pytest tests/ -v --basetemp=.cache/pytest --junitxml=results/test_results.xml
```

Run the 18 comparisons with the currently available inputs:

```bash
python run_experiments.py
```

Start the local web application:

```bash
streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open the local URL printed by Streamlit (normally `http://127.0.0.1:8501`).
Stop the server with **Ctrl+C**. Leave the virtual environment with `deactivate`.

Syntax-only check:

```bash
python -m py_compile app.py filtering.py image_utils.py run_experiments.py
```

## Photographs needed for the final experiment

At implementation time `images/` contained **no photographs**. The initial run
therefore uses these explicitly labeled generated inputs:

- `synthetic_edges.png`: gradient, rectangle, and circle; **not a photograph**.
- `synthetic_texture.png`: periodic texture; **not a photograph**.
- `checkerboard.png`: the required generated checkerboard.

The 18 comparisons using these images are **provisional**, not completion of the
final two-photograph experiment. No photographs were downloaded or fabricated.

Add two of your own photographs to `images/`, preferably:

- `images/photo_01.jpg`: a scene with strong edges and large smooth areas.
- `images/photo_02.jpg`: a scene with fine texture and varied detail.

Then run:

```bash
python run_experiments.py --photos photo_01.jpg photo_02.jpg
```

Use the actual filenames if different. Explicit missing/invalid inputs produce
an error. Without `--photos`, the runner selects the first two non-generated image
filenames alphabetically and records any unused candidates. Only place actual
photographs under those non-generated names. With fewer than two photos, the
runner fills missing slots with labeled synthetic examples and marks the output
provisional. Original image files are never overwritten. Missing synthetic assets
are created once; existing assets are preserved.

The runner produces **3 inputs × 2 filters × 3 sizes = 18 comparisons**. Kernel sizes
are 3, 7, and 15. Gaussian sigma is `kernel_size / 6`: 0.5, 7/6, and 2.5 pixels.
Images undergo EXIF correction, conversion to grayscale, aspect-preserving LANCZOS
resizing to a maximum dimension of 512, then float64 normalization to [0,1].
Processed dimensions and source hashes are recorded. Re-running experiments replaces
the generated metrics, settings, summary, and the same 18 numbered panels.
After a new run, refresh the measured-results section in `report_notes.md` using
the new CSV/settings; it records the initial run rather than silently rewriting itself.

## App walkthrough

1. Select a bundled example, or upload a JPG/JPEG/PNG/TIFF/WebP.
2. Inspect the grayscale preview and disclosed original/processed dimensions.
3. Choose Box or Gaussian, an odd kernel size (1–31), and positive Gaussian sigma.
4. Press **Run both methods**. No batch experiment runs on startup or widget reruns.
5. Compare the spatial/Fourier outputs and raw-array MAE, maximum difference,
   RMSE, and PASS/FAIL status.
6. Inspect the difference map's labeled color scale. The scale is amplified to
   reveal numerical roundoff, which need not represent a visible mismatch.
7. Download output PNGs, the four-image comparison panel, JSON metrics, or the
   raw float64 input/kernel/output arrays as NPZ.

Uploads stay in session memory and are not permanently saved. Changing input or
parameters invalidates the old comparison until the button is pressed again.
Bundled inputs are read only from `module3/images/`; the app needs no files from
another assignment directory. Zero padding can darken image boundaries. Images
and blur outputs share the [0,1] display range; PNG clipping/rounding is display-only.

## Implementation and files

```text
module3/
├── app.py                       # Streamlit interface; one selected comparison
├── filtering.py                 # Explicit spatial and numpy.fft implementations
├── image_utils.py               # EXIF/loading, synthetic inputs, PNG panels
├── run_experiments.py            # Reproducible 18-comparison runner
├── tests/
│   ├── test_filtering.py         # Independent scipy.signal.convolve2d reference
│   ├── test_app.py               # Button-only computation and state invalidation
│   └── test_image_handling.py    # Orientation, resizing, dtype/range checks
├── images/                      # Two user photos plus generated example patterns
├── results/
│   ├── comparison_metrics.csv   # Full-precision raw-array errors and agreement
│   ├── experiment_settings.json # Settings, dimensions, source hashes, versions
│   ├── experiment_summary.txt
│   ├── experiment_run.log
│   ├── app_verification.txt     # Local browser checks
│   ├── test_results.txt         # Captured test output
│   ├── test_results.xml
│   └── panels/                  # Original | spatial | Fourier | abs difference
├── requirements.txt
├── .gitignore
├── README.md
└── report_notes.md              # Theory and actual run notes for a later PDF
```

Spatial convolution loops over kernel entries and adds vectorized image slices,
using the flipped kernel. Fourier convolution uses the ordinary, unshifted
kernel with FFT shape `(H+Kh-1, W+Kw-1)`, multiplies transforms, checks imaginary
roundoff, and crops from `((Kh-1)//2, (Kw-1)//2)` to shape `(H,W)`.
There is no analytic frequency-domain Gaussian and no `fftconvolve` wrapper.
SciPy's `convolve2d` is an independent **test-only** reference.

The fixed agreement criterion is `abs(Fourier-spatial) <= 1e-12 + 1e-10*abs(spatial)`
at every pixel. Metrics are computed before any clipping or display conversion.
Virtual environments, caches, `.DS_Store`, and secrets are ignored; example images
and the small results required for review remain trackable.

## Optional future deployment — not performed

Publication, pushing to GitHub, and deployment require your approval. These are
instructions for a later approved deployment; none has been carried out here.

1. After approving publication, add the `module3/` source, `requirements.txt`,
   bundled examples, and small result files to the chosen GitHub repository/branch.
   Keep environments, caches, and secrets untracked.
2. Sign in to Streamlit Community Cloud and choose **Create app**.
3. Select the repository and branch. Set the entrypoint to **`module3/app.py`**
   when deploying from the shared assignment repository. If Module 3 is the
   repository root, use **`app.py`** instead.
4. Use Advanced settings to select a supported Python version, such as 3.12.
   This app does not require secrets. Ensure `module3/requirements.txt` is included.
5. Deploy, review build/runtime logs, and test uploads, comparisons, and downloads.
6. Fill the URL placeholders in `report_notes.md` and **link Module 3 from the shared
   assignment webpage**. That shared webpage is outside this task's edit scope.

See the official [Community Cloud deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Suggested screen recording (about 2 minutes)

Show the app title and synthetic/photograph label, select the checkerboard, run a
7×7 Box filter, and point out edge darkening. Switch to Gaussian with sigma 7/6,
run again, then explain the tiny errors and amplified difference color bar. Upload
one real photo to show resizing, run a 15×15 comparison, and demonstrate downloading
metrics and a panel. End with the tests and the experiment CSV/status, stating
whether the required two photographs have been supplied.
