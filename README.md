# CSC 8830 Computer Vision — Jieming Li

One Streamlit application serves **Home**, **Module 2**, and **Module 3**. The
repository-root **`app.py` remains the deployment entry point**. Each assignment
keeps its implementation and assets in its own directory.

## Run locally

From the `CSC-8830_computer_vision` repository root, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If you already use the existing Module 3 environment, you can instead run:

```bash
source module3/.venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Use the local URL printed by Streamlit, normally `http://localhost:8501`.
The sidebar and Home links open both assignments in the same application.
Stop with **Ctrl+C**; leave the environment with `deactivate`.

## Pages and behavior

- **Home:** descriptions and links to both assignments.
- **Module 2** (`/module-2`): existing calibrated book measurement, four-corner
  selection, camera-to-book distance, evaluation summary, and PNG/JSON downloads.
  The required decoded image resolution remains 3024 × 4032. Measurement uses
  the same `module2/measure_book.py` functions and final calibration archive.
- **Module 3** (`/module-3`): existing Box/Gaussian spatial and Fourier comparison,
  image upload or bundled examples, numerical metrics, heatmap, and downloads.
  The same `module3/filtering.py` is shared with the experiment runner.

**Switching pages starts a fresh interactive selection. Download outputs before
leaving a module.** Widgets/results have separate `module2.` and `module3.` keys.
Page changes clear previous interactive state, including uploads and click events;
ordinary reruns on a page preserve its current work. Module 3 calculates only when
**Run both methods** is pressed. Module 2 calculations still require four corners
and **Calculate**. Uploaded files and downloadable results stay in session memory.

The shared app does not rerun calibration, batch experiments, or evaluation. It does
not overwrite completed trials, calibration archives, saved figures, or reports.
Module 2 reference photos and ground-truth records are not loaded or exposed by
any webpage.

## Organization

```text
app.py                         # Stable deployment entry point and navigation
home.py                        # Assignment descriptions and page links
requirements.txt               # Includes both modules' dependency lists
pytest.ini                     # Shared and existing Module 3 test discovery
tests/test_shared_app.py        # Navigation, measurement regression, state checks
module2/
  page.py                      # Thin shared-app adapter
  app.py                       # Existing measurement UI; also runs standalone
  measure_book.py               # Shared CLI/web dimension-estimation logic
  measurement_utils.py         # Existing geometry and image-loading helpers
  calibration_output/          # Existing final calibration and reports
  results/                     # Existing completed trials and evaluation figures
  README.md                    # Module 2 setup and all local CLI commands
module3/
  page.py                      # Thin shared-app adapter
  app.py                       # Existing blur UI; also runs standalone
  filtering.py                 # Unchanged spatial and FFT implementations
  image_utils.py               # Existing preprocessing and display helpers
  run_experiments.py            # Existing batch runner
  images/                      # Photographs and generated examples
  results/                     # Existing measured comparisons and figures
  submission/                  # Existing Word report
  tests/                       # Existing correctness and standalone UI tests
  README.md                    # Module 3 setup, experiments, and report workflow
```

Asset locations resolve from each module's `__file__`, independently of the working
directory. Package imports keep modules separate. OpenCV uses the headless 4.x
package for the web server; Module 2's calibration/measurement functions retain
their existing behavior. No system OpenCV GUI library is required by Streamlit.

## Tests and syntax checks

From the repository root with the environment activated:

```bash
python -m pytest -q
python -m py_compile app.py home.py module2/app.py module2/page.py module2/measure_book.py module3/app.py module3/page.py
```

The test suite includes all existing Module 3 correctness tests, regression checks
against all 20 saved Module 2 measurements, and shared-page navigation/state tests.
These checks read completed data without changing it. The shared requirements select Streamlit 1.64 or newer for uploader/page-testing
support. Each module retains its standalone dependency list.

## Local verification

The combined setup passed **107 tests**, including all 81 existing Module 3 tests
and regression checks reproducing all 20 saved Module 2 measurements. Browser
checks covered real four-corner input, measurement/downloads, Box/Gaussian filters,
uploads, direct module links, and repeated page switching. Syntax checks and
`pip check` also passed. Hash checks confirmed 167 protected assets/helpers were
unchanged, including calibration archives, images, completed results, and the
Module 3 report. No push or deployment was performed.

## Standalone module commands

Both original apps remain available independently:

```bash
streamlit run module2/app.py
streamlit run module3/app.py
```

The existing module-specific CLI commands remain documented in
[Module 2 README](module2/README.md) and [Module 3 README](module3/README.md).
For example, to deliberately rerun the Module 3 experiments later:

```bash
python module3/run_experiments.py --photos photo_01.jpg photo_02.jpg
```

That command replaces Module 3 experiment outputs; it is unnecessary for launching
or testing the shared webpage.

## Existing deployment

No push or deployment is performed by this change. When publication is separately
authorized, keep the existing Streamlit app, repository/branch, and main-file path
**`app.py`**. Include the root files, both module folders, and the existing assets
used by their pages. Install the root `requirements.txt`. The same deployment can
then serve Home and both module routes under its existing base URL; a second
Streamlit app is unnecessary.

Navigation uses Streamlit's [st.navigation](https://docs.streamlit.io/develop/api-reference/navigation/st.navigation).
The explicit page-reset behavior accounts for Streamlit's
[inactive-widget cleanup](https://docs.streamlit.io/develop/concepts/multipage-apps/widgets).
