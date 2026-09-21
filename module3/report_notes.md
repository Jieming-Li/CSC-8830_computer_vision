# CSC 8830 Module 3 — report notes

## Objective

Implement image blurring by explicit spatial convolution and by multiplication
of Fourier transforms, then experimentally verify their numerical equivalence.
The same discrete box/Gaussian kernel, zero boundary extension, and centered crop
are essential. The local app supports inspection and downloadable comparisons.

## Spatial convolution

Let the image be `f[r,c]`, zero outside its finite domain, and let `h[u,v]` be the
ordinary stored discrete kernel. The full linear convolution is

\[
g[r,c] = (f*h)[r,c] = \sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1}
 h[u,v] f[r-u,c-v].
\]

For odd kernels, the same-size output is
`y[i,j] = g[i+(Kh-1)/2, j+(Kw-1)/2]`. The implementation zero-pads the image,
flips the kernel on both axes, and sums vectorized slices in loops over kernel
entries. It does not loop over individual image pixels or call a blur wrapper.
Flipping matters for asymmetric kernels, even though the blur kernels are symmetric.

Box weights are `1/(K*K)`. Gaussian weights are samples of
`exp(-(x*x+y*y)/(2*sigma*sigma))` on a centered integer grid, divided by their sum.
That finite normalized array is used unchanged by both implementations.

## Convolution theorem: substitution and separation

Using the 2D discrete-time Fourier transform and zero-extended finite sequences,

\[
G(\omega_x,\omega_y)
=\sum_r\sum_c g[r,c]e^{-i(\omega_xr+\omega_yc)}
=\sum_r\sum_c\sum_u\sum_v h[u,v]f[r-u,c-v]
 e^{-i(\omega_xr+\omega_yc)}.
\]

The effective sums are finite, so they can be reordered. Substitute `a=r-u`,
`b=c-v`, hence `r=a+u` and `c=b+v`:

\[
G=\sum_u\sum_v h[u,v]\sum_a\sum_b f[a,b]
 e^{-i[\omega_x(a+u)+\omega_y(b+v)]}.
\]

Separate the exponential and the sums:

\[
G=\left[\sum_u\sum_v h[u,v]e^{-i(\omega_xu+\omega_yv)}\right]
  \left[\sum_a\sum_b f[a,b]e^{-i(\omega_xa+\omega_yb)}\right]
 = H(\omega_x,\omega_y)F(\omega_x,\omega_y).
\]

Sampling these transforms on the padded DFT grid allows reconstruction with the
inverse DFT: `ifft2(fft2(f) * fft2(h))`. NumPy's default inverse FFT supplies the
normalization factor. This establishes the convolution theorem in exact arithmetic;
floating-point implementations agree within small numerical roundoff.

## Linear vs. circular convolution, boundaries, and cropping

A DFT treats its finite arrays as periodic, so unpadded multiplication produces
circular convolution: values wrap from one border to the opposite border. The
full linear-convolution support is `(H+Kh-1, W+Kw-1)`. Padding both transforms to
exactly this size makes the support fit before periodic repetition, avoiding
wrap-around. The inverse result is cropped from `((Kh-1)//2, (Kw-1)//2)` to `(H,W)`.

The kernel is transformed in its ordinary unshifted array order. No `fftshift`
or `ifftshift` occurs in filtering. A shift is only appropriate for optional
spectrum visualization, which is not needed for this experiment.

Both methods use zero outside the image. A normalized blur preserves a constant
image's interior, but boundary neighborhoods include zeros, so edges can darken.
Comparing one zero-padded method with a reflected or periodically padded method
would test different problems and need not give matching outputs.

## Experimental protocol

Use two photographs and a generated checkerboard, with Box and Gaussian kernels
of sizes 3, 7, 15: 18 comparisons. Gaussian sigma is size/6 (0.5, 7/6, 2.5).
Images are EXIF-corrected, converted to grayscale, resized with aspect ratio
preserved to a maximum dimension of 512, and normalized to float64 [0,1].

All metrics use the raw float64 outputs before clipping, rounding, or PNG conversion:

- MAE = mean(abs(spatial − Fourier)).
- Maximum difference = max(abs(spatial − Fourier)).
- RMSE = sqrt(mean((spatial − Fourier)^2)).
- Agreement requires abs(Fourier − spatial) <= 1e-12 + 1e-10*abs(spatial) at every pixel.

The maximum imaginary residual of the full inverse FFT is checked against
`1e-12 * max(1, max(abs(real(full))))` before taking the real component.
The CSV records each inverse FFT residual and its allowed limit. The settings
JSON records software versions, source hashes, input types, processed dimensions,
and the exact tolerance.

Panels display original, spatial result, and Fourier result at the same [0,1]
range. The absolute-difference panel has its own labeled, amplified color scale.
A colorful heatmap of errors near machine precision is not evidence of a visible
blur mismatch. **These errors measure agreement between implementations, not image
quality relative to the original.** No perceptual quality metric is claimed.

## Actual run and verification

<!-- ACTUAL_RESULTS_START -->
Run: `2026-09-21T21:25:53.647068+00:00` (UTC). **Provisional: no photographs were available.**
The two labeled synthetic replacements and generated checkerboard produced all
**18 of 18 agreeing comparisons**. The required final two-photograph run remains
incomplete; add `images/photo_01.jpg` and `images/photo_02.jpg` and run the command
in the README. No photograph was downloaded or invented.

| Input | Type | Processed width × height |
|---|---|---|
| `synthetic_edges.png` | synthetic placeholder (not a photograph) | 480 × 320 px |
| `synthetic_texture.png` | synthetic placeholder (not a photograph) | 480 × 320 px |
| `checkerboard.png` | generated checkerboard | 512 × 384 px |

No input required resizing in this run. Original image hashes were unchanged.

### Measured agreement

- Largest maximum absolute difference: **4.6629367034256575e-15**.
- Largest MAE across comparisons: **7.9756030145025079e-16**.
- Largest RMSE across comparisons: **1.2712795338700996e-15**.
- Tolerance: `rtol=1e-10`, `atol=1e-12`, unchanged from the correctness tests.
- Differences are in normalized intensity units, consistent with float64 roundoff.

The table below is rounded for readability. `results/comparison_metrics.csv`
preserves full-precision metrics, FFT residuals, crop offsets, and panel paths.

| Input | Filter | Size | Sigma (px) | MAE | Max abs. difference | RMSE | Agrees |
|---|---|---:|---:|---:|---:|---:|:---:|
| `synthetic_edges.png` | box | 3 | — | 1.759423e-16 | 1.110223e-15 | 2.230966e-16 | True |
| `synthetic_edges.png` | box | 7 | — | 2.485080e-16 | 1.665335e-15 | 3.689584e-16 | True |
| `synthetic_edges.png` | box | 15 | — | 7.009892e-16 | 4.662937e-15 | 1.271280e-15 | True |
| `synthetic_edges.png` | gaussian | 3 | 0.5 | 1.888617e-16 | 1.221245e-15 | 2.412553e-16 | True |
| `synthetic_edges.png` | gaussian | 7 | 1.16667 | 1.787061e-16 | 1.221245e-15 | 2.320270e-16 | True |
| `synthetic_edges.png` | gaussian | 15 | 2.5 | 3.582829e-16 | 2.442491e-15 | 5.724619e-16 | True |
| `synthetic_texture.png` | box | 3 | — | 1.775401e-16 | 1.110223e-15 | 2.245679e-16 | True |
| `synthetic_texture.png` | box | 7 | — | 1.792394e-16 | 1.110223e-15 | 2.289412e-16 | True |
| `synthetic_texture.png` | box | 15 | — | 2.199322e-16 | 1.332268e-15 | 2.795803e-16 | True |
| `synthetic_texture.png` | gaussian | 3 | 0.5 | 1.865877e-16 | 1.110223e-15 | 2.391935e-16 | True |
| `synthetic_texture.png` | gaussian | 7 | 1.16667 | 1.743136e-16 | 1.110223e-15 | 2.258497e-16 | True |
| `synthetic_texture.png` | gaussian | 15 | 2.5 | 2.520575e-16 | 1.554312e-15 | 3.261478e-16 | True |
| `checkerboard.png` | box | 3 | — | 2.713920e-16 | 1.665335e-15 | 3.512069e-16 | True |
| `checkerboard.png` | box | 7 | — | 4.600225e-16 | 1.443290e-15 | 5.688857e-16 | True |
| `checkerboard.png` | box | 15 | — | 7.915655e-16 | 2.664535e-15 | 9.787506e-16 | True |
| `checkerboard.png` | gaussian | 3 | 0.5 | 3.124454e-16 | 1.776357e-15 | 3.965178e-16 | True |
| `checkerboard.png` | gaussian | 7 | 1.16667 | 2.185612e-16 | 1.110223e-15 | 2.744833e-16 | True |
| `checkerboard.png` | gaussian | 15 | 2.5 | 7.975603e-16 | 2.886580e-15 | 1.056777e-15 | True |

There are 18 four-image panels in `results/panels/`. The checkerboard Gaussian
15×15 panel was visually inspected: original/spatial/Fourier panels share [0,1],
and the difference color bar explicitly shows its approximately 10⁻¹⁵ scale.

### Software and verification

Python 3.10.2, numpy 2.2.6, scipy 1.15.3, Pillow 12.3.0, matplotlib 3.10.9, streamlit 1.64.0.

**81 tests passed**, 0 failures, 0 errors, 0 skipped. The tests include the independent SciPy reference, asymmetric and rectangular kernels,
impulse and constant-image boundaries, non-square images, identity, invalid parameters,
EXIF orientation, aspect-preserving resizing, and Streamlit computation/state behavior.
Actual output is saved in `results/test_results.txt` and `results/test_results.xml`.

**Local browser verification passed** using headless Chrome. The app started and
served its health endpoint; bundled Box and Gaussian runs showed PASS; changing
parameters invalidated the old result; PNG/JSON/raw float64 NPZ downloads were
checked; an in-memory 1024×768 synthetic upload was resized to 512×384 and
processed successfully. Bundled-image hashes and the image file list were
unchanged. Details are in `results/app_verification.txt`.

Syntax checks passed for `app.py`, `filtering.py`, `image_utils.py`, and
`run_experiments.py`. No publication or deployment was performed.
<!-- ACTUAL_RESULTS_END -->

## Limitations and conclusion

The reference photographs are required to finish the final-image experiment;
synthetic placeholders test correctness but cannot stand in for photographic
results. The initial results section explicitly records which input types were
available. A future run with added photos requires refreshing the report numbers.

Zero-padding edge darkening is expected. Grayscale conversion and resizing alter
the input used for the experiment, so processed dimensions must accompany the
results. Floating-point roundoff depends on array shapes and software/platform.
The spatial method has work proportional to image pixels times kernel entries;
FFT padding has additional memory cost. This assignment does not benchmark
speed or claim that either method always runs faster.

Agreement with independent SciPy reference tests, together with tiny raw-array
differences on the available experiment inputs, supports that the two implementations
compute the same zero-padded linear convolution. It does not establish that a
particular blur is perceptually optimal.

## Publication placeholders

- GitHub repository URL: **[TO BE ADDED AFTER APPROVAL]**
- Deployed Module 3 app URL: **[NOT DEPLOYED]**
- Shared assignment webpage link: **[TO BE ADDED OUTSIDE THIS MODULE AFTER APPROVAL]**

## References

- [NumPy fft2 documentation](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html)
- [SciPy convolve2d documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
