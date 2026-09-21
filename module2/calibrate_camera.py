#!/usr/bin/env python3
"""Calibrate the phone camera from JPEG checkerboard photographs.

Usage (from this project directory):
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python3 calibrate_camera.py
    python3 calibrate_camera.py --single-image IMG_9050.jpeg

Inputs: calibration_images/*.jpg or *.jpeg (case insensitive).
Outputs: calibration_output/calibration.npz, calibration_report.txt, and
         detected_corners/ annotated images.
Paths are resolved relative to this script. Images are decoded with OpenCV's
direct IMREAD_GRAYSCALE decoding and default EXIF orientation handling. Calibration uses original-resolution coordinates. Use the same decoding
convention for subsequent applications of these intrinsics.

The board has 8 x 8 squares, hence 7 x 7 INTERNAL corners. Its square side is
20.06 mm, based on the board filling the vertical iPad's short screen width.
World points and pose translations use millimeters; rotations use radians.
At least three views are required. Diverse board tilts and positions improve
calibration; a low reprojection error alone does not guarantee good calibration.
"""

from pathlib import Path
import sys
import argparse

CHECKERBOARD = (7, 7)  # (columns, rows) of internal corners, NOT squares
SQUARE_SIZE_MM = 20.06
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "calibration_images"
OUTPUT_DIR = BASE_DIR / "calibration_output"


def detect_corners(gray, cv2, np, pattern=CHECKERBOARD, diagnostics=None):
    """Try original, CLAHE, then half-size images; SB precedes classic.

    Stop at the first complete grid. Optional diagnostics records every attempt
    and the largest partial result, which is visualization-only, never calibrated.
    All returned coordinates use the original image's pixel coordinate system.
    """
    info = diagnostics if diagnostics is not None else {}
    info.update(attempts=[], best_corners=None, best_method="No corners returned",
                complete=False)
    pattern_size = pattern  # Calibration always passes CHECKERBOARD = (7, 7).
    height, width = gray.shape

    def variants():
        # Lazy preprocessing: accept original SB immediately when it succeeds.
        yield "original", gray
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        info["clahe"] = clahe
        yield "CLAHE", clahe
        half = cv2.resize(gray, (max(1, width // 2), max(1, height // 2)),
                          interpolation=cv2.INTER_AREA)
        yield "half", half

    for variant, pixels in variants():
        for detector in ("SB", "classic"):
            method = f"{variant} / {detector}"
            print(f"    Trying {method} for {pattern}...", flush=True)
            if detector == "SB" and not hasattr(cv2, "findChessboardCornersSB"):
                print(f"{method}: unavailable (OpenCV has no SB detector)", flush=True)
                info["attempts"].append(f"{method}: unavailable")
                continue
            try:
                if detector == "SB":
                    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
                             | cv2.CALIB_CB_ACCURACY)
                    # The original branch uses exactly the unmodified decoded gray.
                    if variant == "original":
                        found, corners = cv2.findChessboardCornersSB(
                            gray, pattern_size,
                            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
                            | cv2.CALIB_CB_ACCURACY)
                    else:
                        found, corners = cv2.findChessboardCornersSB(pixels, pattern_size, flags)
                else:
                    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                             | cv2.CALIB_CB_FILTER_QUADS)
                    found, corners = cv2.findChessboardCorners(pixels, pattern, flags)
                raw_count = 0 if corners is None else len(corners)
                raw_result = f"{method}: raw_result={found!r}, corner_count={raw_count}"
                print(raw_result, flush=True)
                info["attempts"].append(raw_result)
                # A successful OpenCV return must not silently become rejection.
                if found and not valid_corners(corners, np, pattern_size):
                    raise RuntimeError(f"{method} returned success with invalid corners: "
                                       f"shape={getattr(corners, 'shape', None)}")
                complete = bool(found)
                if complete and detector == "classic":
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                                50, 0.001)
                    corners = cv2.cornerSubPix(pixels, corners, (5, 5), (-1, -1), criteria)
                    complete = valid_corners(corners, np, pattern)
                # SB estimates are already subpixel; classic is refined above.
                scaled = None
                if corners is not None and corners.size and np.isfinite(corners).all():
                    scaled = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2).copy()
                    if variant == "half":
                        # Invert resize's pixel-center mapping. Separate factors
                        # also handle odd original image dimensions correctly.
                        scale = np.array([width / pixels.shape[1], height / pixels.shape[0]],
                                         dtype=np.float32)
                        scaled = (scaled + 0.5) * scale - 0.5
                    best = info["best_corners"]
                    if complete or best is None or len(scaled) > len(best):
                        info.update(best_corners=scaled, best_method=method, complete=complete)
                count = 0 if scaled is None else len(scaled)
                info["attempts"].append(
                    f"{method}: {'DETECTED' if complete else 'not detected'}; {count} corners")
                if complete:
                    message = f"DETECTED: {len(scaled)} corners using {method}"
                    print(message, flush=True)
                    info["attempts"].append(message)
                    return scaled, method, ""
            except cv2.error as exc:
                message = f"{method}: OpenCV error: {exc}"
                print(message, file=sys.stderr, flush=True)
                info["attempts"].append(message)
    return None, None, f"complete {pattern[0]} x {pattern[1]} internal-corner grid not detected"


def valid_corners(corners, np, pattern=CHECKERBOARD):
    """Reject incomplete or nonfinite detector output before calibration."""
    return (corners is not None and corners.shape == (pattern[0] * pattern[1], 1, 2)
            and np.isfinite(corners).all())


def save_diagnostic_panel(path, gray, info, pattern, cv2, np):
    """Save a labeled 2 x 2 preview; partial corners are explicitly identified.

    Diagnostic previews may be resized for readability; calibration points are
    always retained at original resolution. Unreadable inputs get placeholders.
    """
    if gray is None:
        gray = np.zeros((480, 640), dtype=np.uint8)
        missing = " (input unreadable)"
    else:
        missing = ""
    enhanced = info.get("clahe")
    if enhanced is None:
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, threshold = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    corners = info.get("best_corners")
    if corners is not None:
        cv2.drawChessboardCorners(overlay, pattern, corners, info.get("complete", False))
    label = info.get("best_method", "No corners returned")
    if corners is not None:
        label += f" ({len(corners)} corners; " + ("complete)" if info.get("complete") else "partial)")
    tiles = []
    for pixels, title in ((gray, "Original grayscale" + missing),
                          (enhanced, "CLAHE" + missing),
                          (threshold, "Otsu threshold of CLAHE" + missing),
                          (overlay, label)):
        if pixels.ndim == 2:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        scale = min(800 / pixels.shape[1], 600 / pixels.shape[0])
        preview = cv2.resize(pixels, (max(1, round(pixels.shape[1] * scale)),
                                      max(1, round(pixels.shape[0] * scale))))
        tile = np.full((650, 800, 3), 32, dtype=np.uint8)
        tile[50:50 + preview.shape[0], :preview.shape[1]] = preview
        cv2.putText(tile, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    panel = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    if not cv2.imwrite(str(path), panel):
        raise RuntimeError(f"Could not save diagnostic panel: {path}")


def run_single_image(filename, cv2, np):
    """Run detection only; leave calibration, reports, and input files untouched.

    Print raw returns for each attempted method. Once a method succeeds, later
    fallbacks are skipped, just as they are during full calibration.
    """
    if Path(filename).name != filename:
        print("ERROR: --single-image expects a filename inside calibration_images/", file=sys.stderr)
        return 1
    image_path = str(INPUT_DIR / filename)
    try:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Cannot decode image: {image_path}")
        pattern_size = (7, 7)
        print(f"Image: {image_path}\nLoading: IMREAD_GRAYSCALE; "
              f"shape={gray.shape}, dtype={gray.dtype}; OpenCV={cv2.__version__}", flush=True)
        corners, method, reason = detect_corners(gray, cv2, np, pattern=pattern_size)
        if corners is None:
            print(f"REJECTED: {filename}: {reason}", file=sys.stderr)
            return 1
        print(f"Accepted by {method}; remaining fallback methods skipped.", flush=True)
        return 0
    except (OSError, RuntimeError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--single-image", metavar="FILENAME",
                        help="Detect only this input image, print raw results, and skip calibration")
    args = parser.parse_args()
    print("Run command: python3 calibrate_camera.py", flush=True)
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        print(f"ERROR: {exc}. Install dependencies: pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    if args.single_image:
        return run_single_image(args.single_image, cv2, np)

    report = [
        "CAMERA CALIBRATION REPORT",
        "Checkerboard squares: 8 x 8",
        f"Checkerboard internal corners (columns, rows): {CHECKERBOARD}",
        f"Square size: {SQUARE_SIZE_MM} mm; world-coordinate unit: millimeters",
        "Board filled the short screen width of a vertical iPad.",
        "Capture assumption: same rear camera, lens, orientation, resolution, zoom.",
        "Pixel convention: direct IMREAD_GRAYSCALE with default EXIF orientation handling.",
        f"OpenCV version: {cv2.__version__}",
        f"Input directory: {INPUT_DIR}",
    ]
    report_path = OUTPUT_DIR / "calibration_report.txt"
    npz_path = OUTPUT_DIR / "calibration.npz"
    temporary_npz = OUTPUT_DIR / "calibration.pending.npz"
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        corners_dir = OUTPUT_DIR / "detected_corners"
        corners_dir.mkdir(exist_ok=True)
        diagnostics_dir = OUTPUT_DIR / "detection_diagnostics"
        diagnostics_dir.mkdir(exist_ok=True)
        # Preserve the prior calibration until a successful replacement is ready.
        temporary_npz.unlink(missing_ok=True)
        report_path.write_text("\n".join(report) + "\nStatus: RUNNING\n", encoding="utf-8")
        if not INPUT_DIR.is_dir():
            raise RuntimeError(f"Input directory does not exist: {INPUT_DIR}")
        paths = sorted((p for p in INPUT_DIR.iterdir()
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}),
                       key=lambda p: (p.name.lower(), p.name))
        report.append(f"Input image count: {len(paths)}")
        print(f"Found {len(paths)} JPEG images.", flush=True)
        accepted, rejected, image_points, sizes, methods = [], [], [], [], []
        for index, path in enumerate(paths, 1):
            print(f"[{index}/{len(paths)}] Detecting {path.name!r}...", flush=True)
            # Include the original extension to distinguish a.jpg from a.jpeg.
            annotated_path = corners_dir / (path.name + ".corners.png")
            annotated_path.unlink(missing_ok=True)
            diagnostic_path = diagnostics_dir / (path.name + ".panel.png")
            diagnostic_path.unlink(missing_ok=True)
            gray, detection_info = None, {}
            try:
                image_path = str(path)
                gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError("image could not be decoded")
                corners, method, reason = detect_corners(gray, cv2, np, diagnostics=detection_info)
            except (cv2.error, ValueError) as exc:
                print(f"ERROR loading/detecting {path.name!r}: {exc}", file=sys.stderr, flush=True)
                corners, reason = None, str(exc)
            report.append(f"\nDetection attempts for {path.name!r}:")
            report.extend("  " + attempt for attempt in detection_info.get("attempts", []))
            if corners is None:
                save_diagnostic_panel(diagnostic_path, gray, detection_info, CHECKERBOARD, cv2, np)
                rejected.append((path.name, reason))
                print(f"REJECTED: {path.name!r}: {reason}", flush=True)
                continue
            height, width = gray.shape
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(image, CHECKERBOARD, corners, True)
            if not cv2.imwrite(str(annotated_path), image):
                raise RuntimeError(f"Could not save annotated image: {annotated_path}")
            accepted.append(path.name)
            image_points.append(corners)
            sizes.append((width, height))
            methods.append(method)
            print(f"ACCEPTED: {path.name!r} ({width} x {height}; {method})", flush=True)

        report.append(f"\nAccepted images: {len(accepted)}")
        report.extend(f"  {name!r}: {size[0]} x {size[1]} pixels; {method}"
                      for name, size, method in zip(accepted, sizes, methods))
        report.append(f"\nRejected images: {len(rejected)}")
        report.extend(f"  {name!r}: {reason}" for name, reason in rejected)
        if not rejected:
            report.append("  (none)")
        problems = []
        if len(accepted) < 3:
            problems.append(f"Only {len(accepted)} images successfully detected; at least 3 are required")
        if len(set(sizes)) > 1:
            problems.append("Accepted images have inconsistent resolutions: "
                            + ", ".join(f"{w} x {h}" for w, h in sorted(set(sizes))))
        if problems:
            raise RuntimeError(". ".join(problems))
        width, height = sizes[0]
        report.append(f"\nImage resolution (width x height): {width} x {height} pixels")

        # Row-major planar board coordinates. The origin is the first internal
        # corner, Z=0 is the board plane, and adjacent corners are 20.06 mm apart.
        board = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        board[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        board *= SQUARE_SIZE_MM
        object_points = [board.copy() for _ in accepted]
        solver_rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, (width, height), None, None)
        if not all(np.isfinite(value).all() for value in
                   (solver_rms, camera_matrix, dist_coeffs, rvecs, tvecs)):
            raise RuntimeError("Calibration returned nonfinite values; inspect the input views")

        # Each residual is a 2D pixel displacement. RMS is sqrt(sum(dx²+dy²)/N)
        # across ALL corners, not the average of the per-image mean errors.
        means, image_rms = [], []
        total_squared_error, total_points = 0.0, 0
        for obj, observed, rotation, translation in zip(
                object_points, image_points, rvecs, tvecs):
            projected, _ = cv2.projectPoints(obj, rotation, translation,
                                             camera_matrix, dist_coeffs)
            residual = observed.reshape(-1, 2).astype(np.float64) - projected.reshape(-1, 2)
            squared = np.sum(residual ** 2, axis=1)
            means.append(float(np.mean(np.sqrt(squared))))
            image_rms.append(float(np.sqrt(np.mean(squared))))
            total_squared_error += float(np.sum(squared))
            total_points += len(squared)
        rms = float(np.sqrt(total_squared_error / total_points))
        if not np.isfinite(rms):
            raise RuntimeError("Reprojection calculation returned nonfinite values")

        report.extend([
            "\nCamera matrix (pixel coordinates):",
            np.array2string(camera_matrix, precision=10),
            "Distortion coefficients (k1, k2, p1, p2, k3; dimensionless):",
            np.array2string(dist_coeffs.ravel(), precision=10),
            f"OpenCV calibrateCamera RMS: {solver_rms:.8f} pixels",
            f"Overall RMS reprojection error (projectPoints): {rms:.8f} pixels",
            "RMS definition: sqrt(sum of squared 2D corner distances / total corners).",
            "\nPer-image errors (pixels):",
            "Mean error = arithmetic mean of Euclidean 2D corner distances.",
        ])
        for name, mean, view_rms in zip(accepted, means, image_rms):
            line = f"  {name!r}: mean={mean:.8f}, RMS={view_rms:.8f}"
            report.append(line)
            print(line)
        report.append("\nPoses: board-to-camera Rodrigues rotations (radians), translations (mm).")
        for name, rotation, translation in zip(accepted, rvecs, tvecs):
            report.append(f"  {name!r}: rvec={rotation.ravel().tolist()}, tvec_mm={translation.ravel().tolist()}")

        # Plain numeric/string arrays permit loading with allow_pickle=False.
        # Poses, image points, and errors share accepted_filenames ordering.
        np.savez_compressed(
            temporary_npz, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
            image_width=width, image_height=height,
            checkerboard_inner_corners=np.array(CHECKERBOARD, dtype=np.int32),
            square_size_mm=SQUARE_SIZE_MM, world_unit="mm",
            rvecs=np.asarray(rvecs), tvecs=np.asarray(tvecs),
            accepted_filenames=np.asarray(accepted, dtype=str),
            rejected_filenames=np.asarray([n for n, _ in rejected], dtype=str),
            rejection_reasons=np.asarray([r for _, r in rejected], dtype=str),
            object_points_mm=np.asarray(object_points), image_points=np.asarray(image_points),
            rms_reprojection_error=rms, opencv_rms=solver_rms,
            per_image_mean_errors_px=np.asarray(means),
            per_image_rms_errors_px=np.asarray(image_rms),
        )
        temporary_npz.replace(npz_path)
        report.append("\nStatus: SUCCESS")
        print(f"Overall RMS reprojection error: {rms:.8f} pixels")
        print(f"Saved calibration: {npz_path}")
    except (OSError, RuntimeError, cv2.error) as exc:
        report.append(f"\nStatus: FAILED\nERROR: {exc}")
        report.append("Existing calibration.npz, if present, was preserved; it is from a prior run.")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
            print(f"Report: {report_path}")
        except OSError as exc:
            print(f"ERROR: Cannot write report: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
