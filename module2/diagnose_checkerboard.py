#!/usr/bin/env python3
"""Diagnose IMG_9050.jpeg without changing the known calibration board.

Run: python3 diagnose_checkerboard.py
Each candidate uses the same ordered pipeline as calibrate_camera.py.
Panels, complete detections, and an attempt log are saved in
calibration_output/pattern_diagnosis/. Candidate matches may be subgrids and
must not be interpreted as a replacement for the known (7, 7) pattern.
"""
import sys
from calibrate_camera import INPUT_DIR, OUTPUT_DIR, detect_corners, save_diagnostic_panel

CANDIDATES = ((7, 7), (7, 8), (8, 7), (6, 7), (7, 6))


def main():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        print(f"ERROR: {exc}. Install dependencies: pip install -r requirements.txt", file=sys.stderr)
        return 1
    output = OUTPUT_DIR / "pattern_diagnosis"
    source = INPUT_DIR / "IMG_9050.jpeg"
    try:
        output.mkdir(parents=True, exist_ok=True)
        image_path = str(source)
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Cannot read input image: {source}")
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rows = ["Pattern  | Detected | Successful method", "-" * 60]
        details = [f"Input: {source}", "Known calibration pattern remains (7, 7); square size 20.06 mm."]
        for pattern in CANDIDATES:
            print(f"Testing candidate {pattern}", flush=True)
            info = {}
            corners, method, reason = detect_corners(gray, cv2, np, pattern, info)
            tag = f"pattern_{pattern[0]}x{pattern[1]}"
            save_diagnostic_panel(output / f"{tag}.panel.png", gray, info, pattern, cv2, np)
            detected_path = output / f"{tag}.detected.png"
            detected_path.unlink(missing_ok=True)
            if corners is not None:
                annotated = image.copy()
                cv2.drawChessboardCorners(annotated, pattern, corners, True)
                if not cv2.imwrite(str(detected_path), annotated):
                    raise RuntimeError(f"Cannot save {detected_path}")
            rows.append(f"{str(pattern):8} | {'YES' if corners is not None else 'NO':8} | {method or '-'}")
            details.extend([f"\nCandidate {pattern}", *info["attempts"]])
            if reason:
                details.append(reason)
        table = "\n".join(rows)
        print("\n" + table, flush=True)
        print("Candidate matches are diagnostic only; calibration remains (7, 7).")
        (output / "pattern_report.txt").write_text(table + "\n\n" + "\n".join(details) + "\n", encoding="utf-8")
        print(f"Diagnostics saved to: {output}")
        return 0
    except (OSError, RuntimeError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
