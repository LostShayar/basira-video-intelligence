"""Click on a camera frame to get zone / line coordinates for cameras.yaml.

Usage:
    python scripts/draw_zones.py --camera cam_03

    Left-click   add a point
    Enter        print the points (as fractions) to paste into cameras.yaml
    c            clear points
    q / Esc      quit

2 points = a line (start, end). 3 or more points = a zone polygon.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from basira.config import load_cameras  # noqa: E402

WINDOW = "Basira - draw zones (click points, Enter to print, c to clear, q to quit)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", required=True, help="camera id, e.g. cam_01")
    p.add_argument("--frame", type=int, default=30, help="which frame of the video to show")
    args = p.parse_args()

    cams = {c.id: c for c in load_cameras()}
    if args.camera not in cams:
        sys.exit(f"Unknown camera {args.camera}. Options: {', '.join(cams)}")

    cap = cv2.VideoCapture(cams[args.camera].source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read a frame from {cams[args.camera].source}")

    # shrink big frames so the whole image fits on a laptop screen;
    # clicks are converted to fractions, so the shrinking doesn't change the result
    scale = min(1.0, 1100 / frame.shape[1], 620 / frame.shape[0])
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    points: list[tuple[int, int]] = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_click)

    while True:
        view = frame.copy()
        for pt in points:
            cv2.circle(view, pt, 5, (0, 255, 255), -1)
        if len(points) == 2:
            cv2.line(view, points[0], points[1], (60, 220, 60), 2)
        elif len(points) >= 3:
            cv2.polylines(view, [np.array(points)], True, (40, 40, 230), 2)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("c"), ord("C")):
            points.clear()
        if key == 13 and points:  # Enter
            frac = [[round(x / (w - 1), 3), round(y / (h - 1), 3)] for x, y in points]
            if len(frac) == 2:
                print(f"\nstart: {frac[0]}\nend: {frac[1]}\n")
            else:
                print(f"\npolygon: {frac}\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
