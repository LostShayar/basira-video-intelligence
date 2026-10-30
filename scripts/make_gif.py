"""Turn a recorded demo video into a small GIF for the README.

Usage:
    python scripts/run_analytics.py --record data/demo.mp4 --seconds 40
    python scripts/make_gif.py data/demo.mp4 assets/demo/basira-demo.gif --start 8 --seconds 12

Keeps the GIF small enough for GitHub (aim for under ~10 MB): fewer frames per
second, a smaller width, and a shared colour palette.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", help="input .mp4 (e.g. data/demo.mp4)")
    p.add_argument("gif", help="output .gif (e.g. assets/demo/basira-demo.gif)")
    p.add_argument("--start", type=float, default=0, help="start at this second")
    p.add_argument("--seconds", type=float, default=12, help="length of the GIF")
    p.add_argument("--fps", type=float, default=5, help="frames per second in the GIF")
    p.add_argument("--width", type=int, default=900, help="GIF width in pixels")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10
    step = max(1, round(src_fps / args.fps))
    first = int(args.start * src_fps)
    last = int((args.start + args.seconds) * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    frames = []
    for i in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        if (i - first) % step:
            continue
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise SystemExit("No frames read; check --start and --seconds against the video length")

    palette = frames[len(frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif_frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(args.gif, save_all=True, append_images=gif_frames[1:],
                       duration=round(1000 / args.fps), loop=0, optimize=True)
    size_mb = Path(args.gif).stat().st_size / 1e6
    print(f"Saved {args.gif}: {len(gif_frames)} frames, {size_mb:.1f} MB")
    if size_mb > 10:
        print("Tip: that's large for a README. Try --width 700 or --seconds 8.")


if __name__ == "__main__":
    main()
