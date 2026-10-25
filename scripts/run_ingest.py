"""Day 2 demo: play every camera in configs/cameras.yaml side by side.

Usage (from the repo folder):
    python scripts/run_ingest.py                 # opens a window, press q to quit
    python scripts/run_ingest.py --seconds 10    # stop after 10 s
    python scripts/run_ingest.py --headless --seconds 10   # no window, print stats only
    python scripts/run_ingest.py --record assets/demo/day2.mp4 --seconds 10
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from basira.config import load_cameras  # noqa: E402
from basira.ingest.stream import StreamManager  # noqa: E402
from basira.ingest.tiling import tile_frames  # noqa: E402


WINDOW = "Basira - multi-camera view (q or Esc to quit)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="path to cameras.yaml")
    p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until q)")
    p.add_argument("--headless", action="store_true", help="no window, print stats only")
    p.add_argument("--record", default=None, help="save the tiled view to this .mp4")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cameras = load_cameras(args.config) if args.config else load_cameras()
    names = {c.id: c.name for c in cameras}

    writer = None
    start = time.time()
    last_print = 0.0

    with StreamManager(cameras) as manager:
        while True:
            frames = manager.latest_frames()
            stats = manager.stats()
            labels = [f"{cid} | {names[cid]} | {stats[cid]['read_fps']} fps" for cid in frames]
            grid = tile_frames([f.image if f else None for f in frames.values()], labels)

            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    h, w = grid.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                writer.write(grid)

            if not args.headless:
                cv2.imshow(WINDOW, grid)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # q, Q or Esc
                    break
                # closing the window with the X button also stops the program
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                time.sleep(0.05)

            now = time.time()
            if now - last_print >= 2:
                print(" | ".join(f"{cid}: {s['read_fps']} fps, {s['frames']} frames, "
                                 f"{s['dropped']} dropped" for cid, s in stats.items()))
                last_print = now
            if args.seconds and now - start >= args.seconds:
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
