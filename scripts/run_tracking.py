"""Day 3 demo: YOLO detection + ByteTrack tracking on every camera.

Usage (from the repo folder, with .venv active):
    python scripts/run_tracking.py                       # live window, q / Esc / X to quit
    python scripts/run_tracking.py --seconds 20          # stop after 20 s
    python scripts/run_tracking.py --record assets/demo/day3.mp4 --seconds 20
    python scripts/run_tracking.py --headless --seconds 10   # no window, stats only

The first run downloads the YOLO weights (~6 MB) automatically.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from basira.config import load_cameras, load_config  # noqa: E402
from basira.ingest.stream import StreamManager  # noqa: E402
from basira.ingest.tiling import tile_frames  # noqa: E402
from basira.perception.annotate import count_by_class, draw  # noqa: E402
from basira.perception.detector import Detector  # noqa: E402
from basira.perception.tracker import TrackerBank  # noqa: E402

WINDOW = "Basira - detection + tracking (q or Esc to quit)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until q)")
    p.add_argument("--headless", action="store_true", help="no window, print stats only")
    p.add_argument("--record", default=None, help="save the annotated tiled view to this .mp4")
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO input size (smaller = faster)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    cameras = load_cameras()
    names = {c.id: c.name for c in cameras}
    m = cfg["model"]

    detector = Detector(m["weights"], m.get("classes"), m.get("confidence", 0.35),
                        image_size=args.imgsz, device=args.device)
    trackers = TrackerBank(frame_rate=10)  # tracking runs at processing speed, not camera speed

    last_seen = {c.id: -1 for c in cameras}
    annotated = {c.id: None for c in cameras}
    counts = {c.id: {} for c in cameras}
    loop_times: list[float] = []
    writer = None
    start = last_print = time.time()

    with StreamManager(cameras) as manager:
        time.sleep(0.5)  # let cameras connect
        while True:
            t0 = time.time()

            # 1. take the newest frame from each camera (skip cameras with nothing new)
            batch = []
            for cid, frame in manager.latest_frames().items():
                if frame is not None and frame.index != last_seen[cid]:
                    last_seen[cid] = frame.index
                    batch.append(frame)

            # 2. detect on all cameras in one batch, 3. track per camera, 4. draw
            for frame, dets in zip(batch, detector.detect_batch([f.image for f in batch])):
                tracked = trackers.update(frame.camera_id, dets)
                annotated[frame.camera_id] = draw(frame.image, tracked, detector.names)
                counts[frame.camera_id] = count_by_class(tracked, detector.names)

            loop_times.append(time.time() - t0)
            loop_times = loop_times[-30:]
            fps = len(loop_times) / sum(loop_times) if sum(loop_times) else 0.0

            labels = []
            for cid in annotated:
                summary = ", ".join(f"{n} {k}" for k, n in counts[cid].items()) or "none"
                labels.append(f"{cid} | {names[cid]} | {fps:.1f} fps | {summary}")
            grid = tile_frames(list(annotated.values()), labels)

            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    h, w = grid.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                writer.write(grid)

            if not args.headless:
                cv2.imshow(WINDOW, grid)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break

            now = time.time()
            if now - last_print >= 2:
                print(f"processing {fps:.1f} fps | YOLO batch {detector.last_ms:.0f} ms | "
                      + " | ".join(f"{cid}: {counts[cid]}" for cid in counts))
                last_print = now
            if args.seconds and now - start >= args.seconds:
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
