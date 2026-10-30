"""Live demo: detection, tracking, zones, lines, loitering and crowd alerts, privacy blur.

Usage (from the repo folder, with .venv active):
    python scripts/run_analytics.py                    # live window, q / Esc / X to quit
    python scripts/run_analytics.py --seconds 30
    python scripts/run_analytics.py --record assets/demo/analytics.mp4 --seconds 30
    python scripts/run_analytics.py --headless --seconds 20   # no window, events only

Events (zone entries, line crossings) are printed in the terminal as they happen.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from basira.analytics.draw import draw_alert  # noqa: E402
from basira.config import load_cameras, load_config  # noqa: E402
from basira.ingest.stream import StreamManager  # noqa: E402
from basira.ingest.tiling import tile_frames  # noqa: E402
from basira.perception.detector import Detector  # noqa: E402
from basira.pipeline import Pipeline  # noqa: E402

WINDOW = "Basira - live analytics (q or Esc to quit)"
ALERT_SECONDS = 4  # how long a red alert banner stays on a camera


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, cameras = load_config(), load_cameras()
    names = {c.id: c.name for c in cameras}
    m = cfg["model"]
    detector = Detector(m["weights"], m.get("classes"), m.get("confidence", 0.35),
                        image_size=args.imgsz, device=args.device)
    pipeline = Pipeline(cameras, detector, analytics_cfg=cfg.get("analytics", {}),
                        blur_faces=cfg.get("privacy", {}).get("blur_faces", True))

    last_seen = {c.id: -1 for c in cameras}
    shown = {c.id: None for c in cameras}
    counts = {c.id: {} for c in cameras}
    alerts: dict[str, tuple[str, float]] = {}
    loop_times: list[float] = []
    writer = None
    start = time.time()

    with StreamManager(cameras) as manager:
        time.sleep(0.5)
        while True:
            t0 = time.time()
            batch = []
            for cid, frame in manager.latest_frames().items():
                if frame is not None and frame.index != last_seen[cid]:
                    last_seen[cid] = frame.index
                    batch.append(frame)

            for cid, result in pipeline.process(batch).items():
                shown[cid], counts[cid] = result.image, result.counts
                for event in result.events:
                    print(time.strftime("%H:%M:%S"), event)
                    if event.is_alert:
                        alerts[cid] = (event.headline(), time.time())

            loop_times = (loop_times + [time.time() - t0])[-30:]
            fps = len(loop_times) / sum(loop_times) if sum(loop_times) else 0.0

            tiles, labels = [], []
            for cid in shown:
                img = shown[cid]
                if img is not None and cid in alerts and time.time() - alerts[cid][1] < ALERT_SECONDS:
                    img = draw_alert(img, alerts[cid][0])
                tiles.append(img)
                summary = ", ".join(f"{n} {k}" for k, n in counts[cid].items()) or "none"
                labels.append(f"{cid} | {names[cid]} | {fps:.1f} fps | {summary}")
            grid = tile_frames(tiles, labels)

            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    h, w = grid.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                writer.write(grid)

            if not args.headless:
                cv2.imshow(WINDOW, grid)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if args.seconds and time.time() - start >= args.seconds:
                break

    for cid, analytics in pipeline.analytics.items():
        for line in analytics.lines:
            print(f"[{cid}] {line.line_id}: in {line.in_count}, out {line.out_count}")
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
