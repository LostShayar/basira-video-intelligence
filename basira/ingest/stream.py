"""Multi-camera ingestion: one reader thread per camera.

Each CameraStream reads frames with OpenCV in its own thread and pushes them
into a LatestFrameBuffer. StreamManager starts and stops all cameras together.

Video files are played back at their real frame rate (and loop by default), so
a file behaves like a live camera instead of being read as fast as the disk allows.
RTSP / webcam sources reconnect automatically if the connection drops.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2

from basira.config import CameraConfig
from basira.ingest.buffer import Frame, LatestFrameBuffer

log = logging.getLogger(__name__)


def _open_capture(source: str) -> cv2.VideoCapture:
    # "0", "1" ... means a local webcam index
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


class CameraStream:
    def __init__(
        self,
        camera: CameraConfig,
        buffer_size: int = 2,
        loop: bool = True,
        realtime: bool = True,
        reconnect_delay: float = 2.0,
    ):
        self.camera = camera
        self.buffer = LatestFrameBuffer(buffer_size)
        self.loop = loop
        self.realtime = realtime
        self.reconnect_delay = reconnect_delay

        self.frames_read = 0
        self.source_fps = 0.0
        self.connected = False
        self.finished = False
        self._fps_window: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{camera.id}", daemon=True)

    @property
    def is_file(self) -> bool:
        return "://" not in self.camera.source and not self.camera.source.isdigit()

    @property
    def read_fps(self) -> float:
        """Frames per second actually read over the last ~2 seconds."""
        w = self._fps_window
        if len(w) < 2:
            return 0.0
        return (len(w) - 1) / (w[-1] - w[0]) if w[-1] > w[0] else 0.0

    def start(self) -> "CameraStream":
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = _open_capture(self.camera.source)
            if not cap.isOpened():
                self.connected = False
                log.warning("[%s] cannot open %s, retrying in %.0fs",
                            self.camera.id, self.camera.source, self.reconnect_delay)
                if self.is_file and not self.loop:
                    self.finished = True
                    return
                self._stop.wait(self.reconnect_delay)
                continue

            self.connected = True
            self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_period = 1.0 / self.source_fps
            log.info("[%s] connected (%.1f fps)", self.camera.id, self.source_fps)
            next_due = time.perf_counter()

            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok:
                    break
                now = time.time()
                self.buffer.put(Frame(self.camera.id, self.frames_read, now, image))
                self.frames_read += 1
                self._fps_window.append(now)
                while self._fps_window and now - self._fps_window[0] > 2.0:
                    self._fps_window.pop(0)

                if self.realtime and self.is_file:
                    next_due += frame_period
                    delay = next_due - time.perf_counter()
                    if delay > 0:
                        self._stop.wait(delay)
                    else:
                        next_due = time.perf_counter()  # fell behind; don't try to catch up

            cap.release()
            self.connected = False
            if self.is_file and not self.loop:
                self.finished = True
                return
            if not self.is_file:
                log.warning("[%s] stream lost, reconnecting", self.camera.id)
                self._stop.wait(self.reconnect_delay)
            # files with loop=True simply reopen from the start


class StreamManager:
    def __init__(self, cameras: list[CameraConfig], **stream_kwargs):
        self.streams = {c.id: CameraStream(c, **stream_kwargs) for c in cameras}

    def start(self) -> "StreamManager":
        for s in self.streams.values():
            s.start()
        return self

    def stop(self) -> None:
        for s in self.streams.values():
            s.stop()

    def latest_frames(self) -> dict[str, Frame | None]:
        return {cid: s.buffer.latest() for cid, s in self.streams.items()}

    def stats(self) -> dict[str, dict]:
        return {
            cid: {
                "connected": s.connected,
                "read_fps": round(s.read_fps, 1),
                "frames": s.frames_read,
                "dropped": s.buffer.dropped,
            }
            for cid, s in self.streams.items()
        }

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
