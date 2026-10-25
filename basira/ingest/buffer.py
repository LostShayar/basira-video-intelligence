"""A thread-safe frame buffer that keeps only the newest frames.

Real-time video prefers fresh frames over complete ones: if the consumer
(detection) is slower than the camera, we drop the oldest frame instead of
letting a backlog grow and latency climb.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Frame:
    camera_id: str
    index: int          # frame number since the stream started
    timestamp: float    # time.time() when the frame was read
    image: Any          # numpy array (H, W, 3), BGR


class LatestFrameBuffer:
    def __init__(self, maxsize: int = 2):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._frames: deque[Frame] = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self.dropped = 0

    def put(self, frame: Frame) -> None:
        with self._lock:
            if len(self._frames) == self._frames.maxlen:
                self.dropped += 1  # deque(maxlen) discards the oldest on append
            self._frames.append(frame)
            self._new.notify_all()

    def get(self, timeout: float | None = None) -> Frame | None:
        """Pop the oldest buffered frame, waiting up to `timeout` seconds."""
        with self._new:
            if not self._frames and not self._new.wait_for(lambda: bool(self._frames), timeout):
                return None
            return self._frames.popleft()

    def latest(self) -> Frame | None:
        """Return the newest frame without removing it (for display)."""
        with self._lock:
            return self._frames[-1] if self._frames else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
