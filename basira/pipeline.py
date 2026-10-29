"""The per-frame pipeline: detect -> track -> zone and line analytics -> draw.

Kept in one place so the demo scripts, and later the API and event engine,
all run exactly the same logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from basira.analytics.draw import draw_analytics
from basira.analytics.zones import CameraAnalytics, Event
from basira.config import CameraConfig
from basira.ingest.buffer import Frame
from basira.perception.annotate import count_by_class, draw
from basira.perception.detector import Detector
from basira.perception.tracker import TrackerBank


@dataclass
class CameraResult:
    image: np.ndarray            # annotated frame
    counts: dict[str, int]       # objects per class in this frame
    events: list[Event] = field(default_factory=list)


class Pipeline:
    def __init__(self, cameras: list[CameraConfig], detector: Detector, tracking_fps: float = 10):
        self.detector = detector
        self.trackers = TrackerBank(frame_rate=tracking_fps)
        self.analytics = {c.id: CameraAnalytics(c.id, c.zones, c.lines) for c in cameras}

    def process(self, frames: list[Frame]) -> dict[str, CameraResult]:
        results: dict[str, CameraResult] = {}
        if not frames:
            return results
        names = self.detector.names
        for frame, dets in zip(frames, self.detector.detect_batch([f.image for f in frames])):
            tracked = self.trackers.update(frame.camera_id, dets)
            analytics = self.analytics[frame.camera_id]
            events = analytics.update(tracked, frame.image.shape, names)
            image = draw(draw_analytics(frame.image, analytics), tracked, names)
            results[frame.camera_id] = CameraResult(image, count_by_class(tracked, names), events)
        return results
