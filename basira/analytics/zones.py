"""Zone and line analytics on top of tracked detections.

Zones and lines are written in configs/cameras.yaml as fractions of the frame
(0.0 = left/top edge, 1.0 = right/bottom edge), so the same config works for
720p, 1080p or 4K video. They are converted to pixels on the first frame.

Everything here works on tracker IDs, which is why tracking had to come first:
"person #12 entered the restricted zone" only fires once per entry, and a line
crossing is counted once per object.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

FEET = (sv.Position.BOTTOM_CENTER,)  # a person is "in" a zone where their feet are


@dataclass
class Event:
    camera_id: str
    type: str            # zone_entry | line_cross
    track_id: int
    class_name: str
    zone_id: str
    detail: str = ""     # e.g. "in" / "out" for line crossings
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        extra = f" ({self.detail})" if self.detail else ""
        return f"[{self.camera_id}] {self.class_name} #{self.track_id} {self.type} {self.zone_id}{extra}"


def to_pixels(points: list[list[float]], width: int, height: int) -> np.ndarray:
    """[[0.5, 0.25], ...] -> [[640, 180], ...] for a 1280x720 frame."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected a list of [x, y] points, got {points}")
    if pts.min() < 0 or pts.max() > 1:
        raise ValueError(f"zone points must be fractions between 0 and 1, got {points}")
    return np.round(pts * [width - 1, height - 1]).astype(np.int64)


class ZoneMonitor:
    """Counts objects inside a polygon and reports each new entry once.

    An object only counts as having left after it has been missing from the zone
    for `exit_grace` seconds, so a single missed detection (flicker) doesn't
    produce a second "entered" alert.
    """

    def __init__(
        self,
        zone_id: str,
        zone_type: str,
        polygon_px: np.ndarray,
        max_people: int | None = None,
        classes: list[str] | None = None,
        exit_grace: float = 2.0,
    ):
        self.zone_id = zone_id
        self.type = zone_type          # restricted | crowd | loitering
        self.polygon = polygon_px
        self.max_people = max_people
        self.classes = set(classes) if classes else None   # None = every class
        self.exit_grace = exit_grace
        self._zone = sv.PolygonZone(polygon_px, triggering_anchors=FEET)
        self._last_inside: dict[int, float] = {}            # track id -> last time seen inside
        self.count = 0

    def update(self, camera_id: str, detections: sv.Detections, names: dict[int, str],
               now: float | None = None) -> list[Event]:
        now = time.time() if now is None else now
        inside: list[tuple[int, str]] = []
        if len(detections):
            mask = self._zone.trigger(detections)
            for tid, cid in zip(detections.tracker_id[mask], detections.class_id[mask]):
                name = names.get(int(cid), str(cid))
                if self.classes is None or name in self.classes:
                    inside.append((int(tid), name))
        self.count = len(inside)

        events = []
        for tid, name in inside:
            is_new = tid not in self._last_inside
            self._last_inside[tid] = now
            if is_new and self.type == "restricted":
                events.append(Event(camera_id, "zone_entry", tid, name, self.zone_id))

        # forget objects that have really left
        self._last_inside = {t: ts for t, ts in self._last_inside.items() if now - ts <= self.exit_grace}
        return events


class LineCounter:
    """Counts objects crossing a line, in each direction."""

    def __init__(self, line_id: str, start_px: np.ndarray, end_px: np.ndarray, min_frames: int = 3):
        self.line_id = line_id
        self.start = tuple(int(v) for v in start_px)
        self.end = tuple(int(v) for v in end_px)
        # min_frames: an object must stay on the new side this many frames before
        # the crossing counts, so feet jittering on the line aren't counted over and over
        self._line = sv.LineZone(sv.Point(*self.start), sv.Point(*self.end), triggering_anchors=FEET,
                                 minimum_crossing_threshold=min_frames)

    @property
    def in_count(self) -> int:
        return self._line.in_count

    @property
    def out_count(self) -> int:
        return self._line.out_count

    def update(self, camera_id: str, detections: sv.Detections, names: dict[int, str]) -> list[Event]:
        if len(detections) == 0:
            return []
        crossed_in, crossed_out = self._line.trigger(detections)
        events = []
        for direction, mask in (("in", crossed_in), ("out", crossed_out)):
            for tid, cid in zip(detections.tracker_id[mask], detections.class_id[mask]):
                events.append(Event(camera_id, "line_cross", int(tid),
                                    names.get(int(cid), str(cid)), self.line_id, direction))
        return events


class CameraAnalytics:
    """All zones and lines for one camera, built from its config on the first frame."""

    def __init__(self, camera_id: str, zones_cfg: list[dict], lines_cfg: list[dict]):
        self.camera_id = camera_id
        self.zones_cfg = zones_cfg
        self.lines_cfg = lines_cfg
        self.zones: list[ZoneMonitor] = []
        self.lines: list[LineCounter] = []
        self.frame_size: tuple[int, int] | None = None

    def _build(self, width: int, height: int) -> None:
        self.frame_size = (width, height)
        self.zones = [
            ZoneMonitor(z["id"], z.get("type", "zone"), to_pixels(z["polygon"], width, height),
                        z.get("max_people"), z.get("classes"))
            for z in self.zones_cfg
        ]
        self.lines = []
        for ln in self.lines_cfg:
            start, end = to_pixels([ln["start"], ln["end"]], width, height)
            self.lines.append(LineCounter(ln["id"], start, end, ln.get("min_frames", 3)))

    def update(self, detections: sv.Detections, frame_shape: tuple, names: dict[int, str]) -> list[Event]:
        height, width = frame_shape[:2]
        if self.frame_size != (width, height):
            self._build(width, height)
        events: list[Event] = []
        for zone in self.zones:
            events += zone.update(self.camera_id, detections, names)
        for line in self.lines:
            events += line.update(self.camera_id, detections, names)
        return events
