"""Zone and line analytics on top of tracked detections.

Zones and lines are written in configs/cameras.yaml as fractions of the frame
(0.0 = left/top edge, 1.0 = right/bottom edge), so the same config works for
720p, 1080p or 4K video. They are converted to pixels on the first frame.

Everything here works on tracker IDs, which is why tracking had to come first:
"person #12 entered the restricted zone" only fires once per entry, and a line
crossing is counted once per object.

Zone types:
    restricted  alert when an object enters (once per visit)
    loitering   alert when an object stays longer than `dwell_seconds`
    crowd       alert when more than `max_people` are inside (with a cooldown)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

FEET = (sv.Position.BOTTOM_CENTER,)  # a person is "in" a zone where their feet are

ALERT_TYPES = {"zone_entry", "loitering", "crowding"}


@dataclass
class Event:
    camera_id: str
    type: str            # zone_entry | loitering | crowding | line_cross
    track_id: int        # -1 for zone-level events such as crowding
    class_name: str
    zone_id: str
    detail: str = ""     # e.g. "in" / "out" for lines, "34s" for loitering
    timestamp: float = field(default_factory=time.time)

    @property
    def is_alert(self) -> bool:
        return self.type in ALERT_TYPES

    def __str__(self) -> str:
        extra = f" ({self.detail})" if self.detail else ""
        if self.track_id < 0:
            return f"[{self.camera_id}] {self.type} {self.zone_id}{extra}"
        return f"[{self.camera_id}] {self.class_name} #{self.track_id} {self.type} {self.zone_id}{extra}"

    def headline(self) -> str:
        """Short text for the on-screen alert banner."""
        if self.type == "zone_entry":
            return f"ALERT: {self.class_name} #{self.track_id} entered {self.zone_id}"
        if self.type == "loitering":
            return f"ALERT: {self.class_name} #{self.track_id} loitering in {self.zone_id} ({self.detail})"
        if self.type == "crowding":
            return f"ALERT: crowding in {self.zone_id} ({self.detail})"
        return str(self)


def to_pixels(points: list[list[float]], width: int, height: int) -> np.ndarray:
    """[[0.5, 0.25], ...] -> [[640, 180], ...] for a 1280x720 frame."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected a list of [x, y] points, got {points}")
    if pts.min() < 0 or pts.max() > 1:
        raise ValueError(f"zone points must be fractions between 0 and 1, got {points}")
    return np.round(pts * [width - 1, height - 1]).astype(np.int64)


class ZoneMonitor:
    """Tracks who is inside a polygon, for how long, and raises alerts by zone type.

    An object only counts as having left after it has been missing from the zone
    for both `exit_grace` seconds AND `exit_frames` frames. So a missed detection
    (flicker) or a slow, stuttering camera doesn't restart its visit or produce a
    second alert.
    """

    def __init__(
        self,
        zone_id: str,
        zone_type: str,
        polygon_px: np.ndarray,
        max_people: int | None = None,
        classes: list[str] | None = None,
        exit_grace: float = 2.0,
        exit_frames: int = 3,
        dwell_seconds: float = 30.0,
        cooldown_seconds: float = 20.0,
    ):
        self.zone_id = zone_id
        self.type = zone_type
        self.polygon = polygon_px
        self.max_people = max_people
        self.classes = set(classes) if classes else None   # None = every class
        self.exit_grace = exit_grace
        self.exit_frames = exit_frames
        self.dwell_seconds = dwell_seconds
        self.cooldown_seconds = cooldown_seconds
        self._zone = sv.PolygonZone(polygon_px, triggering_anchors=FEET)
        self._entered_at: dict[int, float] = {}    # track id -> when this visit started
        self._last_inside: dict[int, float] = {}   # track id -> last time seen inside
        self._missed: dict[int, int] = {}          # track id -> frames in a row not seen inside
        self._loiter_alerted: set[int] = set()     # ids already alerted this visit
        self._last_crowd_alert = -1e9
        self.count = 0
        self.longest_dwell = 0.0                   # seconds, for display

    def dwell(self, track_id: int, now: float) -> float:
        start = self._entered_at.get(track_id)
        return 0.0 if start is None else now - start

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

        events: list[Event] = []
        for tid, name in inside:
            is_new = tid not in self._last_inside
            if is_new:
                self._entered_at[tid] = now
            self._last_inside[tid] = now
            self._missed[tid] = 0

            if is_new and self.type == "restricted":
                events.append(Event(camera_id, "zone_entry", tid, name, self.zone_id))

            if self.type == "loitering" and tid not in self._loiter_alerted:
                stayed = self.dwell(tid, now)
                if stayed >= self.dwell_seconds:
                    self._loiter_alerted.add(tid)
                    events.append(Event(camera_id, "loitering", tid, name, self.zone_id, f"{stayed:.0f}s"))

        if (self.type == "crowd" and self.max_people and self.count > self.max_people
                and now - self._last_crowd_alert >= self.cooldown_seconds):
            self._last_crowd_alert = now
            events.append(Event(camera_id, "crowding", -1, "person", self.zone_id,
                                f"{self.count}/{self.max_people}"))

        # forget objects that have really left (and reset their visit)
        current_ids = {tid for tid, _ in inside}
        for t in self._last_inside:
            if t not in current_ids:
                self._missed[t] = self._missed.get(t, 0) + 1
        gone = [t for t, ts in self._last_inside.items()
                if now - ts > self.exit_grace and self._missed.get(t, 0) >= self.exit_frames]
        for t in gone:
            self._last_inside.pop(t, None)
            self._missed.pop(t, None)
            self._entered_at.pop(t, None)
            self._loiter_alerted.discard(t)

        current = [tid for tid, _ in inside]
        self.longest_dwell = max((self.dwell(t, now) for t in current), default=0.0)
        return events


class LineCounter:
    """Counts objects crossing a line, in each direction.

    Two protections against over-counting:
      - min_frames: an object must stay on the new side this many frames, so feet
        jittering on the line aren't counted over and over;
      - recount_after: the same track ID is counted at most once per line within
        this many seconds, so a slow car crawling on the line in traffic
        (in, out, in) is counted once.
    """

    def __init__(self, line_id: str, start_px: np.ndarray, end_px: np.ndarray,
                 min_frames: int = 3, recount_after: float = 10.0):
        self.line_id = line_id
        self.start = tuple(int(v) for v in start_px)
        self.end = tuple(int(v) for v in end_px)
        self.recount_after = recount_after
        self._line = sv.LineZone(sv.Point(*self.start), sv.Point(*self.end), triggering_anchors=FEET,
                                 minimum_crossing_threshold=min_frames)
        self._counted_at: dict[int, float] = {}   # track id -> when it was last counted
        self.in_count = 0
        self.out_count = 0

    def update(self, camera_id: str, detections: sv.Detections, names: dict[int, str],
               now: float | None = None) -> list[Event]:
        now = time.time() if now is None else now
        if len(detections) == 0:
            return []
        crossed_in, crossed_out = self._line.trigger(detections)
        events = []
        for direction, mask in (("in", crossed_in), ("out", crossed_out)):
            for tid, cid in zip(detections.tracker_id[mask], detections.class_id[mask]):
                tid = int(tid)
                if now - self._counted_at.get(tid, -1e9) < self.recount_after:
                    continue  # already counted this object recently
                self._counted_at[tid] = now
                if direction == "in":
                    self.in_count += 1
                else:
                    self.out_count += 1
                events.append(Event(camera_id, "line_cross", tid,
                                    names.get(int(cid), str(cid)), self.line_id, direction))
        # keep memory small
        self._counted_at = {t: ts for t, ts in self._counted_at.items() if now - ts < self.recount_after}
        return events


class CameraAnalytics:
    """All zones and lines for one camera, built from its config on the first frame.

    `defaults` holds settings from the `analytics:` section of cameras.yaml;
    a zone can override them (e.g. `dwell_seconds: 10` on one zone).
    """

    def __init__(self, camera_id: str, zones_cfg: list[dict], lines_cfg: list[dict],
                 defaults: dict | None = None):
        self.camera_id = camera_id
        self.zones_cfg = zones_cfg
        self.lines_cfg = lines_cfg
        d = defaults or {}
        self.dwell_seconds = d.get("loitering", {}).get("threshold_seconds", 30)
        self.cooldown_seconds = d.get("events", {}).get("cooldown_seconds", 20)
        self.zones: list[ZoneMonitor] = []
        self.lines: list[LineCounter] = []
        self.frame_size: tuple[int, int] | None = None

    def _build(self, width: int, height: int) -> None:
        self.frame_size = (width, height)
        self.zones = [
            ZoneMonitor(
                z["id"], z.get("type", "zone"), to_pixels(z["polygon"], width, height),
                max_people=z.get("max_people"),
                classes=z.get("classes"),
                dwell_seconds=z.get("dwell_seconds", self.dwell_seconds),
                cooldown_seconds=z.get("cooldown_seconds", self.cooldown_seconds),
            )
            for z in self.zones_cfg
        ]
        self.lines = []
        for ln in self.lines_cfg:
            start, end = to_pixels([ln["start"], ln["end"]], width, height)
            self.lines.append(LineCounter(ln["id"], start, end, ln.get("min_frames", 3),
                                          ln.get("recount_after", 10.0)))

    def update(self, detections: sv.Detections, frame_shape: tuple, names: dict[int, str],
               now: float | None = None) -> list[Event]:
        height, width = frame_shape[:2]
        if self.frame_size != (width, height):
            self._build(width, height)
        events: list[Event] = []
        for zone in self.zones:
            events += zone.update(self.camera_id, detections, names, now)
        for line in self.lines:
            events += line.update(self.camera_id, detections, names, now)
        return events
