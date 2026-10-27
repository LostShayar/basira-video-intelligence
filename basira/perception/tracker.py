"""Multi-object tracking with ByteTrack, one tracker per camera.

Detection answers "there is a person here in this frame".
Tracking answers "this is the same person #7 we saw in earlier frames".

Each camera needs its own tracker, because IDs and motion history from one
camera must never be matched against another camera's boxes.

ByteTrack comes from Roboflow's `trackers` package (Supervision 0.31 removed
its built-in copy).
"""
from __future__ import annotations

import supervision as sv
from trackers import ByteTrackTracker


class CameraTracker:
    def __init__(self, frame_rate: float = 25.0, lost_seconds: float = 1.5):
        # lost_track_buffer is in frames: how long a person can be hidden
        # (behind a pole, another person) and still keep the same ID.
        self.tracker = ByteTrackTracker(
            frame_rate=frame_rate,
            lost_track_buffer=max(1, round(frame_rate * lost_seconds)),
            track_activation_threshold=0.5,  # start a new track from a fairly confident box
            high_conf_det_threshold=0.5,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Returns only confirmed tracks, each with a stable `tracker_id`."""
        tracked = self.tracker.update(detections)
        if tracked.tracker_id is None or len(tracked) == 0:
            return tracked
        return tracked[tracked.tracker_id >= 0]  # -1 = not confirmed yet


class TrackerBank:
    """Creates trackers lazily, one per camera id."""

    def __init__(self, frame_rate: float = 25.0, lost_seconds: float = 1.5):
        self.frame_rate = frame_rate
        self.lost_seconds = lost_seconds
        self._trackers: dict[str, CameraTracker] = {}

    def update(self, camera_id: str, detections: sv.Detections) -> sv.Detections:
        if camera_id not in self._trackers:
            self._trackers[camera_id] = CameraTracker(self.frame_rate, self.lost_seconds)
        return self._trackers[camera_id].update(detections)
