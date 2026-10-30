import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from basira.analytics.zones import CameraAnalytics, LineCounter, ZoneMonitor  # noqa: E402
from basira.perception.privacy import blur_heads, head_boxes  # noqa: E402

NAMES = {0: "person", 2: "car"}
SQUARE = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])


def obj(track_id, x_center, y_bottom, cls=0):
    return sv.Detections(
        xyxy=np.array([[x_center - 20, y_bottom - 80, x_center + 20, y_bottom]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([cls]),
        tracker_id=np.array([track_id]),
    )


def many(n, cls=0):
    return sv.Detections.merge([obj(i + 1, 110 + i * 6, 200, cls) for i in range(n)])


# ---------- loitering ----------

def test_loitering_alerts_after_dwell_time_once():
    zone = ZoneMonitor("platform", "loitering", SQUARE, dwell_seconds=10)
    events = []
    for t in range(0, 25):                       # person stands still for 24 s
        events += zone.update("cam", obj(3, 200, 200), NAMES, now=t)
    assert len(events) == 1
    assert events[0].type == "loitering" and events[0].track_id == 3
    assert events[0].detail == "10s"


def test_short_visit_is_not_loitering():
    zone = ZoneMonitor("platform", "loitering", SQUARE, dwell_seconds=10)
    events = []
    for t in range(0, 6):                        # 5 s inside, then leaves
        events += zone.update("cam", obj(3, 200, 200), NAMES, now=t)
    for t in range(6, 20):
        events += zone.update("cam", obj(3, 500, 200), NAMES, now=t)
    assert events == []


def test_leaving_resets_the_visit():
    zone = ZoneMonitor("platform", "loitering", SQUARE, dwell_seconds=10)
    for t in range(0, 8):
        zone.update("cam", obj(3, 200, 200), NAMES, now=t)
    for t in range(8, 12):                       # gone 4 s (> 2 s grace): visit ends
        zone.update("cam", obj(3, 500, 200), NAMES, now=t)
    events = []
    for t in range(12, 20):                      # back for 8 s: still under 10 s
        events += zone.update("cam", obj(3, 200, 200), NAMES, now=t)
    assert events == []
    assert zone.longest_dwell == pytest.approx(7)


# ---------- crowding ----------

def test_crowding_alert_with_cooldown():
    zone = ZoneMonitor("plaza", "crowd", SQUARE, max_people=3, cooldown_seconds=20)
    assert zone.update("cam", many(3), NAMES, now=0) == []          # at the limit: fine
    first = zone.update("cam", many(5), NAMES, now=1)               # over the limit
    assert len(first) == 1 and first[0].type == "crowding" and first[0].detail == "5/3"
    assert zone.update("cam", many(5), NAMES, now=10) == []         # still crowded, in cooldown
    assert len(zone.update("cam", many(5), NAMES, now=22)) == 1     # cooldown over: alert again


# ---------- line recount ----------

def test_slow_car_on_the_line_counted_once():
    line = LineCounter("road", np.array([0, 200]), np.array([400, 200]), min_frames=1)
    t = 0.0
    for y in (180, 220, 180, 220, 180, 220):     # crawling back and forth across y=200
        line.update("cam", obj(226, 200, y, cls=2), NAMES, now=t)
        t += 0.5
    assert line.in_count + line.out_count == 1


def test_same_object_counted_again_after_recount_window():
    line = LineCounter("road", np.array([0, 200]), np.array([400, 200]), min_frames=1, recount_after=10)
    line.update("cam", obj(5, 200, 180), NAMES, now=0)
    line.update("cam", obj(5, 200, 220), NAMES, now=1)          # counted
    line.update("cam", obj(5, 200, 180), NAMES, now=15)         # 14 s later, crosses back
    assert line.in_count + line.out_count == 2


# ---------- config defaults ----------

def test_slow_camera_does_not_reset_visits():
    zone = ZoneMonitor("platform", "loitering", SQUARE, dwell_seconds=10)
    events = []
    for t in range(0, 30, 3):                    # one frame every 3 s (very slow)
        events += zone.update("cam", obj(3, 200, 200), NAMES, now=t)
    assert len(events) == 1                      # visit kept going; alert fired once


def test_zone_overrides_default_dwell():
    cam = CameraAnalytics(
        "cam_03",
        [{"id": "platform", "type": "loitering", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
          "dwell_seconds": 5}],
        [],
        defaults={"loitering": {"threshold_seconds": 30}},
    )
    events = []
    for t in range(0, 8):
        events += cam.update(obj(1, 320, 240), (480, 640, 3), NAMES, now=t)
    assert [e.type for e in events] == ["loitering"]


# ---------- privacy ----------

def test_head_box_is_top_of_person_only():
    dets = sv.Detections.merge([obj(1, 200, 300), obj(2, 400, 300, cls=2)])
    boxes = head_boxes(dets, {0}, (480, 640, 3))
    assert len(boxes) == 1                       # the car gets no blur box
    x1, y1, x2, y2 = boxes[0]
    assert y1 == 220 and y2 < 240                # top ~22% of an 80 px tall box


def test_blur_changes_only_the_head_area():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    out = blur_heads(img, obj(1, 200, 300), {0})
    assert not np.array_equal(out[222:236, 190:210], img[222:236, 190:210])   # head blurred
    assert np.array_equal(out[260:300, 180:220], img[260:300, 180:220])       # body untouched
    assert np.array_equal(out[0:100, 400:640], img[0:100, 400:640])           # background untouched
