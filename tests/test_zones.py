import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from basira.analytics.zones import CameraAnalytics, LineCounter, ZoneMonitor, to_pixels  # noqa: E402

NAMES = {0: "person", 2: "car"}


def person(track_id, x_center, y_bottom, cls=0):
    """A 40x80 box whose feet (bottom centre) are at (x_center, y_bottom)."""
    return sv.Detections(
        xyxy=np.array([[x_center - 20, y_bottom - 80, x_center + 20, y_bottom]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([cls]),
        tracker_id=np.array([track_id]),
    )


def test_fractions_become_pixels():
    px = to_pixels([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], 1281, 721)
    assert px.tolist() == [[0, 0], [1280, 720], [640, 360]]


def test_pixel_values_in_config_are_rejected():
    with pytest.raises(ValueError):
        to_pixels([[820, 410], [1180, 410], [1180, 700]], 1280, 720)


def square_zone(zone_type="restricted"):
    return ZoneMonitor("box", zone_type, np.array([[100, 100], [300, 100], [300, 300], [100, 300]]))


def test_restricted_entry_fires_once_per_entry():
    zone = square_zone()
    assert zone.update("cam", person(7, 50, 200), NAMES, now=0) == []          # outside
    events = zone.update("cam", person(7, 200, 200), NAMES, now=1)             # steps in
    assert len(events) == 1 and events[0].track_id == 7 and events[0].type == "zone_entry"
    assert zone.update("cam", person(7, 210, 200), NAMES, now=2) == []         # still inside: no repeat
    for t in (3, 4, 5, 6):                                                      # out for 3 s, 4 frames
        zone.update("cam", person(7, 50, 200), NAMES, now=t)
    assert len(zone.update("cam", person(7, 200, 200), NAMES, now=7)) == 1     # re-enters: fires again


def test_one_missed_frame_does_not_double_alert():
    zone = square_zone()
    assert len(zone.update("cam", person(7, 200, 200), NAMES, now=0)) == 1
    empty = sv.Detections.empty()
    empty.tracker_id = np.array([], dtype=int)
    zone.update("cam", empty, NAMES, now=0.1)                                   # YOLO missed one frame
    assert zone.update("cam", person(7, 200, 200), NAMES, now=0.2) == []       # same visit, no new alert


def test_zone_class_filter():
    zone = ZoneMonitor("tracks", "restricted", np.array([[100, 100], [300, 100], [300, 300], [100, 300]]),
                       classes=["person"])
    bus = person(9, 200, 200, cls=2)
    assert zone.update("cam", bus, NAMES, now=0) == []
    assert len(zone.update("cam", person(5, 200, 200), NAMES, now=1)) == 1


def test_crowd_zone_counts_but_does_not_alert():
    zone = square_zone("crowd")
    dets = sv.Detections.merge([person(1, 150, 200), person(2, 250, 250), person(3, 500, 200)])
    assert zone.update("cam", dets, NAMES) == []
    assert zone.count == 2


def test_jitter_on_the_line_is_not_counted():
    line = LineCounter("mid", np.array([200, 0]), np.array([200, 400]))
    for x in (150, 170, 195, 205, 195, 205, 195, 205, 195):            # wobbling across x=200
        line.update("cam", person(1, x, 300), NAMES)
    assert line.in_count + line.out_count == 0


def test_line_counts_each_direction():
    line = LineCounter("mid", np.array([200, 0]), np.array([200, 400]))   # vertical line at x=200
    for x in (150, 170, 190, 210, 230, 250, 270):                       # person 1 walks right
        line.update("cam", person(1, x, 300), NAMES)
    for x in (270, 250, 230, 210, 190, 170, 150):                       # person 2 walks left
        line.update("cam", person(2, x, 300), NAMES)
    assert line.in_count + line.out_count == 2
    assert line.in_count == 1 and line.out_count == 1


def test_camera_analytics_builds_from_fractions():
    cam = CameraAnalytics(
        "cam_03",
        [{"id": "tracks", "type": "restricted", "polygon": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]}],
        [],
    )
    events = cam.update(person(4, 320, 240), (480, 640, 3), NAMES)    # centre of a 640x480 frame
    assert [str(e) for e in events] == ["[cam_03] person #4 zone_entry tracks"]
