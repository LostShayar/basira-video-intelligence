import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from basira.perception.annotate import count_by_class, labels_for  # noqa: E402
from basira.perception.detector import class_ids_for  # noqa: E402
from basira.perception.tracker import TrackerBank  # noqa: E402

NAMES = {0: "person", 2: "car"}


def box(x, y, w=40, h=80, cls=0, conf=0.9):
    return sv.Detections(
        xyxy=np.array([[x, y, x + w, y + h]], dtype=float),
        confidence=np.array([conf]),
        class_id=np.array([cls]),
    )


def test_class_names_map_to_ids():
    assert class_ids_for(NAMES, ["car", "person"]) == [2, 0]


def test_unknown_class_raises():
    with pytest.raises(ValueError):
        class_ids_for(NAMES, ["dragon"])


def test_moving_person_keeps_same_id():
    bank = TrackerBank(frame_rate=10)
    ids = []
    for step in range(15):  # person walks right 5 px per frame
        tracked = bank.update("cam_01", box(100 + step * 5, 200))
        if len(tracked):
            ids.append(int(tracked.tracker_id[0]))
    assert len(ids) >= 10
    assert len(set(ids)) == 1


def test_two_people_get_different_ids():
    bank = TrackerBank(frame_rate=10)
    tracked = None
    for step in range(8):
        dets = sv.Detections.merge([box(100 + step * 4, 200), box(500 - step * 4, 200)])
        tracked = bank.update("cam_01", dets)
    assert len(set(tracked.tracker_id.tolist())) == 2


def test_cameras_have_separate_trackers():
    bank = TrackerBank(frame_rate=10)
    for step in range(8):
        a = bank.update("cam_01", box(100 + step * 5, 200))
        b = bank.update("cam_02", box(100 + step * 5, 200))
    assert len(a) == 1 and len(b) == 1
    assert bank._trackers["cam_01"] is not bank._trackers["cam_02"]


def test_labels_and_counts():
    dets = sv.Detections.merge([box(0, 0), box(100, 0, cls=2)])
    dets.tracker_id = np.array([7, 12])
    assert labels_for(dets, NAMES) == ["person #7", "car #12"]
    assert count_by_class(dets, NAMES) == {"person": 1, "car": 1}
