"""Draw boxes and 'person #7' style labels on a frame."""
from __future__ import annotations

import numpy as np
import supervision as sv

# colour by track id, so each person/vehicle keeps its own colour
_box = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
_label = sv.LabelAnnotator(text_scale=0.5, text_padding=4, color_lookup=sv.ColorLookup.TRACK)


def labels_for(detections: sv.Detections, names: dict[int, str]) -> list[str]:
    labels = []
    for class_id, track_id in zip(detections.class_id, detections.tracker_id):
        name = names.get(int(class_id), str(class_id))
        labels.append(f"{name} #{int(track_id)}" if track_id is not None else name)
    return labels


def draw(image: np.ndarray, detections: sv.Detections, names: dict[int, str]) -> np.ndarray:
    out = image.copy()
    if len(detections) == 0:
        return out
    out = _box.annotate(out, detections)
    return _label.annotate(out, detections, labels=labels_for(detections, names))


def count_by_class(detections: sv.Detections, names: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_id in detections.class_id:
        name = names.get(int(class_id), str(class_id))
        counts[name] = counts.get(name, 0) + 1
    return counts
