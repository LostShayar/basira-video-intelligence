"""Privacy: blur the head area of every person before a frame is shown or saved.

Basira never identifies people, so it doesn't need a face model. Instead it
blurs the top part of each person's box, where the head is. This works from
any angle (faces turned away, small faces far from the camera) where a face
detector would miss, and costs almost nothing.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

HEAD_FRACTION = 0.22   # top 22% of a standing person's box is roughly the head


def head_boxes(detections: sv.Detections, person_class_ids: set[int], frame_shape: tuple) -> np.ndarray:
    """Return [x1, y1, x2, y2] boxes covering the head of each detected person."""
    h, w = frame_shape[:2]
    boxes = []
    for (x1, y1, x2, y2), cid in zip(detections.xyxy, detections.class_id):
        if int(cid) not in person_class_ids:
            continue
        bw, bh = x2 - x1, y2 - y1
        hx1 = x1 + bw * 0.15
        hx2 = x2 - bw * 0.15
        hy2 = y1 + bh * HEAD_FRACTION
        boxes.append([max(0, hx1), max(0, y1), min(w, hx2), min(h, hy2)])
    return np.array(boxes, dtype=int).reshape(-1, 4)


def blur_heads(image: np.ndarray, detections: sv.Detections, person_class_ids: set[int]) -> np.ndarray:
    """Return a copy of `image` with every person's head area blurred."""
    out = image.copy()
    for x1, y1, x2, y2 in head_boxes(detections, person_class_ids, image.shape):
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = out[y1:y2, x1:x2]
        k = max(3, (min(roi.shape[:2]) // 2) | 1)   # odd kernel that scales with head size
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out
