"""Object detection with Ultralytics YOLO.

One shared model serves every camera: frames from all cameras are sent in a
single batch, which is much faster than one call per camera.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import supervision as sv

log = logging.getLogger(__name__)


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Detector:
    def __init__(
        self,
        weights: str = "yolo11n.pt",
        classes: list[str] | None = None,
        confidence: float = 0.35,
        image_size: int = 640,
        device: str = "auto",
    ):
        from ultralytics import YOLO  # imported here so tests without YOLO still run

        self.model = YOLO(weights)
        self.device = pick_device(device)
        self.confidence = confidence
        self.image_size = image_size
        self.names: dict[int, str] = self.model.names
        self.class_ids = class_ids_for(self.names, classes) if classes else None
        self.last_ms = 0.0
        log.info("YOLO %s on %s, classes=%s", weights, self.device, classes or "all")

    def detect_batch(self, images: list[np.ndarray]) -> list[sv.Detections]:
        if not images:
            return []
        start = time.perf_counter()
        results = self.model.predict(
            images,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )
        self.last_ms = (time.perf_counter() - start) * 1000
        return [sv.Detections.from_ultralytics(r) for r in results]


def class_ids_for(names: dict[int, str], wanted: list[str]) -> list[int]:
    """Turn class names like ['person', 'car'] into the model's numeric ids."""
    lookup = {name: idx for idx, name in names.items()}
    missing = [w for w in wanted if w not in lookup]
    if missing:
        raise ValueError(f"Unknown classes {missing}. Model knows: {sorted(lookup)}")
    return [lookup[w] for w in wanted]
