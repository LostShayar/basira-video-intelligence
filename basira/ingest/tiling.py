"""Combine several camera frames into one grid image for display."""
from __future__ import annotations

import math

import cv2
import numpy as np


def tile_frames(frames: list, labels: list[str], tile_w: int = 640, tile_h: int = 360):
    """frames: list of BGR images or None. Returns one BGR image."""
    n = max(len(frames), 1)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for i, (img, label) in enumerate(zip(frames, labels)):
        r, c = divmod(i, cols)
        if img is None:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            cv2.putText(tile, "NO SIGNAL", (tile_w // 2 - 90, tile_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            tile = cv2.resize(img, (tile_w, tile_h))
        cv2.rectangle(tile, (0, 0), (tile_w, 30), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
    return grid
