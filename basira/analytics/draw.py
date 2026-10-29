"""Draw zones, lines and their live counts on a frame."""
from __future__ import annotations

import cv2
import numpy as np

from basira.analytics.zones import CameraAnalytics

# BGR colours by zone type
ZONE_COLOURS = {
    "restricted": (40, 40, 230),   # red
    "crowd": (230, 150, 30),       # blue
    "loitering": (30, 200, 230),   # amber
}
LINE_COLOUR = (60, 220, 60)        # green


def _label(img, text, org, colour):
    # text grows with the frame, so it stays readable after tiling (1280 px wide -> scale 1.0)
    scale = max(0.6, img.shape[1] / 1280)
    thick = max(2, round(scale * 2))
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - h - 10), (x + w + 10, y + 4), colour, -1)
    cv2.putText(img, text, (x + 5, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)


def draw_analytics(image: np.ndarray, analytics: CameraAnalytics) -> np.ndarray:
    out = image.copy()
    overlay = out.copy()
    for zone in analytics.zones:
        colour = ZONE_COLOURS.get(zone.type, (200, 200, 200))
        cv2.fillPoly(overlay, [zone.polygon], colour)
    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)  # transparent fill

    for zone in analytics.zones:
        colour = ZONE_COLOURS.get(zone.type, (200, 200, 200))
        cv2.polylines(out, [zone.polygon], True, colour, 2)
        x, y = zone.polygon.min(axis=0)
        text = f"{zone.zone_id}: {zone.count}"
        if zone.max_people:
            text += f"/{zone.max_people}"
        _label(out, text, (int(x) + 6, int(y) + int(40 * max(0.6, out.shape[1] / 1280))), colour)

    for line in analytics.lines:
        cv2.line(out, line.start, line.end, LINE_COLOUR, 3)
        mx, my = (line.start[0] + line.end[0]) // 2, (line.start[1] + line.end[1]) // 2
        _label(out, f"{line.line_id}  in {line.in_count} | out {line.out_count}", (max(5, mx - 150), max(40, my - 12)), LINE_COLOUR)
    return out


def draw_alert(image: np.ndarray, text: str) -> np.ndarray:
    """Red banner at the bottom of the frame for the latest alert."""
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(0.8, w / 1100)
    bar = int(50 * scale)
    cv2.rectangle(out, (0, h - bar), (w, h), (30, 30, 200), -1)
    cv2.putText(out, text, (14, h - int(bar * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(2, round(scale * 2)))
    return out
