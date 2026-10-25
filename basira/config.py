"""Load configs/cameras.yaml into plain Python objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cameras.yaml"


@dataclass
class CameraConfig:
    id: str
    name: str
    source: str
    zones: list = field(default_factory=list)
    lines: list = field(default_factory=list)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def load_cameras(path: str | Path = DEFAULT_CONFIG) -> list[CameraConfig]:
    cfg = load_config(path)
    cameras = []
    for c in cfg["cameras"]:
        source = c["source"]
        # Relative file paths are resolved from the repo root, so the script
        # works no matter which folder you run it from.
        if "://" not in source and not Path(source).is_absolute():
            source = str(ROOT / source)
        cameras.append(
            CameraConfig(
                id=c["id"],
                name=c.get("name", c["id"]),
                source=source,
                zones=c.get("zones", []),
                lines=c.get("lines", []),
            )
        )
    return cameras
