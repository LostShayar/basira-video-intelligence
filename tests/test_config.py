"""Smoke tests so CI is green from Day 1. Real analytics tests arrive in Week 1."""
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cameras.yaml"


def load():
    with CONFIG.open() as f:
        return yaml.safe_load(f)


def test_config_has_cameras():
    cfg = load()
    assert len(cfg["cameras"]) >= 1


def test_camera_ids_are_unique():
    ids = [c["id"] for c in load()["cameras"]]
    assert len(ids) == len(set(ids))


def test_zone_polygons_have_at_least_three_points():
    for cam in load()["cameras"]:
        for zone in cam.get("zones", []):
            assert len(zone["polygon"]) >= 3, f"{cam['id']}/{zone['id']}"


def test_privacy_blur_is_on_by_default():
    assert load()["privacy"]["blur_faces"] is True
