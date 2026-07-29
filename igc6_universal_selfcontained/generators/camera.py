"""2D camera / viewport transform.

Pure math, no dependency on the renderer. Converts between world-space scene
coordinates and screen-space pixel coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .scene_spec import BoundingBox


@dataclass
class CameraSpec:
    viewport_width: int = 640
    viewport_height: int = 480
    world_x: float = 0.0
    world_y: float = 0.0
    zoom: float = 1.0
    rotation: float = 0.0  # degrees


def world_to_screen(wx: float, wy: float, camera: CameraSpec) -> tuple[float, float]:
    dx = (wx - camera.world_x) * camera.zoom
    dy = (wy - camera.world_y) * camera.zoom
    if camera.rotation:
        rad = math.radians(camera.rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        dx, dy = dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r
    return (dx + camera.viewport_width / 2.0, dy + camera.viewport_height / 2.0)


def screen_to_world(sx: float, sy: float, camera: CameraSpec) -> tuple[float, float]:
    dx = sx - camera.viewport_width / 2.0
    dy = sy - camera.viewport_height / 2.0
    if camera.rotation:
        rad = math.radians(-camera.rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        dx, dy = dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r
    return (dx / camera.zoom + camera.world_x, dy / camera.zoom + camera.world_y)


def aabb_in_view(bbox: BoundingBox, camera: CameraSpec) -> bool:
    """True if ``bbox`` (world space) overlaps the camera's viewport.

    ponytail: ignores rotation for the culling test (uses the axis-aligned
    world-space viewport rect) — a conservative approximation that may keep
    a few extra objects near the edges when the camera is rotated, never
    drops one that should be visible.
    """

    half_w = camera.viewport_width / 2.0 / camera.zoom
    half_h = camera.viewport_height / 2.0 / camera.zoom
    view = BoundingBox(
        camera.world_x - half_w,
        camera.world_y - half_h,
        camera.world_x + half_w,
        camera.world_y + half_h,
    )
    return bbox.overlaps(view)


if __name__ == "__main__":
    cam = CameraSpec(viewport_width=640, viewport_height=480, world_x=100, world_y=50, zoom=2.0)
    sx, sy = world_to_screen(100, 50, cam)
    assert (sx, sy) == (320.0, 240.0)  # camera center maps to viewport center
    wx, wy = screen_to_world(sx, sy, cam)
    assert (round(wx, 6), round(wy, 6)) == (100.0, 50.0)

    rot_cam = CameraSpec(viewport_width=100, viewport_height=100, rotation=90)
    rx, ry = world_to_screen(10, 0, rot_cam)
    assert abs(rx - 50.0) < 1e-6 and abs(ry - 60.0) < 1e-6

    near = BoundingBox(90, 40, 110, 60)
    far = BoundingBox(10_000, 10_000, 10_010, 10_010)
    assert aabb_in_view(near, cam)
    assert not aabb_in_view(far, cam)

    print("OK — camera transforms round-trip and cull correctly")
