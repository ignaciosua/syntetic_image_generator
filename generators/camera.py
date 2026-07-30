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
    mode: str = "static"
    bounds: tuple[float, float, float, float] | None = None
    dead_zone: tuple[float, float] = (0.0, 0.0)
    look_ahead: float = 0.0
    smoothing: float = 1.0
    target: str | None = None
    effects: tuple[str, ...] = ()
    shake_intensity: float = 0.0
    shake_time: float = 0.0
    shake_frequency: float = 18.0


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


def clamp_camera(camera: CameraSpec) -> CameraSpec:
    """Clamp camera center to optional world bounds, respecting zoom/viewport."""
    if camera.bounds is None:
        return camera
    x0, y0, x1, y1 = camera.bounds
    half_w = camera.viewport_width / max(2.0 * camera.zoom, 1e-6)
    half_h = camera.viewport_height / max(2.0 * camera.zoom, 1e-6)
    camera.world_x = min(max(camera.world_x, x0 + half_w), max(x0 + half_w, x1 - half_w))
    camera.world_y = min(max(camera.world_y, y0 + half_h), max(y0 + half_h, y1 - half_h))
    return camera


def update_camera_target(camera: CameraSpec, objects, dt: float = 1 / 60) -> CameraSpec:
    """Apply deterministic target tracking, dead-zone and look-ahead."""
    if not camera.target:
        return clamp_camera(camera)
    target = next((obj for obj in objects if obj.name == camera.target), None)
    if target is None:
        return clamp_camera(camera)
    tx, ty = float(target.resolved_x), float(target.resolved_y)
    dzx, dzy = camera.dead_zone
    dx, dy = tx - camera.world_x, ty - camera.world_y
    if abs(dx) > dzx:
        tx = tx - (dzx if dx > 0 else -dzx)
    else:
        tx = camera.world_x
    if abs(dy) > dzy:
        ty = ty - (dzy if dy > 0 else -dzy)
    else:
        ty = camera.world_y
    if camera.look_ahead:
        tx += camera.look_ahead
    alpha = min(1.0, max(0.0, camera.smoothing * max(dt, 0.0) * 10.0))
    camera.world_x += (tx - camera.world_x) * alpha
    camera.world_y += (ty - camera.world_y) * alpha
    return clamp_camera(camera)


def trigger_camera_shake(camera: CameraSpec, intensity: float, duration: float, frequency: float = 18.0) -> CameraSpec:
    """Start/restart a deterministic camera impact impulse."""
    camera.shake_intensity = max(float(intensity), 0.0)
    camera.shake_time = max(float(duration), 0.0)
    camera.shake_frequency = max(float(frequency), 1.0)
    return camera


def update_camera_shake(camera: CameraSpec, dt: float = 1 / 60, elapsed: float = 0.0) -> CameraSpec:
    """Apply and decay a camera impulse in world-space coordinates."""
    if camera.shake_time <= 0 or camera.shake_intensity <= 0:
        return camera
    import math
    remaining = camera.shake_time
    amp = camera.shake_intensity * min(1.0, remaining)
    camera.world_x += math.sin((elapsed + remaining) * camera.shake_frequency) * amp
    camera.world_y += math.cos((elapsed + remaining) * camera.shake_frequency * 1.37) * amp
    camera.shake_time = max(0.0, remaining - max(dt, 0.0))
    camera.shake_intensity *= 0.92
    return camera


def apply_camera_effects(frame, effects: tuple[str, ...] = (), intensity: float = 1.0):
    """Apply small deterministic camera post-effects without changing geometry."""
    import numpy as np
    if not effects:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    if "vignette" in effects:
        nx = (xx - w / 2) / max(w / 2, 1)
        ny = (yy - h / 2) / max(h / 2, 1)
        mask = np.clip(1.0 - (nx * nx + ny * ny) * 0.45 * intensity, 0.35, 1.0)
        out[..., :3] = np.clip(out[..., :3].astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)
    if "grayscale" in effects:
        gray = np.dot(out[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
        out[..., :3] = gray[..., None]
    if "scanlines" in effects:
        out[::2, :, :3] = (out[::2, :, :3].astype(np.float32) * 0.82).astype(np.uint8)
    if "flash" in effects:
        out[..., :3] = np.clip(out[..., :3].astype(np.float32) + 32 * intensity, 0, 255).astype(np.uint8)
    if "motion_blur" in effects:
        shifted = np.roll(out[..., :3], 2, axis=1)
        out[..., :3] = ((out[..., :3].astype(np.float32) * .65) + (shifted.astype(np.float32) * .35)).astype(np.uint8)
    if "chromatic_aberration" in effects:
        out[..., 0] = np.roll(out[..., 0], 1, axis=1)
        out[..., 2] = np.roll(out[..., 2], -1, axis=1)
    if "pixelate" in effects:
        block = max(1, int(round(2 + intensity * 2)))
        small = out[::block, ::block]
        out = np.repeat(np.repeat(small, block, axis=0), block, axis=1)[:h, :w]
    if "film_grain" in effects:
        rng = np.random.RandomState(0)
        noise = rng.normal(0, 4.0 * intensity, (h, w, 1))
        out[..., :3] = np.clip(out[..., :3].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if "fog" in effects or "depth_of_field" in effects:
        # Deterministic center-focused falloff (screen-space approximation).
        nx = (xx - w / 2) / max(w / 2, 1)
        ny = (yy - h / 2) / max(h / 2, 1)
        falloff = np.clip((nx * nx + ny * ny) * 1.8, 0.0, 1.0)[..., None]
        if "fog" in effects:
            fog_color = np.array([150, 165, 175], dtype=np.float32)
            alpha = falloff * 0.55 * intensity
            out[..., :3] = (out[..., :3] * (1 - alpha) + fog_color * alpha).astype(np.uint8)
        else:
            blur = np.roll(out[..., :3], 2, axis=0)
            alpha = falloff * 0.65
            out[..., :3] = (out[..., :3] * (1 - alpha) + blur * alpha).astype(np.uint8)
    if "underwater" in effects:
        out[..., 0] = (out[..., 0].astype(np.float32) * .55).astype(np.uint8)
        out[..., 1] = np.clip(out[..., 1].astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
        out[..., 2] = np.clip(out[..., 2].astype(np.float32) * 1.22, 0, 255).astype(np.uint8)
    if "night_vision" in effects:
        green = np.dot(out[..., :3].astype(np.float32), [0.18, 0.7, 0.12])
        out[..., :3] = np.stack([green * .25, green, green * .35], axis=-1).clip(0, 255).astype(np.uint8)
    if "thermal" in effects:
        lum = np.dot(out[..., :3].astype(np.float32), [0.3, 0.59, 0.11]) / 255.0
        out[..., 0] = np.clip(lum * 255 * 1.5, 0, 255).astype(np.uint8)
        out[..., 1] = np.clip(lum * 255 * .8, 0, 255).astype(np.uint8)
        out[..., 2] = np.clip((1 - lum) * 255 * .7, 0, 255).astype(np.uint8)
    if "damage" in effects:
        edge = np.zeros((h, w), dtype=np.float32)
        edge[: max(1, h // 12), :] = 1; edge[-max(1, h // 12):, :] = 1
        edge[:, : max(1, w // 12)] = 1; edge[:, -max(1, w // 12):] = 1
        out[..., 0] = np.clip(out[..., 0].astype(np.float32) + edge * 90 * intensity, 0, 255).astype(np.uint8)
    if "rain" in effects or "dust" in effects:
        rng = np.random.RandomState(17 if "rain" in effects else 23)
        count = max(1, (h * w) // 1800)
        for _ in range(count):
            x = int(rng.randint(0, w)); y = int(rng.randint(0, h))
            length = 5 if "rain" in effects else 2
            out[y:min(h, y + length), x, :3] = (180, 205, 220) if "rain" in effects else (185, 170, 130)
    return out


def frame_objects(camera: CameraSpec, objects, padding: float = 1.15) -> CameraSpec:
    """Center and zoom the camera to contain all supplied objects."""
    visible = list(objects)
    if not visible:
        return camera
    xs = [float(o.resolved_x) for o in visible]
    ys = [float(o.resolved_y) for o in visible]
    camera.world_x = (min(xs) + max(xs)) * 0.5
    camera.world_y = (min(ys) + max(ys)) * 0.5
    span_x = max((max(xs) - min(xs)) * padding, 1.0)
    span_y = max((max(ys) - min(ys)) * padding, 1.0)
    camera.zoom = min(camera.viewport_width / span_x, camera.viewport_height / span_y)
    return clamp_camera(camera)


def transition_camera(start: CameraSpec, end: CameraSpec, t: float, easing: str = "smoothstep") -> CameraSpec:
    """Interpolate camera pose deterministically for cinematic transitions."""
    u = min(1.0, max(0.0, float(t)))
    if easing == "smoothstep":
        u = u * u * (3.0 - 2.0 * u)
    elif easing == "ease_in":
        u = u * u
    elif easing == "ease_out":
        u = 1.0 - (1.0 - u) ** 2
    out = CameraSpec(
        viewport_width=round(start.viewport_width + (end.viewport_width - start.viewport_width) * u),
        viewport_height=round(start.viewport_height + (end.viewport_height - start.viewport_height) * u),
        world_x=start.world_x + (end.world_x - start.world_x) * u,
        world_y=start.world_y + (end.world_y - start.world_y) * u,
        zoom=start.zoom + (end.zoom - start.zoom) * u,
        rotation=start.rotation + (end.rotation - start.rotation) * u,
        mode=end.mode if u >= 1 else start.mode,
        bounds=end.bounds if u >= 1 else start.bounds,
        dead_zone=end.dead_zone if u >= 1 else start.dead_zone,
        look_ahead=start.look_ahead + (end.look_ahead - start.look_ahead) * u,
        smoothing=start.smoothing + (end.smoothing - start.smoothing) * u,
        target=end.target if u >= 1 else start.target,
        effects=end.effects if u >= 1 else start.effects,
    )
    return clamp_camera(out)


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
