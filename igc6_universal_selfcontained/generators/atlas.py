"""Sprite stamping onto RGBA canvases and flipbook frame selection.

Pure numpy compositing. This never touches ``_render_object`` — it draws
directly onto whatever RGBA ``np.ndarray`` canvas the caller passes in
(typically the output of ``make_scene_raster(..., mode="rgba")`` or a
``TilemapRenderer``/``LayerManager`` canvas), so it cannot change what
``make_scene`` itself produces.
"""

from __future__ import annotations

import numpy as np

from .scene_spec import AtlasRegion, AtlasSpec, FlipbookClip


def stamp_sprite(
    canvas: np.ndarray,
    atlas: AtlasSpec,
    region: AtlasRegion,
    x: float,
    y: float,
    scale: float = 1.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    """Alpha-composite ``region`` of ``atlas.image`` onto ``canvas`` at (x, y).

    ``canvas`` and ``atlas.image`` are H×W×4 uint8 RGBA arrays. ``(x, y)`` is
    where the region's pivot lands. Returns ``canvas`` (mutated in place, also
    returned for chaining).
    """

    src = atlas.image[region.y : region.y + region.height, region.x : region.x + region.width]
    if flip_x:
        src = src[:, ::-1]
    if flip_y:
        src = src[::-1, :]

    if scale != 1.0:
        new_h = max(1, round(region.height * scale))
        new_w = max(1, round(region.width * scale))
        yi = (np.arange(new_h) * region.height // new_h).clip(0, region.height - 1)
        xi = (np.arange(new_w) * region.width // new_w).clip(0, region.width - 1)
        src = src[yi][:, xi]

    h, w = src.shape[:2]
    px = int(round(x - w * region.pivot_x))
    py = int(round(y - h * region.pivot_y))

    dst_x0, dst_y0 = max(px, 0), max(py, 0)
    dst_x1, dst_y1 = min(px + w, canvas.shape[1]), min(py + h, canvas.shape[0])
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return canvas  # fully off-canvas

    src_x0, src_y0 = dst_x0 - px, dst_y0 - py
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)

    src_slice = src[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    dst_slice = canvas[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)

    alpha = src_slice[..., 3:4] / 255.0
    out = src_slice[..., :3] * alpha + dst_slice[..., :3] * (1.0 - alpha)
    out_a = src_slice[..., 3:4] + dst_slice[..., 3:4] * (1.0 - alpha)

    canvas[dst_y0:dst_y1, dst_x0:dst_x1, :3] = out.clip(0, 255).astype(np.uint8)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1, 3:4] = out_a.clip(0, 255).astype(np.uint8)
    return canvas


def flipbook_region(clip: FlipbookClip, elapsed_ms: float) -> AtlasRegion:
    """The frame of ``clip`` active at ``elapsed_ms`` into playback."""

    n = len(clip.frames)
    idx = int(elapsed_ms // clip.frame_duration_ms)
    if clip.loop:
        idx %= n
    else:
        idx = min(idx, n - 1)
    return clip.frames[idx]


if __name__ == "__main__":
    atlas_img = np.zeros((16, 16, 4), np.uint8)
    atlas_img[0:8, 0:8] = (255, 0, 0, 255)  # red 8x8 region at origin
    regions = (AtlasRegion(name="red", x=0, y=0, width=8, height=8),)
    atlas = AtlasSpec(image=atlas_img, regions=regions, default_region="red")
    assert atlas.get("red").width == 8

    canvas = np.zeros((32, 32, 4), np.uint8)
    stamp_sprite(canvas, atlas, atlas.get("red"), x=16, y=16)
    assert tuple(canvas[16, 16]) == (255, 0, 0, 255)
    assert tuple(canvas[0, 0]) == (0, 0, 0, 0)  # untouched corner

    clip = FlipbookClip(frames=(atlas.get("red"), atlas.get("red")), frame_duration_ms=100, loop=True)
    assert flipbook_region(clip, 50) is clip.frames[0]
    assert flipbook_region(clip, 150) is clip.frames[1]
    assert flipbook_region(clip, 250) is clip.frames[0]  # wraps

    print("OK — sprite stamping alpha-composites and flipbook frame selection wraps correctly")
