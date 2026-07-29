"""
synthetic_image_generator.py — Deterministic progressive visual generator.

Single responsibility: given an integer index and native canvas dimensions,
return a float32 RGB image. The historical default remains 32×32×3.
Zero disk. Zero network. 100% deterministic for the same index and dimensions.

154 levels of progressive visual complexity:
  1D → 2D → 3D → hollow → edges → palettes → textures →
  depth → creatures → flora → buildings → elements → scenes → chaos →
  styles → capture pipeline → vehicles → faces → optics → natural texture →
  fractals → man-made grids → sky → weather → reflections → codec artifacts

Usage:
  import:  from synthetic_image_generator import make_image
  CLI:     python3 synthetic_image_generator.py --demo
           python3 synthetic_image_generator.py --single 42
"""

import numpy as np
import hashlib
import math
import sys
import os
import argparse
import multiprocessing
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn, idctn
from scipy.ndimage import map_coordinates, binary_fill_holes

# ── Constants ──
PRIMO = 2654435761   # golden-ratio prime for hash seed
DEFAULT_HEIGHT, DEFAULT_WIDTH, C = 32, 32, 3
MIN_NATIVE_DIMENSION = 32
MAX_CANVAS_DIMENSION = 4096
MAX_CANVAS_PIXELS = 8_388_608
H, W = DEFAULT_HEIGHT, DEFAULT_WIDTH
_SCALE_INDEXED_GEOMETRY = False
# Level schedule is shared by all six generators so that idx → level is the same
# everywhere; see levels.py for what it used to be and why that broke cross-modal
# learning. This file's own weighted table is gone with it — uneven per-level budgets
# were six different opinions that could not be reconciled into one schedule.
if __package__:
    from ._primitive_ir import PrimitiveOp, command as _primitive_command
    from ._rendergraph import RenderGraph
    from .levels import (
        N_LEVELS,
        LEVEL_NAMES,
        LEVEL_TABLE,
        N_BLOCKS,
        SAMPLES_PER_LEVEL,
        level_block as _level_block,
        level_of as _level_of,
        level_start as _level_start,
    )
    from .scene import scene as _scene, apply_visual as _apply_scene
    from .scene_spec import (
        Background,
        LightSpec,
        LayoutRelation,
        LayoutSpec,
        MaterialSpec,
        ObjectSpec,
        PostSpec,
        RasterSpec,
        SceneSpec,
    )
    from .materials import evaluate_material
    from .layout import resolve_layout
    from .wave import (
        WAVE_LEVELS,
        theta as wave_theta,
        render_image as render_wave_image,
    )
else:
    # Historical scripts add ``generators/`` directly to sys.path and import
    # this file as a top-level module. Keep that supported while the same files
    # are also mapped to the installed ``synthetic_image_generator`` package.
    from _primitive_ir import PrimitiveOp, command as _primitive_command
    from _rendergraph import RenderGraph
    from levels import (
        N_LEVELS,
        LEVEL_NAMES,
        LEVEL_TABLE,
        N_BLOCKS,
        SAMPLES_PER_LEVEL,
        level_block as _level_block,
        level_of as _level_of,
        level_start as _level_start,
    )
    from scene import scene as _scene, apply_visual as _apply_scene
    from scene_spec import (
        Background,
        LightSpec,
        LayoutRelation,
        LayoutSpec,
        MaterialSpec,
        ObjectSpec,
        PostSpec,
        RasterSpec,
        SceneSpec,
    )
    from materials import evaluate_material
    from layout import resolve_layout
    from wave import (
        WAVE_LEVELS,
        theta as wave_theta,
        render_image as render_wave_image,
    )

CONTENT_POOL = [4, 7, 9, 15, 21, 28, 29, 37, 49, 53, 59, 64, 68]   # style base (levels 72-79)
# full-frame levels only: nothing on a black background, and placement is canvas-relative
# so they survive the resolution swap used by the capture-pipeline levels
FRAME_POOL = [7, 21, 28, 29, 33, 37, 43, 44, 64, 66, 68, 69, 71]
# + new levels that never re-enter make_image (no recursion) — base for the 32px post levels
# levels whose colors/edges already come from a base render — don't post-process twice
POST_LEVELS = set(range(72, 82)) | {86, 87, 88, 89, 96, 98, 116, 143}
RECURSIVE_LEVELS = {129, 130, 133, 135, 136}   # symmetry/diffraction/refraction filters that
                                                # recurse on any *other* level, see below
RICH_POOL = FRAME_POOL + [82, 83, 84, 85, 90, 91, 92, 94, 95, 97, 99,
                          100, 101, 103, 104, 105, 106, 109, 110, 111,
                          112, 115, 117, 118, 119, 120]   # 113/114 excluded: slowest levels

# ═══════════════════════════════════════════════════════════════
# Internal helpers (not part of public API)
# ═══════════════════════════════════════════════════════════════

def _rng(idx):
    return np.random.RandomState(
        int.from_bytes(hashlib.sha256(f"{idx}_{PRIMO}".encode()).digest()[:4], "little")
    )

def _rand_light(rng):
    L = np.array([rng.uniform(-1, 1), rng.uniform(-0.5, 0.8), rng.uniform(0.2, 1)])
    return L / (np.linalg.norm(L) + 1e-8)

def _rand_light_col(rng):
    """Real light sources aren't always neutral-white at fixed strength: golden hour, tungsten,
    overcast blue, neon/stage gels, harsh sun vs dim room. Most renders still stay plain white —
    this is a rare-ish tint+intensity multiplier applied to whatever the surface color already is."""
    intensity = rng.uniform(0.5, 1.6) if rng.randint(0, 3) == 0 else 1.0
    if rng.randint(0, 4) == 0:                        # colored light source, not just brightness
        tint = rng.randn(C).astype(np.float32) * rng.uniform(0.1, 0.5)
        col = np.clip(1.0 + tint - tint.mean(), 0.35, 1.8)
    else:
        col = np.ones(C, np.float32)
    return (col * intensity).astype(np.float32)

def _phong(Nx, Ny, Nz, L):
    diff = np.maximum(0, Nx * L[0] + Ny * L[1] + Nz * L[2])
    spec = np.maximum(0, 2 * Nz * Nz - 1) ** 8
    return np.clip(0.12 + 0.65 * diff + 0.35 * spec, 0, 1)

def _perlin(rng, octaves=3):
    n = np.zeros((H, W), np.float32)
    for o in range(octaves):
        s = 2 ** o
        # Ceil division covers odd and non-power-of-two canvas dimensions.
        nh, nw = max(1, math.ceil(H / s)), max(1, math.ceil(W / s))
        b = rng.randn(nh, nw).astype(np.float32)
        b = np.repeat(
            np.repeat(b, s, axis=0)[:H], s, axis=1
        )[:, :W]
        n += b / (2 ** o)
    mn, mx = n.min(), n.max()
    return (n - mn) / (mx - mn + 1e-8)

def _tex_at(tex, x, y):
    """Bilinear sample from a (H,W) texture array at float coordinates."""
    x = np.clip(x, 0, W - 1.001)
    y = np.clip(y, 0, H - 1.001)
    ix = np.asarray(x).astype(np.intp)
    iy = np.asarray(y).astype(np.intp)
    fx, fy = x - ix, y - iy
    return (tex[iy, ix] * (1 - fx) * (1 - fy) +
            tex[iy, np.minimum(ix + 1, W - 1)] * fx * (1 - fy) +
            tex[np.minimum(iy + 1, H - 1), ix] * (1 - fx) * fy +
            tex[np.minimum(iy + 1, H - 1),
                np.minimum(ix + 1, W - 1)] * fx * fy)

def _edge_sobel(ch):
    gx = np.abs(np.diff(ch, axis=1, append=ch[:, -1:]))
    gy = np.abs(np.diff(ch, axis=0, append=ch[-1:, :]))
    g = np.sqrt(gx ** 2 + gy ** 2)
    return (g - g.min()) / (g.max() - g.min() + 1e-8)

def _bresenham(img, x0, y0, x1, y1, thick, color):
    h, w = img.shape[:2]
    scale = _indexed_geometry_scales()[2]
    if scale != 1.0:
        thick = max(
            0, int(round(((2 * int(thick) + 1) * scale - 1) * 0.5))
        )
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        for ddy in range(-thick, thick + 1):
            for ddx in range(-thick, thick + 1):
                px, py = x0 + ddx, y0 + ddy
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


@lru_cache(maxsize=8)
def _cached_ogrid(height, width):
    """Return immutable broadcast grids reused by geometry-heavy primitives."""
    yy, xx = np.ogrid[:height, :width]
    yy.setflags(write=False)
    xx.setflags(write=False)
    return yy, xx


def _canvas_ogrid():
    return _cached_ogrid(H, W)


def _indexed_geometry_scales():
    if not _SCALE_INDEXED_GEOMETRY:
        return 1.0, 1.0, 1.0
    scale_x = W / DEFAULT_WIDTH
    scale_y = H / DEFAULT_HEIGHT
    return scale_x, scale_y, min(scale_x, scale_y)


def _sphere(cx, cy, R):
    R *= _indexed_geometry_scales()[2]
    yy, xx = _canvas_ogrid()
    dd = ((xx - cx) ** 2 + (yy - cy) ** 2) / R ** 2
    m = dd <= 1.0
    return m, np.where(m, (xx - cx) / R, 0), np.where(m, (yy - cy) / R, 0), np.where(m, np.sqrt(np.maximum(0, 1.0 - dd)), 0)

def _ellipse(cx, cy, rx, ry):
    scale_x, scale_y, _scale = _indexed_geometry_scales()
    rx, ry = rx * scale_x, ry * scale_y
    yy, xx = _canvas_ogrid()
    return ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) <= 1

def _cylinder_side(cx, cy, R):
    R *= _indexed_geometry_scales()[2]
    yy, xx = _canvas_ogrid()
    dd = ((xx - cx) ** 2 + (yy - cy) ** 2) / R ** 2
    m = dd <= 1.0
    rr = np.sqrt(np.maximum(dd, 0)) * R
    return m, np.where(m, (xx - cx) / (rr + 1e-8), 0), np.where(m, (yy - cy) / (rr + 1e-8), 0)

def _organic_blob(rng, cx, cy, rx, ry):
    scale_x, scale_y, _scale = _indexed_geometry_scales()
    rx, ry = rx * scale_x, ry * scale_y
    yy, xx = _canvas_ogrid()
    ang = np.arctan2(yy - cy, xx - cx)
    na = 24
    a = np.linspace(0, 2 * math.pi, na)
    p = rng.randn(na).astype(np.float32) * 0.3
    p = np.convolve(p, np.ones(4) / 4, mode='same')
    warp = np.interp(ang % (2 * math.pi), a, p, period=2 * math.pi).reshape(H, W)
    wrx, wry = rx * (1 + warp), ry * (1 + warp)
    d = np.sqrt(((xx - cx) ** 2 / wrx ** 2 + (yy - cy) ** 2 / wry ** 2))
    m = d <= 1.0
    return m, np.where(m, (xx - cx) / (wrx + 1e-8), 0), np.where(m, (yy - cy) / (wry + 1e-8), 0), np.where(m, np.sqrt(np.maximum(0, 1.0 - d ** 2)), 0)

def _splat(rng, cx, cy, R):
    R *= _indexed_geometry_scales()[2]
    yy, xx = _canvas_ogrid()
    rx, ry = R, R * rng.uniform(0.5, 1.8)
    ang = rng.uniform(0, math.pi)
    xr = (xx - cx) * math.cos(ang) + (yy - cy) * math.sin(ang)
    yr = -(xx - cx) * math.sin(ang) + (yy - cy) * math.cos(ang)
    d = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    for _ in range(rng.randint(2, 5)):
        bx = rng.uniform(-R * 0.7, R * 0.7)
        by = rng.uniform(-R * 0.7, R * 0.7)
        br = rng.uniform(R * 0.3, R * 0.9)
        nd = np.sqrt((xr / rx - bx / (rx + 1e-8)) ** 2 + (yr / ry - by / (ry + 1e-8)) ** 2) / (br / R)
        d = np.minimum(d, nd)
    m = d <= 1.0
    return m, np.where(m, -(xx - cx) / (R + 1e-8), 0), np.where(m, -(yy - cy) / (R + 1e-8), 0), np.where(m, np.sqrt(np.maximum(0, 1.0 - d)), 0)

def _hollow_ellipse(cx, cy, rx, ry, th):
    scale_x, scale_y, scale = _indexed_geometry_scales()
    rx, ry, th = rx * scale_x, ry * scale_y, th * scale
    yy, xx = _canvas_ogrid()
    do = ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2)
    tirx, tiry = rx - th, ry - th
    if tirx < 0.5 or tiry < 0.5:
        return do <= 1.0
    return (do <= 1.0) & (((xx - cx) ** 2 / tirx ** 2 + (yy - cy) ** 2 / tiry ** 2) > 1.0)

def _hollow_rect(x, y, ww, hh, th):
    scale_x, scale_y, scale = _indexed_geometry_scales()
    ww, hh, th = ww * scale_x, hh * scale_y, th * scale
    x0, x1 = max(0, int(x)), min(W, int(x + ww))
    y0, y1 = max(0, int(y)), min(H, int(y + hh))
    xi0, xi1 = int(x0 + th), int(x1 - th)
    yi0, yi1 = int(y0 + th), int(y1 - th)
    full = np.zeros((H, W), dtype=bool)
    full[y0:y1, x0:x1] = True
    if xi0 < xi1 and yi0 < yi1:
        full[yi0:yi1, xi0:xi1] = False
    return full

def _hollow_blob(rng, cx, cy, rx, ry, th=None):
    """Organic outline of constant width in pixels (a proportional ring reads as a donut)."""
    scale_x, scale_y, scale = _indexed_geometry_scales()
    rx, ry = rx * scale_x, ry * scale_y
    yy, xx = _canvas_ogrid()
    ang = np.arctan2(yy - cy, xx - cx)
    na = 24
    a = np.linspace(0, 2 * math.pi, na)
    p = rng.randn(na).astype(np.float32) * 0.3
    p = np.convolve(p, np.ones(4) / 4, mode='same')
    warp = np.interp(ang % (2 * math.pi), a, p, period=2 * math.pi).reshape(H, W)
    wrx, wry = rx * (1 + warp), ry * (1 + warp)
    d = np.sqrt(((xx - cx) ** 2 / wrx ** 2 + (yy - cy) ** 2 / wry ** 2))
    th = (rng.uniform(0.8, 2.4) if th is None else th) * scale
    frac = np.clip(1.0 - th / np.maximum(np.minimum(wrx, wry), 1e-3), 0.05, 0.95)
    return (d <= 1.0) & (d > frac)

# ── color palettes ──

def _mono(rng):
    v = rng.uniform(0.1, 0.9)
    return np.ones(3) * v

def _warm(rng):
    return np.array([rng.uniform(0.6, 1), rng.uniform(0.1, 0.4), rng.uniform(0.0, 0.2)])

def _cool(rng):
    return np.array([rng.uniform(0.0, 0.2), rng.uniform(0.1, 0.5), rng.uniform(0.6, 1)])

def _comp(rng):
    c1 = np.array([rng.uniform(0, 1), rng.uniform(0.3, 0.8), rng.uniform(0.4, 0.9)])
    return c1, 1.0 - c1

def _tri(rng):
    return tuple(np.array([rng.uniform(0, 1), rng.uniform(0.3, 0.8), rng.uniform(0.4, 0.9)]) for _ in range(3))

# ── directional textures ──

def _brushed(rng, angle=None, freq=None):
    """angle/freq in [0,1], driven explicitly by the typed AST (schema.Op.BRUSHED);
    left None for every other (legacy) caller, which keeps the original random draw.
    freq's 1.8-4.2 band is disjoint from `_wood`'s 0.3-1.6 and `_striated`'s 5-14 —
    the three used to share one overlapping range, which is why they were visually
    indistinguishable from each other in the executable-language audit."""
    ang = rng.uniform(0, math.pi) if angle is None else float(angle) * math.pi
    if freq is None:
        broad_f, fine_f = rng.uniform(0.3, 1.5), rng.uniform(3, 14)
    else:
        broad_f = 1.8 + float(freq) * 2.4
        fine_f = broad_f * rng.uniform(2.5, 4.5)
    yy, xx = np.ogrid[:H, :W]
    proj = xx * math.cos(ang) + yy * math.sin(ang)
    fine = np.sin(proj * fine_f + _perlin(rng, 2) * rng.uniform(0.2, 1.2))
    broad = np.sin(proj * broad_f + rng.uniform(0, 6))
    streak = (0.5 + 0.5 * fine) * (0.55 + 0.45 * broad)          # parallel streaks, no isotropic noise
    return (streak - streak.min()) / (streak.max() - streak.min() + 1e-8)

def _wood(rng, angle=None, freq=None):
    """Growth rings stretched along the trunk axis, plus knots — anisotropic by construction.
    angle/freq in [0,1] from the typed AST; None for legacy callers (original random draw).
    freq's 0.3-1.6 band is disjoint from `_brushed`'s 1.8-4.2 and `_striated`'s 5-14."""
    yy, xx = np.ogrid[:H, :W]
    ang = rng.uniform(0, math.pi) if angle is None else float(angle) * math.pi
    ring_f = rng.uniform(0.35, 4.5) if freq is None else 0.3 + float(freq) * 1.3
    ca, sa = math.cos(ang), math.sin(ang)
    cx, cy = rng.uniform(-W, 2 * W), rng.uniform(-H, 2 * H)
    u = (xx - cx) * ca + (yy - cy) * sa
    v = (-(xx - cx) * sa + (yy - cy) * ca) / rng.uniform(2.5, 12)      # rings squashed along the grain
    dist = np.sqrt(u ** 2 + v ** 2)
    grain = np.sin(dist * ring_f + _perlin(rng, 3) * rng.uniform(0.5, 4)) * 0.5 + 0.5
    for _ in range(rng.randint(0, 3)):                                  # knots
        kx, ky = rng.uniform(0, W), rng.uniform(0, H)
        kd = np.sqrt((xx - kx) ** 2 + (yy - ky) ** 2)
        grain = np.maximum(grain, np.exp(-kd / rng.uniform(1, 3)) * np.sin(kd * 2) ** 2)
    return (grain - grain.min()) / (grain.max() - grain.min() + 1e-8)

def _marble(rng):
    """Sharp branching veins over a flat matrix — thin filaments, not smooth mottling."""
    n = _curl_warp(rng, gaussian_filter(_perlin(rng, rng.randint(2, 4)), rng.uniform(0.7, 1.8)),
                   rng.uniform(1, 6))
    n = (n - n.min()) / (n.max() - n.min() + 1e-8)
    v = np.abs(np.sin(n * rng.uniform(3, 11) + rng.uniform(0, 6)))
    veins = (1.0 - v) ** rng.uniform(2, 7)                             # thin bright filaments
    veins = veins + rng.uniform(0.0, 0.4) * (1.0 - np.abs(np.sin(n * rng.uniform(2, 6)))) ** 5
    return (veins - veins.min()) / (veins.max() - veins.min() + 1e-8)

def _striated(rng, angle=None, freq=None):
    """angle/freq in [0,1] from the typed AST; None for legacy callers (original random draw).
    freq's 5-14 band is disjoint from `_wood`'s 0.3-1.6 and `_brushed`'s 1.8-4.2."""
    yy, xx = np.ogrid[:H, :W]
    ang = rng.uniform(0, math.pi) if angle is None else float(angle) * math.pi
    line_f = rng.uniform(0.5, 14) if freq is None else 5.0 + float(freq) * 9.0
    proj = xx * math.cos(ang) + yy * math.sin(ang)
    return np.clip(np.sin(proj * line_f) * 0.5 + 0.5 + _perlin(rng, 3) * rng.uniform(0.05, 0.8), 0, 1)

# ── post-processing ──

def _depth_fog(img, rng):
    yy = np.arange(H, dtype=np.float32) / H
    s = rng.uniform(0.2, 0.8)
    fog = rng.uniform(0.3, 0.8, C)
    for cc in range(C):
        img[:, :, cc] = img[:, :, cc] * (1 - yy.reshape(-1, 1) * s) + fog[cc] * yy.reshape(-1, 1) * s

def _add_grain(img, rng):
    return np.clip(img + rng.randn(H, W, C).astype(np.float32) * rng.uniform(0.01, 0.06), 0, 1)

def _chromatic_ab(img, rng):
    out = img.copy()
    if rng.randint(0, 2) == 0:
        return out
    for ch, (dy, dx) in enumerate([(rng.randint(-1, 2), rng.randint(-1, 2)) for _ in range(2)]):
        out[:, :, ch] = np.roll(img[:, :, ch], dy, axis=0)
        out[:, :, ch] = np.roll(out[:, :, ch], dx, axis=1)
    return out

def _gauss_blur(img, rng):
    sigma = rng.uniform(0.3, 1.5)
    out = img.copy()
    for cc in range(C):
        out[:, :, cc] = gaussian_filter(img[:, :, cc], sigma)
    return out

def _glass(img, rng, mask):
    alpha = rng.uniform(0.3, 0.7)
    col = rng.uniform(0.3, 1.0, C)
    for cc in range(C):
        img[:, :, cc][mask] = (1 - alpha) * img[:, :, cc][mask] + alpha * col[cc]
    cx, cy = rng.randint(2, W - 2), rng.randint(2, H - 2)
    R = rng.uniform(1, 4)
    yy, xx = np.ogrid[:H, :W]
    hm = ((xx - cx) ** 2 + (yy - cy) ** 2) <= R ** 2
    for cc in range(C):
        img[:, :, cc][hm & mask] = np.minimum(1, img[:, :, cc][hm & mask] + 0.3)

def _perspective_grid(rng):
    img = np.zeros((H, W, C), np.float32)
    vp_x = rng.uniform(W * 0.2, W * 0.8)
    vp_y = rng.uniform(H * 0.1, H * 0.35)
    sky = rng.uniform(0.1, 0.5, C)
    gnd = rng.uniform(0.05, 0.35, C)
    for r in range(H):
        img[r, :] = sky * (1 - r / max(H, 1)) + gnd * 0.1 * (r / max(H, 1))
    gnd_y = int(vp_y)
    if gnd_y < H:
        img[gnd_y:, :] = gnd
    for _ in range(rng.randint(4, 12)):
        x0 = rng.uniform(0, W)
        _bresenham(img, int(vp_x), int(vp_y), int(x0), H - 1, 0, rng.uniform(0.2, 0.7, C))
    for i in range(rng.randint(3, 7)):
        y = int(vp_y + (H - vp_y) * (i + 1) / (rng.randint(3, 7) + 1))
        if y < H:
            for x in range(W):
                img[y, x] = rng.uniform(0.3, 0.7, C)
    tex = _perlin(rng, 3) * 0.3
    for cc in range(C):
        img[gnd_y:, :, cc] = np.clip(img[gnd_y:, :, cc] + tex[gnd_y:] * 0.5, 0, 1)
    return img

# ── drawing helpers ──

_RIM = 0.0            # subsurface / rim-light strength
_IRID = 0.0           # iridescence phase (0 = off)
_SUBJECT_GAIN = 1.0   # A recursive base render leaves its own value behind for
                      # post levels. Public access is isolated by _RENDER_LOCK.
_LIGHT_COL = np.ones(3, np.float32)   # light source tint+intensity, see _rand_light_col()
# The legacy renderer mutates the canvas dimensions and lighting state while it
# recursively renders supersampled/post-processing levels.  Keep that exact
# numerical contract, but isolate it from concurrent public calls.  Batch
# parallelism uses processes, so each worker owns an independent renderer state.
_RENDER_LOCK = threading.RLock()

def _draw_3d(img, mask, Nx, Ny, Nz, light, color):
    lit = _phong(Nx, Ny, Nz, light)
    col = np.asarray(color, np.float32) * _SUBJECT_GAIN * _LIGHT_COL
    rim = (1.0 - np.abs(Nz)) ** 3 * _RIM if _RIM else None     # subsurface / backlit edge glow
    for cc in range(C):
        v = col[..., cc] * lit if col.ndim == 3 else col[cc] * lit
        if _IRID:                                              # thin-film interference
            v = v * (0.62 + 0.5 * np.sin(Nz * 5.5 + cc * 2.1 + _IRID))
        if rim is not None:
            v = v + rim * (col[..., cc] if col.ndim == 3 else col[cc])
        img[:, :, cc][mask] = np.clip(v[mask], 0, 1)

def _fill(img, mask, color):
    for cc in range(C):
        img[mask, cc] = color[cc]

def _fill_alpha(img, mask, color, alpha):
    for cc in range(C):
        img[mask, cc] = (1 - alpha) * img[mask, cc] + alpha * color[cc]

def _1f_bg(rng, brightness=1.0):
    fx = np.fft.fftfreq(H).reshape(-1, 1)
    fy = np.fft.fftfreq(W).reshape(1, -1)
    freqs = np.sqrt(fx ** 2 + fy ** 2)
    freqs[0, 0] = 0.5 / H
    alpha_val = rng.uniform(0.7, 1.5)
    spec = 1.0 / (freqs ** alpha_val + 0.01)
    spec /= spec.max()
    bg = np.zeros((H, W, C), np.float32)
    for cc in range(C):
        ph = rng.uniform(0, 2 * math.pi, (H, W))
        F = spec * (np.cos(ph) + 1j * np.sin(ph))
        x = np.fft.ifft2(F).real
        vmin, vmax = x.min(), x.max()
        bg[:, :, cc] = ((x - vmin) / (vmax - vmin + 1e-8)).astype(np.float32) * brightness
    return bg

# ═══════════════════════════════════════════════════════════════
# Creatures (2D + 3D)
# ═══════════════════════════════════════════════════════════════

def _critter_2d(rng, img, ccol, typ):
    bx, by = W // 2 + rng.randint(-9, 10), H // 2 + rng.randint(-8, 9)
    sz = rng.uniform(1.8, 11.0)          # from a distant speck to a subject that overflows
    if typ == 0:  # insect
        ang = rng.uniform(-0.3, 0.3)
        for i in range(3):
            cx = bx - (i - 1) * sz * 1.2 * math.cos(ang)
            cy = by - (i - 1) * sz * 1.2 * math.sin(ang)
            m = _ellipse(cx, cy, sz * (0.45 + 0.1 * i), sz * (0.45 + 0.1 * i))
            _fill(img, m, ccol * rng.uniform(0.7, 1.0))
        for s in [-1, 1]:
            _bresenham(img, int(bx), int(by - sz), int(bx + s * 2), int(by - sz - 3), 0, ccol)
        for i in range(2):
            for s in [-1, 1]:
                _bresenham(img, int(bx), int(by), int(bx + s * 5), int(by + (i - 0.5) * 4), 1, ccol * 0.8)
    elif typ == 1:  # 4-legged
        _fill(img, _ellipse(bx, by, sz * 1.5, sz), ccol)
        _fill(img, _ellipse(bx + sz * 1.3, by - sz * 0.3, sz * 0.7, sz * 0.6), ccol * 1.1)
        for s in [-1, 1]:
            _fill(img, _ellipse(bx + sz * 1.3 + s, by - sz * 0.8, 1.0, 1.5), ccol * 0.7)
        for lx in [bx - sz, bx + sz]:
            for s in [-1, 1]:
                _bresenham(img, int(lx), int(by + sz * 0.5), int(lx), int(by + sz + 2), int(sz * 0.4), ccol * 0.85)
        _bresenham(img, int(bx - sz * 1.2), int(by), int(bx - sz * 1.8), int(by - sz * 0.2), 1, ccol * 0.6)
    elif typ == 2:  # bird
        _fill(img, _ellipse(bx, by, sz, sz * 0.7), ccol)
        _fill(img, _ellipse(bx + sz * 0.8, by - sz * 0.3, sz * 0.5, sz * 0.45), ccol * 1.05)
        _bresenham(img, int(bx + sz * 0.8), int(by - sz * 0.3), int(bx + sz * 1.4), int(by - sz * 0.3), 1, ccol * 0.5)
        for s in [-1, 1]:
            for t in np.linspace(0, 1, 8):
                px = int(bx + (bx + s * sz * 2.5 - bx) * t + rng.uniform(-1, 1))
                py = int(by - sz * 0.3 + (by - sz * 2 - by + sz * 0.3) * t + rng.uniform(-1, 1))
                if 0 <= px < W and 0 <= py < H:
                    img[max(0, py - 1):min(H, py + 2), max(0, px - 1):min(W, px + 2)] = ccol * 0.8
    elif typ == 3:  # fish
        _fill(img, _ellipse(bx, by, sz * 1.8, sz * 0.5), ccol)
        for s in [-1, 1]:
            _bresenham(img, int(bx - sz * 1.5), int(by), int(bx - sz * 2), int(by + s * sz * 1.2), 1, ccol * 0.7)
        _fill(img, _ellipse(bx + sz, by - sz * 0.15, 1.2, 1.0), np.array([0.1, 0.1, 0.1]))
    elif typ == 4:  # spider
        _fill(img, _ellipse(bx, by, sz * 0.8, sz), ccol * 0.5)
        _fill(img, _ellipse(bx, by - sz * 1.1, sz * 0.5, sz * 0.5), ccol * 0.7)
        for i in range(4):
            ang = math.pi / 6 + i * math.pi / 6
            for s in [-1, 1]:
                _bresenham(img, int(bx), int(by - sz * 0.5),
                           int(bx + s * sz * 2.5 * math.cos(ang)),
                           int(by + sz * 2.5 * math.sin(ang)), 0, ccol * 0.6)


def _critter_3d(rng, img, ccol, typ):
    """Creature built in 3D object-space and shot from a random camera angle."""
    L = _rand_light(rng)
    c = np.asarray(ccol, np.float32)
    parts = []
    if typ == 0:      # humanoid
        parts = [('s', (0, -0.95, 0), 0.42, c * 1.05),
                 ('s', (0, -0.35, 0), 0.5, c), ('s', (0, 0.2, 0), 0.45, c * 0.95)]
        for sd in (-1, 1):
            parts.append(('t', (sd * 0.35, -0.55, 0), (sd * 0.75, 0.25, rng.uniform(-0.3, 0.3)), 0.13, 0.75, c * 0.9))
            parts.append(('t', (sd * 0.22, 0.5, 0), (sd * 0.28, 1.35, rng.uniform(-0.2, 0.2)), 0.16, 0.75, c * 0.85))
    elif typ == 1:    # quadruped
        parts = [('s', (0, 0, -0.85), 0.42, c * 1.05), ('s', (0, 0, -0.3), 0.5, c),
                 ('s', (0, 0.02, 0.35), 0.52, c), ('s', (0, 0.05, 0.9), 0.4, c * 0.95)]
        for sd in (-1, 1):
            for z in (-0.35, 0.75):
                parts.append(('t', (sd * 0.28, 0.35, z), (sd * 0.32, 1.15, z + rng.uniform(-0.15, 0.15)),
                              0.12, 0.7, c * 0.85))
        parts.append(('t', (0, -0.1, 1.15), (0, -0.5, 1.7), 0.07, 0.4, c * 0.8))
    elif typ == 2:    # bird
        parts = [('s', (0, 0, 0), 0.5, c), ('s', (0, -0.4, -0.6), 0.3, c * 1.05),
                 ('t', (0, -0.4, -0.75), (0, -0.35, -1.15), 0.07, 0.3, np.array([0.8, 0.6, 0.2]) * c.mean() * 2),
                 ('t', (0, 0.05, 0.4), (0, 0.15, 1.2), 0.12, 0.35, c * 0.85)]
        for sd in (-1, 1):
            parts.append(('t', (sd * 0.3, -0.1, 0), (sd * 1.25, -0.35, rng.uniform(-0.3, 0.3)), 0.14, 0.3, c * 0.9))
    elif typ == 3:    # fish
        parts = [('s', (0, 0, -0.55), 0.4, c * 1.05), ('s', (0, 0, 0), 0.52, c), ('s', (0, 0, 0.6), 0.38, c * 0.95),
                 ('t', (0, 0, 1.0), (0, rng.uniform(-0.5, 0.5), 1.5), 0.22, 0.25, c * 0.85)]
        for sd in (-1, 1):
            parts.append(('t', (sd * 0.25, 0.1, 0), (sd * 0.7, 0.45, 0.25), 0.1, 0.25, c * 0.8))
        parts.append(('t', (0, -0.45, -0.1), (0, -0.85, 0.3), 0.1, 0.25, c * 0.8))
    else:             # abstract blob-creature
        for _ in range(3 + rng.randint(0, 4)):
            parts.append(('s', (rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)),
                          rng.uniform(0.25, 0.65), c * rng.uniform(0.8, 1.15)))
    return _render_parts(img, parts, L, rng, size=rng.uniform(5, 11),
                          anchor=(0, W / 2 + rng.uniform(-5, 5), H / 2 + rng.uniform(-4, 4)))


# ═══════════════════════════════════════════════════════════════
# Flora / Buildings / Elements
# ═══════════════════════════════════════════════════════════════

def _tree(rng, img, ccol, typ):
    bx, by = rng.randint(2, W - 2), rng.randint(6, H + 6)
    ht = rng.uniform(4, 30)          # from a distant sapling to a canopy that overflows the frame
    tw = rng.uniform(1.5, 3.5)
    m_trunk = _ellipse(bx, by + ht * 0.2, tw, ht * 0.5)
    _fill(img, m_trunk, ccol * 0.4)
    if typ == 0:  # round canopy
        for _ in range(rng.randint(2, 5)):
            cx = bx + rng.uniform(-ht * 0.3, ht * 0.3)
            cy = by - ht * 0.3 + rng.uniform(-ht * 0.2, ht * 0.1)
            m = _ellipse(cx, cy, rng.uniform(2.5, ht * 0.5), rng.uniform(2.5, ht * 0.4))
            _fill(img, m, ccol * rng.uniform(0.7, 1.2))
    elif typ == 1:  # pine/conifer
        for i in range(rng.randint(3, 6)):
            y = by - ht * 0.4 - i * ht * 0.12
            w = ht * 0.5 * (1 - i * 0.15)
            m = _ellipse(bx, y, w, ht * 0.12)
            _fill(img, m, ccol * rng.uniform(0.6, 1.0))
    else:  # palm
        _bresenham(img, int(bx), int(by + ht * 0.3), int(bx + rng.randint(-4, 5)), int(by - ht * 0.2), 1, ccol * 0.5)
        _bresenham(img, int(bx), int(by + ht * 0.3), int(bx + rng.randint(-4, 5)), int(by - ht * 0.2), 1, ccol * 0.5)
        _bresenham(img, int(bx), int(by + ht * 0.3), int(bx + rng.randint(-3, 4)), int(by - ht * 0.15), 1, ccol * 0.5)


def _bush(rng, img, ccol):
    cx, cy = rng.randint(1, W - 1), rng.randint(6, H + 4)
    R = rng.uniform(2, 16)
    for _ in range(rng.randint(3, 7)):
        xx = rng.uniform(cx - R * 0.6, cx + R * 0.6)
        yy = rng.uniform(cy - R * 0.4, cy + R * 0.3)
        m = _ellipse(xx, yy, rng.uniform(1.2, R * 0.4), rng.uniform(1.2, R * 0.35))
        _fill(img, m, ccol * rng.uniform(0.7, 1.1))


def _flower(rng, img, ccol):
    cx, cy = rng.randint(5, W - 5), rng.randint(5, H - 5)
    R = rng.uniform(2, 5)
    _bresenham(img, int(cx), int(cy), int(cx), int(cy + R * 1.2), 1, ccol * 0.3)
    n_petals = rng.randint(4, 8)
    for i in range(n_petals):
        ang = 2 * math.pi * i / n_petals
        px, py = cx + math.cos(ang) * R * 0.5, cy + math.sin(ang) * R * 0.5
        m = _ellipse(px, py, R * 0.25, R * 0.35)
        _fill(img, m, ccol * rng.uniform(0.8, 1.3))
    _fill(img, _ellipse(cx, cy, R * 0.2, R * 0.2), rng.uniform(0.6, 1.0, C) * 0.3)


def _building(rng, img, ccol, typ):
    """Axonometric solid with real faces — was a stack of ellipses."""
    L = _rand_light(rng)
    c = np.asarray(ccol, np.float32)
    bw = rng.uniform(5, 14)
    bh = rng.uniform(6, 24) if typ == 1 else rng.uniform(5, 15)
    x0 = rng.uniform(0, max(1.0, W - bw))
    y1 = rng.uniform(H * 0.5, H - 1)
    y0 = max(0.0, y1 - bh)
    dx = rng.uniform(1.5, 5) * (1 if rng.randint(0, 2) else -1)
    dy = -rng.uniform(0.8, 3.5)
    front, top, side = _box3d(img, x0, y0, x0 + bw, y1, dx, dy, c, L)
    if typ == 0:      # facade with lit windows
        lit = np.clip(c * rng.uniform(1.5, 3.0) + rng.uniform(0.1, 0.5), 0, 1)
        gx, gy = rng.randint(2, 5), rng.randint(3, 6)
        for wy in range(int(y0) + 2, int(y1) - 1, gy):
            for wx in range(int(x0) + 1, int(x0 + bw) - 1, gx):
                wm = _rect(wx, wy, wx + max(1, gx - 2), wy + max(1, gy - 2)) & front
                _fill(img, wm, np.clip(lit * rng.uniform(0.2, 1.0), 0, 1))
    elif typ == 1:    # tower + spire
        _tube(img, x0 + bw / 2, y0, x0 + bw / 2 + dx * 0.5, max(0.0, y0 - rng.uniform(3, 8)),
              rng.uniform(0.5, 1.2), np.clip(c * 0.8, 0, 1), L, taper=0.2)
    else:             # pitched roof
        roof = np.clip(c * rng.uniform(0.4, 0.8), 0, 1)
        rh = max(2, int(bh * 0.35))
        lit = _phong(np.zeros((H, W), np.float32), -np.ones((H, W), np.float32),
                     np.zeros((H, W), np.float32), L)
        for i in range(rh):
            wr = bw * 0.5 * (1 - i / rh)
            m = _rect(x0 + bw / 2 - wr, y0 - i - 1, x0 + bw / 2 + wr, y0 - i)
            for ch in range(C):
                img[:, :, ch][m] = np.clip(roof[ch] * lit[m], 0, 1)


def _water(rng, img):
    sky = np.array([rng.uniform(0.0, 0.2), rng.uniform(0.0, 0.3), rng.uniform(0.2, 0.6)])
    water_col = np.array([rng.uniform(0.0, 0.15), rng.uniform(0.1, 0.4), rng.uniform(0.5, 0.9)])
    hz = rng.randint(H // 3, 2 * H // 3)
    for r in range(hz):
        img[r, :] = sky * (1 - r / max(hz, 1)) + water_col * 0.2
    tex = _perlin(rng, 3)
    for r in range(hz, H):
        wave = 0.5 + 0.5 * np.sin(r * rng.uniform(0.3, 0.7) + tex[r] * 4)
        for cc in range(C):
            img[r, :, cc] = water_col[cc] + wave * 0.2 * sky[cc]
    if hz > 2:
        for cc in range(C):
            img[hz:, :, cc] += tex[hz:] * rng.uniform(0.1, 0.25)


def _fire(rng, img, zoom=1.0, base_y=None):
    """Turbulent 2D flame: perlin-warped cone, core→mid→fringe layers, additive emissive.
    zoom < 1 = distant speck; zoom > 1 = close-up filling frame."""
    z = zoom
    bx = rng.uniform(5, W - 5)
    if base_y is not None:
        by = base_y
    else:
        by = rng.uniform(H * 0.48, H - 3)
    ht = rng.uniform(14, 26) * z
    wd = rng.uniform(4, 9) * z
    tex = _perlin(rng, 3)
    # dark fuel base (log / ember bed) — blend in-place
    a = _soft_disc(bx, by + 1.5, rng.uniform(2.0, 4.5) * z)
    ba = a * rng.uniform(0.5, 0.75)
    bc = np.array([0.06, 0.03, 0.01])
    for cc in range(C):
        img[:, :, cc] = np.clip(img[:, :, cc] * (1 - ba) + ba * bc[cc], 0, 1)
    # 3 emissive layers: core (white-hot), mid (orange), fringe (red)
    for (c_lo, c_hi, wf, a_scale) in [
        (np.array([1.0, 1.0, 0.85]),  np.array([1.0, 0.92, 0.25]), 0.55, 1.0),
        (np.array([1.0, 0.78, 0.10]), np.array([1.0, 0.30, 0.0]),  0.88, 0.90),
        (np.array([1.0, 0.35, 0.02]), np.array([0.65, 0.04, 0.0]), 1.15, 0.70),
    ]:
        for i in range(int(ht)):
            y = int(by - i)
            if y < 0:
                break
            t = i / max(ht, 1)
            w = wd * wf * max(0.0, 1.0 - t * 0.75)
            n_pts = max(3, int(w * 6))
            xs = np.linspace(bx - w, bx + w, n_pts)
            valid = (np.trunc(xs).astype(np.intp) >= 0) & (
                np.trunc(xs).astype(np.intp) < W
            )
            if not valid.any():
                continue
            xs = xs[valid]
            noise = _tex_at(tex, xs, y)
            offsets = (noise - 0.5) * wd * 0.5
            target_x = np.clip(xs + offsets, 0, W - 1).astype(np.intp)
            distance = np.abs(xs - bx) / max(w, 0.1)
            colors = (
                c_lo.reshape(1, C) * (1 - distance).reshape(-1, 1)
                + c_hi.reshape(1, C) * distance.reshape(-1, 1)
            )
            alpha = (
                a_scale
                * (1.0 - t ** 1.3)
                * np.maximum(0.0, 1.0 - distance ** 0.55)
            )
            np.add.at(img[y], target_x, colors * alpha.reshape(-1, 1))
            np.clip(img[y], 0, 1, out=img[y])
    # embers
    for _ in range(rng.randint(6, 16)):
        ex = bx + rng.uniform(-wd * 0.5, wd * 0.5)
        ey = by - ht * rng.uniform(0.6, 1.3)
        if 0 <= ex < W and 0 <= ey < H:
            a = _soft_disc(ex, ey, rng.uniform(0.1, 0.55))
            ecol = np.array([1.0, rng.uniform(0.35, 0.8), rng.uniform(0.0, 0.06)])
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] + a * rng.uniform(0.5, 1.0) * ecol[cc], 0, 1)
    # thin smoke
    if ht > 16:
        for _ in range(rng.randint(0, 3)):
            sx = bx + rng.uniform(-wd * 0.25, wd * 0.25)
            sy = by - ht * rng.uniform(0.82, 1.08)
            a = _soft_disc(sx, sy, rng.uniform(0.8, 2.5) * z)
            sa = a * rng.uniform(0.04, 0.14)
            sc = np.array([0.10, 0.09, 0.08])
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] * (1 - sa) + sa * sc[cc], 0, 1)


def _fire_3d(rng, img, L, zoom=1.0, base_y=None):
    """Realistic 3D flame: emissive tubes (no phong — fire is light source).
    Inner core (white-thick) + outer envelope (orange-red thin) + embers + smoke.
    zoom < 1 = distant; zoom > 1 = close-up."""
    z = zoom
    bx = rng.uniform(5, W - 5)
    if base_y is not None:
        by = base_y
    else:
        by = rng.uniform(H * 0.45, H - 3)
    ht = rng.uniform(14, 26) * z
    wd_factor = z
    tex = _perlin(rng, 3)
    # fuel base glow (dark, blends in-place)
    a = _soft_disc(bx, by + 1.5, rng.uniform(2.2, 4.8) * z)
    ba = a * rng.uniform(0.55, 0.8)
    bc = np.array([0.05, 0.02, 0.0])
    for cc in range(C):
        img[:, :, cc] = np.clip(img[:, :, cc] * (1 - ba) + ba * bc[cc], 0, 1)
    # ---- inner core: thick white-hot strands ----
    n_core = rng.randint(5, 12)
    for _ in range(n_core):
        x, y = bx + rng.uniform(-1.5 * z, 1.5 * z), by
        ang = -math.pi / 2 + rng.uniform(-0.35, 0.35)
        n_seg = rng.randint(4, 7)
        seg_len = ht / n_seg * rng.uniform(0.7, 1.2)
        r0 = rng.uniform(1.2, 2.5) * z
        curl = rng.choice([-1, 1]) * rng.uniform(0.04, 0.18)
        for s in range(n_seg):
            t = s / max(n_seg - 1, 1)
            noise = _tex_at(tex, x, y)
            ang += curl + (noise - 0.5) * 0.25 + rng.uniform(-0.04, 0.04)
            x1 = x + math.cos(ang) * seg_len
            y1 = y + math.sin(ang) * seg_len
            r = r0 * max(0.1, 1 - t * 0.88)
            col = np.array([1.0, 0.94 + 0.06 * (1 - t), 0.7 + 0.3 * (1 - t)])
            _fire_tube(img, x, y, x1, y1, max(0.4 * z, r), col, taper=0.25 + 0.75 * (1 - t))
            x, y = x1, y1
    # ---- outer envelope: thinner, more turbulent, orange→red ----
    n_env = rng.randint(8, 22)
    for _ in range(n_env):
        x, y = bx + rng.uniform(-1.8 * z, 1.8 * z), by
        ang = -math.pi / 2 + rng.uniform(-0.55, 0.55)
        n_seg = rng.randint(3, 6)
        seg_len = ht / n_seg * rng.uniform(0.8, 1.4)
        r0 = rng.uniform(0.4, 1.3) * z
        curl = rng.choice([-1, 1]) * rng.uniform(0.08, 0.35)
        phase = rng.uniform(0, 6)
        for s in range(n_seg):
            t = s / max(n_seg - 1, 1)
            noise = _tex_at(tex, x, y)
            ang += curl + math.sin(s * 0.7 + phase) * 0.18 + (noise - 0.5) * 0.45
            x1 = x + math.cos(ang) * seg_len
            y1 = y + math.sin(ang) * seg_len
            r = r0 * max(0.08, 1 - t * 0.92)
            col = np.array([1.0, 0.5 + 0.4 * (1 - t), 0.02 + 0.2 * (1 - t)])
            _fire_tube(img, x, y, x1, y1, max(0.18 * z, r), col, taper=0.15 + 0.85 * (1 - t))
            x, y = x1, y1
    # ---- embers: additive bright specks ----
    for _ in range(rng.randint(6, 18)):
        ex = bx + rng.uniform(-ht * 0.28, ht * 0.28)
        ey = by - ht * rng.uniform(0.55, 1.25)
        if 0 <= ex < W and 0 <= ey < H:
            a = _soft_disc(ex, ey, rng.uniform(0.1, 0.45) * z)
            ecol = np.array([1.0, rng.uniform(0.3, 0.8), rng.uniform(0.0, 0.06)])
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] + a * rng.uniform(0.5, 1.0) * ecol[cc], 0, 1)
    # ---- thin smoke wisps at the top ----
    if ht > 16:
        for _ in range(rng.randint(1, 4)):
            sx = bx + rng.uniform(-ht * 0.1, ht * 0.1)
            sy = by - ht * rng.uniform(0.85, 1.12)
            a = _soft_disc(sx, sy, rng.uniform(0.7, 2.5) * z)
            sa = a * rng.uniform(0.03, 0.12)
            sc = np.array([0.10, 0.08, 0.07])
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] * (1 - sa) + sa * sc[cc], 0, 1)


def _fire_bg(rng, img):
    """Vary fire backgrounds: night forest, indoor fireplace, campfire dusk, desert, plain dark.
    ponytail: 5 styles is enough for variety at 32px — more would be indistinguishable."""
    t = rng.randint(0, 6)
    if t == 0:    # night forest — dark teal/brown low-light
        sky = np.array([rng.uniform(0.0, 0.05), rng.uniform(0.01, 0.08), rng.uniform(0.04, 0.12)])
        gnd = np.array([rng.uniform(0.03, 0.1), rng.uniform(0.04, 0.12), rng.uniform(0.01, 0.04)])
        _sky_horizon(img, sky, gnd, rng)
        # tree silhouettes
        for _ in range(rng.randint(0, 3)):
            tx = rng.uniform(0, W)
            th = rng.uniform(6, H)
            _bresenham(img, int(tx), H - 1, int(tx + rng.uniform(-4, 4)), max(0, H - int(th)), int(rng.uniform(1, 2)),
                       np.array([0.02, 0.02, 0.01]))
    elif t == 1:  # indoor fireplace — warm brick tones, dark walls
        for y in range(H):
            t_y = y / max(H - 1, 1)
            img[y, :] = np.array([0.10, 0.04, 0.02]) * (1 - t_y) + np.array([0.04, 0.02, 0.01]) * t_y
        tex = _perlin(rng, 2)
        for cc in range(C):
            img[:, :, cc] += tex * rng.uniform(0.0, 0.05)
    elif t == 2:  # campfire at dusk — purple-orange horizon
        sky = np.array([rng.uniform(0.05, 0.15), rng.uniform(0.02, 0.08), rng.uniform(0.08, 0.2)])
        gnd = np.array([rng.uniform(0.1, 0.25), rng.uniform(0.05, 0.15), rng.uniform(0.02, 0.08)])
        _sky_horizon(img, sky, gnd, rng)
    elif t == 3:  # desert night — warm sand + dark sky
        sky = np.array([rng.uniform(0.01, 0.04), rng.uniform(0.01, 0.05), rng.uniform(0.06, 0.15)])
        gnd = np.array([rng.uniform(0.08, 0.18), rng.uniform(0.06, 0.14), rng.uniform(0.03, 0.08)])
        _sky_horizon(img, sky, gnd, rng)
    elif t == 4:  # plain dark — minimal ambient (original style)
        bg_tex = _perlin(rng, 2)
        for y in range(H):
            t_y = y / max(H - 1, 1)
            img[y, :] = np.array([0.02, 0.01, 0.005]) * (1 - t_y) + np.array([0.06, 0.02, 0.01]) * t_y
            img[y] += bg_tex[y].reshape(W, 1) * 0.03 * t_y
    else:  # brick/stone wall — vertical lines with noise
        wall_col = np.array([rng.uniform(0.06, 0.14), rng.uniform(0.03, 0.08), rng.uniform(0.02, 0.05)])
        tex = _perlin(rng, 2)
        img[:, :] = wall_col
        img[:, :, :] += tex.reshape(H, W, 1) * rng.uniform(0.02, 0.08)
        # horizontal mortar lines
        for _ in range(rng.randint(1, 3)):
            hy = rng.randint(4, H - 4)
            img[hy:hy + 1, :] += rng.uniform(0.01, 0.04)


def _sky_horizon(img, sky, gnd, rng):
    """Gradient sky→ground with a randomised horizon line."""
    hz = rng.randint(H // 3, 2 * H // 3)
    for r in range(H):
        t = min(1.0, max(0.0, (r - hz) / max(H - hz, 1)))
        img[r, :] = sky * (1 - t) + gnd * t


def _gas_cloud(rng, img):
    n_blobs = rng.randint(4, 10)
    ccol = rng.uniform(0.4, 0.9, C)
    for _ in range(n_blobs):
        cx, cy = rng.uniform(2, W - 2), rng.uniform(2, H - 2)
        R = rng.uniform(3, 9)
        tex = _perlin(rng, 3)
        alpha = rng.uniform(0.2, 0.6)
        yy, xx = np.ogrid[:H, :W]
        dd = ((xx - cx) ** 2 + (yy - cy) ** 2) / R ** 2
        m = dd <= 1.0
        g = 1.0 - dd / (dd[m].max() + 1e-8)
        for cc in range(C):
            img[:, :, cc][m] = np.clip(
                (1 - alpha) * img[:, :, cc][m] + alpha * ccol[cc] * g[m] * tex[m], 0, 1
            )

# ═══════════════════════════════════════════════════════════════
# Photo-realism helpers (levels 80-99)
# ═══════════════════════════════════════════════════════════════

def _rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = True
    return m

@contextmanager
def _canvas_size(width, height):
    """Temporarily select one native canvas while the render lock is held."""
    global H, W
    old_height, old_width = H, W
    H, W = int(height), int(width)
    try:
        yield
    finally:
        H, W = old_height, old_width


@contextmanager
def _indexed_geometry_mode(enabled):
    """Scale canonical primitive sizes only for non-default indexed renders."""
    global _SCALE_INDEXED_GEOMETRY
    previous = _SCALE_INDEXED_GEOMETRY
    _SCALE_INDEXED_GEOMETRY = bool(enabled)
    try:
        yield
    finally:
        _SCALE_INDEXED_GEOMETRY = previous


def _render_at(width, height, idx, lvl, step=7):
    """Render one level on a width×height canvas (temporary global swap).

    Public render entry points hold ``_RENDER_LOCK`` while this legacy
    module-global H/W swap is active.
    """
    with _canvas_size(width, height):
        return _render_image(idx, step, force_level=lvl)

def _base_render(rng, size=None):
    """A random content level, optionally rendered at higher resolution."""
    pool = FRAME_POOL if size else RICH_POOL
    lvl = pool[rng.randint(0, len(pool))]
    idx = _level_start(lvl, rng) + rng.randint(0, SAMPLES_PER_LEVEL)
    if size is None:
        return _render_image(idx, 7, force_level=lvl)
    width, height = size
    return _render_at(width, height, idx, lvl)

def _box_down(img, f):
    source_h, source_w = img.shape[:2]
    h, w = math.ceil(source_h / f), math.ceil(source_w / f)
    pad_h, pad_w = h * f - source_h, w * f - source_w
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return img.reshape(h, f, w, f, C).mean(axis=(1, 3))

def _motion_blur(img, rng):
    ang = rng.uniform(0, math.pi)
    n = rng.randint(3, 7)
    out = np.zeros_like(img)
    for t in range(n):
        d = t - n // 2
        out += np.roll(np.roll(img, int(round(d * math.sin(ang))), axis=0),
                       int(round(d * math.cos(ang))), axis=1)
    return out / n

def _natural_color(img, rng, strength=None):
    """Real photos have highly correlated RGB + moderate saturation + white balance.

    strength (None for legacy callers = the original random magnitude) directly sets
    how far this pulls toward desaturated + white-balanced. Previously `sat`/`wb` were
    an independent rng draw and executor._apply_filter blended the result again by a
    *separate* strength — two independent dials on the same "how much" question could
    both land low and cancel the effect almost entirely.
    """
    lum = img.mean(axis=-1, keepdims=True)
    if strength is None:
        sat = rng.uniform(0.2, 0.75)
        wb = np.array([rng.uniform(0.82, 1.18), rng.uniform(0.95, 1.05), rng.uniform(0.82, 1.18)])
    else:
        s = float(strength)
        sat = 0.85 - 0.7 * s                      # s=1 -> strongly desaturated
        sign = 1.0 if rng.randint(0, 2) else -1.0
        shift = 0.05 + 0.35 * s
        wb = np.array([1.0 + sign * shift, 1.0, 1.0 - sign * shift * 0.3])
    return np.clip((lum * (1 - sat) + img * sat) * wb, 0, 1)

def _tone_curve(img, rng, strength=None):
    """Exposure / gamma / S-curve / crushed blacks / blown highlights.

    strength (None for legacy callers = original random magnitude) sets how far
    exposure/gamma/contrast sit from neutral (1.0), instead of an independent rng
    draw that executor._apply_filter's separate blend could dilute right back down.
    """
    if strength is None:
        exposure, gamma, contrast = rng.uniform(0.55, 1.7), rng.uniform(0.55, 1.9), rng.uniform(0.6, 1.7)
    else:
        s = float(strength)
        exposure = float(np.clip(1.0 + rng.uniform(-1, 1) * 0.7 * s, 0.3, 1.7))
        gamma = float(np.clip(1.0 + rng.uniform(-1, 1) * 0.7 * s, 0.3, 1.9))
        contrast = 1.0 + rng.uniform(0, 1) * 0.7 * s
    out = np.clip(img * exposure, 0, 1) ** gamma
    piv = rng.uniform(0.3, 0.65)
    out = np.clip(piv + (out - piv) * contrast, 0, 1)
    return out

def _vignette(img, rng, strength=None):
    s = rng.uniform(0.2, 0.7) if strength is None else strength
    yy, xx = _canvas_ogrid()
    d = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (0.5 * math.hypot(H, W))
    return np.clip(img * (1 - s * d ** 2).reshape(H, W, 1), 0, 1)

def _soft_disc(cx, cy, R):
    """Antialiased disc alpha — the only way to get sub-pixel edges at 32px."""
    R *= _indexed_geometry_scales()[2]
    yy, xx = _canvas_ogrid()
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(R - d + 0.5, 0, 1).astype(np.float32)

def _blend(img, alpha, color):
    a = alpha.reshape(H, W, 1)
    color_array = np.asarray(color)
    if color_array.ndim == 1:
        return np.clip(img * (1 - a) + color_array.reshape(1, 1, C) * a, 0, 1)
    return np.clip(img * (1 - a) + color_array * a, 0, 1)

def _photo_color(rng, lo=0.05, hi=0.95):
    """A plausible photographic color: channels correlated around a luminance, rarely saturated."""
    lum = rng.uniform(lo, hi)
    tint = rng.randn(3).astype(np.float32) * rng.uniform(0.02, 0.3)
    return np.clip(lum + tint - tint.mean(), 0.02, 1.0)

def _backdrop(rng, hi_key=False):
    """Background field. Real photos have no pure-black canvas — and often a white one."""
    n = _perlin(rng, rng.randint(2, 5)).reshape(H, W, 1)
    if hi_key:      # studio white, snow, blown-out sky: bright and smooth
        base, amp = _photo_color(rng, 0.62, 1.0), rng.uniform(0.02, 0.25)
    else:
        base, amp = _photo_color(rng, 0.04, 0.55), rng.uniform(0.25, 1.0)
    return np.clip(base.reshape(1, 1, C) * (1 - amp + amp * n), 0, 1)

def _sky_gradient(rng, img, hz=None, sun=True):
    """The sky types photos actually contain — overcast and milky are the common ones."""
    t = rng.randint(0, 5)
    if t == 0:      # clear blue
        top = np.array([rng.uniform(0.08, 0.3), rng.uniform(0.28, 0.5), rng.uniform(0.55, 0.95)])
        bot = np.array([rng.uniform(0.45, 0.8), rng.uniform(0.55, 0.85), rng.uniform(0.75, 1.0)])
    elif t == 1:    # overcast grey
        g = rng.uniform(0.3, 0.72)
        top = np.ones(3) * g * rng.uniform(0.8, 0.95) + np.array([0, 0, 0.04])
        bot = np.ones(3) * min(1.0, g * rng.uniform(1.05, 1.3))
    elif t == 2:    # milky white haze
        top = np.ones(3) * rng.uniform(0.65, 0.92)
        bot = np.ones(3) * rng.uniform(0.85, 1.0)
    elif t == 3:    # sunset / golden hour
        top = np.array([rng.uniform(0.25, 0.5), rng.uniform(0.15, 0.35), rng.uniform(0.3, 0.6)])
        bot = np.array([rng.uniform(0.8, 1.0), rng.uniform(0.35, 0.7), rng.uniform(0.12, 0.4)])
    else:           # dusk / night
        top = np.array([rng.uniform(0.02, 0.12), rng.uniform(0.03, 0.15), rng.uniform(0.08, 0.3)])
        bot = np.array([rng.uniform(0.08, 0.3), rng.uniform(0.1, 0.3), rng.uniform(0.18, 0.45)])
    hz = H if hz is None else hz
    for r in range(H):
        img[r, :] = top + (bot - top) * min(1.0, r / max(hz, 1))
    if sun and rng.randint(0, 2) == 0:
        sx, sy = rng.uniform(0, W), rng.uniform(0, hz * 0.7)
        yy, xx = np.ogrid[:H, :W]
        glow = np.exp(-np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / rng.uniform(3, 10))
        warm = np.array([1.0, rng.uniform(0.75, 1.0), rng.uniform(0.5, 1.0)])
        img[:] = np.clip(img + glow.reshape(H, W, 1) * warm.reshape(1, 1, C) * rng.uniform(0.3, 0.9), 0, 1)
    return img

def _contact_shadow(img, cx, cy, r, rng):
    a = _soft_disc(cx, cy, r) * rng.uniform(0.3, 0.7)
    a = a * np.clip((np.arange(H).reshape(H, 1) - cy) / max(r, 1) + 1.0, 0, 1)  # only below
    return np.clip(img * (1 - a.reshape(H, W, 1) * 0.8), 0, 1)

def _branch(img, rng, x, y, ang, length, thick, color, depth):
    """Recursive self-similar structure: branches, cracks, lightning, deltas."""
    if depth <= 0 or length < 1.2:
        return
    x1 = x + math.cos(ang) * length
    y1 = y + math.sin(ang) * length
    _bresenham(img, int(x), int(y), int(x1), int(y1), int(thick), color)
    for _ in range(rng.randint(2, 4)):
        _branch(img, rng, x1, y1, ang + rng.uniform(-0.9, 0.9),
                length * rng.uniform(0.5, 0.8), max(0, thick - 1),
                color * rng.uniform(0.7, 1.05), depth - 1)

def _tube(img, x0, y0, x1, y1, r, color, L, taper=1.0):
    """Shaded cylinder between two points — the 3D counterpart of a 2D stroke."""
    scale_x, scale_y, scale = _indexed_geometry_scales()
    if scale_x != 1.0 or scale_y != 1.0:
        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        x0, x1 = mx + (x0 - mx) * scale_x, mx + (x1 - mx) * scale_x
        y0, y1 = my + (y0 - my) * scale_y, my + (y1 - my) * scale_y
        r *= scale
    yy, xx = _canvas_ogrid()
    dx, dy = x1 - x0, y1 - y0
    ln = math.sqrt(dx * dx + dy * dy) + 1e-8
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (ln * ln), 0, 1)
    px, py = x0 + t * dx, y0 + t * dy
    rr = r * (1 - t * (1 - taper)) + 1e-8
    d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    m = d <= rr
    nx, ny = -dy / ln, dx / ln                       # in-plane perpendicular
    s = np.where(m, np.clip(((xx - px) * nx + (yy - py) * ny) / rr, -1, 1), 0)
    _draw_3d(img, m, s * nx, s * ny, np.sqrt(np.maximum(0, 1 - s ** 2)), L, color)
    return m

def _fire_tube(img, x0, y0, x1, y1, r, color, taper=1.0):
    """Emissive tube — additive, no phong (fire IS the light source)."""
    scale_x, scale_y, scale = _indexed_geometry_scales()
    if scale_x != 1.0 or scale_y != 1.0:
        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        x0, x1 = mx + (x0 - mx) * scale_x, mx + (x1 - mx) * scale_x
        y0, y1 = my + (y0 - my) * scale_y, my + (y1 - my) * scale_y
        r *= scale
    yy, xx = _canvas_ogrid()
    dx, dy = x1 - x0, y1 - y0
    ln = math.sqrt(dx * dx + dy * dy) + 1e-8
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (ln * ln), 0, 1)
    px, py = x0 + t * dx, y0 + t * dy
    rr = r * (1 - t * (1 - taper)) + 1e-8
    d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    alpha = np.clip(1.0 - d / rr, 0, 1).reshape(H, W, 1)
    alpha = alpha ** 1.8                                    # softer falloff toward edges
    c = np.asarray(color).reshape(1, 1, C)
    return np.clip(img + alpha * c, 0, 1, out=img)

def _para(x0, x1, y, dx, dy):
    """Mask of the parallelogram swept from the horizontal segment (x0..x1, y) along (dx, dy)."""
    m = np.zeros((H, W), bool)
    n = int(max(abs(dx), abs(dy))) * 2 + 2
    for i in range(n):
        t = i / (n - 1)
        r = int(round(y + t * dy))
        if 0 <= r < H:
            m[r, max(0, int(round(x0 + t * dx))):max(0, int(round(x1 + t * dx)))] = True
    return m

def _para_v(y0, y1, x, dx, dy):
    """Same, sweeping a vertical segment (x, y0..y1)."""
    m = np.zeros((H, W), bool)
    n = int(max(abs(dx), abs(dy))) * 2 + 2
    for i in range(n):
        t = i / (n - 1)
        c = int(round(x + t * dx))
        if 0 <= c < W:
            m[max(0, int(round(y0 + t * dy))):max(0, int(round(y1 + t * dy))), c] = True
    return m

def _box3d(img, x0, y0, x1, y1, dx, dy, color, L, faces=True):
    """Axonometric solid: front face + cap + side, each with its own constant normal.

    dx/dy sign = where the camera is: right/left of the object, above/below it.
    """
    front = _rect(x0, y0, x1, y1)
    top = _para(x0, x1, y0 if dy < 0 else y1, dx, dy)          # roof seen from above, floor from below
    side = _para_v(y0, y1, x1 if dx > 0 else x0, dx, dy)
    if faces:
        for m, n in ((top, (0, -1 if dy < 0 else 1, 0)), (side, (1 if dx > 0 else -1, 0, 0))):
            lit = _phong(np.full((H, W), n[0], np.float32), np.full((H, W), n[1], np.float32),
                         np.full((H, W), n[2], np.float32), L)
            for ch in range(C):
                surface = color[..., ch][m] if np.asarray(color).ndim == 3 else color[ch]
                img[:, :, ch][m] = np.clip(surface * _LIGHT_COL[ch] * lit[m], 0, 1)
    lit = _phong(np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), np.ones((H, W), np.float32), L)
    for ch in range(C):
        surface = color[..., ch][front] if np.asarray(color).ndim == 3 else color[ch]
        img[:, :, ch][front] = np.clip(surface * _LIGHT_COL[ch] * lit[front], 0, 1)
    return front, top, side

def _ground(rng, img, hz, gcol=None, hi_key=False):
    """Sky + textured ground plane. Returns the horizon color (for atmospheric fade)."""
    if hi_key:                      # subject on a plain bright field: studio, snow, overcast
        img[:] = _backdrop(rng, True)
        return img[max(hz - 1, 0)].mean(axis=0)
    _sky_gradient(rng, img, hz=max(hz, 1), sun=rng.randint(0, 2) == 0)
    g = _photo_color(rng, 0.05, 0.9) if gcol is None else gcol   # includes snow / sand / pavement
    if hz < H:
        img[hz:] = np.clip(g.reshape(1, 1, C) * (0.7 + 0.6 * _perlin(rng, 4)[hz:].reshape(-1, W, 1)), 0, 1)
    return img[min(hz, H - 1)].mean(axis=0)

def _rot3(pts, ay, ax):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ca, sa = math.cos(ay), math.sin(ay)
    x, z = x * ca - z * sa, x * sa + z * ca
    cb, sb = math.cos(ax), math.sin(ax)
    y, z = y * cb - z * sb, y * sb + z * cb
    return np.stack([x, y, z], axis=-1)

def _project(pts, dist, fov):
    """Perspective projection to image coords + per-point scale."""
    s = 1.0 / (pts[:, 2] + dist)
    return W / 2 + pts[:, 0] * s * fov, H / 2 + pts[:, 1] * s * fov, s * dist

def _render_parts(img, parts, L, rng, yaw=None, pitch=None, dist=None, size=None, anchor=None,
                  allow_closeup=True):
    """Draw an object defined in 3D object-space from an arbitrary camera angle.

    parts: ('s', (x,y,z), r, color)                          → sphere
           ('t', (x0,y0,z0), (x1,y1,z1), r, taper, color)    → cylinder
    Spheres and cylinders project to spheres and cylinders, so a random yaw/pitch is
    all it takes to see the object from any side. Painter's algorithm on rotated depth.
    Returns the mask of each part (index-aligned with `parts`).
    """
    yaw = rng.uniform(0, 2 * math.pi) if yaw is None else yaw
    pitch = rng.uniform(-0.55, 0.55) if pitch is None else pitch
    dist = rng.uniform(3.0, 5.0) if dist is None else dist
    size = rng.uniform(7, 15) if size is None else size
    if allow_closeup and rng.randint(0, 4) == 0:  # optional close-up for legacy indexed renders
        size *= rng.uniform(1.8, 3.2)
    fov = size * dist                                  # size = pixels per unit at z=0
    pts, first = [], []
    for p in parts:
        first.append(len(pts))
        pts.append(p[1])
        if p[0] == 't':
            pts.append(p[2])
    P = _rot3(np.asarray(pts, np.float32), yaw, pitch)
    s = 1.0 / np.maximum(P[:, 2] + dist, 0.3)
    u = W / 2 + P[:, 0] * s * fov
    v = H / 2 + P[:, 1] * s * fov
    if anchor is not None:
        k, ax, ay = anchor
        u += ax - u[first[k]]
        v += ay - v[first[k]]
    masks = [None] * len(parts)
    depth = [P[first[k], 2] if parts[k][0] == 's' else (P[first[k], 2] + P[first[k] + 1, 2]) / 2
             for k in range(len(parts))]
    for k in sorted(range(len(parts)), key=lambda i: -depth[i]):
        p, i0 = parts[k], first[k]
        if p[0] == 's':
            R = max(0.7, p[2] * s[i0] * fov)
            m, Nx, Ny, Nz = _sphere(u[i0], v[i0], R)
            _draw_3d(img, m, Nx, Ny, Nz, L, np.clip(p[3], 0, 1))
        else:
            r = max(0.45, p[3] * (s[i0] + s[i0 + 1]) / 2 * fov)
            m = _tube(img, u[i0], v[i0], u[i0 + 1], v[i0 + 1], r, np.clip(p[5], 0, 1), L, taper=p[4])
        masks[k] = m
    if rng.randint(0, 3) == 0:              # spotted / patched coat (frogs, cats, horses)
        union = np.zeros((H, W), bool)
        for m in masks:
            union |= m
        spots = union & (_perlin(rng, rng.randint(3, 5)) > rng.uniform(0.5, 0.72))
        k = rng.uniform(0.3, 0.75) if rng.randint(0, 2) else rng.uniform(1.3, 2.2)
        for ch in range(C):
            img[:, :, ch][spots] = np.clip(img[:, :, ch][spots] * k, 0, 1)
    return masks

OBJECT_KINDS = ('tree', 'conifer', 'palm', 'bush', 'flower', 'plane', 'insect', 'spider', 'blob', 'human')

def _object_parts(rng, kind):
    """Parts of one object in 3D object-space, roughly 1 unit tall, base near y=0.

    Shared by the dedicated levels and by the coherent-scene level, so objects of
    different categories can share a frame, a light and a ground plane.
    """
    if kind == 'tree':
        bark = np.array([rng.uniform(0.2, 0.5), rng.uniform(0.12, 0.35), rng.uniform(0.05, 0.2)])
        leaf = np.array([rng.uniform(0.05, 0.3), rng.uniform(0.3, 0.85), rng.uniform(0.05, 0.3)])
        parts = [('t', (0, 0, 0), (rng.uniform(-0.2, 0.2), -1.0, rng.uniform(-0.2, 0.2)),
                  rng.uniform(0.07, 0.14), 0.6, bark)]
        for _ in range(rng.randint(3, 7)):
            parts.append(('s', (rng.uniform(-0.5, 0.5), -1.15 + rng.uniform(-0.3, 0.25), rng.uniform(-0.5, 0.5)),
                          rng.uniform(0.35, 0.6), leaf * rng.uniform(0.75, 1.25)))
        return parts
    if kind == 'conifer':
        bark = np.array([rng.uniform(0.2, 0.5), rng.uniform(0.12, 0.32), rng.uniform(0.05, 0.2)])
        leaf = np.array([rng.uniform(0.03, 0.25), rng.uniform(0.25, 0.8), rng.uniform(0.05, 0.3)])
        parts = [('t', (0, 0, 0), (0, -1.1, 0), rng.uniform(0.06, 0.12), 0.7, bark)]
        n = rng.randint(4, 7)
        for i in range(n):
            parts.append(('s', (0, -0.45 - 0.75 * i / n, 0), 0.55 * (1 - 0.55 * i / n), leaf * (0.7 + 0.3 * i / n)))
        return parts
    if kind == 'palm':
        bark = np.array([rng.uniform(0.2, 0.5), rng.uniform(0.12, 0.32), rng.uniform(0.05, 0.2)])
        leaf = np.array([rng.uniform(0.03, 0.25), rng.uniform(0.25, 0.8), rng.uniform(0.05, 0.3)])
        mid = (rng.uniform(-0.2, 0.2), -0.6, rng.uniform(-0.2, 0.2))
        top = (mid[0] + rng.uniform(-0.25, 0.25), -1.15, mid[2] + rng.uniform(-0.25, 0.25))
        parts = [('t', (0, 0, 0), mid, rng.uniform(0.06, 0.11), 0.85, bark),
                 ('t', mid, top, rng.uniform(0.05, 0.09), 0.8, bark)]
        nf = rng.randint(4, 8)
        for k in range(nf):
            a = 2 * math.pi * k / nf + rng.uniform(-0.3, 0.3)
            fl = rng.uniform(0.5, 0.85)
            parts.append(('t', top, (top[0] + math.cos(a) * fl, top[1] + rng.uniform(0.05, 0.4),
                                     top[2] + math.sin(a) * fl), rng.uniform(0.07, 0.13), 0.2,
                          leaf * rng.uniform(0.7, 1.2)))
        return parts
    if kind == 'bush':
        leaf = np.array([rng.uniform(0.05, 0.3), rng.uniform(0.3, 0.85), rng.uniform(0.05, 0.3)])
        return [('s', (rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.1), rng.uniform(-0.7, 0.7)),
                 rng.uniform(0.35, 0.7), leaf * rng.uniform(0.7, 1.25)) for _ in range(rng.randint(4, 9))]
    if kind == 'flower':
        stem = np.array([rng.uniform(0.05, 0.25), rng.uniform(0.3, 0.7), rng.uniform(0.05, 0.2)])
        pet = _photo_color(rng, 0.3, 1.0)
        parts = [('t', (0, 0, 0), (rng.uniform(-0.3, 0.3), -1.0, rng.uniform(-0.3, 0.3)),
                  rng.uniform(0.04, 0.08), 0.7, stem)]
        head = parts[0][2]
        n = rng.randint(5, 9)
        for k in range(n):
            a = 2 * math.pi * k / n
            parts.append(('s', (head[0] + math.cos(a) * 0.42, head[1], head[2] + math.sin(a) * 0.42),
                          0.24, pet * rng.uniform(0.8, 1.2)))
        parts.append(('s', head, 0.18, _photo_color(rng, 0.4, 1.0) * np.array([1.0, 0.9, 0.3])))
        return parts
    if kind == 'plane':
        col = np.ones(3) * rng.uniform(0.5, 1.0)
        wl, sweep = rng.uniform(0.9, 1.4), rng.uniform(0.0, 0.45)
        parts = [('t', (0, 0, -1.0), (0, 0, 1.0), rng.uniform(0.1, 0.16), 0.5, col)]
        for sd in (-1, 1):
            parts.append(('t', (0, 0, 0.1), (sd * wl, rng.uniform(-0.1, 0.1), 0.1 - sweep),
                          rng.uniform(0.07, 0.12), 0.35, col * 0.85))
            parts.append(('t', (0, 0, -0.85), (sd * 0.4, 0, -1.0), 0.06, 0.4, col * 0.75))
        parts.append(('t', (0, 0, -0.85), (0, -0.45, -1.05), 0.05, 0.45, col * 0.7))
        return parts
    if kind == 'insect':
        col = _photo_color(rng, 0.15, 0.9)
        parts = []
        for i in range(3):
            z = 0.35 - i * 0.35
            for sd in (-1, 1):
                knee = (sd * 0.75, 0.15, z + rng.uniform(-0.15, 0.15))
                parts.append(('t', (sd * 0.2, 0.05, z), knee, 0.055, 0.7, col * 0.7))
                parts.append(('t', knee, (sd * 1.1, 0.75, z + rng.uniform(-0.2, 0.2)), 0.045, 0.6, col * 0.6))
        for z, r in ((-0.75, 0.3), (-0.25, 0.38), (0.5, 0.45)):
            parts.append(('s', (0, 0, z), r * rng.uniform(0.9, 1.15), col * rng.uniform(0.8, 1.15)))
        for sd in (-1, 1):
            parts.append(('t', (sd * 0.12, -0.1, -0.9), (sd * 0.5, -0.55, -1.5), 0.04, 0.3, col * 0.6))
        return parts
    if kind == 'spider':
        col = _photo_color(rng, 0.1, 0.8)
        parts = []
        for k in range(4):
            a = 0.35 + k * 0.55
            for sd in (-1, 1):
                knee = (sd * math.sin(a) * 0.85, -0.35, math.cos(a) * 0.85)
                parts.append(('t', (0, 0, 0), knee, 0.05, 0.7, col * 0.75))
                parts.append(('t', knee, (sd * math.sin(a) * 1.5, 0.55, math.cos(a) * 1.5), 0.04, 0.5, col * 0.6))
        parts.append(('s', (0, -0.05, 0.45), rng.uniform(0.5, 0.7), col))
        parts.append(('s', (0, 0, -0.35), rng.uniform(0.3, 0.42), col * 1.15))
        return parts
    if kind == 'human':
        skin = _photo_color(rng, 0.15, 0.9)
        parts = [
            ('s', (0, -0.85, 0), 0.16, skin * 1.05),                          # head
            ('s', (0, -0.55, 0), 0.19, skin),                                 # chest
            ('s', (0, -0.25, 0), 0.17, skin * 0.95),                          # waist
        ]
        for sd in (-1, 1):
            parts.append(('t', (sd * 0.15, -0.62, 0), (sd * 0.32, -0.2, rng.uniform(-0.1, 0.1)),
                          0.055, 0.8, skin * 0.9))                            # arm
            parts.append(('t', (sd * 0.09, -0.1, 0), (sd * 0.12, 0.55, rng.uniform(-0.05, 0.05)),
                          0.07, 0.75, skin * 0.85))                           # leg
        return parts
    col = _photo_color(rng, 0.15, 0.95)      # 'blob': generic rounded object
    return [('s', (rng.uniform(-0.5, 0.5), rng.uniform(-0.6, 0.1), rng.uniform(-0.5, 0.5)),
             rng.uniform(0.35, 0.75), col * rng.uniform(0.85, 1.15)) for _ in range(rng.randint(1, 4))]


# ── imaging regimes beyond "opaque solid lit from the front" ──

def _blackbody(T):
    """Approximate sRGB of a Planck emitter at T kelvin. Stars, lava, hot metal, filaments."""
    t = np.clip(T, 1000.0, 15000.0) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.47 * math.log(t) - 161.12
    else:
        r = 329.7 * (t - 60) ** -0.1332
        g = 288.12 * (t - 60) ** -0.0755
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.52 * math.log(t - 10) - 305.04
    return np.clip(np.array([r, g, b], np.float32) / 255.0, 0, 1)

def _point_source(img, x, y, flux, color, core=0.7, halo=3.0, spikes=False):
    """A point of light as an optical system really renders it: core + halo (+ diffraction spikes)."""
    scale = _indexed_geometry_scales()[2]
    core, halo = core * scale, halo * scale
    yy, xx = np.ogrid[:H, :W]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    psf = np.exp(-d2 / (2 * core ** 2)) + 0.18 * np.exp(-d2 / (2 * halo ** 2))
    if spikes:
        sp = (np.exp(-((xx - x) ** 2) / (2 * 0.45 ** 2)) * np.exp(-np.abs(yy - y) / halo) +
              np.exp(-((yy - y) ** 2) / (2 * 0.45 ** 2)) * np.exp(-np.abs(xx - x) / halo))
        psf = psf + 0.35 * sp
    img += psf.reshape(H, W, 1) * np.asarray(color, np.float32).reshape(1, 1, C) * flux
    return img

def _poisson(img, rng, photons=None):
    """Shot noise: in the low-photon regime the noise IS the signal's square root."""
    ph = rng.uniform(6, 140) if photons is None else photons
    return np.clip(rng.poisson(np.clip(img, 0, 1) * ph).astype(np.float32) / ph, 0, 1)

def _reaction_diffusion_initial(rng):
    """Prepare one deterministic Gray-Scott state without evolving it."""
    height, width = H, W                    # periodic boundary: no crop needed
    # only presets that actually develop within a few hundred steps (measured, dt must stay at 1.0)
    f, k = [(0.029, 0.057), (0.055, 0.062), (0.039, 0.058),
            (0.026, 0.051), (0.037, 0.060), (0.042, 0.059)][rng.randint(0, 6)]
    U = np.ones((height, width), np.float32)
    # seed from structured noise instead of a few dots: mature-looking patterns in far fewer steps
    seed = np.repeat(
        np.repeat(
            rng.rand(height // 4 + 1, width // 4 + 1).astype(np.float32),
            4, axis=0,
        ),
        4, axis=1,
    )[:height, :width]
    V = (seed > rng.uniform(0.55, 0.75)).astype(np.float32) * 0.25
    U -= V * 2
    V += rng.rand(height, width).astype(np.float32) * 0.02
    steps = int(rng.randint(200, 340))
    return U, V, float(f), float(k), steps


def _reaction_diffusion_evolve(U, V, f, k, steps):
    """Evolve a prepared state on CPU using the byte-exact legacy order."""
    height, width = U.shape
    Du, Dv, dt = 0.16, 0.08, 1.0            # dt>1 blows the pattern away, verified empirically
    previous_y = np.r_[height - 1, np.arange(height - 1)]
    following_y = np.r_[np.arange(1, height), 0]
    previous_x = np.r_[width - 1, np.arange(width - 1)]
    following_x = np.r_[np.arange(1, width), 0]
    for _ in range(steps):
        # Direct periodic indexing preserves the legacy addition order while
        # avoiding four general-purpose ``np.roll`` calls per field and step.
        lu = (
            U[previous_y, :] + U[following_y, :]
            + U[:, previous_x] + U[:, following_x] - 4 * U
        )
        lv = (
            V[previous_y, :] + V[following_y, :]
            + V[:, previous_x] + V[:, following_x] - 4 * V
        )
        uvv = U * V * V
        U += dt * (Du * lu - uvv + f * (1 - U))
        V += dt * (Dv * lv + uvv - (f + k) * V)
    return (V - V.min()) / (V.max() - V.min() + 1e-8)


def _reaction_diffusion(rng):
    """Gray-Scott: spots, stripes, coral, mazes — how living surfaces actually pattern themselves."""
    return _reaction_diffusion_evolve(*_reaction_diffusion_initial(rng))

def _curl_warp(rng, field, strength=None):
    """Advect a field along a divergence-free flow: filaments, swirls, turbulence.

    The same primitive covers milk in coffee, smoke, hurricanes and spiral galaxies.
    """
    strength = rng.uniform(2, 7) if strength is None else strength
    pot = gaussian_filter(_perlin(rng, 3), rng.uniform(0.8, 2.0))
    gy, gx = np.gradient(pot)
    vx, vy = -gy, gx                                   # curl of a scalar potential
    n = np.sqrt(vx ** 2 + vy ** 2).max() + 1e-8
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    out = field
    for _ in range(rng.randint(1, 4)):
        out = map_coordinates(out, [yy + vy / n * strength, xx + vx / n * strength],
                              order=1, mode='reflect')
    return out

def _radial_warp(img, cx, cy, fn):
    """Warp the image radially about a center: droplet, fisheye, glass ball, gravitational lens."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    rn = fn(r)
    sx, sy = cx + dx / r * rn, cy + dy / r * rn
    out = np.empty_like(img)
    for ch in range(C):
        out[:, :, ch] = map_coordinates(img[:, :, ch], [sy, sx], order=1, mode='reflect')
    return out

def _lut(scalar, stops):
    """Map a [0,1] scalar field through control colors — thermal, depth and other false color."""
    x = np.clip(scalar, 0, 1)
    pos = np.linspace(0, 1, len(stops))
    out = np.empty((H, W, C), np.float32)
    for ch in range(C):
        out[:, :, ch] = np.interp(x, pos, [c[ch] for c in stops])
    return out

def _speckle(rng):
    """Coherent illumination: random phase through an aperture, intensity = |field|^2."""
    ap = rng.uniform(0.08, 0.3)
    fy = np.fft.fftfreq(H).reshape(-1, 1)
    fx = np.fft.fftfreq(W).reshape(1, -1)
    mask = (np.sqrt(fx ** 2 + fy ** 2) < ap)
    ph = rng.uniform(0, 2 * math.pi, (H, W))
    F = mask * (np.cos(ph) + 1j * np.sin(ph))
    a = np.abs(np.fft.ifft2(F)) ** 2
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def _closed_curve(rng, n=80, dim=3, harmonics=4, wobble=0.5):
    """A closed, smooth, irregular loop. Fourier series in the parameter, so it closes exactly
    and has no corners; the first harmonic stays dominant so it does not self-intersect.

    Unlike _organic_blob (a radial r(theta)) this can have necks and re-entrant lobes.
    """
    t = np.linspace(0, 2 * math.pi, n, endpoint=False).astype(np.float32)
    pts = np.zeros((n, 3), np.float32)
    pts[:, 0], pts[:, 1] = np.cos(t), np.sin(t)
    for ax in range(dim):
        for k in range(2, harmonics + 2):
            amp = wobble ** (k - 1) * rng.uniform(0.3, 1.0)
            pts[:, ax] += amp * np.cos(k * t + rng.uniform(0, 2 * math.pi))
    pts[:, :2] -= pts[:, :2].mean(0)
    r = np.sqrt((pts[:, :2] ** 2).sum(1)).max() + 1e-6
    return pts / r


def V3(x, y, z):
    return np.array([x, y, z], np.float32)


def _skeleton(rng, n=None, dim=2, humanoid=False):
    """A connected stick figure: chain with branches. The reach goes from a single blob
    to a limbed body, because the number of segments is what sets the number of lobes."""
    if humanoid:                       # proof that a limbed body is inside the support
        return [(np.array([0, .55, 0], np.float32), np.array([0, -.15, 0], np.float32), .17),
                (np.array([0, -.2, 0], np.float32), np.array([0, -.5, 0], np.float32), .13),
                (np.array([0, -.05, 0], np.float32), np.array([-.5, .2, 0], np.float32), .07),
                (np.array([0, -.05, 0], np.float32), np.array([.5, .2, 0], np.float32), .07),
                (np.array([-.1, .5, 0], np.float32), np.array([-.2, 1.0, 0], np.float32), .08),
                (np.array([.1, .5, 0], np.float32), np.array([.2, 1.0, 0], np.float32), .08)]
    n = rng.randint(1, 13) if n is None else n
    branch = rng.uniform(0.15, 0.65)      # low = worm, high = limbed body
    pts = [np.zeros(3, np.float32)]
    segs = []
    for i in range(n):
        a_ = pts[rng.randint(0, len(pts))] if (i and rng.rand() < branch) else pts[-1]
        d = rng.randn(3).astype(np.float32)
        if dim == 2:
            d[2] = 0.0
        d /= np.linalg.norm(d) + 1e-8
        b_ = a_ + d * rng.uniform(0.25, 0.95)
        segs.append((a_, b_, rng.uniform(0.06, 0.34)))
        pts.append(b_)
    if not humanoid and rng.randint(0, 2) == 0:      # sprout a fan of thin tips (fingers, tentacles)
        base = pts[-1]
        nf = rng.randint(3, 7)
        spread = rng.uniform(0.4, 1.1)
        along = pts[-1] - pts[max(0, len(pts) - 2)]
        along = along / (np.linalg.norm(along) + 1e-8)
        perp = np.cross(along, V3(0, 0, 1) if dim == 3 else V3(0, 0, 1))
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        rt = rng.uniform(0.03, 0.08)
        for k in range(nf):
            off = (k - (nf - 1) / 2) / max(nf - 1, 1)
            tip = base + along * rng.uniform(0.35, 0.7) + perp * off * spread
            if dim == 3:
                tip = tip + V3(0, 0, 1) * rng.uniform(-0.2, 0.2)
            segs.append((base + perp * off * spread * 0.4, tip, rt))
    return segs


def _meta_field(segs, scale, cx, cy, smooth):
    """Smooth union of capsules. Its zero level set is closed, corner-free and hole-free by
    construction, and unlike a radial r(theta) it can have necks, limbs and concavities."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    acc = None
    for a_, b_, r in segs:
        x0, y0 = cx + a_[0] * scale, cy + a_[1] * scale
        x1, y1 = cx + b_[0] * scale, cy + b_[1] * scale
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy + 1e-8
        u = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L2, 0, 1)
        zr = 1.0 + 0.35 * (a_[2] + b_[2]) / 2          # z thickens what is nearer
        d = np.sqrt((xx - x0 - u * dx) ** 2 + (yy - y0 - u * dy) ** 2) - r * scale * max(zr, 0.3)
        if acc is None:
            acc = d
        else:
            m = np.minimum(acc, d)                       # stable soft-min (logsumexp)
            acc = m - np.log(np.exp((m - acc) * smooth) + np.exp((m - d) * smooth)) / smooth
    return acc


def _curve_dist(pts):
    """Distance from every pixel to the closed polyline (for constant-width strokes)."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    d = np.full((H, W), 1e9, np.float32)
    q = np.roll(pts, -1, axis=0)
    for i in range(len(pts)):
        x0, y0 = pts[i, 0], pts[i, 1]
        dx, dy = q[i, 0] - x0, q[i, 1] - y0
        L2 = dx * dx + dy * dy + 1e-8
        u = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L2, 0, 1)
        d = np.minimum(d, np.sqrt((xx - x0 - u * dx) ** 2 + (yy - y0 - u * dy) ** 2))
    return d


def _habitat(rng, img, hi_key=False):
    """Somewhere for a creature to be. A subject always floating on the same field is redundant."""
    m = rng.randint(0, 6)
    if m == 0:
        _ground(rng, img, rng.randint(4, 22), None, hi_key)
    elif m == 1:
        img[:] = gaussian_filter(_1f_bg(rng, rng.uniform(0.3, 0.9)), sigma=(1.5, 1.5, 0))   # bokeh
    elif m == 2:
        img[:] = _backdrop(rng, True)                                                        # bright field
    elif m == 3:
        _water(rng, img)
    elif m == 4:
        v = _perlin(rng, rng.randint(3, 5)).reshape(H, W, 1)                                 # foliage / litter
        img[:] = np.clip(_photo_color(rng, 0.05, 0.6).reshape(1, 1, C) * (0.4 + 1.2 * v), 0, 1)
    else:
        _sky_gradient(rng, img)
    return img


# A compact 5x7 bitmap font of REAL known symbols (digits, latin, math, punctuation,
# arrows). Embedded as data so it is deterministic and font-free. Rows are 5-bit ints.
_FONT_ART = {
 "0":"01100 10010 10110 11010 10010 10010 01100","1":"00100 01100 00100 00100 00100 00100 01110",
 "2":"01100 10010 00010 00100 01000 10000 11110","3":"11110 00010 00100 00110 00010 10010 01100",
 "4":"00010 00110 01010 10010 11111 00010 00010","5":"11111 10000 11100 00010 00010 10010 01100",
 "6":"00110 01000 10000 11100 10010 10010 01100","7":"11111 00010 00100 00100 01000 01000 01000",
 "8":"01100 10010 10010 01100 10010 10010 01100","9":"01100 10010 10010 01110 00010 00100 01100",
 "A":"01100 10010 10010 11110 10010 10010 10010","B":"11100 10010 10010 11100 10010 10010 11100",
 "C":"01110 10000 10000 10000 10000 10000 01110","D":"11100 10010 10010 10010 10010 10010 11100",
 "E":"11110 10000 10000 11100 10000 10000 11110","F":"11110 10000 10000 11100 10000 10000 10000",
 "G":"01110 10000 10000 10110 10010 10010 01110","H":"10010 10010 10010 11110 10010 10010 10010",
 "I":"01110 00100 00100 00100 00100 00100 01110","J":"00110 00010 00010 00010 10010 10010 01100",
 "K":"10010 10100 11000 10000 11000 10100 10010","L":"10000 10000 10000 10000 10000 10000 11110",
 "M":"10001 11011 10101 10101 10001 10001 10001","N":"10010 11010 11010 10110 10110 10010 10010",
 "O":"01100 10010 10010 10010 10010 10010 01100","P":"11100 10010 10010 11100 10000 10000 10000",
 "Q":"01100 10010 10010 10010 10110 01100 00011","R":"11100 10010 10010 11100 11000 10100 10010",
 "S":"01110 10000 10000 01100 00010 00010 11100","T":"11111 00100 00100 00100 00100 00100 00100",
 "U":"10010 10010 10010 10010 10010 10010 01100","V":"10001 10001 10001 01010 01010 00100 00100",
 "W":"10001 10001 10001 10101 10101 11011 10001","X":"10001 01010 00100 00100 00100 01010 10001",
 "Y":"10001 01010 00100 00100 00100 00100 00100","Z":"11111 00010 00100 01000 10000 10000 11111",
 "+":"00000 00100 00100 11111 00100 00100 00000","-":"00000 00000 00000 11111 00000 00000 00000",
 "=":"00000 00000 11111 00000 11111 00000 00000","/":"00001 00010 00100 00100 01000 10000 10000",
 "*":"00000 10101 01110 11111 01110 10101 00000","#":"01010 11111 01010 01010 11111 01010 00000",
 "<":"00010 00100 01000 10000 01000 00100 00010",">":"01000 00100 00010 00001 00010 00100 01000",
 "^":"00100 01010 10001 00000 00000 00000 00000","@":"01110 10001 10111 10101 10111 10000 01110",
 "?":"01100 10010 00010 00100 00100 00000 00100","!":"00100 00100 00100 00100 00100 00000 00100",
 ".":"00000 00000 00000 00000 00000 01100 01100",",":"00000 00000 00000 00000 01100 00100 01000",
 ":":"00000 01100 01100 00000 01100 01100 00000",";":"00000 01100 01100 00000 01100 00100 01000",
 "(":"00010 00100 01000 01000 01000 00100 00010",")":"01000 00100 00010 00010 00010 00100 01000",
 "[":"01110 01000 01000 01000 01000 01000 01110","]":"01110 00010 00010 00010 00010 00010 01110",
 "%":"11001 11010 00100 01000 10000 01011 10011","&":"01100 10010 10100 01000 10101 10010 01101",
 "~":"00000 00000 01000 10101 00010 00000 00000",'"'"'":"01100 01100 01000 00000 00000 00000 00000",
 "gG":"11111 10000 10000 10000 10000 10000 10000","gD":"00100 00100 01010 01010 10001 10001 11111",
 "gTh":"01110 10001 10001 11111 10001 10001 01110","gL":"00100 00100 01010 01010 10001 10001 10001",
 "gX":"11111 00000 00000 01110 00000 00000 11111","gP":"11111 10001 10001 10001 10001 10001 10001",
 "gS":"11111 01000 00100 00010 00100 01000 11111","gF":"00100 01110 10101 10101 10101 01110 00100",
 "gPs":"10101 10101 10101 01110 00100 00100 00100","gW":"01110 10001 10001 10001 01010 01010 11011",
 "rF":"10000 10110 11100 10110 10010 10000 10000","rTh":"10000 11000 10100 10010 10100 11000 10000",
 "rA":"10001 10010 10100 11000 10100 10010 10001","rR":"11000 10100 11000 10100 10010 10001 10000",
 "rK":"00001 00010 00100 01000 00100 00010 00001","rH":"10001 10001 11111 10001 10001 10001 10001",
 "rN":"10000 10001 10010 10100 01000 10000 10000","rM":"10001 11011 10101 10001 10001 10001 10001",
}
_FONT = {c: np.array([[int(b) for b in row] for row in art.split()], np.uint8)
         for c, art in _FONT_ART.items()}
_FONT_KEYS = list(_FONT)


def _stamp_glyph(ch, cell):
    """Return a `cell`x`cell` float mask of a real glyph, nearest-scaled from the 5x7 bitmap."""
    g = _FONT[ch]
    yi = (np.arange(cell) * 7 // cell)
    xi = (np.arange(cell) * 5 // cell)
    return g[yi][:, xi].astype(np.float32)


def _glyph(rng, gx, gy, cell, kind):


    """One synthetic character on a `cell`-px grid at (gx,gy). No fonts: strokes are drawn
    procedurally to imitate the STATISTICS of a script — stroke width, junctions, density —
    not any real character. Kinds: 0 alpha/latin, 1 CJK-like, 2 cuneiform, 3 digits/7-seg, 4 runic.
    """
    m = np.zeros((H, W), bool)
    yy, xx = np.ogrid[:H, :W]
    def seg(x0, y0, x1, y1, t):
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy + 1e-6
        u = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L2, 0, 1)
        return ((xx - x0 - u * dx) ** 2 + (yy - y0 - u * dy) ** 2) <= t * t
    def dot(x, y, r):
        return (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    pad = cell * 0.16
    x0, y0 = gx + pad, gy + pad
    w, h = cell - 2 * pad, cell - 2 * pad
    t = max(0.6, cell * rng.uniform(0.06, 0.13))
    def gp(cx, cy):                                    # grid point on a 3x3 lattice
        return x0 + cx * w / 2, y0 + cy * h / 2
    if kind == 1:      # CJK-like: dense box with internal strokes, high stroke count
        for a, b in [((0, 0), (2, 0)), ((0, 0), (0, 2)), ((2, 0), (2, 2)), ((0, 2), (2, 2))]:
            if rng.rand() < 0.85:
                m |= seg(*gp(*a), *gp(*b), t)
        for _ in range(rng.randint(2, 5)):
            r = rng.randint(0, 3)
            if rng.rand() < 0.5:
                m |= seg(*gp(0, r), *gp(2, r), t)      # horizontal internal stroke
            else:
                c = rng.randint(0, 3)
                m |= seg(*gp(c, 0), *gp(c, 2), t)      # vertical internal stroke
    elif kind == 2:    # cuneiform: wedge marks (a thick short seg + a tick)
        for _ in range(rng.randint(3, 7)):
            cx, cy = rng.uniform(x0, x0 + w), rng.uniform(y0, y0 + h)
            ang = rng.choice([0, math.pi / 2, math.pi / 4, -math.pi / 4])
            ln = cell * rng.uniform(0.15, 0.35)
            ex, ey = cx + math.cos(ang) * ln, cy + math.sin(ang) * ln
            m |= seg(cx, cy, ex, ey, t * 1.4)
            m |= dot(cx, cy, t * 1.6)                  # the wedge head
    elif kind == 3:    # digits / seven-segment
        segs7 = [((0, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 1), (2, 2)), ((0, 2), (2, 2)),
                 ((0, 1), (0, 2)), ((0, 0), (0, 1)), ((0, 1), (2, 1))]
        on = rng.rand(7) < rng.uniform(0.45, 0.8)
        on[0] = on[0] or rng.rand() < 0.5
        for k, (a, b) in enumerate(segs7):
            if on[k]:
                m |= seg(*gp(*a), *gp(*b), t)
    elif kind == 4:    # runic: a vertical stave with angled twigs
        m |= seg(*gp(1, 0), *gp(1, 2), t)
        for _ in range(rng.randint(2, 5)):
            r = rng.uniform(0, 2)
            s2 = rng.choice([-1, 1])
            m |= seg(x0 + w / 2, y0 + r * h / 2, x0 + w / 2 + s2 * w / 2,
                     y0 + (r + rng.choice([-0.6, 0.6])) * h / 2, t)
    else:              # latin/alpha: 2-4 strokes between random lattice nodes + optional curve
        nodes = [(rng.randint(0, 3), rng.randint(0, 3)) for _ in range(rng.randint(3, 5))]
        for i in range(len(nodes) - 1):
            m |= seg(*gp(*nodes[i]), *gp(*nodes[i + 1]), t)
        if rng.rand() < 0.4:
            m |= dot(*gp(rng.randint(0, 3), rng.randint(0, 3)), t * 1.3)
    return m


def _jpeg_artifacts(img, rng, strength=None):
    """8×8 block DCT quantization + chroma subsampling — the artifacts real datasets carry.

    strength (None for legacy callers = original random magnitude) sets the
    quantization step directly, instead of an independent rng draw that
    executor._apply_filter's separate blend could dilute right back down.
    """
    q = rng.uniform(0.04, 0.22) if strength is None else 0.05 + float(strength) * 0.35
    nb = H // 8
    out = img.copy()
    # chroma subsample: average 2×2 on the two chroma-ish channels
    lum = out.mean(axis=-1, keepdims=True)
    chroma = out - lum
    chroma = np.repeat(np.repeat(_box_down(chroma + 0.5, 2) - 0.5, 2, axis=0), 2, axis=1)
    out = np.clip(lum + chroma[:H, :W], 0, 1)
    if nb >= 1:
        blocks = out[:nb * 8, :nb * 8].reshape(nb, 8, nb, 8, C)
        co = dctn(blocks, axes=(1, 3), norm='ortho')
        step_q = q * (1 + np.arange(8).reshape(1, 8, 1, 1, 1) + np.arange(8).reshape(1, 1, 1, 8, 1))
        co = np.round(co / step_q) * step_q
        out[:nb * 8, :nb * 8] = idctn(co, axes=(1, 3), norm='ortho').reshape(nb * 8, nb * 8, C)
    return np.clip(out, 0, 1)


# ═══════════════════════════════════════════════════════════════
# Main generator
# ═══════════════════════════════════════════════════════════════

def _render_image(idx: int, step: int = 7, force_level: int | None = None) -> np.ndarray:
    """Generate a deterministic 32×32×3 image from an integer index.

    Args:
        idx:  Integer seed — same idx always returns the same image.
        step: Skip between consecutive hash-indices (default 7).
        force_level: If set, use this level instead of deriving from idx.

    Returns:
        float32 numpy array of shape (32, 32, 3), values in [0, 1].
    """
    img = np.zeros((H, W, C), np.float32)
    block = _level_block(idx)
    lvl = _level_of(idx)
    lidx = idx % SAMPLES_PER_LEVEL
    rng2 = _rng(lidx * max(1, step) + block * 100000)   # block, not level: duplicates differ

    if force_level is not None:
        lvl = force_level
    if lvl in WAVE_LEVELS:
        # Plane-wave family: no rng at all, the image is a closed-form function of a
        # Sobol-sampled θ that also renders the audio and video for this index.
        return render_wave_image(wave_theta(idx), *WAVE_LEVELS[lvl], h=H, w=W)

    # ── 72-79: STYLE FILTERS ──
    if 72 <= lvl <= 79:
        style_idx = lvl - 72
        base_lvl = CONTENT_POOL[(lidx * 3 + lvl * 7) % len(CONTENT_POOL)]
        base_idx = _level_start(base_lvl, rng2) + rng2.randint(0, SAMPLES_PER_LEVEL)
        # Generate base image by recursing (safe: base_lvl < 72, so no infinite loop)
        base_img = _render_image(base_idx, step, force_level=base_lvl)
        img = _apply_style(base_img, style_idx, rng2)
        return np.clip(img, 0, 1).astype(np.float32)

    # ── 129-130, 133, 135-136: RECURSIVE FILTERS (symmetry / diffraction / refraction over a
    # random other level) ──
    if lvl in RECURSIVE_LEVELS:
        base_lvl = rng2.randint(0, N_LEVELS)
        while base_lvl in RECURSIVE_LEVELS:          # never recurse into another recursive filter
            base_lvl = rng2.randint(0, N_LEVELS)
        base_idx = _level_start(base_lvl, rng2) + rng2.randint(0, SAMPLES_PER_LEVEL)
        base_img = _render_image(base_idx, step, force_level=base_lvl)
        if lvl == 129:
            img = _mirror_fold(base_img, rng2.uniform(0, math.pi))
        elif lvl == 130:
            img = _kaleidoscope(base_img, rng2.randint(4, 11), rng2.uniform(0, 2 * math.pi))
        elif lvl == 133:
            img = _diffraction_2d(base_img, rng2.uniform(0, math.pi), rng2.uniform(1.5, 4.0),
                                  rng2.uniform(0.8, 2.2))
        elif lvl == 135:
            strength = rng2.uniform(1.5, 4.0)
            dxf = (_perlin(rng2, rng2.randint(3, 5)) - 0.5) * strength
            dyf = (_perlin(rng2, rng2.randint(3, 5)) - 0.5) * strength
            img = _refraction_2d(base_img, dxf, dyf)
        else:  # 136
            L = _rand_light(rng2)
            cx, cy = rng2.uniform(7, W - 7), rng2.uniform(6, H - 6)
            R = rng2.uniform(5, 11)
            mag = rng2.uniform(1.6, 3.0) * (-1 if rng2.rand() < 0.4 else 1)  # sometimes inverted: water droplet
            img = _refraction_3d(base_img, cx, cy, R, mag, L)
        return np.clip(img, 0, 1).astype(np.float32)

    global _SUBJECT_GAIN, _RIM, _IRID, _LIGHT_COL
    hi_key = rng2.randint(0, 4) == 0                 # bright background, like studio / snow / sky
    if lvl not in POST_LEVELS and rng2.randint(0, 8) > 0:
        img += _backdrop(rng2, hi_key)               # never start from pure black
    # a dark subject against a light background is most of what CIFAR contains
    _SUBJECT_GAIN = rng2.uniform(0.1, 0.5) if (hi_key and rng2.randint(0, 2) == 0) else 1.0
    _RIM = rng2.uniform(0.15, 0.7) if rng2.randint(0, 5) == 0 else 0.0     # skin, wax, leaves, clouds
    _IRID = rng2.uniform(0.1, 6.2) if rng2.randint(0, 12) == 0 else 0.0    # soap film, beetle, oil
    _LIGHT_COL = _rand_light_col(rng2)               # golden hour, tungsten, overcast, neon, harsh/dim

    # ── 0-3: 1D ──
    if lvl == 0:
        for _ in range(1 + rng2.randint(0, 4)):
            px, py = rng2.randint(0, W), rng2.randint(0, H)
            img[py, px] = rng2.uniform(0.3, 1.0, C)
    elif lvl == 1:
        for _ in range(3 + rng2.randint(0, 10)):
            px, py = rng2.randint(0, W), rng2.randint(0, H)
            sz = rng2.randint(0, 2)
            cc = rng2.uniform(0.2, 1.0, C)
            for dy in range(-sz, sz + 1):
                for dx in range(-sz, sz + 1):
                    if 0 <= px + dx < W and 0 <= py + dy < H:
                        img[py + dy, px + dx] = cc
    elif lvl == 2:
        for _ in range(1 + rng2.randint(0, 3)):
            cc = rng2.uniform(0.2, 1.0, C)
            _bresenham(img, rng2.randint(0, W), rng2.randint(0, H),
                       rng2.randint(0, W), rng2.randint(0, H), rng2.randint(0, 2), cc)
    elif lvl == 3:
        pts = [(rng2.randint(0, W), rng2.randint(0, H)) for _ in range(3 + rng2.randint(0, 5))]
        cc = rng2.uniform(0.2, 1.0, C)
        for i in range(len(pts) - 1):
            _bresenham(img, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], rng2.randint(0, 1), cc)

    # ── 4-8: 2D shapes ──
    elif lvl == 4:
        for _ in range(1 + rng2.randint(0, 2)):
            cc = rng2.uniform(0.1, 0.9, C)
            cx, cy = rng2.randint(5, W - 5), rng2.randint(5, H - 5)
            rx, ry = rng2.randint(2, 11), rng2.randint(2, 11)
            t = rng2.randint(0, 3)
            if t == 0:
                m = _ellipse(cx, cy, rx, ry)
            elif t == 1:
                x, y = rng2.randint(0, W - 10), rng2.randint(0, H - 10)
                ww, hh = rng2.randint(3, 16), rng2.randint(3, 16)
                m = _ellipse(x + ww / 2, y + hh / 2, ww / 2, hh / 2)
            else:
                m, _, _, _ = _splat(rng2, cx, cy, max(rx, ry))
            _fill(img, m, cc)

    elif lvl == 5:
        for _ in range(1 + rng2.randint(0, 3)):
            cc = rng2.uniform(0.2, 1.0, C)
            cx, cy = rng2.randint(6, W - 6), rng2.randint(6, H - 6)
            rx, ry = rng2.randint(3, 11), rng2.randint(3, 11)
            t = rng2.randint(0, 4)
            if t == 0:
                m = _hollow_ellipse(cx, cy, rx, ry, rng2.uniform(0.8, 2.2))
            elif t == 1:
                m = _hollow_rect(rng2.randint(2, W - 12), rng2.randint(2, H - 12),
                                 rng2.randint(5, 16), rng2.randint(5, 16), rng2.randint(1, 3))
            elif t == 2:
                m = _hollow_blob(rng2, cx, cy, rx, ry)
            else:
                na = rng2.randint(3, 8)
                rrs = rng2.uniform(2, max(3, rx), na)
                angs = np.sort(rng2.uniform(0, 2 * math.pi, na))
                pts = [(int(cx + rrs[i] * math.cos(angs[i])), int(cy + rrs[i] * math.sin(angs[i]))) for i in range(na)]
                tmp = np.zeros((H, W, C), np.float32)
                for i in range(na):
                    _bresenham(tmp, pts[i][0], pts[i][1], pts[(i + 1) % na][0], pts[(i + 1) % na][1], 1, (1, 1, 1))
                m = tmp[:, :, 0] > 0
            _fill(img, m, cc)

    elif lvl == 6:
        cx, cy = rng2.randint(5, W - 5), rng2.randint(5, H - 5)
        rx, ry = rng2.randint(3, 10), rng2.randint(3, 10)
        cb = rng2.uniform(0.0, 0.4, C)
        yy, xx = np.ogrid[:H, :W]
        dd = np.sqrt(((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2))
        m = dd <= 1.0
        g = 1.0 - dd / (dd[m].max() + 1e-8)
        for cc in range(C):
            img[m, cc] = np.clip(cb[cc] + g[m] * 0.7, 0, 1)

    elif lvl == 7:
        bg = _perlin(rng2, 3) * 0.3
        for cc in range(C):
            img[:, :, cc] = bg
        for _ in range(1 + rng2.randint(0, 3)):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(4, H - 4)
            r = rng2.uniform(3, 9)
            m, _, _, _ = _organic_blob(rng2, cx, cy, r, r * rng2.uniform(0.7, 1.6))
            tex = _perlin(rng2, 3)
            cc = rng2.uniform(0.1, 0.8, C)
            for ch in range(C):
                img[m, ch] = np.clip(cc[ch] + tex[m] * 0.5, 0, 1)

    elif lvl == 8:
        img = _1f_bg(rng2)
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        e = _edge_sobel(gray)
        for cc in range(C):
            img[:, :, cc] = np.where(e > rng2.uniform(0.3, 0.7), 1.0, 0.0)

    # ── 9-14: color palettes ──
    elif lvl == 9:
        pal = _mono(rng2)
        for _ in range(1 + rng2.randint(0, 3)):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(4, H - 4)
            r = rng2.uniform(3, 9)
            m, _, _, _ = _organic_blob(rng2, cx, cy, r, r * rng2.uniform(0.7, 2.0))
            shade = rng2.uniform(0.3, 1.0)
            for cc in range(C):
                img[m, cc] = pal[cc] * shade
        bg = _perlin(rng2, 3) * 0.15
        for cc in range(C):
            img[:, :, cc] = np.maximum(img[:, :, cc], bg * pal[cc])

    elif lvl == 10:
        pal = np.array([rng2.uniform(0, 1), rng2.uniform(0.3, 0.7), rng2.uniform(0.5, 0.9)])
        c1, c2, c3 = pal, pal * 0.6 + 0.1, pal * 1.1
        cx1, cy1 = rng2.randint(5, W - 5), rng2.randint(5, H - 5)
        r1 = rng2.uniform(4, 12)
        cx2, cy2 = rng2.randint(5, W - 5), rng2.randint(5, H - 5)
        r2 = rng2.uniform(3, 9)
        m1, _, _, _ = _splat(rng2, cx1, cy1, r1)
        m2, _, _, _ = _organic_blob(rng2, cx2, cy2, r2, r2 * 1.3)
        _fill(img, m1, c1)
        _fill(img, m2, c2)
        _fill(img, ~(m1 | m2), c3 * 0.3)

    elif lvl == 11:
        c1, c2 = _comp(rng2)
        img[:, :] = c1 * 0.3 + c2 * 0.1
        for i in range(2 + rng2.randint(0, 3)):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(4, H - 4)
            R = rng2.uniform(3, 8)
            m, _, _, _ = _splat(rng2, cx, cy, R)
            _fill(img, m, c1 if i % 2 == 0 else c2)

    elif lvl == 12:
        p1, p2, p3 = _tri(rng2)
        img[:, :] = (p1 + p2 + p3) * 0.15
        for i, pal in enumerate([p1, p2, p3]):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(4, H - 4)
            R = rng2.uniform(3, 9)
            m, _, _, _ = _organic_blob(rng2, cx, cy, R, R)
            _fill(img, m, pal)

    elif lvl == 13:
        pal = _warm(rng2)
        for _ in range(2 + rng2.randint(0, 4)):
            cx, cy = rng2.randint(3, W - 3), rng2.randint(3, H - 3)
            R = rng2.uniform(2, 8)
            m, _, _, _ = _splat(rng2, cx, cy, R)
            shade = rng2.uniform(0.5, 1.0)
            for cc in range(C):
                img[m, cc] = pal[cc] * shade
        bg = _perlin(rng2, 3) * 0.2
        for cc in range(C):
            img[:, :, cc] = np.maximum(img[:, :, cc], bg * pal[cc] * 0.5)

    elif lvl == 14:
        pal = _cool(rng2)
        for _ in range(2 + rng2.randint(0, 4)):
            cx, cy = rng2.randint(3, W - 3), rng2.randint(3, H - 3)
            R = rng2.uniform(2, 8)
            m, _, _, _ = _splat(rng2, cx, cy, R)
            shade = rng2.uniform(0.5, 1.0)
            for cc in range(C):
                img[m, cc] = pal[cc] * shade
        bg = _perlin(rng2, 3) * 0.2
        for cc in range(C):
            img[:, :, cc] = np.maximum(img[:, :, cc], bg * pal[cc] * 0.5)

    # ── 15-21: 3D geometry ──
    elif lvl == 15:
        cx, cy = rng2.randint(6, W - 6), rng2.randint(6, H - 6)
        R = rng2.uniform(5, 13)
        m, Nx, Ny, Nz = _sphere(cx, cy, R)
        _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), rng2.uniform(0.1, 1.0, C))

    elif lvl == 16:
        cx, cy = W / 2, H / 2
        s = rng2.uniform(9, 18)
        hs = s / 2
        yy, xx = np.ogrid[:H, :W]
        mf = (xx >= cx - hs) & (xx <= cx + hs) & (yy >= cy - hs) & (yy <= cy + hs)
        mt = (yy >= cy - hs) & (yy <= cy) & (xx >= cx - hs) & (xx <= cx + hs)
        ml = (xx >= cx - hs) & (xx <= cx) & (yy >= cy - hs) & (yy <= cy + hs)
        L = _rand_light(rng2)
        cc = rng2.uniform(0.1, 1.0, C)
        for ch in range(C):
            img[:, :, ch][mf] = cc[ch] * _phong(np.where(mf, 0, 0), np.where(mf, 0, 0), np.where(mf, 1, 0), L)[mf]
            img[:, :, ch][mt] = np.maximum(img[:, :, ch][mt], cc[ch] * _phong(np.where(mt, 0, 0), np.where(mt, -1, 0), np.where(mt, 0, 0), L)[mt] * 0.75)
            img[:, :, ch][ml] = np.maximum(img[:, :, ch][ml], cc[ch] * _phong(np.where(ml, -1, 0), np.where(ml, 0, 0), np.where(ml, 0, 0), L)[ml] * 0.55)

    elif lvl == 17:
        cx, cy = rng2.randint(6, W - 6), rng2.randint(8, H - 8)
        R = rng2.uniform(4, 10)
        m, Nx, Ny = _cylinder_side(cx, cy, R)
        L = _rand_light(rng2)
        lit = _phong(Nx, Ny, np.zeros_like(Nx), L)
        cc = rng2.uniform(0.1, 1.0, C)
        for ch in range(C):
            img[:, :, ch][m] = cc[ch] * lit[m]
        tcy = int(cy - R * 1.5 * 0.7)
        if tcy > 2:
            yy, xx = np.ogrid[:H, :W]
            tdd = ((xx - cx) ** 2 + (yy - tcy) ** 2) / R ** 2
            tm = tdd <= 1.0
            tlit = _phong(np.where(tm, 0, 0), np.where(tm, 0, 0), np.where(tm, 1, 0), L)
            for ch in range(C):
                img[:, :, ch][tm] = np.maximum(img[:, :, ch][tm], cc[ch] * tlit[tm])

    elif lvl == 18:
        cx, cy = rng2.randint(10, W - 10), rng2.randint(10, H - 10)
        R = rng2.uniform(6, 10)
        r = R * rng2.uniform(0.25, 0.45)
        yy, xx = np.ogrid[:H, :W]
        dx, dy = (xx - cx) / R, (yy - cy) / R
        rr = np.sqrt(dx ** 2 + dy ** 2)
        dR = rr - 1.0
        dz_sq = (r / R) ** 2 - dR ** 2
        m = (dz_sq >= 0) & (rr > 0.15)
        Nz = np.where(m, np.sqrt(np.maximum(0, dz_sq)) / (r / R), 0)
        Nr = np.where(m, dR / (rr + 1e-8), 0)
        Nx = np.where(m, dx / (rr + 1e-8) * Nr, 0)
        Ny = np.where(m, dy / (rr + 1e-8) * Nr, 0)
        _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), rng2.uniform(0.1, 1.0, C))

    elif lvl == 19:
        for _ in range(1 + rng2.randint(0, 2)):
            cx, cy = rng2.randint(6, W - 6), rng2.randint(6, H - 6)
            rx, ry = rng2.randint(4, 10), rng2.randint(4, 10)
            m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, rx, ry)
            _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), rng2.uniform(0.1, 1.0, C))

    elif lvl == 20:
        for _ in range(1 + rng2.randint(0, 3)):
            cx, cy = rng2.randint(6, W - 6), rng2.randint(6, H - 6)
            R = rng2.uniform(4, 9)
            m, Nx, Ny, Nz = _splat(rng2, cx, cy, R)
            _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), rng2.uniform(0.1, 1.0, C))

    elif lvl == 21:
        bg = _perlin(rng2, 3) * 0.15
        for cc in range(C):
            img[:, :, cc] = bg
        for _ in range(2 + rng2.randint(0, 3)):
            cx, cy = rng2.randint(5, W - 5), rng2.randint(5, H - 5)
            R = rng2.uniform(3, 9)
            cc = rng2.uniform(0.1, 1.0, C)
            t = rng2.randint(0, 3)
            L = _rand_light(rng2)
            if t == 0:
                m, Nx, Ny, Nz = _sphere(cx, cy, R)
            elif t == 1:
                m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.7, 1.5))
            else:
                m, Nx, Ny, Nz = _splat(rng2, cx, cy, R)
            _draw_3d(img, m, Nx, Ny, Nz, L, cc)

    # ── 22-25: directional textures ──
    elif lvl == 22:
        tex = _brushed(rng2)
        cc = rng2.uniform(0.1, 0.4, C)
        for ch in range(C):
            img[:, :, ch] = cc[ch] + tex * rng2.uniform(0.3, 0.6)
    elif lvl == 23:
        tex = _wood(rng2)
        cc = rng2.uniform(0.3, 0.6, C)
        for ch in range(C):
            img[:, :, ch] = cc[ch] * 0.4 + tex * cc[ch] * 0.8
    elif lvl == 24:
        tex = _marble(rng2)
        c1 = rng2.uniform(0.6, 1.0, C)
        c2 = rng2.uniform(0.1, 0.3, C)
        for ch in range(C):
            img[:, :, ch] = c1[ch] * tex + c2[ch] * (1 - tex)
    elif lvl == 25:
        tex = _striated(rng2)
        c1 = rng2.uniform(0.3, 0.8, C)
        c2 = rng2.uniform(0.1, 0.4, C)
        for ch in range(C):
            img[:, :, ch] = c1[ch] * tex + c2[ch] * (1 - tex)

    # ── 26-31: depth / perspective / scene ──
    elif lvl == 26:
        img = _1f_bg(rng2)
        for _ in range(1 + rng2.randint(0, 2)):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(H // 3, 2 * H // 3)
            R = rng2.uniform(4, 9)
            m, Nx, Ny, Nz = _sphere(cx, cy, R)
            _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), rng2.uniform(0.1, 1.0, C))
        _depth_fog(img, rng2)

    elif lvl == 27:
        img = _perspective_grid(rng2)
        vp_y = rng2.uniform(H * 0.1, H * 0.35)
        for _ in range(rng2.randint(1, 4)):
            cx = rng2.randint(4, W - 4)
            cy = rng2.randint(int(vp_y) + 4, H - 4)
            R = rng2.uniform(2, 6) * (1 - cy / H)
            m, Nx, Ny, Nz = _sphere(cx, cy, R)
            _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), rng2.uniform(0.1, 1.0, C))

    elif lvl == 28:
        img = _1f_bg(rng2, 0.5)
        for _ in range(2 + rng2.randint(0, 5)):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(4, H - 4)
            R = rng2.uniform(2, 9)
            cc = rng2.uniform(0.1, 1.0, C)
            t = rng2.randint(0, 4)
            L = _rand_light(rng2)
            if t == 0:
                m, Nx, Ny, Nz = _sphere(cx, cy, R)
            elif t == 1:
                m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.6, 2.0))
            elif t == 2:
                m, Nx, Ny, Nz = _splat(rng2, cx, cy, R)
            else:
                m, _, _, _ = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.5, 2.0))
                Nx = Ny = Nz = np.zeros((H, W))
            lit = _phong(Nx, Ny, Nz, L) if t != 3 else np.ones((H, W)) * 0.7
            for ch in range(C):
                img[:, :, ch][m] = np.clip(cc[ch] * lit[m], 0, 1)

    elif lvl == 29:
        sky = 0.3 + np.array([rng2.uniform(0.0, 0.3), rng2.uniform(0.0, 0.4), rng2.uniform(0.3, 0.7)])
        gnd = np.array([rng2.uniform(0.1, 0.45)] * 3)
        hz = rng2.randint(H // 3, 2 * H // 3)
        for r in range(H):
            img[r, :] = sky * (1 - r / max(hz, 1)) + gnd * 0.15 * (r / max(hz, 1))
        img[hz:, :] = gnd
        tex = _perlin(rng2, 4) * 0.3
        for cc in range(C):
            img[:, :, cc] = np.clip(img[:, :, cc] + tex * 0.4, 0, 1)
        for _ in range(rng2.randint(2, 5)):
            cx = rng2.randint(3, W - 3)
            cy = rng2.randint(2, min(hz - 1, H - 5))
            R = rng2.uniform(2, 7)
            cc = rng2.uniform(0.1, 1.0, C)
            t = rng2.randint(0, 3)
            L = _rand_light(rng2)
            if t == 0:
                m, Nx, Ny, Nz = _sphere(cx, cy, R)
            elif t == 1:
                m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.6, 1.8))
            else:
                m, Nx, Ny, Nz = _splat(rng2, cx, cy, R)
            _draw_3d(img, m, Nx, Ny, Nz, L, cc)

    elif lvl == 30:   # low sun: the cast shadow is the subject, not the object
        L = _rand_light(rng2)
        hz = rng2.randint(2, H // 3)
        _ground(rng2, img, hz, None, hi_key)
        sdx = -L[0] / (abs(L[1]) + 0.25)                     # shadow runs opposite the light
        ln = rng2.uniform(6, 20) / (abs(L[1]) + 0.35)        # low sun -> very long shadow
        for _ in range(rng2.randint(1, 4)):
            cx, cy = rng2.uniform(4, W - 4), rng2.uniform(hz + 4, H - 2)
            R = rng2.uniform(1.5, 5)
            sh = np.zeros((H, W), bool)
            for t in np.linspace(0, 1, 14):
                sh |= _ellipse(cx + sdx * ln * t, cy + rng2.uniform(0.1, 0.4) * ln * t * 0.25,
                               R * (1 - 0.55 * t), R * 0.4 * (1 - 0.55 * t))
            k = rng2.uniform(0.25, 0.6)
            for ch in range(C):
                img[:, :, ch][sh] *= k
            m, Nx, Ny, Nz = _sphere(cx, cy - R, R)
            _draw_3d(img, m, Nx, Ny, Nz, L, _photo_color(rng2, 0.15, 0.95))

    elif lvl == 31:   # contact and occlusion: a pile of touching bodies, AO in the crevices
        img[:] = _backdrop(rng2, hi_key)
        L = _rand_light(rng2)
        n = rng2.randint(3, 8)
        cx, cy = rng2.uniform(10, W - 10), rng2.uniform(12, H - 6)
        union = np.zeros((H, W), bool)
        for i in range(n):                       # stacked / heaped, so they actually touch
            R = rng2.uniform(2.5, 7)
            x = cx + rng2.uniform(-8, 8)
            y = cy - i * rng2.uniform(1.5, 4.0) + rng2.uniform(-2, 2)
            if rng2.randint(0, 2):
                m, Nx, Ny, Nz = _sphere(x, y, R)
            else:
                m, Nx, Ny, Nz = _organic_blob(rng2, x, y, R, R * rng2.uniform(0.7, 1.4))
            _draw_3d(img, m, Nx, Ny, Nz, L, _photo_color(rng2, 0.12, 0.95))
            union |= m
        occ = gaussian_filter(union.astype(np.float32), rng2.uniform(1.5, 3.5))
        k = rng2.uniform(0.35, 0.85)
        for ch in range(C):                      # darkest where the most geometry surrounds a point
            img[:, :, ch][union] *= np.clip(1 - k * occ[union], 0.15, 1.0)
        img = _contact_shadow(img, cx, cy + 3, rng2.uniform(5, 11), rng2)

    # ── 32-36: composites + postFX ──
    elif lvl == 32:
        bg = _perlin(rng2, 4) * 0.2
        for cc in range(C):
            img[:, :, cc] = bg
        for _ in range(rng2.randint(0, 4)):
            cc = rng2.uniform(0.2, 1.0, C)
            _bresenham(img, rng2.randint(0, W), rng2.randint(0, H),
                       rng2.randint(0, W), rng2.randint(0, H), rng2.randint(0, 1), cc)
        for _ in range(rng2.randint(1, 3)):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(4, H - 4)
            r = rng2.uniform(2, 8)
            m, _, _, _ = _organic_blob(rng2, cx, cy, r, r)
            _fill_alpha(img, m, rng2.uniform(0.1, 0.8, C), rng2.uniform(0.3, 0.7))
        for _ in range(rng2.randint(1, 4)):
            cx, cy = rng2.randint(5, W - 5), rng2.randint(5, H - 5)
            R = rng2.uniform(3, 9)
            t = rng2.randint(0, 3)
            L = _rand_light(rng2)
            cc = rng2.uniform(0.1, 1.0, C)
            if t == 0:
                m, Nx, Ny, Nz = _sphere(cx, cy, R)
            elif t == 1:
                m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.6, 2.0))
            else:
                m, Nx, Ny, Nz = _splat(rng2, cx, cy, R)
            _draw_3d(img, m, Nx, Ny, Nz, L, cc)

    elif lvl == 33:
        img = _1f_bg(rng2, 0.6)
        for _ in range(rng2.randint(1, 3)):
            cc = rng2.uniform(0.3, 1.0, C)
            cx, cy = rng2.randint(6, W - 6), rng2.randint(6, H - 6)
            rx, ry = rng2.randint(3, 9), rng2.randint(3, 9)
            m = _hollow_blob(rng2, cx, cy, rx, ry)
            _fill(img, m, cc)
        for _ in range(rng2.randint(1, 3)):
            cx, cy = rng2.randint(5, W - 5), rng2.randint(5, H - 5)
            R = rng2.uniform(3, 8)
            t = rng2.randint(0, 3)
            L = _rand_light(rng2)
            cc = rng2.uniform(0.1, 1.0, C)
            if t == 0:
                m, Nx, Ny, Nz = _sphere(cx, cy, R)
            elif t == 1:
                m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.6, 1.8))
            else:
                m, Nx, Ny, Nz = _splat(rng2, cx, cy, R)
            if rng2.randint(0, 3) == 0:
                _glass(img, rng2, m)
            else:
                _draw_3d(img, m, Nx, Ny, Nz, L, cc)

    elif lvl == 34:
        img = _1f_bg(rng2)
        for _ in range(rng2.randint(1, 3)):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(4, H - 4)
            R = rng2.uniform(4, 9)
            m, _, _, _ = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.7, 1.4))
            _glass(img, rng2, m)

    elif lvl == 35:
        img = _1f_bg(rng2)
        for _ in range(1 + rng2.randint(0, 2)):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(4, H - 4)
            R = rng2.uniform(4, 9)
            m, Nx, Ny, Nz = _sphere(cx, cy, R)
            _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), rng2.uniform(0.1, 1.0, C))
        img = _gauss_blur(img, rng2)

    elif lvl == 36:   # sensor artifacts: heavy grain, strong CA, banding, hot pixels, shear
        img[:] = _backdrop(rng2, hi_key)
        for _ in range(rng2.randint(1, 4)):
            cx, cy = rng2.uniform(3, W - 3), rng2.uniform(3, H - 3)
            R = rng2.uniform(2, 9)
            m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.6, 1.6))
            _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), _photo_color(rng2, 0.1, 0.95))
        img = np.clip(img + rng2.randn(H, W, C).astype(np.float32) * rng2.uniform(0.05, 0.24), 0, 1)
        for ch in (0, 2):                                    # chromatic aberration up to 2 px
            img[:, :, ch] = np.roll(np.roll(img[:, :, ch], rng2.randint(-2, 3), 0), rng2.randint(-2, 3), 1)
        if rng2.randint(0, 2) == 0:                          # fixed-pattern row banding
            band = np.sin(np.arange(H) * rng2.uniform(0.4, 3.0) + rng2.uniform(0, 6))
            img = np.clip(img + band.reshape(H, 1, 1) * rng2.uniform(0.03, 0.14), 0, 1)
        for _ in range(rng2.randint(0, 14)):                 # hot / dead pixels
            img[rng2.randint(0, H), rng2.randint(0, W)] = float(rng2.randint(0, 2))
        if rng2.randint(0, 3) == 0:                          # rolling shutter shear
            k = rng2.uniform(-0.35, 0.35)
            for r in range(H):
                img[r] = np.roll(img[r], int(r * k), axis=0)

    # ── 37-44: isolated features ──
    elif lvl == 37:
        img = _1f_bg(rng2)
    elif lvl == 38:
        n = _perlin(rng2, rng2.randint(1, 6)) ** rng2.uniform(0.3, 3.0)
        if rng2.randint(0, 3) == 0:
            n = gaussian_filter(n, rng2.uniform(0.8, 2.5))       # isotropic and soft: no direction,
            n = (n - n.min()) / (n.max() - n.min() + 1e-8)       # that is what separates it from 22
        c1, c2 = _photo_color(rng2, 0.02, 0.5), _photo_color(rng2, 0.3, 1.0)
        img = c1.reshape(1, 1, C) + (c2 - c1).reshape(1, 1, C) * n.reshape(H, W, 1)
        if rng2.randint(0, 3) == 0:
            img = np.clip(img * (0.4 + 1.2 * _perlin(rng2, 1).reshape(H, W, 1)), 0, 1)   # large-scale falloff
    elif lvl == 39:
        for _ in range(rng2.randint(1, 4)):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(4, H - 4)
            R = rng2.uniform(4, 10)
            t = rng2.randint(0, 4)
            L = _rand_light(rng2)
            cc = rng2.uniform(0.2, 1.0, C)
            if t == 0:
                m, Nx, Ny, Nz = _sphere(cx, cy, R)
            elif t == 1:
                m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.6, 2.0))
            elif t == 2:
                m, Nx, Ny, Nz = _splat(rng2, cx, cy, R)
            else:
                m = _hollow_blob(rng2, cx, cy, R, R * rng2.uniform(0.7, 1.5))
                Nx = Ny = Nz = np.zeros((H, W))
            lit = _phong(Nx, Ny, Nz, L) if t != 3 else np.ones((H, W)) * 0.7
            for ch in range(C):
                img[:, :, ch][m] = np.clip(cc[ch] * lit[m], 0, 1)
    elif lvl == 40:
        for _ in range(rng2.randint(1, 4)):
            cc = rng2.uniform(0.3, 1.0, C)
            cx, cy = rng2.randint(6, W - 6), rng2.randint(6, H - 6)
            rx, ry = rng2.randint(3, 10), rng2.randint(3, 10)
            t = rng2.randint(0, 3)
            if t == 0:
                m = _hollow_ellipse(cx, cy, rx, ry, rng2.uniform(0.6, 2.5))
            elif t == 1:
                m = _hollow_blob(rng2, cx, cy, rx, ry)
            else:
                na = rng2.randint(3, 8)
                rrs = rng2.uniform(2, max(3, rx), na)
                angs = np.sort(rng2.uniform(0, 2 * math.pi, na))
                pts = [(int(cx + rrs[i] * math.cos(angs[i])), int(cy + rrs[i] * math.sin(angs[i]))) for i in range(na)]
                tmp = np.zeros((H, W, C), np.float32)
                for i in range(na):
                    _bresenham(tmp, pts[i][0], pts[i][1], pts[(i + 1) % na][0], pts[(i + 1) % na][1], 1, (1, 1, 1))
                m = tmp[:, :, 0] > 0
            _fill(img, m, cc)
    elif lvl == 41:
        img = _1f_bg(rng2)
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        e = _edge_sobel(gray)
        for cc in range(C):
            img[:, :, cc] = np.where(e > rng2.uniform(0.15, 0.5), rng2.uniform(0.3, 1.0), 0.0)
    elif lvl == 42:
        t = rng2.randint(0, 4)
        yy, xx = np.ogrid[:H, :W]
        if t == 0:
            cx, cy = rng2.uniform(0, W), rng2.uniform(0, H)
            R = rng2.uniform(10, 25)
            g = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
        elif t == 1:
            g = yy / H
        elif t == 2:
            g = xx / W
        else:
            g = (xx / W + yy / H) / 2
        g = np.clip(g, 0, 1)
        c0 = rng2.uniform(0.0, 0.3, C)
        c1 = rng2.uniform(0.5, 1.0, C)
        for cc in range(C):
            img[:, :, cc] = (1 - g) * c0[cc] + g * c1[cc]
    elif lvl == 43:
        pts = [(rng2.uniform(0, W), rng2.uniform(0, H)) for _ in range(rng2.randint(5, 15))]
        yy, xx = np.ogrid[:H, :W]
        dmin = np.full((H, W), 1e9)
        for px, py in pts:
            d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
            dmin = np.minimum(dmin, d)
        dmin = (dmin - dmin.min()) / (dmin.max() - dmin.min() + 1e-8)
        cc = rng2.uniform(0.2, 0.9, C)
        for ch in range(C):
            img[:, :, ch] = dmin * cc[ch]
        d2 = np.full((H, W), 1e9)
        for px, py in pts:
            d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
            d2 = np.where(d > dmin, d2, np.where(d < d2, d, d2))
        boundary = ((d2 - dmin) < rng2.uniform(0.5, 1.5)).astype(np.float32)
        for ch in range(C):
            img[:, :, ch] = np.maximum(img[:, :, ch], boundary * 0.8)
    elif lvl == 44:
        t = rng2.randint(0, 3)
        yy, xx = np.ogrid[:H, :W]
        if t == 0:
            step = rng2.randint(2, 8)
            p = ((xx // step) + (yy // step)) % 2
        elif t == 1:
            step = rng2.randint(2, 8)
            ang = rng2.uniform(0, math.pi)
            xxr = xx * math.cos(ang) + yy * math.sin(ang)
            p = (xxr // step) % 2
        else:
            cx, cy = rng2.uniform(0, W), rng2.uniform(0, H)
            step = rng2.randint(3, 10)
            p = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) // step) % 2
        p = p.astype(np.float32)
        c0 = rng2.uniform(0.0, 0.3, C)
        c1 = rng2.uniform(0.6, 1.0, C)
        for ch in range(C):
            img[:, :, ch] = (1 - p) * c0[ch] + p * c1[ch]

    # ── 45-46: fog-blobs, texture+grain ──
    elif lvl == 45:
        bg = _perlin(rng2, 4) * 0.2
        for cc in range(C):
            img[:, :, cc] = bg
        for _ in range(2 + rng2.randint(0, 4)):
            cx = rng2.randint(3, W - 3)
            cy = rng2.randint(3, H - 3)
            R = rng2.uniform(3, 8)
            m, Nx, Ny, Nz = _splat(rng2, cx, cy, R)
            _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), rng2.uniform(0.1, 1.0, C))
        _depth_fog(img, rng2)
    elif lvl == 46:
        t = rng2.randint(0, 4)
        tex, cc = None, None
        if t == 0:
            tex = _brushed(rng2)
            cc = rng2.uniform(0.1, 0.4, C)
        elif t == 1:
            tex = _wood(rng2)
            cc = rng2.uniform(0.3, 0.6, C)
        elif t == 2:
            tex = _marble(rng2)
            cc = rng2.uniform(0.6, 1.0, C)
        else:
            tex = _striated(rng2)
            cc = rng2.uniform(0.3, 0.8, C)
        tex = np.clip(tex, 0, 1) ** rng2.uniform(0.4, 2.6)          # contrast of the pattern
        c2 = _photo_color(rng2, 0.05, 0.95)
        for ch in range(C):
            img[:, :, ch] = cc[ch] * (1 - tex) + c2[ch] * tex
        if rng2.randint(0, 3) == 0:        # the texture lives on a lit curved body, not on the frame
            L = _rand_light(rng2)
            m, Nx, Ny, Nz = _organic_blob(rng2, W / 2 + rng2.uniform(-5, 5), H / 2 + rng2.uniform(-5, 5),
                                          rng2.uniform(7, 16), rng2.uniform(7, 16))
            lit = _phong(Nx, Ny, Nz, L)
            body, img = img.copy(), _backdrop(rng2, hi_key)
            for ch in range(C):
                img[:, :, ch][m] = np.clip(body[:, :, ch][m] * lit[m], 0, 1)
        elif rng2.randint(0, 3) == 0:      # lit from one side instead of flat
            g = np.linspace(rng2.uniform(0.3, 1.0), rng2.uniform(0.3, 1.0), W).reshape(1, W, 1)
            img = np.clip(img * g, 0, 1)
        img = _add_grain(img, rng2)

    # ── 48-57: CREATURES (2D + 3D) ──
    elif 48 <= lvl <= 52:
        typ = lvl - 48
        _habitat(rng2, img, hi_key)
        cc = _photo_color(rng2, 0.15, 0.95)
        _critter_2d(rng2, img, cc, typ)
        if rng2.randint(0, 2) == 0:
            pal = _mono(rng2)
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            for ch in range(C):
                img[:, :, ch] = gray * pal[ch] / (pal.mean() + 1e-8)
    elif 53 <= lvl <= 57:
        typ = lvl - 53
        _habitat(rng2, img, hi_key)
        # shadow must stay small relative to the creature (size 5-11px): a wide independent
        # radius used to swallow small subjects (bird/fish/abstract) under a grey disc
        img = _contact_shadow(img, W / 2 + rng2.uniform(-6, 6), H / 2 + rng2.uniform(0, 9),
                              rng2.uniform(2, 5), rng2)
        cc = _photo_color(rng2, 0.15, 0.95)
        _critter_3d(rng2, img, cc, typ)
        if rng2.randint(0, 2) == 0:
            pal = _mono(rng2)
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            for ch in range(C):
                img[:, :, ch] = gray * pal[ch] / (pal.mean() + 1e-8)

    # ── 58: Texture-wrapped creatures ──
    elif lvl == 58:
        typ = rng2.randint(0, 5)
        _habitat(rng2, img, hi_key)
        cc = _photo_color(rng2, 0.15, 0.95)
        parts_masks = _critter_3d(rng2, img, cc, typ)
        t = rng2.randint(0, 4)
        if t == 0:
            tex = _brushed(rng2)
        elif t == 1:
            tex = _wood(rng2)
        elif t == 2:
            tex = _marble(rng2)
        else:
            tex = _striated(rng2)
        # texture must wrap the creature only — applying it over the whole frame drowned the
        # (already small) subject in a wall of wood/marble/stripe pattern
        body = np.zeros((H, W), bool)
        for pm in parts_masks:
            body |= pm
        for ch in range(C):
            img[:, :, ch][body] = np.clip(img[:, :, ch][body] * (0.6 + 0.4 * tex[body]), 0, 1)

    # ── 59-63: FLORA & BUILDINGS ──
    elif lvl in (59, 60, 61):
        season = rng2.randint(0, 4)
        if season == 0:      # spring / summer green
            fol = np.array([rng2.uniform(0.05, 0.3), rng2.uniform(0.35, 0.9), rng2.uniform(0.05, 0.3)])
        elif season == 1:    # autumn
            fol = np.array([rng2.uniform(0.55, 1.0), rng2.uniform(0.25, 0.6), rng2.uniform(0.02, 0.2)])
        elif season == 2:    # dry / dead
            fol = np.array([rng2.uniform(0.35, 0.7), rng2.uniform(0.3, 0.55), rng2.uniform(0.15, 0.35)])
        else:                # snow-laden / frosted
            fol = np.array([rng2.uniform(0.6, 0.95)] * 3) * rng2.uniform(0.8, 1.1)
        for _ in range(rng2.randint(1, 4)):
            _tree(rng2, img, fol * rng2.uniform(0.8, 1.2), lvl - 59)
    elif lvl == 62:
        _bush(rng2, img, np.array([rng2.uniform(0.1, 0.3), rng2.uniform(0.5, 0.9), rng2.uniform(0.1, 0.3)]))
        if rng2.randint(0, 3) == 0:
            _flower(rng2, img, rng2.uniform(0.4, 1.0, C))
    elif lvl == 63:
        _building(rng2, img, rng2.uniform(0.2, 0.7, C), rng2.randint(0, 3))
        if rng2.randint(0, 2) == 0:
            sky = rng2.uniform(0.3, 0.7, C)
            for r in range(H):
                img[r, :] = np.maximum(img[r, :], sky * (r / H) * 0.4)

    # ── 64-66: ELEMENTS ──
    elif lvl == 64:
        _water(rng2, img)
    elif lvl == 65:
        L = _rand_light(rng2)
        zoom = 2.0 ** rng2.uniform(-1.0, 1.2)          # 0.5x (distant) → 2.3x (close-up)
        # background: 5 diverse fire settings (plus ~1/6 chance of old dark style via _fire_bg)
        _fire_bg(rng2, img)
        base_y = None
        if zoom > 1.6:                                  # close-up: push flame center lower so top fills frame
            base_y = rng2.uniform(H * 0.55, H - 2)
        elif zoom < 0.7:                                # distant: raise flame center toward horizon
            base_y = rng2.uniform(H * 0.55, H - 1)
        n_fires = rng2.randint(1, 3)
        for _ in range(n_fires):
            if rng2.randint(0, 2) == 0:
                _fire(rng2, img, zoom=zoom, base_y=base_y)
            else:
                _fire_3d(rng2, img, L, zoom=zoom, base_y=base_y)
        # subtle bloom
        bloom = gaussian_filter(img, sigma=(rng2.uniform(0.4, 0.9),) * 2 + (0,))
        img = np.clip(img + bloom * rng2.uniform(0.08, 0.28), 0, 1)
    elif lvl == 66:
        sky = np.array([rng2.uniform(0.3, 0.6), rng2.uniform(0.3, 0.7), rng2.uniform(0.5, 0.9)])
        img[:, :] = sky
        _gas_cloud(rng2, img)

    # ── 67: Element combos ──
    elif lvl == 67:
        combo = rng2.randint(0, 3)
        if combo == 0:
            _water(rng2, img)
            fire_rng = _rng(lidx * 7 + 999999)
            for _ in range(rng2.randint(1, 2)):
                _fire(fire_rng, img)
        elif combo == 1:
            sky = np.array([rng2.uniform(0.2, 0.5), rng2.uniform(0.3, 0.6), rng2.uniform(0.4, 0.7)])
            gnd = np.array([rng2.uniform(0.05, 0.2)] * 3)
            hz = rng2.randint(H // 3, 2 * H // 3)
            for r in range(H):
                img[r, :] = sky * (1 - r / max(hz, 1)) + gnd * 0.3 * (r / max(hz, 1))
            img[hz:, :] = gnd
            _gas_cloud(rng2, img)
        else:
            img[:, :, :] = rng2.uniform(0.02, 0.1)
            for _ in range(rng2.randint(1, 3)):
                _fire(rng2, img)
            _gas_cloud(rng2, img)

    # ── 68-70: FULL SCENES ──
    elif lvl == 68:
        sky = np.array([rng2.uniform(0.1, 0.3), rng2.uniform(0.2, 0.5), rng2.uniform(0.4, 0.7)])
        gnd = np.array([rng2.uniform(0.05, 0.25), rng2.uniform(0.2, 0.5), rng2.uniform(0.05, 0.2)])
        hz = rng2.randint(H // 3, 2 * H // 3)
        for r in range(H):
            img[r, :] = sky * (1 - r / max(hz, 1)) + gnd * (r / max(hz, 1))
        img[hz:, :] = gnd
        tex = _perlin(rng2, 4) * 0.3
        for cc in range(C):
            img[:, :, cc] = np.clip(img[:, :, cc] + tex * 0.3, 0, 1)
        if rng2.randint(0, 3) == 0:
            _water(rng2, img)          # before the flora: _water repaints the whole frame
        for _ in range(rng2.randint(2, 5)):
            _tree(rng2, img, np.array([rng2.uniform(0.05, 0.2), rng2.uniform(0.3, 0.7), rng2.uniform(0.05, 0.2)]), rng2.randint(0, 3))
        for _ in range(rng2.randint(1, 4)):
            _bush(rng2, img, np.array([rng2.uniform(0.1, 0.3), rng2.uniform(0.4, 0.8), rng2.uniform(0.1, 0.3)]))

    elif lvl == 69:
        sky = np.array([rng2.uniform(0.1, 0.4), rng2.uniform(0.1, 0.4), rng2.uniform(0.3, 0.7)])
        gnd = np.array([rng2.uniform(0.1, 0.3)] * 3)
        hz = rng2.randint(H // 3, 2 * H // 3)
        for r in range(H):
            img[r, :] = sky * (1 - r / max(hz, 1)) + gnd * (r / max(hz, 1))
        img[hz:, :] = gnd
        for _ in range(rng2.randint(2, 5)):
            _building(rng2, img, rng2.uniform(0.1, 0.6, C), rng2.randint(0, 3))
        if rng2.randint(0, 2) == 0:
            _tree(rng2, img, np.array([rng2.uniform(0.1, 0.25), rng2.uniform(0.4, 0.7), rng2.uniform(0.1, 0.25)]), 0)

    elif lvl == 70:
        bg = _perlin(rng2, 4) * 0.25
        for cc in range(C):
            img[:, :, cc] = bg
        env = rng2.randint(0, 4)
        if env == 0:
            _water(rng2, img)
        elif env == 1:
            _tree(rng2, img, np.array([rng2.uniform(0.1, 0.3), rng2.uniform(0.5, 0.8), rng2.uniform(0.1, 0.3)]), 0)
        elif env == 2:
            _building(rng2, img, rng2.uniform(0.2, 0.6, C), rng2.randint(0, 3))
        else:
            _gas_cloud(rng2, img)
        for _ in range(rng2.randint(1, 3)):
            if rng2.randint(0, 2) == 0:
                _critter_2d(rng2, img, rng2.uniform(0.2, 0.9, C), rng2.randint(0, 5))
            else:
                _critter_3d(rng2, img, rng2.uniform(0.2, 0.9, C), rng2.randint(0, 5))

    # ── 80-81: CAPTURE PIPELINE ──
    elif lvl == 80:  # supersample + box downsample → antialiased edges
        img = _box_down(_base_render(rng2, size=(W * 2, H * 2)), 2)
        if rng2.randint(0, 2) == 0:
            img = _add_grain(img, rng2)

    elif lvl == 81:  # random crop of a 2× render → off-center framing
        big = _base_render(rng2, size=(W * 2, H * 2))
        y0, x0 = rng2.randint(0, H + 1), rng2.randint(0, W + 1)
        img = np.ascontiguousarray(big[y0:y0 + H, x0:x0 + W])

    # ── 82-84: VEHICLES (man-made, hard edges, wheels/wings/hulls) ──
    elif lvl == 82:  # car / truck / bus, side view
        gy = rng2.randint(20, 29)
        img[:gy] = rng2.uniform(0.3, 0.85, C)
        img[gy:] = rng2.uniform(0.12, 0.4, C)
        body = rng2.uniform(0.1, 0.95, C)
        glass = np.array([0.55, 0.7, 0.88]) * rng2.uniform(0.6, 1.3)
        typ = rng2.randint(0, 3)
        x0, x1 = rng2.randint(1, 6), rng2.randint(24, 31)
        by1 = gy - 2
        by0 = by1 - (5 if typ == 0 else 7 if typ == 1 else 9)
        _fill(img, _rect(x0, by0, x1, by1), body)
        if typ == 0:
            cw, cx0 = (x1 - x0) * 0.45, x0 + (x1 - x0) * 0.3
            _fill(img, _rect(cx0, by0 - 3, cx0 + cw, by0), np.clip(body * 0.9, 0, 1))
            _fill(img, _rect(cx0 + 1, by0 - 2, cx0 + cw - 1, by0 - 1), np.clip(glass, 0, 1))
        elif typ == 1:
            _fill(img, _rect(x0, by0 - 4, x0 + 7, by0), np.clip(body * 0.85, 0, 1))
            _fill(img, _rect(x0 + 8, by0 - 6, x1, by0), rng2.uniform(0.25, 0.9, C))
            _fill(img, _rect(x0 + 1, by0 - 3, x0 + 5, by0 - 1), np.clip(glass, 0, 1))
        else:
            for wx in range(int(x0) + 2, int(x1) - 2, 4):
                _fill(img, _rect(wx, by0 + 1, wx + 3, by0 + 4), np.clip(glass * rng2.uniform(0.7, 1.2), 0, 1))
        for wx in (x0 + (x1 - x0) * 0.22, x0 + (x1 - x0) * 0.78):
            img = _blend(img, _soft_disc(wx, by1, rng2.uniform(1.8, 2.9)), np.array([0.06, 0.06, 0.08]))
            img = _blend(img, _soft_disc(wx, by1, rng2.uniform(0.6, 1.1)), np.array([0.5, 0.5, 0.55]))
        if 0 <= by0 < H:
            img[by0, int(x0):int(x1)] = np.clip(body * 1.6, 0, 1)  # roof specular
        img = _contact_shadow(img, (x0 + x1) / 2, by1 + 1, (x1 - x0) * 0.5, rng2)

    elif lvl == 83:  # airplane on sky
        _sky_gradient(rng2, img)
        col = np.ones(3) * rng2.uniform(0.55, 1.0) * rng2.uniform(0.8, 1.1)
        cx, cy = rng2.uniform(10, 22), rng2.uniform(10, 22)
        ang = rng2.uniform(-0.6, 0.6)
        yy, xx = np.ogrid[:H, :W]
        xr = (xx - cx) * math.cos(ang) + (yy - cy) * math.sin(ang)
        yr = -(xx - cx) * math.sin(ang) + (yy - cy) * math.cos(ang)
        fl, fw = rng2.uniform(7, 13), rng2.uniform(1.2, 2.5)
        body = (xr / fl) ** 2 + (yr / fw) ** 2 <= 1
        wing = ((xr - rng2.uniform(-2, 2)) / rng2.uniform(1.5, 3)) ** 2 + (yr / rng2.uniform(6, 12)) ** 2 <= 1
        tail = ((xr + fl * 0.8) / 1.8) ** 2 + (yr / rng2.uniform(3, 5.5)) ** 2 <= 1
        _fill(img, wing, np.clip(col * 0.72, 0, 1))
        _fill(img, tail, np.clip(col * 0.62, 0, 1))
        _fill(img, body, np.clip(col, 0, 1))
        under = body & (yr > 0)
        for ch in range(C):
            img[:, :, ch][under] *= rng2.uniform(0.55, 0.8)  # consistent underside shading
        for ch in range(C):
            img[:, :, ch][body & (np.abs(yr) < 0.7)] = np.clip(col[ch] * rng2.uniform(0.2, 0.5), 0, 1)

    elif lvl == 84:  # ship on water
        hz = rng2.randint(12, 21)
        _sky_gradient(rng2, img, hz=hz)
        sea = np.array([rng2.uniform(0.02, 0.16), rng2.uniform(0.1, 0.36), rng2.uniform(0.3, 0.75)])
        tex = _perlin(rng2, 3)
        img[hz:] = np.clip(sea.reshape(1, 1, C) * (0.6 + 0.8 * tex[hz:].reshape(-1, W, 1)), 0, 1)
        hull = rng2.uniform(0.1, 0.85, C)
        x0, x1 = rng2.randint(2, 10), rng2.randint(20, 31)
        top = hz - rng2.randint(1, 4)
        nh = rng2.randint(3, 6)
        for i in range(nh):
            r = top + i
            if 0 <= r < H:
                img[r, max(0, int(x0 + i * 1.2)):max(0, int(x1 - i * 1.2))] = np.clip(hull * (1 - 0.09 * i), 0, 1)
        sw = (x1 - x0) * 0.3
        _fill(img, _rect(x0 + sw, top - 4, x0 + 2 * sw, top), np.clip(hull * 1.4, 0, 1))
        _bresenham(img, int((x0 + x1) / 2), int(top - 4), int((x0 + x1) / 2), max(0, int(top - 10)), 0,
                   np.clip(hull * 0.55, 0, 1))
        for i in range(1, rng2.randint(3, 8)):  # wake / reflection
            r = top + nh + i
            if r < H:
                a = rng2.uniform(0.15, 0.45) / i
                img[r] = np.clip(img[r] * (1 - a) + hull * a, 0, 1)

    # ── 85: HEAD CLOSE-UP (frontal symmetric, or profile with neck spilling off-frame) ──
    elif lvl == 85:
        img = gaussian_filter(_1f_bg(rng2, 0.6), sigma=(1.6, 1.6, 0))
        if hi_key:
            img = np.clip(img * 0.3 + _photo_color(rng2, 0.65, 1.0).reshape(1, 1, C) * 0.75, 0, 1)
        skin = _photo_color(rng2, 0.18, 0.9)
        L = _rand_light(rng2)
        eye = np.array([0.05, 0.05, 0.09])
        profile = rng2.randint(0, 3) > 0
        parts = [('s', (0, 0, 0), 1.0, skin)]                                        # head
        for s_ in (-1, 1):   # -z is towards the camera: eyes and muzzle in front, ears behind
            parts.append(('s', (s_ * 0.8, -0.65, 0.15), rng2.uniform(0.3, 0.5), skin * 0.85))    # ears
            parts.append(('s', (s_ * 0.34, -0.18, -0.88), rng2.uniform(0.16, 0.24), eye))        # eyes
            parts.append(('s', (s_ * 0.34 - 0.06, -0.24, -1.0), 0.06, np.ones(3) * 0.95))        # catchlight
        parts.append(('s', (0, 0.42, -0.92), rng2.uniform(0.14, 0.26), skin * rng2.uniform(0.35, 0.6)))
        if profile:   # elongated muzzle + neck running out of frame, like the horse crops in CIFAR
            for i in range(1, 4):
                parts.append(('s', (0, 0.15 + 0.12 * i, -0.9 - 0.42 * i), 0.62 - 0.11 * i,
                              skin * rng2.uniform(0.85, 1.1)))
            for i in range(1, 4):
                parts.append(('s', (rng2.uniform(-0.15, 0.15), 0.75 + 0.62 * i, 0.5 + 0.35 * i),
                              0.72 + 0.06 * i, skin * rng2.uniform(0.8, 1.0)))       # neck
            yaw = rng2.uniform(1.15, 1.95) * (1 if rng2.randint(0, 2) else -1)
            size = rng2.uniform(14, 24)          # head alone overflows the 32px frame
            ax = W / 2 + rng2.uniform(-7, 7)
            ay = H / 2 + rng2.uniform(-8, 4)
        else:
            yaw, size = rng2.uniform(-1.0, 1.0), rng2.uniform(9, 14)
            ax, ay = W / 2 + rng2.uniform(-3, 3), H / 2 + rng2.uniform(-2, 3)
        masks = _render_parts(img, parts, L, rng2, yaw=yaw, pitch=rng2.uniform(-0.4, 0.4),
                              size=size, dist=rng2.uniform(3.5, 5.0), anchor=(0, ax, ay))
        tex = _brushed(rng2)
        fur = masks[0] & ~(masks[2] | masks[6])                                       # head minus eyes
        for ch in range(C):
            img[:, :, ch][fur] = np.clip(img[:, :, ch][fur] * (0.75 + 0.45 * tex[fur]), 0, 1)

    # ── 86-89: OPTICS & CAMERA RESPONSE ──
    elif lvl == 86:  # depth of field: blurred background + bokeh + sharp subject
        img = gaussian_filter(_base_render(rng2), sigma=(rng2.uniform(1.2, 3.0),) * 2 + (0,))
        for _ in range(rng2.randint(2, 8)):
            a = _soft_disc(rng2.uniform(0, W), rng2.uniform(0, H), rng2.uniform(1.5, 4.5)) * rng2.uniform(0.15, 0.5)
            img = _blend(img, a, np.clip(rng2.uniform(0.5, 1.2, C), 0, 1))
        cx, cy, R = rng2.uniform(8, 24), rng2.uniform(10, 26), rng2.uniform(6, 12)
        t = rng2.randint(0, 3)
        if t == 0:
            m, Nx, Ny, Nz = _sphere(cx, cy, R)
        elif t == 1:
            m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.6, 1.5))
        else:
            m, Nx, Ny, Nz = _splat(rng2, cx, cy, R)
        _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), rng2.uniform(0.15, 1.0, C))

    elif lvl == 87:  # motion blur (camera shake / moving subject)
        img = _motion_blur(_base_render(rng2), rng2)
        if rng2.randint(0, 2) == 0:
            img = _add_grain(img, rng2)

    elif lvl == 88:  # natural color statistics: correlated RGB, moderate saturation, white balance
        img = _natural_color(_base_render(rng2), rng2)
        if rng2.randint(0, 2) == 0:
            img = _gauss_blur(img, rng2)

    elif lvl == 89:  # exposure / gamma / contrast curve + lens vignette
        img = _vignette(_tone_curve(_base_render(rng2), rng2), rng2)

    # ── 90: BACKLIT SILHOUETTE ──
    elif lvl == 90:
        _sky_gradient(rng2, img)
        img = np.clip(img * rng2.uniform(1.0, 1.5), 0, 1)
        dark = np.ones(3) * rng2.uniform(0.01, 0.1)
        gy = rng2.randint(20, 31)
        img[gy:] = dark * rng2.uniform(0.8, 2.2)
        t = rng2.randint(0, 3)
        if t == 0:
            for _ in range(rng2.randint(1, 4)):
                _tree(rng2, img, dark, rng2.randint(0, 3))
        elif t == 1:
            _critter_2d(rng2, img, dark, rng2.randint(0, 5))
        else:
            for _ in range(rng2.randint(1, 3)):
                _building(rng2, img, dark, rng2.randint(0, 3))
        sil = (img.mean(axis=-1) < 0.15).astype(np.float32)
        halo = np.clip(gaussian_filter(sil, 1.3) - sil, 0, 1)
        img = np.clip(img + halo.reshape(H, W, 1) * rng2.uniform(0.1, 0.45), 0, 1)

    # ── 91-92: NATURAL MICRO-TEXTURE ──
    elif lvl == 91:  # grass / foliage field with depth-scaled strokes
        hz = rng2.randint(0, 15)
        base = np.array([rng2.uniform(0.05, 0.3), rng2.uniform(0.2, 0.62), rng2.uniform(0.05, 0.28)])
        if hz > 0:
            _sky_gradient(rng2, img, hz=hz, sun=False)
        img[hz:] = np.clip(base * 0.6, 0, 1)
        for _ in range(rng2.randint(70, 160)):
            x, y = rng2.uniform(0, W), rng2.uniform(hz, H)
            depth = (y - hz) / max(H - hz, 1)
            ln = 1.5 + depth * rng2.uniform(2, 6)
            a = -math.pi / 2 + rng2.uniform(-0.65, 0.65)
            col = np.clip(base * rng2.uniform(0.5, 1.5) * (0.55 + 0.6 * depth), 0, 1)
            _bresenham(img, int(x), int(y), int(x + math.cos(a) * ln), int(y + math.sin(a) * ln), 0, col)
        haze = np.clip(1 - (np.arange(H) - hz) / max(H - hz, 1), 0, 1).reshape(H, 1, 1) * rng2.uniform(0.1, 0.4)
        img = np.clip(img * (1 - haze) + img[max(hz - 1, 0)].mean(axis=0).reshape(1, 1, C) * haze, 0, 1)

    elif lvl == 92:  # fur / feather: shading-modulated high-frequency + noisy silhouette
        img = gaussian_filter(_1f_bg(rng2, 0.5), sigma=(1.5, 1.5, 0))
        R = rng2.uniform(7, 13)
        cx, cy = W / 2 + rng2.uniform(-4, 4), H / 2 + rng2.uniform(-3, 5)
        m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.6, 1.3))
        col = rng2.uniform(0.15, 0.9, C)
        _draw_3d(img, m, Nx, Ny, Nz, _rand_light(rng2), col)
        fur = _brushed(rng2) * 0.6 + _perlin(rng2, 5) * 0.4
        for ch in range(C):
            img[:, :, ch][m] = np.clip(img[:, :, ch][m] * (0.6 + 0.8 * fur[m]), 0, 1)
        strand = (gaussian_filter(m.astype(np.float32), 1.0) > 0.15) & (~m) & (_perlin(rng2, 5) > 0.55)
        for ch in range(C):
            img[:, :, ch][strand] = np.clip(col[ch] * rng2.uniform(0.4, 0.9), 0, 1)

    # ── 93: FRACTAL / SELF-SIMILAR STRUCTURE ──
    elif lvl == 93:
        mode = rng2.randint(0, 3)
        if mode == 0:  # bare branches against sky
            _sky_gradient(rng2, img, sun=False)
            col = np.ones(3) * rng2.uniform(0.08, 0.35)
            _branch(img, rng2, W / 2 + rng2.uniform(-6, 6), H - 1, -math.pi / 2 + rng2.uniform(-0.3, 0.3),
                    rng2.uniform(6, 11), 1, col, 4)
        elif mode == 1:  # lightning / cracks
            img[:] = rng2.uniform(0.02, 0.2, C)
            col = np.clip(rng2.uniform(0.7, 1.0, C) + 0.2, 0, 1)
            _branch(img, rng2, rng2.uniform(8, 24), 0, math.pi / 2 + rng2.uniform(-0.4, 0.4),
                    rng2.uniform(5, 9), 0, col, 5)
            img = np.clip(img + gaussian_filter(img, (1.5, 1.5, 0)) * 0.6, 0, 1)
        else:  # river delta / veins over texture
            img[:] = _perlin(rng2, 4).reshape(H, W, 1) * rng2.uniform(0.3, 0.8, C)
            _branch(img, rng2, W / 2, H - 1, -math.pi / 2, rng2.uniform(6, 10), 1, rng2.uniform(0.4, 1.0, C), 4)

    # ── 94: MAN-MADE REPETITION (bricks / windows / fences) ──
    elif lvl == 94:
        t = rng2.randint(0, 3)
        shear = rng2.uniform(-0.35, 0.35)
        if t == 0:
            base = np.array([rng2.uniform(0.35, 0.8), rng2.uniform(0.15, 0.45), rng2.uniform(0.1, 0.35)])
            img[:] = rng2.uniform(0.5, 0.85, C)
            bh_, bw_ = rng2.randint(3, 6), rng2.randint(5, 11)
            for r0 in range(0, H, bh_):
                off = (r0 // bh_ % 2) * (bw_ // 2)
                for c0 in range(-bw_, W, bw_):
                    x = c0 + off + int(shear * r0)
                    _fill(img, _rect(x + 1, r0 + 1, x + bw_ - 1, r0 + bh_ - 1),
                          np.clip(base * rng2.uniform(0.75, 1.25), 0, 1))
        elif t == 1:
            img[:] = rng2.uniform(0.12, 0.6, C)
            gx, gy = rng2.randint(4, 9), rng2.randint(4, 10)
            lit = rng2.uniform(0.45, 1.0, C)
            for r0 in range(rng2.randint(0, 4), H - 2, gy):
                for c0 in range(rng2.randint(0, 4), W - 2, gx):
                    _fill(img, _rect(c0, r0, c0 + gx - 2, r0 + gy - 2), np.clip(lit * rng2.uniform(0.15, 1.0), 0, 1))
        else:
            img[:] = _perlin(rng2, 3).reshape(H, W, 1) * rng2.uniform(0.3, 0.9, C)
            col = rng2.uniform(0.3, 0.9, C)
            for x in range(rng2.randint(0, 3), W, rng2.randint(3, 6)):
                _fill(img, _rect(x, rng2.randint(2, 9), x + 1 + rng2.randint(0, 2), H - rng2.randint(0, 6)), col)
            for y in (rng2.randint(8, 15), rng2.randint(18, 27)):
                _fill(img, _rect(0, y, W, y + 2), np.clip(col * 0.8, 0, 1))
        img = np.clip(img * (0.75 + 0.5 * _perlin(rng2, 4).reshape(H, W, 1)), 0, 1)

    # ── 95: SKY + CLOUDS ──
    elif lvl == 95:
        hz = rng2.randint(18, 33)
        _sky_gradient(rng2, img, hz=min(hz, H))
        cl = gaussian_filter(_perlin(rng2, 3), rng2.uniform(0.6, 1.4))   # clouds are smooth, not speckle
        a = np.clip((cl - np.percentile(cl, rng2.uniform(35, 70))) * rng2.uniform(3.0, 9.0), 0, 1)
        a = a * rng2.uniform(0.6, 1.0)
        img = _blend(img, a, np.clip(rng2.uniform(0.72, 1.0, C), 0, 1))
        if hz < H:
            gnd = np.array([rng2.uniform(0.05, 0.3), rng2.uniform(0.1, 0.4), rng2.uniform(0.05, 0.25)])
            img[hz:] = np.clip(gnd.reshape(1, 1, C) * (0.7 + 0.6 * _perlin(rng2, 3)[hz:].reshape(-1, W, 1)), 0, 1)

    # ── 96: WEATHER / ATMOSPHERE ──
    elif lvl == 96:
        img = _base_render(rng2)
        haze = rng2.uniform(0.15, 0.5)
        img = np.clip(img * (1 - haze) + np.clip(rng2.uniform(0.5, 0.95, C), 0, 1) * haze, 0, 1)
        t = rng2.randint(0, 3)
        if t == 0:  # rain streaks
            a = rng2.uniform(-0.5, 0.5)
            for _ in range(rng2.randint(20, 60)):
                x, y, ln = rng2.uniform(0, W), rng2.uniform(0, H), rng2.uniform(2, 6)
                _bresenham(img, int(x), int(y), int(x + math.sin(a) * ln), int(y + math.cos(a) * ln), 0,
                           np.clip(img[int(y) % H, int(x) % W] * 0.5 + 0.55, 0, 1))
        elif t == 1:  # snow
            for _ in range(rng2.randint(15, 45)):
                d = _soft_disc(rng2.uniform(0, W), rng2.uniform(0, H), rng2.uniform(0.5, 1.8)) * rng2.uniform(0.4, 0.95)
                img = _blend(img, d, np.ones(3) * rng2.uniform(0.85, 1.0))
        else:  # dust / light shafts
            yy, xx = np.ogrid[:H, :W]
            ang = rng2.uniform(0, math.pi)
            shaft = np.sin((xx * math.cos(ang) + yy * math.sin(ang)) * rng2.uniform(0.2, 0.6)) * 0.5 + 0.5
            img = np.clip(img + shaft.reshape(H, W, 1) * rng2.uniform(0.05, 0.22), 0, 1)

    # ── 97: REFLECTION SYMMETRY (water / wet floor) ──
    elif lvl == 97:
        hz = rng2.randint(13, 23)
        top = np.zeros((H, W, C), np.float32)
        _sky_gradient(rng2, top, hz=hz)
        t = rng2.randint(0, 3)
        if t == 0:
            for _ in range(rng2.randint(1, 4)):
                _tree(rng2, top, np.array([rng2.uniform(0.05, 0.25), rng2.uniform(0.3, 0.7), rng2.uniform(0.05, 0.2)]),
                      rng2.randint(0, 3))
        elif t == 1:
            for _ in range(rng2.randint(1, 3)):
                _building(rng2, top, rng2.uniform(0.15, 0.7, C), rng2.randint(0, 3))
        else:
            _critter_3d(rng2, top, rng2.uniform(0.2, 0.9, C), rng2.randint(0, 5))
        img[:hz] = top[:hz]
        refl = top[:hz][::-1]
        nrows = min(H - hz, refl.shape[0])
        img[hz:hz + nrows] = refl[:nrows] * rng2.uniform(0.5, 0.85)
        if hz + nrows < H:
            img[hz + nrows:] = img[hz + nrows - 1]
        pn = _perlin(rng2, 3)
        amp = rng2.uniform(1, 4)
        for i, r in enumerate(range(hz, H)):
            img[r] = np.roll(img[r], int(round((pn[r, 0] - 0.5) * amp * (1 + i * 0.15))), axis=0)
        img[hz:] = np.clip(img[hz:] * (0.75 + 0.5 * pn[hz:].reshape(-1, W, 1)), 0, 1)

    # ── 98: CODEC ARTIFACTS (what real datasets actually carry) ──
    elif lvl == 98:
        img = _jpeg_artifacts(_base_render(rng2), rng2)
        if rng2.randint(0, 2) == 0:
            img = _add_grain(img, rng2)

    # ── 99: PHYSICALLY COHERENT SCENE (one light, one ground, depth-consistent) ──
    elif lvl == 99:
        L = _rand_light(rng2)                      # ONE light for every object in the frame
        hz = rng2.randint(9, 19)
        _sky_gradient(rng2, img, hz=hz)
        gnd = _photo_color(rng2, 0.1, 0.45)
        img[hz:] = np.clip(gnd.reshape(1, 1, C) * (0.7 + 0.7 * _perlin(rng2, 4)[hz:].reshape(-1, W, 1)), 0, 1)
        fog = img[hz].mean(axis=0)
        for y in np.sort(rng2.uniform(hz + 1, H - 1, rng2.randint(2, 6))):   # far to near: they occlude
            depth = 1 - (y - hz) / max(H - hz, 1)
            x = rng2.uniform(3, W - 3)
            size = rng2.uniform(4, 11) * (1 - 0.6 * depth)                   # farther = smaller
            img = _contact_shadow(img, x, y, max(1.5, size * 0.5), rng2)
            kind = OBJECT_KINDS[rng2.randint(0, len(OBJECT_KINDS))]
            masks = _render_parts(img, _object_parts(rng2, kind), L, rng2,
                                  pitch=rng2.uniform(-0.4, 0.3), anchor=(0, x, y), size=size)
            m = np.zeros((H, W), bool)
            for k in masks:
                m |= k
            f = depth * rng2.uniform(0.3, 0.8)                               # atmospheric perspective
            for ch in range(C):
                img[:, :, ch][m] = np.clip(img[:, :, ch][m] * (1 - f) + fog[ch] * f, 0, 1)

    # ── 100-102: FLORA IN 3D (counterpart of the flat levels 59-62) ──
    elif lvl == 100:  # round tree, seen from a random camera angle
        L = _rand_light(rng2)
        hz = rng2.randint(6, 17)
        _ground(rng2, img, hz, _photo_color(rng2, 0.1, 0.45) * np.array([0.8, 1.1, 0.7]), hi_key)
        bx, by = rng2.uniform(7, W - 7), rng2.uniform(hz + 2, H - 2)
        img = _contact_shadow(img, bx, by, rng2.uniform(3.5, 8), rng2)
        _render_parts(img, _object_parts(rng2, 'tree'), L, rng2, pitch=rng2.uniform(-0.5, 0.3),
                      anchor=(0, bx, by), size=rng2.uniform(7, 15))

    elif lvl == 101:  # conifer / palm, random camera angle
        L = _rand_light(rng2)
        hz = rng2.randint(6, 17)
        _ground(rng2, img, hz, _photo_color(rng2, 0.1, 0.45) * np.array([0.8, 1.1, 0.7]), hi_key)
        bx, by = rng2.uniform(7, W - 7), rng2.uniform(hz + 2, H - 2)
        img = _contact_shadow(img, bx, by, rng2.uniform(3, 7), rng2)
        _render_parts(img, _object_parts(rng2, 'conifer' if rng2.randint(0, 2) else 'palm'), L, rng2,
                      pitch=rng2.uniform(-0.5, 0.3), anchor=(0, bx, by), size=rng2.uniform(7, 14))

    elif lvl == 102:  # bush / flower, random camera angle (petals foreshorten with the view)
        L = _rand_light(rng2)
        hz = rng2.randint(8, 20)
        _ground(rng2, img, hz, _photo_color(rng2, 0.1, 0.45) * np.array([0.8, 1.1, 0.7]), hi_key)
        bx, by = rng2.uniform(7, W - 7), rng2.uniform(hz + 2, H - 2)
        img = _contact_shadow(img, bx, by, rng2.uniform(3, 6), rng2)
        _render_parts(img, _object_parts(rng2, 'bush' if rng2.randint(0, 2) else 'flower'), L, rng2,
                      anchor=(0, bx, by), size=rng2.uniform(6, 13))

    # ── 103: BUILDING IN 3D (counterpart of the flat level 63) ──
    elif lvl == 103:
        L = _rand_light(rng2)
        hz = rng2.randint(14, 26)
        _ground(rng2, img, hz, np.ones(3) * rng2.uniform(0.1, 0.35))
        for _ in range(rng2.randint(1, 4)):
            bw, bh = rng2.uniform(6, 14), rng2.uniform(8, 22)
            x0 = rng2.uniform(0, W - bw)
            y1 = hz + rng2.uniform(0, max(1, H - hz - 2))
            y0 = max(0, y1 - bh)
            dx = rng2.uniform(2, 6) * rng2.choice([-1, 1])
            dy = rng2.uniform(0.5, 4) * (1 if rng2.randint(0, 5) == 0 else -1)
            wall = rng2.uniform(0.15, 0.75, C)
            img = _contact_shadow(img, x0 + bw / 2, y1, bw * 0.7, rng2)   # before, or it eats the facade
            front, top, side = _box3d(img, x0, y0, x0 + bw, y1, dx, dy, wall, L)
            lit = np.clip(rng2.uniform(0.5, 1.0, C) + 0.1, 0, 1)
            gx, gy = rng2.randint(2, 5), rng2.randint(3, 6)
            for wy in range(int(y0) + 2, int(y1) - 1, gy):
                for wx in range(int(x0) + 1, int(x0 + bw) - 1, gx):
                    wm = _rect(wx, wy, wx + max(1, gx - 2), wy + max(1, gy - 2)) & front
                    _fill(img, wm, np.clip(lit * rng2.uniform(0.15, 1.0), 0, 1))

    # ── 104-106: VEHICLES IN 3D (counterpart of the flat levels 82-84) ──
    elif lvl == 104:  # car / truck in 3D
        L = _rand_light(rng2)
        gy = rng2.randint(20, 29)
        _ground(rng2, img, gy, np.ones(3) * rng2.uniform(0.1, 0.35), hi_key)
        body = rng2.uniform(0.1, 0.95, C)
        x0, x1 = rng2.uniform(1, 6), rng2.uniform(23, 30)
        by1 = gy - 2
        bh = rng2.uniform(4.5, 8)
        dx = rng2.uniform(2, 5) * (1 if rng2.randint(0, 2) else -1)
        dy = rng2.uniform(0.8, 3.5) * (1 if rng2.randint(0, 4) == 0 else -1)   # 1 in 4: low camera
        img = _contact_shadow(img, (x0 + x1) / 2, by1 + 1, (x1 - x0) * 0.5, rng2)
        _box3d(img, x0, by1 - bh, x1, by1, dx, dy, body, L)
        cw = (x1 - x0) * rng2.uniform(0.4, 0.6)
        cx0 = x0 + (x1 - x0) * rng2.uniform(0.15, 0.35)
        cf, _, _ = _box3d(img, cx0, by1 - bh - rng2.uniform(3, 5), cx0 + cw, by1 - bh, dx * 0.7, dy * 0.7,
                          np.clip(body * 0.9, 0, 1), L)
        glass = np.array([0.5, 0.68, 0.9]) * rng2.uniform(0.6, 1.3)
        _fill(img, cf & _rect(cx0 + 1, by1 - bh - rng2.uniform(2.5, 4), cx0 + cw - 1, by1 - bh - 1),
              np.clip(glass, 0, 1))
        for wx in (x0 + (x1 - x0) * 0.22, x0 + (x1 - x0) * 0.78):
            m, Nx, Ny, Nz = _sphere(wx, by1, rng2.uniform(1.8, 2.9))
            _draw_3d(img, m, Nx, Ny, Nz, L, np.array([0.12, 0.12, 0.14]))

    elif lvl == 105:  # airplane, random camera angle (from below, above, head-on, banking)
        L = _rand_light(rng2)
        _sky_gradient(rng2, img)
        _render_parts(img, _object_parts(rng2, 'plane'), L, rng2, size=rng2.uniform(6, 13),
                      anchor=(0, rng2.uniform(9, W - 9), rng2.uniform(9, H - 9)))

    elif lvl == 106:  # ship in 3D (hull + superstructure as shaded boxes on water)
        L = _rand_light(rng2)
        hz = rng2.randint(12, 21)
        _sky_gradient(rng2, img, hz=hz)
        sea = np.array([rng2.uniform(0.02, 0.16), rng2.uniform(0.1, 0.36), rng2.uniform(0.3, 0.75)])
        tex = _perlin(rng2, 3)
        img[hz:] = np.clip(sea.reshape(1, 1, C) * (0.6 + 0.8 * tex[hz:].reshape(-1, W, 1)), 0, 1)
        hull = rng2.uniform(0.1, 0.85, C)
        x0, x1 = rng2.uniform(2, 9), rng2.uniform(21, 30)
        top = hz - rng2.uniform(0, 3)
        dx = rng2.uniform(2, 5) * (1 if rng2.randint(0, 2) else -1)
        dy = -rng2.uniform(0.4, 2.5)
        _box3d(img, x0, top - rng2.uniform(3, 5), x1, top, dx, dy, hull, L)
        sw = (x1 - x0) * rng2.uniform(0.25, 0.4)
        sx = x0 + (x1 - x0) * rng2.uniform(0.2, 0.5)
        _box3d(img, sx, top - rng2.uniform(7, 10), sx + sw, top - rng2.uniform(3, 5), dx * 0.6, dy * 0.6,
               np.clip(hull * 1.35, 0, 1), L)
        _tube(img, (x0 + x1) / 2, top - 8, (x0 + x1) / 2, max(0, top - 14), rng2.uniform(0.5, 1.0),
              np.clip(hull * 0.6, 0, 1), L)
        for i in range(1, rng2.randint(3, 8)):  # reflection on the water
            r = int(top + i)
            if r < H:
                a = rng2.uniform(0.15, 0.45) / i
                img[r] = np.clip(img[r] * (1 - a) + hull * a, 0, 1)

    # ── 107-108: INSECT / SPIDER IN 3D (counterpart of the flat levels 48 and 52) ──
    elif lvl in (107, 108):  # insect / spider, random camera angle and habitat
        L = _rand_light(rng2)
        bgm = rng2.randint(0, 4)
        if bgm == 0:
            img = gaussian_filter(_1f_bg(rng2, 0.45), sigma=(1.4, 1.4, 0))     # macro bokeh
        elif bgm == 1:
            _ground(rng2, img, rng2.randint(4, 20), None, hi_key)              # on the ground
        elif bgm == 2:
            img[:] = _backdrop(rng2, True)                                     # on a bright surface
        else:
            v = _reaction_diffusion(rng2).reshape(H, W, 1)                     # on bark / leaf / lichen
            img[:] = np.clip(_photo_color(rng2, 0.05, 0.5).reshape(1, 1, C) * (0.5 + v),
                             0, 1)
        kind = 'insect' if lvl == 107 else 'spider'
        parts = _object_parts(rng2, kind)
        img = _contact_shadow(img, W / 2, H / 2 + rng2.uniform(0, 6), rng2.uniform(3, 9), rng2)
        _render_parts(img, parts, L, rng2, size=rng2.uniform(4, 16),
                      anchor=(len(parts) - (3 if lvl == 107 else 2),
                              W / 2 + rng2.uniform(-7, 7), H / 2 + rng2.uniform(-7, 7)))

    # ── 109: WIREFRAME IN 3D (counterpart of the flat hollows, levels 5 and 40) ──
    elif lvl == 109:
        L = _rand_light(rng2)
        img[:] = _backdrop(rng2, hi_key) if rng2.randint(0, 2) else \
            _perlin(rng2, 4).reshape(H, W, 1) * rng2.uniform(0.05, 0.3, C)
        t = rng2.randint(0, 3)
        if t == 0:  # cube
            v = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], np.float32)
            e = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(np.abs(v[i] - v[j])) == 2]
        elif t == 1:  # pyramid
            v = np.array([[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1], [0, -1.2, 0]], np.float32)
            e = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
        else:  # ground grid
            g = rng2.randint(3, 5)
            v = np.array([[i / (g - 1) * 2 - 1, 0.8, j / (g - 1) * 2 - 1] for i in range(g) for j in range(g)],
                         np.float32)
            e = [(i * g + j, i * g + j + 1) for i in range(g) for j in range(g - 1)] + \
                [(i * g + j, (i + 1) * g + j) for i in range(g - 1) for j in range(g)]
        p = _rot3(v * rng2.uniform(0.7, 1.2), rng2.uniform(0, 6.28), rng2.uniform(-0.7, 0.7))
        u, vv, sc = _project(p, rng2.uniform(2.6, 4.5), rng2.uniform(9, 34))   # from far frame to close-up
        col = rng2.uniform(0.3, 1.0, C)
        for i, j in sorted(e, key=lambda ij: -(p[ij[0], 2] + p[ij[1], 2])):   # far edges first
            r = max(0.45, 0.9 * (sc[i] + sc[j]) / 2)                          # thicker when near
            _tube(img, u[i], vv[i], u[j], vv[j], r, np.clip(col * (0.55 + 0.5 * sc[i]), 0, 1), L)

    # ── 110: TEXTURE ON A 3D SURFACE (pattern follows the geometry, not the frame) ──
    elif lvl == 110:
        L = _rand_light(rng2)
        hz = rng2.randint(8, 19)
        _sky_gradient(rng2, img, hz=hz)
        yy, xx = np.ogrid[:H, :W]
        v = np.maximum(yy - hz, 0.4) / H
        z = 1.0 / (v + 1e-3)
        u = (xx - W / 2) * z * rng2.uniform(0.02, 0.09)
        pat = rng2.randint(0, 3)
        su, sv = rng2.uniform(0.6, 2.2), rng2.uniform(0.4, 1.6)
        if pat == 0:
            t = ((np.floor(u / su) + np.floor(z * 0.12 / sv)) % 2).astype(np.float32)
        elif pat == 1:
            t = (np.sin(u / su * 3) * 0.5 + 0.5).astype(np.float32)
        else:
            t = ((np.sin(u / su * 3) * np.sin(z * 0.12 / sv * 3)) > 0).astype(np.float32)
        c1, c2 = rng2.uniform(0.1, 0.9, C), rng2.uniform(0.1, 0.9, C)
        t = np.broadcast_to(t, (H, W)).reshape(H, W, 1)
        floor = c1.reshape(1, 1, C) * t + c2.reshape(1, 1, C) * (1 - t)
        haze = np.clip(1 - np.broadcast_to(v, (H, W)).reshape(H, W, 1) * 2.5, 0, 1)
        horizon = img[max(hz - 1, 0)].mean(axis=0).reshape(1, 1, C)
        img[hz:] = np.clip((floor * haze + horizon * (1 - haze))[hz:], 0, 1)
        # same pattern wrapped on a lit sphere: texture and shading share the geometry
        cx, cy, R = rng2.uniform(8, W - 8), rng2.uniform(6, hz + 8), rng2.uniform(5, 11)
        img = _contact_shadow(img, cx, cy + R, R * 1.2, rng2)
        m, Nx, Ny, Nz = _sphere(cx, cy, R)
        uu = np.arctan2(Nx, Nz + 1e-6) / math.pi
        vvw = np.arcsin(np.clip(Ny, -1, 1)) / (math.pi / 2)
        ku, kv = rng2.uniform(2, 6), rng2.uniform(2, 6)
        tt = ((np.floor(uu * ku) + np.floor(vvw * kv)) % 2).astype(np.float32)
        lit = _phong(Nx, Ny, Nz, L)
        for ch in range(C):
            img[:, :, ch][m] = np.clip((c1[ch] * tt + c2[ch] * (1 - tt))[m] * lit[m], 0, 1)

    # ── 111: PARTICLE FIELDS IN DEPTH (swarm, dust, bubbles, snow in a beam) ──
    elif lvl == 111:
        img[:] = _backdrop(rng2, hi_key) if rng2.randint(0, 3) == 0 else \
            _perlin(rng2, 3).reshape(H, W, 1) * _photo_color(rng2, 0.03, 0.3).reshape(1, 1, C)
        dense = rng2.randint(0, 2) == 0
        n = rng2.randint(60, 260) if dense else rng2.randint(14, 60)
        p = rng2.uniform(-1, 1, (n, 3)).astype(np.float32)
        if rng2.randint(0, 2) == 0:                     # clustered, not uniform: swarms clump
            nc = rng2.randint(1, 5)
            ctr = rng2.uniform(-0.8, 0.8, (nc, 3)).astype(np.float32)
            p = ctr[rng2.randint(0, nc, n)] + p * rng2.uniform(0.12, 0.45)
        p = _rot3(p, rng2.uniform(0, 6.28), rng2.uniform(-0.7, 0.7))
        dist = rng2.uniform(2.4, 4.2)
        u, v, sc = _project(p, dist, rng2.uniform(10, 22))
        col = _photo_color(rng2, 0.35, 1.0)
        base = rng2.uniform(0.35, 1.6) * (0.6 if dense else 1.0)
        varied = rng2.randint(0, 2) == 0
        for i in np.argsort(-p[:, 2]):                  # painter's algorithm: far particles first
            b = float(np.clip(sc[i], 0.2, 1.6))
            R = max(0.35, b * base * (rng2.uniform(0.4, 2.2) if varied else 1.0))
            c = col * b if not varied else np.clip(col * b * rng2.uniform(0.6, 1.4), 0, 1)
            img = _blend(img, _soft_disc(u[i], v[i], R) * min(1.0, b), c)
        if rng2.randint(0, 2) == 0:
            _depth_fog(img, rng2)

    # ── 112: EMISSIVE POINT SOURCES (night sky, city lights, fluorescence) ──
    elif lvl == 112:
        img[:] = _photo_color(rng2, 0.01, 0.12).reshape(1, 1, C) * (0.4 + 0.6 * _perlin(rng2, 3).reshape(H, W, 1))
        if rng2.randint(0, 3) == 0:      # nebulosity / airglow behind the field
            neb = _curl_warp(rng2, _perlin(rng2, 3))
            img = np.clip(img + neb.reshape(H, W, 1) * _photo_color(rng2, 0.1, 0.5).reshape(1, 1, C) * 0.6, 0, 1)
        spikes = rng2.randint(0, 2) == 0
        for _ in range(rng2.randint(6, 70)):
            flux = rng2.uniform(0.03, 1.0) ** 2.2      # magnitude distribution: many faint, few bright
            T = rng2.uniform(2200, 11000)
            img = _point_source(img, rng2.uniform(0, W), rng2.uniform(0, H), flux * 2.2, _blackbody(T),
                                core=rng2.uniform(0.45, 0.9), halo=rng2.uniform(1.5, 5),
                                spikes=spikes and flux > 0.4)
        img = _poisson(np.clip(img, 0, 1), rng2, rng2.uniform(20, 400))

    # ── 113: REACTION-DIFFUSION (skins, coral, fungus, lichen, chemistry) ──
    elif lvl == 113:
        v = _reaction_diffusion(rng2)
        c1, c2 = _photo_color(rng2, 0.05, 0.6), _photo_color(rng2, 0.3, 0.98)
        t = v.reshape(H, W, 1)
        if rng2.randint(0, 2) == 0:
            t = (v > rng2.uniform(0.35, 0.65)).astype(np.float32).reshape(H, W, 1)   # hard spots
        img = c1.reshape(1, 1, C) * (1 - t) + c2.reshape(1, 1, C) * t
        if rng2.randint(0, 2) == 0:      # wrapped on a lit body instead of flat
            L = _rand_light(rng2)
            m, Nx, Ny, Nz = _organic_blob(rng2, W / 2 + rng2.uniform(-4, 4), H / 2 + rng2.uniform(-4, 4),
                                          rng2.uniform(8, 15), rng2.uniform(8, 15))
            lit = _phong(Nx, Ny, Nz, L)
            bg = _backdrop(rng2, hi_key)
            body = img.copy()
            img = bg
            for ch in range(C):
                img[:, :, ch][m] = np.clip(body[:, :, ch][m] * lit[m], 0, 1)

    # ── 114: TRANSMISSION (backlit: microscopy, leaves, jellyfish, x-ray) ──
    elif lvl == 114:
        back = _photo_color(rng2, 0.55, 1.0)
        img[:] = back.reshape(1, 1, C) * (0.85 + 0.15 * _perlin(rng2, 2).reshape(H, W, 1))
        thick = np.zeros((H, W), np.float32)
        for _ in range(rng2.randint(1, 7)):           # overlapping bodies accumulate optical depth
            cx, cy = rng2.uniform(2, W - 2), rng2.uniform(2, H - 2)
            r = rng2.uniform(3, 12)
            m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, r, r * rng2.uniform(0.5, 1.8))
            thick += np.where(m, Nz * rng2.uniform(0.6, 2.2), 0)
        if rng2.randint(0, 3) == 0:
            thick += _reaction_diffusion(rng2) * rng2.uniform(0.3, 1.2)   # internal structure
        absorb = _photo_color(rng2, 0.2, 1.0) * rng2.uniform(0.6, 3.0)    # Beer-Lambert, per channel
        img = np.clip(img * np.exp(-thick.reshape(H, W, 1) * absorb.reshape(1, 1, C)), 0, 1)
        if rng2.randint(0, 4) == 0:                                        # x-ray / negative look
            img = np.clip(1.0 - img.mean(axis=-1, keepdims=True), 0, 1).repeat(C, axis=-1)
        img = _poisson(img, rng2, rng2.uniform(120, 900))

    # ── 115: TURBULENCE & SPIRAL (smoke, fluids, storms, galaxies) ──
    elif lvl == 115:
        if rng2.randint(0, 2) == 0:      # advected turbulence
            base = _perlin(rng2, rng2.randint(3, 5))
            f = _curl_warp(rng2, base, rng2.uniform(3, 9))
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            c1, c2 = _photo_color(rng2, 0.02, 0.4), _photo_color(rng2, 0.4, 1.0)
            img = c1.reshape(1, 1, C) + (c2 - c1).reshape(1, 1, C) * (f ** rng2.uniform(0.6, 2.2)).reshape(H, W, 1)
        else:                            # logarithmic spiral: galaxy, shell, whirlpool
            cx, cy = W / 2 + rng2.uniform(-4, 4), H / 2 + rng2.uniform(-4, 4)
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            dx, dy = xx - cx, yy - cy
            ar = rng2.uniform(0.35, 1.0)                       # inclination
            dyr = dy / ar
            r = np.sqrt(dx ** 2 + dyr ** 2) + 0.5
            th = np.arctan2(dyr, dx)
            arms = rng2.randint(2, 5)
            pitch = rng2.uniform(1.2, 3.5)
            band = 0.5 + 0.5 * np.cos(arms * th - pitch * np.log(r))
            disc = np.exp(-r / rng2.uniform(4, 10))
            f = np.clip(band ** rng2.uniform(1.5, 4) * disc + disc * 0.6, 0, 1)
            f = _curl_warp(rng2, f, rng2.uniform(0.5, 2.5))
            img = f.reshape(H, W, 1) * _blackbody(rng2.uniform(3500, 9000)).reshape(1, 1, C)
            img = np.clip(img + np.exp(-r / rng2.uniform(0.8, 2.2)).reshape(H, W, 1) * 0.9, 0, 1)
            for _ in range(rng2.randint(0, 30)):
                img = _point_source(img, rng2.uniform(0, W), rng2.uniform(0, H),
                                    rng2.uniform(0.05, 0.6), _blackbody(rng2.uniform(3000, 10000)), core=0.5)
            img = _poisson(np.clip(img, 0, 1), rng2, rng2.uniform(30, 300))

    # ── 116: NON-LINEAR OPTICS (droplet, glass ball, fisheye, gravitational lens) ──
    elif lvl == 116:
        base = _base_render(rng2)
        cx, cy = rng2.uniform(8, W - 8), rng2.uniform(8, H - 8)
        mode = rng2.randint(0, 3)
        if mode == 0:      # droplet / glass ball: magnifies inside a radius
            R = rng2.uniform(6, 14)
            kk = rng2.uniform(0.25, 0.8)
            img = _radial_warp(base, cx, cy, lambda r: np.where(r < R, r * kk, r))
            edge = np.abs(np.sqrt((np.mgrid[:H, :W][1] - cx) ** 2 + (np.mgrid[:H, :W][0] - cy) ** 2) - R) < 1.2
            img = np.clip(img + edge.reshape(H, W, 1) * rng2.uniform(0.1, 0.4), 0, 1)
        elif mode == 1:    # fisheye / barrel
            kk = rng2.uniform(0.0015, 0.006) * (1 if rng2.randint(0, 2) else -1)
            img = _radial_warp(base, W / 2, H / 2, lambda r: r * (1 + kk * r ** 2))
        else:              # black hole: deflection ~ 1/r, dark disk, photon ring, hot disc
            rs = rng2.uniform(2.5, 6.5)
            img = _radial_warp(base, cx, cy, lambda r: r + rs * rs * 2.2 / np.maximum(r, 0.8))
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            if rng2.randint(0, 2) == 0:                        # accretion disc, doppler-brightened
                disc = np.exp(-np.abs(np.abs(yy - cy) - 0.0) / rng2.uniform(0.8, 2.0)) * \
                       np.exp(-np.abs(r - rs * rng2.uniform(1.8, 3.2)) / rng2.uniform(1.0, 2.5))
                side = 0.5 + 0.5 * np.sign(xx - cx) * rng2.uniform(0.3, 0.9)
                img = np.clip(img + (disc * side).reshape(H, W, 1) *
                              _blackbody(rng2.uniform(6000, 14000)).reshape(1, 1, C) * 1.6, 0, 1)
            ring = np.exp(-np.abs(r - rs * 1.5) / 0.7)
            img = np.clip(img + ring.reshape(H, W, 1) * rng2.uniform(0.3, 0.9), 0, 1)
            img[r < rs] = 0.0

    # ── 117: HOT EMISSIVE MATTER (lava, plasma, filament, ember) ──
    elif lvl == 117:
        heat = _curl_warp(rng2, gaussian_filter(_perlin(rng2, rng2.randint(2, 4)), rng2.uniform(0.6, 1.6)),
                          rng2.uniform(1, 6))
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8) ** 1.0
        heat = np.clip(heat ** rng2.uniform(0.8, 3.0), 0, 1)
        Tlo, Thi = rng2.uniform(1000, 2500), rng2.uniform(3000, 12000)
        img = np.zeros((H, W, C), np.float32)
        for i in range(8):                       # temperature ramp through the Planck locus
            lo, hi = i / 8, (i + 1) / 8
            m = (heat >= lo) & (heat < hi)
            img[m] = _blackbody(Tlo + (Thi - Tlo) * (i / 7.0)) * (0.15 + 1.5 * (i / 7.0))
        img = np.clip(img * (0.4 + 0.9 * heat.reshape(H, W, 1)), 0, 1)
        glow = gaussian_filter(img, sigma=(rng2.uniform(1.0, 2.5),) * 2 + (0,))
        img = np.clip(img + glow * rng2.uniform(0.4, 1.0), 0, 1)     # bloom around blown highlights

    # ── 118: NON-OPTICAL IMAGING (thermal, x-ray, electron microscope, depth) ──
    elif lvl == 118:
        field = _perlin(rng2, rng2.randint(2, 5))
        for _ in range(rng2.randint(1, 5)):
            cx, cy = rng2.uniform(2, W - 2), rng2.uniform(2, H - 2)
            r = rng2.uniform(3, 12)
            m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, r, r * rng2.uniform(0.5, 1.6))
            field = np.where(m, np.clip(field * 0.3 + Nz * rng2.uniform(0.5, 1.0), 0, 1), field * 0.85)
        field = (field - field.min()) / (field.max() - field.min() + 1e-8)
        mode = rng2.randint(0, 4)
        if mode == 0:      # thermal / infrared false color
            img = _lut(field, [np.zeros(3), np.array([0.25, 0.0, 0.4]), np.array([0.85, 0.1, 0.15]),
                               np.array([1.0, 0.65, 0.0]), np.ones(3)])
        elif mode == 1:    # depth map
            img = _lut(field, [np.array([0.15, 0.0, 0.35]), np.array([0.0, 0.6, 0.75]),
                               np.array([0.8, 0.95, 0.2]), np.ones(3)]) if rng2.randint(0, 2) \
                else np.repeat(field.reshape(H, W, 1), C, axis=2)
        elif mode == 2:    # electron microscope: grey, no color, hard directional shading
            g = np.clip(field * 1.3 - 0.15, 0, 1)
            e = _edge_sobel(g)
            g = np.clip(g * (0.7 + 0.6 * np.roll(g, 1, axis=1)) + e * 0.5, 0, 1)
            img = np.repeat(g.reshape(H, W, 1), C, axis=2)
        else:              # radiograph: transmission, inverted, high dynamic range
            img = np.repeat(np.clip(np.exp(-field * rng2.uniform(1.5, 4.5)), 0, 1).reshape(H, W, 1), C, axis=2)
        img = _poisson(img, rng2, rng2.uniform(60, 900))

    # ── 119: COHERENT LIGHT (speckle, interference, diffraction, moire) ──
    elif lvl == 119:
        mode = rng2.randint(0, 4)
        col = _photo_color(rng2, 0.3, 1.0)
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        if mode == 0:      # laser speckle
            f = _speckle(rng2) ** rng2.uniform(0.4, 1.2)
        elif mode == 1:    # two-source interference fringes
            x1, y1 = rng2.uniform(-20, W + 20), rng2.uniform(-20, H + 20)
            x2, y2 = rng2.uniform(-20, W + 20), rng2.uniform(-20, H + 20)
            d = np.sqrt((xx - x1) ** 2 + (yy - y1) ** 2) - np.sqrt((xx - x2) ** 2 + (yy - y2) ** 2)
            f = 0.5 + 0.5 * np.cos(d * rng2.uniform(0.6, 3.0))
        elif mode == 2:    # Airy rings around a point aperture
            r = np.sqrt((xx - rng2.uniform(8, W - 8)) ** 2 + (yy - rng2.uniform(8, H - 8)) ** 2) + 1e-3
            k = rng2.uniform(0.8, 2.5)
            f = (np.sin(k * r) / (k * r)) ** 2
            f = f / (f.max() + 1e-8)
        else:              # moire between two rotated gratings
            a1, a2 = rng2.uniform(0, math.pi), rng2.uniform(0, math.pi)
            k1, k2 = rng2.uniform(1.0, 3.0), rng2.uniform(1.0, 3.0)
            g1 = 0.5 + 0.5 * np.cos((xx * math.cos(a1) + yy * math.sin(a1)) * k1)
            g2 = 0.5 + 0.5 * np.cos((xx * math.cos(a2) + yy * math.sin(a2)) * k2)
            f = g1 * g2
        # Airy/speckle fields concentrate almost all their energy into a handful of pixels —
        # left as-is, ~1 in 4 samples rendered as a near-blank frame with a single bright dot.
        # Stretching to the field's own 2nd-98th percentile makes the (real, physical) faint
        # fringes around that peak visible instead of clipping them to black.
        lo, hi = np.percentile(f, 2), np.percentile(f, 98)
        f = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)
        img = f.reshape(H, W, 1) * col.reshape(1, 1, C)
        if rng2.randint(0, 3) == 0:      # chromatic dispersion of the fringes
            for ch in range(C):
                img[:, :, ch] = map_coordinates(img[:, :, ch], np.mgrid[:H, :W] * (1 + (ch - 1) * 0.02),
                                                order=1, mode='reflect')
        img = np.clip(img, 0, 1)

    # ── 120: CLOSED ORGANIC SHAPE (outline, filled, 3D volume, or 3D tube loop) ──
    elif lvl == 120:
        img[:] = _backdrop(rng2, hi_key)
        col = _photo_color(rng2, 0.1, 0.95)
        mode = rng2.randint(0, 4)
        if mode == 3:      # a closed loop must be a loop: Fourier curve as a 3D tube
            L = _rand_light(rng2)
            p = _closed_curve(rng2, n=rng2.randint(28, 52), dim=3,
                              harmonics=rng2.randint(2, 6), wobble=rng2.uniform(0.25, 0.75))
            p = _rot3(p, rng2.uniform(0, 6.28), rng2.uniform(-1.2, 1.2))
            dist = rng2.uniform(2.6, 4.5)
            u, v, sc = _project(p, dist, rng2.uniform(9, 17) * dist)
            r = rng2.uniform(0.7, 2.4)
            for i in np.argsort(-(p[:, 2] + np.roll(p[:, 2], -1)) / 2):
                j = (i + 1) % len(p)
                _tube(img, u[i], v[i], u[j], v[j], max(0.45, r * (sc[i] + sc[j]) / 2),
                      np.clip(col * (0.6 + 0.5 * sc[i]), 0, 1), L)
        else:
            segs = _skeleton(rng2, dim=2 if mode < 2 else 3)
            scale = rng2.uniform(6, 17)
            cx, cy = W / 2 + rng2.uniform(-4, 4), H / 2 + rng2.uniform(-4, 4)
            f = _meta_field(segs, scale, cx, cy, rng2.uniform(0.6, 8.0))
            inside = f < 0
            if not inside.any():
                inside = f < f.min() * 0.5
            if mode == 0:      # outline only, constant width, hollow interior
                th = rng2.uniform(0.5, 2.2)
                img = _blend(img, np.clip(th - np.abs(f) + 0.5, 0, 1), col)
            elif mode == 1:    # flat filled silhouette
                img = _blend(img, np.clip(0.5 - f, 0, 1), col)
            else:              # solid 3D volume: depth from the field, smooth closed surface
                L = _rand_light(rng2)
                dep = np.clip(-f / max(scale * rng2.uniform(0.25, 0.6), 1e-3), 0, 1)
                Nz = np.sqrt(np.clip(1 - (1 - dep) ** 2, 0, 1))
                gy, gx = np.gradient(gaussian_filter(dep, 0.8))
                nrm = np.sqrt(gx ** 2 + gy ** 2 + 1e-6)
                _draw_3d(img, inside, -gx / nrm * (1 - Nz), -gy / nrm * (1 - Nz), Nz, L, col)
                img = _contact_shadow(img, cx, cy + scale * 0.9, scale * 0.8, rng2)

    # ── 121: TEXT / SYMBOLS (signage, plates, screens, documents, inscriptions) ──
    elif lvl == 121:
        kind = rng2.randint(0, 5)
        layout = rng2.randint(0, 3)
        if layout == 0:                     # sign / screen: ink on a solid or gradient plate
            _sky_gradient(rng2, img, sun=False) if rng2.randint(0, 3) == 0 else \
                img.__setitem__((slice(None),), _photo_color(rng2, 0.15, 0.95).reshape(1, 1, C))
            ink = _photo_color(rng2, 0.02, 0.85)
            if abs(float(img.mean() - ink.mean())) < 0.25:      # keep ink readable against plate
                ink = np.clip(1 - img.mean() + rng2.uniform(-0.1, 0.1), 0, 1) * np.ones(3)
        elif layout == 1:                   # carved / embossed: inscription on stone/metal
            base = rng2.randint(0, 3)
            img[:] = (_marble(rng2) if base == 0 else _wood(rng2) if base == 1
                      else _brushed(rng2)).reshape(H, W, 1) * _photo_color(rng2, 0.15, 0.7).reshape(1, 1, C)
            ink = None                      # drawn as a shadow/highlight below
        else:                               # dense document / code: many small glyphs
            img[:] = _photo_color(rng2, 0.55, 1.0).reshape(1, 1, C) * (0.9 + 0.1 * _perlin(rng2, 3).reshape(H, W, 1))
            ink = _photo_color(rng2, 0.02, 0.3)
        cell = rng2.choice([7, 8, 10, 13, 16, 22]) if layout != 2 else rng2.choice([5, 6, 7])
        cell = int(cell)
        jitter = cell * rng2.uniform(0.0, 0.12)
        y = rng2.uniform(0, cell * 0.4)
        while y < H - cell * 0.5:
            x = rng2.uniform(0, cell * 0.4)
            while x < W - cell * 0.5:
                if rng2.rand() < (0.9 if layout == 2 else 0.8):
                    gm = _glyph(rng2, x + rng2.uniform(-jitter, jitter), y + rng2.uniform(-jitter, jitter),
                                cell, kind if rng2.rand() < 0.85 else rng2.randint(0, 5))
                    if ink is None:                     # carved: dark groove + light lip
                        for ch in range(C):
                            img[:, :, ch][gm] = np.clip(img[:, :, ch][gm] * rng2.uniform(0.3, 0.6), 0, 1)
                        lip = gm & ~np.roll(gm, 1, 0)
                        for ch in range(C):
                            img[:, :, ch][lip] = np.clip(img[:, :, ch][lip] + 0.3, 0, 1)
                    else:
                        for ch in range(C):
                            img[:, :, ch][gm] = ink[ch]
                x += cell + rng2.uniform(0, cell * 0.3)
            y += cell + rng2.uniform(0, cell * 0.25)
        if rng2.randint(0, 2) == 0:
            img = gaussian_filter(img, sigma=(rng2.uniform(0.3, 0.8),) * 2 + (0,))
        if rng2.randint(0, 3) == 0:
            img = _add_grain(img, rng2)

    # ── 122: KNOWN SYMBOLS + AUGMENTATION (real glyphs, distorted for diversity) ──
    elif lvl == 122:
        bg = _photo_color(rng2, 0.1, 0.95)
        img[:] = bg.reshape(1, 1, C)
        if rng2.randint(0, 3) == 0:
            img[:] = np.clip(img * (0.6 + 0.8 * _perlin(rng2, 3).reshape(H, W, 1)), 0, 1)
        ink = _photo_color(rng2, 0.02, 0.95)
        if abs(float(bg.mean() - ink.mean())) < 0.3:              # keep it readable
            ink = np.clip(1 - bg, 0, 1)
        single = rng2.randint(0, 2) == 0                          # one big char, or a short string
        n = 1 if single else rng2.randint(2, 6)
        cell = rng2.uniform(14, 30) if single else rng2.uniform(7, 15)
        total = n * cell
        ox = (W - total) / 2 + rng2.uniform(-4, 4)
        oy = (H - cell) / 2 + rng2.uniform(-5, 5)
        # one shared affine warp for the whole string: rotation + shear + perspective-ish scale
        ang = rng2.uniform(-0.5, 0.5)
        shear = rng2.uniform(-0.4, 0.4)
        ca, sa = math.cos(ang), math.sin(ang)
        cxp, cyp = W / 2, H / 2
        acc = np.zeros((H, W), np.float32)
        for i in range(n):
            ch = _FONT_KEYS[rng2.randint(0, len(_FONT_KEYS))]
            gm = _stamp_glyph(ch, int(round(cell)))
            gh, gw = gm.shape
            gy0, gx0 = oy, ox + i * cell
            ys, xs = np.where(gm > 0.5)
            px = gx0 + xs + 0.5
            py = gy0 + ys + 0.5
            px = px + shear * (py - cyp)                          # shear
            dx, dy = px - cxp, py - cyp                           # rotate about center
            rx = cxp + dx * ca - dy * sa
            ry = cyp + dx * sa + dy * ca
            ii = np.clip(np.round(ry).astype(int), 0, H - 1)
            jj = np.clip(np.round(rx).astype(int), 0, W - 1)
            acc[ii, jj] = 1.0
        thick = rng2.randint(0, 2)
        if thick:                                                # stroke dilation
            acc = np.clip(acc + np.roll(acc, 1, 0) + np.roll(acc, 1, 1), 0, 1)
        acc = gaussian_filter(acc, rng2.uniform(0.3, 1.1))       # elastic-ish softening
        if rng2.randint(0, 3) == 0:                              # elastic warp
            acc = _curl_warp(rng2, acc, rng2.uniform(0.5, 2.5))
        for ch in range(C):
            img[:, :, ch] = np.clip(img[:, :, ch] * (1 - acc) + ink[ch] * acc, 0, 1)
        aug = rng2.randint(0, 5)
        if aug == 0:
            img = _add_grain(img, rng2)
        elif aug == 1:
            img = _chromatic_ab(img, rng2)
        elif aug == 2:                                           # partial occlusion
            ox2, oy2 = rng2.randint(0, W), rng2.randint(0, H)
            r = rng2.uniform(3, 9)
            m, _, _, _ = _organic_blob(rng2, ox2, oy2, r, r)
            _fill(img, m, _photo_color(rng2, 0.1, 0.9))
        elif aug == 3:                                           # backlit / low contrast
            img = np.clip(img * rng2.uniform(0.5, 0.9) + rng2.uniform(0.05, 0.3), 0, 1)
        if rng2.randint(0, 2) == 0:
            img = gaussian_filter(img, sigma=(rng2.uniform(0.3, 0.9),) * 2 + (0,))

    # ── 123: SINGLE CLEAN SYMBOL (glyph atlas across scripts, minimal distortion) ──
    elif lvl == 123:
        bg = _photo_color(rng2, 0.08, 0.97)
        img[:] = bg.reshape(1, 1, C)
        ink = _photo_color(rng2, 0.02, 0.97)
        if abs(float(bg.mean() - ink.mean())) < 0.35:            # guarantee readable contrast
            ink = np.clip(1 - bg, 0, 1)
        cell = int(rng2.uniform(20, 31))                          # one big glyph filling the frame
        gx = (W - cell) / 2 + rng2.uniform(-2, 2)
        gy = (H - cell) / 2 + rng2.uniform(-2, 2)
        src = rng2.randint(0, 10)
        if src < 6:              # real bitmap glyph: latin, digits, greek, runic, punctuation, math
            ch = _FONT_KEYS[rng2.randint(0, len(_FONT_KEYS))]
            gm = np.zeros((H, W), np.float32)
            stamp = _stamp_glyph(ch, cell)
            y0, x0 = int(round(gy)), int(round(gx))
            sh, sw = stamp.shape
            ys, xs = np.where(stamp > 0.5)
            yy = np.clip(y0 + ys, 0, H - 1); xx = np.clip(x0 + xs, 0, W - 1)
            gm[yy, xx] = 1.0
        elif src < 8:            # CJK-like dense character (real hanzi are unresolvable at 32px anyway)
            gm = _glyph(rng2, gx, gy, cell, 1).astype(np.float32)
        elif src < 9:            # cuneiform
            gm = _glyph(rng2, gx, gy, cell, 2).astype(np.float32)
        else:                    # devanagari-like: a top bar with hanging strokes
            gm = _glyph(rng2, gx, gy, cell, 0).astype(np.float32)
            bar = _rect(gx, gy, gx + cell, gy + max(1, cell * 0.14))
            gm[bar] = 1.0
        # minimal augmentation only: tiny rotation, optional 1px thickening, light AA
        ang = rng2.uniform(-0.18, 0.18)
        if abs(ang) > 0.02:
            ca, sa = math.cos(ang), math.sin(ang)
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            dx, dy = xx - W / 2, yy - H / 2
            gm = map_coordinates(gm, [H / 2 + (-sa * dx + ca * dy), W / 2 + (ca * dx + sa * dy)],
                                 order=1, mode='constant')
        if rng2.randint(0, 3) == 0:
            gm = np.clip(gm + np.roll(gm, 1, 0) + np.roll(gm, 1, 1), 0, 1)   # bold
        gm = gaussian_filter(gm, rng2.uniform(0.3, 0.7))         # gentle anti-aliasing, nothing more
        for ch2 in range(C):
            img[:, :, ch2] = np.clip(img[:, :, ch2] * (1 - gm) + ink[ch2] * gm, 0, 1)

    # ── 124: INDOOR SCENE (room + furniture — the one common real-photo setting we didn't have) ──
    elif lvl == 124:
        L = _rand_light(rng2)
        wall = _photo_color(rng2, 0.25, 0.85)
        floor = _photo_color(rng2, 0.1, 0.55)
        hz = rng2.randint(H // 4, H // 2)             # wall/floor line
        for r in range(H):
            img[r, :] = wall if r < hz else floor
        img[:hz] = np.clip(img[:hz] * (0.85 + 0.3 * (1 - np.arange(hz).reshape(-1, 1, 1) / max(hz, 1))), 0, 1)
        if rng2.randint(0, 3) == 0:                    # a window: sky-toned rect on the wall
            wx0, wx1 = sorted(rng2.uniform(2, W - 2, 2))
            wy0, wy1 = hz * rng2.uniform(0.1, 0.3), hz * rng2.uniform(0.6, 0.9)
            sky = np.array([rng2.uniform(0.5, 0.8), rng2.uniform(0.6, 0.85), rng2.uniform(0.7, 1.0)])
            _fill(img, _rect(wx0, wy0, wx1, wy1), sky)
        for _ in range(rng2.randint(1, 4)):            # furniture: axonometric boxes on the floor
            bw = rng2.uniform(4, 13)
            typ = rng2.randint(0, 3)                   # 0 low+wide (table), 1 tall+narrow (cabinet), 2 cube (stool)
            bh = bw * (0.35 if typ == 0 else 2.2 if typ == 1 else 0.9)
            x0 = rng2.uniform(0, max(1.0, W - bw))
            y1 = rng2.uniform(hz + bh * 0.3, H - 1)
            y0 = max(hz - 1.0, y1 - bh)
            dx = rng2.uniform(1.5, 4) * (1 if rng2.randint(0, 2) else -1)
            dy = -rng2.uniform(0.6, 2.5)
            wood = rng2.rand() < 0.6
            color = np.array([rng2.uniform(0.3, 0.55), rng2.uniform(0.18, 0.35), rng2.uniform(0.08, 0.22)]) \
                if wood else _photo_color(rng2, 0.1, 0.9)
            img = _contact_shadow(img, x0 + bw / 2, y1, bw * 0.5, rng2)
            _box3d(img, x0, y0, x0 + bw, y1, dx, dy, color, L)
        if rng2.randint(0, 2) == 0:                    # rug / mat patch on the floor
            rx0, rx1 = sorted(rng2.uniform(2, W - 2, 2))
            ry0, ry1 = sorted(rng2.uniform(hz + 1, H - 1, 2))
            _fill(img, _rect(rx0, ry0, rx1, ry1), _photo_color(rng2, 0.15, 0.75))

    # ── 125: FULL-BODY HUMAN FIGURE(S) — a dedicated level; before this, 'human' only
    # existed as 1-of-5 inside the generic creature block and never in group scenes ──
    elif lvl == 125:
        L = _rand_light(rng2)
        hz = rng2.randint(10, 24)
        _ground(rng2, img, hz, None, hi_key)
        crowd = rng2.randint(0, 4) == 0                 # occasionally a real multitude, not just 1-3
        n = rng2.randint(8, 21) if crowd else rng2.randint(1, 4)
        for i in range(n):
            bx = rng2.uniform(2, W - 2)
            if crowd:
                by = rng2.uniform(hz + 1, H - 2)
                depth = (by - hz) / max(H - hz, 1)       # 0 at the horizon (far), 1 at the camera (near)
                size = rng2.uniform(2.5, 6) * (0.5 + 0.5 * depth)
            else:
                by = rng2.uniform(max(hz + 3, H * 0.4), H - 2)
                size = rng2.uniform(7, 15) * (1 - 0.15 * i)  # later ones slightly smaller: depth cue
            img = _contact_shadow(img, bx, by, size * 0.18, rng2)
            _render_parts(img, _object_parts(rng2, 'human'), L, rng2,
                          pitch=rng2.uniform(-0.2, 0.2), anchor=(0, bx, by), size=size)

    # ── 126: UNDERWATER SCENE (depth-tinted water, light shafts, seaweed, fish, bubbles —
    # before this, "water" was only a flat backdrop, never an immersive scene) ──
    elif lvl == 126:
        top = np.array([rng2.uniform(0.05, 0.25), rng2.uniform(0.35, 0.65), rng2.uniform(0.45, 0.75)])
        bot = top * rng2.uniform(0.15, 0.4)
        for r in range(H):
            img[r, :] = top + (bot - top) * (r / H)
        if rng2.randint(0, 2) == 0:                     # god rays from the surface
            yy, xx = np.ogrid[:H, :W]
            ang = rng2.uniform(-0.3, 0.3)
            shaft = np.clip(np.sin((xx * math.cos(ang) + yy * math.sin(ang)) * rng2.uniform(0.15, 0.4)) * 0.5 + 0.5, 0, 1)
            fall = np.clip(1 - np.arange(H) / H, 0, 1).reshape(H, 1)
            img = np.clip(img + (shaft * fall).reshape(H, W, 1) * rng2.uniform(0.08, 0.25), 0, 1)
        if rng2.randint(0, 2) == 0:                     # seaweed fronds swaying up from the floor
            for _ in range(rng2.randint(2, 6)):
                x = rng2.uniform(2, W - 2)
                ht = rng2.uniform(6, 18)
                col = np.array([rng2.uniform(0.05, 0.25), rng2.uniform(0.3, 0.6), rng2.uniform(0.05, 0.2)])
                phase = rng2.uniform(0, 6)
                for i in range(int(ht)):
                    y = H - 1 - i
                    if y < 0:
                        break
                    x += math.sin(i * 0.5 + phase) * 0.6
                    if 0 <= int(x) < W:
                        img[y, int(x)] = col * rng2.uniform(0.7, 1.1)
        for _ in range(rng2.randint(1, 3)):              # a fish or two
            _critter_3d(rng2, img, _photo_color(rng2, 0.2, 0.95), 3)
        if rng2.randint(0, 2) == 0:                      # rising bubbles
            for _ in range(rng2.randint(5, 20)):
                img = _blend(img, _soft_disc(rng2.uniform(0, W), rng2.uniform(0, H), rng2.uniform(0.4, 1.3))
                             * rng2.uniform(0.2, 0.5), np.ones(3) * rng2.uniform(0.7, 1.0))

    # ── 127: FLOWER FIELD (many blooms, saturated colors — before this, _flower() only ever
    # appeared as a single decorative accent on a bush, never as the hero subject) ──
    elif lvl == 127:
        base = np.array([rng2.uniform(0.1, 0.35), rng2.uniform(0.3, 0.6), rng2.uniform(0.05, 0.3)])
        img[:] = np.clip(base + (_perlin(rng2, 4).reshape(H, W, 1) - 0.5) * 0.2, 0, 1)  # foliage backdrop
        for _ in range(rng2.randint(6, 16)):
            if rng2.rand() < 0.75:                       # vivid petal colors: real blooms rarely muted
                col = rng2.uniform(0.3, 1.0, C)
                col[rng2.randint(0, C)] *= rng2.uniform(0.3, 0.7)
            else:
                col = _photo_color(rng2, 0.6, 1.0)        # occasional white/pastel bloom
            _flower(rng2, img, col)

    # ── 128: CLUSTERED ORGANIC PORE/VOID TEXTURE (honeycomb, coral, sponge, seed pods —
    # existing textures (grass/fur/fractal/noise) never produce this densely-packed circular
    # cluster pattern) ──
    elif lvl == 128:
        base = _photo_color(rng2, 0.25, 0.75)
        # every additive shift below is clamped to base's per-channel headroom (both directions)
        # so R/G/B always move by the same amount and clip together — an unclamped clip(base+x,0,1)
        # desaturates channels unevenly and (compounded over ~100 overlapping discs) turns the
        # whole texture into rainbow noise
        hi_head, lo_head = (1.0 - base).min(), (base - 0.02).min()
        wobble = min(0.15, 2 * hi_head, 2 * lo_head)
        img[:] = base + (_perlin(rng2, 4).reshape(H, W, 1) - 0.5) * wobble
        bump = rng2.rand() < 0.5           # raised bumps (coral/pod) vs recessed holes (honeycomb/sponge)
        headroom = hi_head if bump else lo_head

        # organic rejection-sampling packing (not a grid): lets small and large pores mix freely
        # in the same image and still cluster/touch, which a fixed cell size can't do
        n_pores = rng2.randint(4, 140)      # a handful up to a dense cluster
        placed = []                          # (cx, cy, r)
        attempts = 0
        while len(placed) < n_pores and attempts < n_pores * 15:
            attempts += 1
            r = rng2.uniform(0.6, 4.5)       # pinholes and large pits/bumps in the same cluster
            cx, cy = rng2.uniform(0, W), rng2.uniform(0, H)
            if any(math.hypot(cx - px, cy - py) < (r + pr) * 0.55 for px, py, pr in placed):
                continue                     # allow touching/overlap (real clusters are tight), not full stacking
            placed.append((cx, cy, r))

        for cx, cy, r in placed:
            d = _soft_disc(cx, cy, r)
            delta = min(rng2.uniform(0.2, 0.4), headroom) * (1 if bump else -1)
            shade = base + delta
            img = _blend(img, d * rng2.uniform(0.6, 1.0), shade)
            rim = np.clip(d - _soft_disc(cx, cy, r * 0.6), 0, 1)
            rim_col = base - delta * 0.5     # rim shades opposite: recessed lip / bump highlight edge
            img = _blend(img, rim * 0.5, rim_col)
            if r > 1.1 and rng2.rand() < 0.35:   # some pores hold visible content, most stay empty
                # content + its AO ring must stay well clear of the pore's own rim zone (r*0.6,
                # see above) so it can never touch or rebase the pore's boundary
                cr = r * rng2.uniform(0.12, 0.3)
                ring_w = cr * rng2.uniform(0.25, 0.45)
                max_offset = max(0.0, r * 0.58 - cr - ring_w)
                off_ang, off_mag = rng2.uniform(0, 2 * math.pi), rng2.uniform(0, max_offset)
                ccx, ccy = cx + math.cos(off_ang) * off_mag, cy + math.sin(off_ang) * off_mag
                # ambient-occlusion seam: content sits recessed inside the pore, so the crevice
                # where its edge meets the inner wall gets less light than the open pore mouth —
                # a dark ring at that junction, not the content color itself
                ao = np.clip(_soft_disc(ccx, ccy, cr + ring_w) - _soft_disc(ccx, ccy, cr), 0, 1) * d
                img = _blend(img, ao, base * 0.15)
                content_col = rng2.uniform(0.05, 1.0, C)   # unrelated color: reads as foreign material
                # mask by the pore's own alpha `d`, not just kept-inside-radius math: _soft_disc's
                # antialiased edge feathers ~0.5px past its nominal radius, so without this the
                # content disc could poke past the pore's own soft boundary
                img = _blend(img, _soft_disc(ccx, ccy, cr) * d, content_col)

    # ── 131: X-RAY / RADIOGRAPH — transmission imaging, not reflected light.
    # No Phong, no light source. Dense material (bone) = bright because it blocks the beam.
    # Overlapping bone deposits stack additively (thicker parts = whiter, like real xray).
    # Dark border = collimator cutoff. Gray fog between bones = soft tissue scatter.
    # 8 body-part types: skull, chest, hand, pelvis, knee, dental, spine, foot.
    elif lvl == 131:
        # ── film base + collimator vignette ──
        img[:] = rng2.uniform(0.0, 0.05, C)
        yy, xx = np.ogrid[:H, :W]
        ed = np.minimum(np.minimum(xx, W - 1 - xx), np.minimum(yy, H - 1 - yy)).astype(np.float32)
        vignette = np.clip(ed / rng2.uniform(1.2, 5.0), 0, 1)
        for c in range(C):
            img[:, :, c] *= vignette
        # ── soft tissue: low-opacity fog over the exposed area ──
        img = _blend(img, np.ones((H, W), np.float32) * rng2.uniform(0.05, 0.18),
                     np.ones(3) * rng2.uniform(0.07, 0.22))
        # ── bone composite mask (additive: overlap = higher density = brighter on film) ──
        bone = np.zeros((H, W), np.float32)
        bp = rng2.randint(0, 8)
        bright = rng2.uniform(0.68, 0.98)
        alpha = rng2.uniform(0.52, 0.72)

        if bp == 0:                                                     # SKULL (AP or lateral)
            # cranium — big round dome
            cx, cy = rng2.uniform(14, 18), rng2.uniform(7, 12)
            cr = rng2.uniform(7, 10)
            bone += _soft_disc(cx, cy, cr) * rng2.uniform(0.8, 1.0)
            # jaw / mandible
            jx, jy = cx + rng2.uniform(-2, 2), cy + cr * rng2.uniform(0.7, 1.0)
            bone += _soft_disc(jx, jy, cr * rng2.uniform(0.5, 0.75)) * rng2.uniform(0.6, 0.9)
            # cheekbones
            for sx in (-1, 1):
                bone += _soft_disc(cx + sx * cr * 0.55, cy + cr * 0.35,
                                  cr * rng2.uniform(0.25, 0.4)) * rng2.uniform(0.55, 0.8)
            # eye sockets — subtract density (no bone = dark on film)
            for sx in (-1, 1):
                ox = cx + sx * cr * rng2.uniform(0.25, 0.42)
                oy = cy - cr * rng2.uniform(0.0, 0.25)
                bone = np.clip(bone - _soft_disc(ox, oy, rng2.uniform(1.4, 2.8)) * rng2.uniform(0.5, 0.9), 0, 1)
            # nasal cavity
            bone = np.clip(bone - _soft_disc(cx, cy + cr * rng2.uniform(0.15, 0.4),
                                             rng2.uniform(0.9, 2.2)) * rng2.uniform(0.35, 0.7), 0, 1)
            # teeth row — small bright dots along jaw
            for i in range(rng2.randint(4, 9)):
                tx = jx + (i - 3.5) * rng2.uniform(1.0, 2.0)
                ty = jy + rng2.uniform(1, 3.5)
                bone += _soft_disc(tx, ty, rng2.uniform(0.35, 0.75)) * rng2.uniform(0.55, 0.9)

        elif bp == 1:                                                   # CHEST PA
            sx, sy = rng2.uniform(14, 18), rng2.uniform(10, 17)        # spine center
            # spine — vertical stack of overlapping discs
            n_vert = rng2.randint(5, 9)
            for i in range(n_vert):
                vy = sy - 3 + i * rng2.uniform(2.0, 4.5)
                bone += _soft_disc(sx, vy, rng2.uniform(1.0, 2.0)) * rng2.uniform(0.7, 0.95)
            # ribs — chains of overlapping discs in gentle arcs from spine outward
            n_ribs = rng2.randint(5, 8)
            for i in range(n_ribs):
                ry0 = sy - 2 + i * (14.0 / n_ribs)
                for side in (-1, 1):
                    rib_w = rng2.uniform(5, 10)                         # how far the rib reaches
                    rib_drop = rng2.uniform(1.5, 5)                     # downward sag
                    rib_arch = rng2.uniform(0.5, 3.5)                   # upward bow
                    n_seg = rng2.randint(3, 6)
                    for j in range(n_seg):
                        t = j / (n_seg - 1)
                        rx = sx + side * t * rib_w
                        ry = ry0 + t * rib_drop - math.sin(t * math.pi) * rib_arch
                        bone += _soft_disc(rx, ry, rng2.uniform(0.45, 0.85)) * rng2.uniform(0.48, 0.72)
            # clavicles — two short horizontal-ish lines at the top of the spine
            for side in (-1, 1):
                cx0 = sx + side * rng2.uniform(1.5, 3.5)
                cy0 = sy - rng2.uniform(5, 10)
                for j in range(rng2.randint(3, 5)):
                    bone += _soft_disc(cx0 + side * j * rng2.uniform(1.2, 3.0),
                                      cy0 + j * rng2.uniform(-0.5, 0.5),
                                      rng2.uniform(0.4, 0.75)) * rng2.uniform(0.55, 0.8)

        elif bp == 2:                                                   # HAND
            wx, wy = rng2.uniform(10, 22), rng2.uniform(18, 24)        # wrist
            # wrist — cluster of small discs
            for _ in range(rng2.randint(4, 8)):
                bone += _soft_disc(wx + rng2.uniform(-2, 2), wy + rng2.uniform(-1.5, 1.5),
                                  rng2.uniform(0.8, 1.8)) * rng2.uniform(0.6, 0.85)
            # 5 metacarpal + phalange chains fanning upward
            for fi in range(5):
                base_x = wx + (fi - 2) * rng2.uniform(1.5, 3.5)
                base_y = wy - rng2.uniform(1, 3)
                n_seg = rng2.randint(2, 4)                              # metacarpal + 1-2 phalanges
                px, py = base_x, base_y
                for k in range(n_seg):
                    px += rng2.uniform(-0.5, 0.5)
                    py -= rng2.uniform(2.5, 5.5)                       # move up
                    r = rng2.uniform(0.5, 1.2) * (1.0 - k * 0.2)       # distal bones thinner
                    bone += _soft_disc(px, py, r) * rng2.uniform(0.55, 0.85)
                # knuckle — slightly larger dot at each joint
                if k > 0:
                    bone += _soft_disc(px, py, r * 1.25) * rng2.uniform(0.45, 0.65)

        elif bp == 3:                                                   # PELVIS AP
            cx, cy = rng2.uniform(14, 18), rng2.uniform(13, 17)
            # iliac wings — two large tilted discs
            for sx in (-1, 1):
                ix = cx + sx * rng2.uniform(4, 7)
                iy = cy + rng2.uniform(-3, 1)
                bone += _soft_disc(ix, iy, rng2.uniform(3.5, 6)) * rng2.uniform(0.65, 0.9)
            # sacrum — triangular-ish central piece (approximated as a cluster)
            for _ in range(rng2.randint(2, 4)):
                bone += _soft_disc(cx + rng2.uniform(-1.5, 1.5), cy + rng2.uniform(1, 4),
                                  rng2.uniform(1.0, 2.2)) * rng2.uniform(0.6, 0.85)
            # pubic symphysis — small discs at the bottom
            for sx in (-1, 1):
                bone += _soft_disc(cx + sx * rng2.uniform(1.5, 3), cy + rng2.uniform(5, 8),
                                  rng2.uniform(0.6, 1.2)) * rng2.uniform(0.55, 0.8)

        elif bp == 4:                                                   # KNEE (lateral)
            kx, ky = rng2.uniform(13, 19), rng2.uniform(10, 14)
            # femoral condyles — two large overlapping discs
            for sx in (-1, 1):
                bone += _soft_disc(kx + sx * rng2.uniform(2.5, 4.5), ky,
                                  rng2.uniform(2.5, 4.5)) * rng2.uniform(0.7, 0.9)
            # tibial plateau — a flatter wider disc below
            bone += _soft_disc(kx, ky + rng2.uniform(4, 7), rng2.uniform(3, 5)) * rng2.uniform(0.6, 0.82)
            # patella — small bright dot above the joint
            bone += _soft_disc(kx + rng2.uniform(-1, 1), ky - rng2.uniform(1, 3),
                              rng2.uniform(1.0, 1.8)) * rng2.uniform(0.7, 0.9)
            # joint space gap — narrow dark band between femur and tibia
            gap_y = ky + rng2.uniform(2.5, 4.5)
            gap_mask = np.abs(np.arange(H, dtype=np.float32).reshape(-1, 1) - gap_y) < rng2.uniform(0.5, 1.2)
            bone = np.clip(bone - gap_mask.astype(np.float32) * rng2.uniform(0.4, 0.7), 0, 1)

        elif bp == 5:                                                   # DENTAL (panoramic / bitewing)
            dx, dy = rng2.uniform(12, 20), rng2.uniform(12, 20)
            arch_rad = rng2.uniform(4, 9)
            n_teeth = rng2.randint(7, 15)
            for i in range(n_teeth):
                ang = math.pi * rng2.uniform(0.4, 0.6) + i * rng2.uniform(0.15, 0.3)
                tx = dx + arch_rad * math.cos(ang)
                ty = dy + arch_rad * math.sin(ang) * rng2.uniform(0.7, 1.3)
                tr = rng2.uniform(0.5, 1.2)
                bone += _soft_disc(tx, ty, tr) * rng2.uniform(0.6, 0.95)
                # tooth root — a smaller disc extending below the crown
                bone += _soft_disc(tx, ty + tr * 1.5, tr * rng2.uniform(0.5, 0.8)) * rng2.uniform(0.45, 0.7)

        elif bp == 6:                                                   # SPINE (lateral)
            sx, sy = rng2.uniform(12, 20), rng2.uniform(3, 8)          # top of vertebral column
            n_vert = rng2.randint(6, 12)
            for i in range(n_vert):
                vy = sy + i * rng2.uniform(2.0, 3.5)
                # vertebral body — squarish (approximated as disc)
                bone += _soft_disc(sx, vy, rng2.uniform(1.2, 2.5)) * rng2.uniform(0.68, 0.92)
                # spinous process — small posterior spike
                bone += _soft_disc(sx + rng2.uniform(2, 4), vy + rng2.uniform(-1, 1),
                                  rng2.uniform(0.5, 1.1)) * rng2.uniform(0.5, 0.75)
                # intervertebral disc space — thin dark gap
                gap_y = vy + rng2.uniform(0.8, 1.8)
                gap_mask = np.abs(np.arange(H, dtype=np.float32).reshape(-1, 1) - gap_y) < rng2.uniform(0.3, 0.7)
                bone = np.clip(bone - gap_mask.astype(np.float32) * rng2.uniform(0.3, 0.55), 0, 1)

        else:                                                           # bp == 7: FOOT (lateral)
            fx, fy = rng2.uniform(6, 12), rng2.uniform(16, 24)         # heel
            # calcaneus — large blocky disc at the heel
            bone += _soft_disc(fx, fy, rng2.uniform(3, 5.5)) * rng2.uniform(0.7, 0.92)
            # talus + tarsals — midfoot cluster
            for _ in range(rng2.randint(3, 6)):
                bone += _soft_disc(fx + rng2.uniform(2, 5), fy - rng2.uniform(1, 4),
                                  rng2.uniform(0.9, 2.0)) * rng2.uniform(0.58, 0.82)
            # metatarsals — 5 rays fanning rightward toward the toes
            for ri in range(5):
                mx = fx + rng2.uniform(2, 6)
                my = fy - rng2.uniform(2, 6)
                for k in range(rng2.randint(2, 4)):
                    mx += rng2.uniform(1.5, 4.0)
                    my -= rng2.uniform(0.3, 1.5)
                    bone += _soft_disc(mx, my, rng2.uniform(0.4, 0.9)) * rng2.uniform(0.5, 0.78)

        # ── deposit the accumulated bone mask onto the image ──
        bone_val = np.clip(bone * alpha, 0, 1)
        img = _blend(img, bone_val, np.ones(3) * bright)

    # ── 132: STAINED MICROSCOPY (cells + nuclei, 3 stain palettes — a different visual regime
    # from the X-ray level above: false-color/structural rather than grayscale-silhouette) ──
    elif lvl == 132:
        L = _rand_light(rng2)
        stain = rng2.randint(0, 3)
        if stain == 0:        # H&E: pink/purple cells on a pale pink field
            img[:] = np.array([rng2.uniform(0.75, 0.9), rng2.uniform(0.6, 0.75), rng2.uniform(0.68, 0.85)])
            cell_col = lambda: np.array([rng2.uniform(0.55, 0.75), rng2.uniform(0.3, 0.5), rng2.uniform(0.55, 0.75)])
            nuc_col = lambda: np.array([rng2.uniform(0.25, 0.45), rng2.uniform(0.05, 0.2), rng2.uniform(0.4, 0.6)])
        elif stain == 1:      # fluorescence: bright cells on black
            img[:] = rng2.uniform(0.0, 0.04, C)
            cell_col = lambda: np.array([rng2.uniform(0.1, 0.3), rng2.uniform(0.5, 0.9), rng2.uniform(0.1, 0.3)])
            nuc_col = lambda: np.array([rng2.uniform(0.3, 0.5), rng2.uniform(0.3, 0.5), rng2.uniform(0.7, 1.0)])
        else:                 # plain grayscale microscopy
            g = rng2.uniform(0.55, 0.8)
            img[:] = np.array([g, g, g])
            cell_col = lambda: np.array([rng2.uniform(0.3, 0.55)] * 3)
            nuc_col = lambda: np.array([rng2.uniform(0.1, 0.25)] * 3)
        n_cells = rng2.randint(4, 18)
        placed = []
        attempts = 0
        while len(placed) < n_cells and attempts < n_cells * 15:
            attempts += 1
            r = rng2.uniform(1.5, 5.0)
            cx, cy = rng2.uniform(0, W), rng2.uniform(0, H)
            if any(math.hypot(cx - px, cy - py) < (r + pr) * 0.7 for px, py, pr in placed):
                continue                   # cells can touch/cluster but not fully stack
            placed.append((cx, cy, r))
        for cx, cy, r in placed:
            m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, r, r * rng2.uniform(0.8, 1.2))
            _draw_3d(img, m, Nx, Ny, Nz, L, cell_col())
            if rng2.rand() < 0.8:          # most cells show a nucleus, not all
                nr = r * rng2.uniform(0.25, 0.4)
                noff = r * 0.15
                ncx, ncy = cx + rng2.uniform(-noff, noff), cy + rng2.uniform(-noff, noff)
                membrane = _soft_disc(cx, cy, r)   # circular approx of the blob's own footprint,
                img = _blend(img, _soft_disc(ncx, ncy, nr) * membrane, nuc_col())  # keeps nucleus inside

    # ── 134: DIFFRACTION-3D (grating/thin-film iridescent surface + a specular glint with
    # diffraction spikes — self-contained, unlike 133/135/136 which recurse on a base image) ──
    elif lvl == 134:
        img[:] = rng2.uniform(0.02, 0.15, C)             # dark field: glint/iridescence needs contrast
        L = _rand_light(rng2)
        cx, cy = rng2.uniform(8, W - 8), rng2.uniform(7, H - 7)
        R = rng2.uniform(6, 12)
        _IRID = rng2.uniform(1.5, 6.2)                    # force strong (normally rare) iridescence
        if rng2.rand() < 0.5:
            m, Nx, Ny, Nz = _sphere(cx, cy, R)
        else:
            m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.6, 1.3))
        _draw_3d(img, m, Nx, Ny, Nz, L, rng2.uniform(0.3, 0.6, C))
        lit = _phong(Nx, Ny, Nz, L)
        ys, xs = np.where(m)
        if len(xs):                                       # glint at the brightest point on the surface
            i = np.argmax(lit[ys, xs])
            img = _point_source(img, xs[i], ys[i], rng2.uniform(0.6, 1.4), np.ones(3),
                                core=rng2.uniform(0.3, 0.6), halo=rng2.uniform(1.0, 2.5), spikes=True)

    # ── 137: HAIR / FUR STRANDS IN 3D (individually rendered shaded strands, groomed vs
    # tangled — level 92's "fur" is a shading texture painted onto a blob's surface, not
    # actual strands with their own depth; this is a distinct rendering regime) ──
    elif lvl == 137:
        L = _rand_light(rng2)
        # base_col's range is chosen opposite the background's so strands can never blend into
        # it — drawing both independently (as the first version did) let dark-on-dark or
        # light-on-light pairs slip through, making some samples read as an empty black square
        if rng2.rand() < 0.5:
            img[:] = rng2.uniform(0.0, 0.08, C)
            base_col = _photo_color(rng2, 0.35, 1.0)
        else:
            img[:] = _photo_color(rng2, 0.6, 0.95)
            base_col = _photo_color(rng2, 0.02, 0.45)
        chaotic = rng2.rand() < 0.5              # groomed/combed vs tangled/matted
        edge = rng2.randint(0, 4)                # which side the strands grow from
        root_ang = (math.pi / 2, -math.pi / 2, 0.0, math.pi)[edge]     # points inward
        # camera-distance proxy: <1 zoomed out (many fine strands), >1 zoomed in (few thick
        # ones filling the frame) — without this every sample reads at the same texture scale
        zoom = rng2.uniform(0.5, 2.3)
        n_strands = int(np.clip(rng2.randint(20, 50) / zoom, 6, 70))
        bg_lum = float(img[0, 0].mean())            # background is a flat fill, any pixel will do
        mask_total = np.zeros((H, W), bool)
        for _ in range(n_strands):
            if edge == 0:   x, y = rng2.uniform(0, W), 0.0
            elif edge == 1: x, y = rng2.uniform(0, W), H - 1.0
            elif edge == 2: x, y = 0.0, rng2.uniform(0, H)
            else:           x, y = W - 1.0, rng2.uniform(0, H)
            ang = root_ang + rng2.uniform(-0.5, 0.5)
            n_seg = rng2.randint(3, 6)
            seg_len = rng2.uniform(6, 22) * zoom / n_seg
            r0 = max(0.15, rng2.uniform(0.35, 0.75) * zoom)
            col = np.clip(base_col * rng2.uniform(0.6, 1.2), 0, 1)
            phase = rng2.uniform(0, 6)
            curl = rng2.choice([-1, 1]) * rng2.uniform(0.15, 0.45)
            for s in range(n_seg):
                if chaotic:                       # large random turns (+ a persistent curl bias):
                    ang += curl + rng2.uniform(-0.5, 0.5)   # tangled, self-crossing, ringlet-like
                else:                              # small jitter + shared gentle wave: combed flow
                    ang += math.sin(s * 0.8 + phase) * 0.15 + rng2.uniform(-0.05, 0.05)
                x1 = x + math.cos(ang) * seg_len
                y1 = y + math.sin(ang) * seg_len
                mask_total |= _tube(img, x, y, x1, y1, r0 * (1 - s / n_seg * 0.7), col, L)
                x, y = x1, y1

        # the disjoint base_col/background ranges above only bound the *unlit* color — per-image
        # globals (_LIGHT_COL, _SUBJECT_GAIN) and Phong's own dim end (as low as 0.12x) can still
        # crush a strand's final rendered brightness back down near the background, especially at
        # low zoom where strands are thin and mostly in shadow. Guarantee real contrast here.
        if mask_total.any():
            strand_lum = float(img[mask_total].mean())
            if abs(strand_lum - bg_lum) < 0.15:
                sign = 1.0 if strand_lum >= bg_lum else -1.0
                img[mask_total] = np.clip(img[mask_total] + sign * 0.15, 0, 1)

    # ── 139: GLASS IN 3D — a true 3D glass object: the background refracts through it,
    # the back-face is visible (darker), a specular highlight sits at the brightest front
    # spot, and the rim has a fresnel-like brightening. NOT recursive like 136 — this
    # renders its own 3D shape and warps its own background, self-contained.
    elif lvl == 139:
        bg_col = _photo_color(rng2, 0.08, 0.7)
        img[:] = bg_col.reshape(1, 1, C) * (0.65 + 0.35 * _perlin(rng2, 3).reshape(H, W, 1))
        L = _rand_light(rng2)
        cx, cy = rng2.uniform(8, W - 8), rng2.uniform(7, H - 7)
        R = rng2.uniform(5, 11)
        shape = rng2.randint(0, 3)
        if shape == 0:
            m, Nx, Ny, Nz = _sphere(cx, cy, R)
        elif shape == 1:
            m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.7, 1.3))
        else:
            m, Nx, Ny, Nz = _splat(rng2, cx, cy, R)
        # ── refractive warp of the background through the object ──
        mag = rng2.uniform(1.4, 2.8) * (-1 if rng2.rand() < 0.35 else 1)  # inverted: droplet-like
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        sx, sy = cx + (xx - cx) / mag, cy + (yy - cy) / mag
        for ch in range(C):
            warped = map_coordinates(img[:, :, ch], [sy, sx], order=1, mode='nearest')
            img[:, :, ch] = np.where(m, warped, img[:, :, ch])
        # ── back-face visible through the front: darker, shows internal volume ──
        back_face = m & (Nz > 0)
        if back_face.any():
            for ch in range(C):
                img[:, :, ch][back_face] = np.clip(img[:, :, ch][back_face] * rng2.uniform(0.45, 0.75), 0, 1)
        # ── phong-lit tint overlay: the glass body itself is faintly visible ──
        # (raised the low end: at 0.12 the tint + rim were too weak against a similarly-toned
        # background — the sphere shape only read clearly in ~2/3 of samples)
        glass_tint = np.array([0.82, 0.9, rng2.uniform(0.75, 1.0)]) * rng2.uniform(0.2, 0.36)
        lit = _phong(Nx, Ny, Nz, L)
        for ch in range(C):
            img[:, :, ch][m] = np.clip(img[:, :, ch][m] + glass_tint[ch] * lit[m], 0, 1)
        # ── fresnel-like rim brightening: edges catch/reflect more light ──
        rim = (1.0 - np.abs(Nz)) ** 3 * rng2.uniform(0.25, 0.55)
        for ch in range(C):
            img[:, :, ch][m] = np.clip(img[:, :, ch][m] + rim[m], 0, 1)
        # ── specular highlight at the brightest-lit point on the front face ──
        front = m & (Nz < 0)
        if front.any():
            yy_f, xx_f = np.where(front)
            i = np.argmax(lit[yy_f, xx_f])
            hx, hy = xx_f[i], yy_f[i]
            h_d = _soft_disc(hx, hy, rng2.uniform(0.8, 2.5)) * rng2.uniform(0.6, 0.95)
            img = _blend(img, h_d, np.ones(3))

    # ── 140: CAUSTICS — the pattern of concentrated light that a refractive object casts
    # onto a surface below it. Not the object itself — the light pattern it produces.
    # The only way to get this without ray-tracing is to approximate with a voronoi-like
    # intensity field modulated by a refractive warp of the same background field. ──
    elif lvl == 140:
        bg_col = _photo_color(rng2, 0.35, 0.8)
        img[:] = bg_col.reshape(1, 1, C) * (0.7 + 0.3 * _perlin(rng2, 3).reshape(H, W, 1))
        # ── a "virtual lens field" — bilinear-interpolated smooth gradient that simulates
        # the lens' refractive power across the image (stronger near the axis, falls off) ──
        cx, cy = rng2.uniform(8, W - 8), rng2.uniform(7, H - 7)
        R = rng2.uniform(7, 14)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
        lens = np.exp(-d ** 2 * 2.5)          # gaussian lens: strongest at center, falls to zero at edge
        # ── caustic network: the lens concentrates perlin noise into bright threads ──
        strength = rng2.uniform(0.6, 2.0)
        warp_field = _perlin(rng2, rng2.randint(3, 5))
        warp_field = _curl_warp(rng2, warp_field, rng2.uniform(1.5, 4.5))
        warp_field = (warp_field - warp_field.min()) / (warp_field.max() - warp_field.min() + 1e-8)
        # lens concentrates the warp: high lens=more refractive concentration=brighter caustic threads
        caustic = np.clip(warp_field * (0.3 + lens * strength * rng2.uniform(2.5, 5.5)), 0, 1.3)
        caustic = caustic ** rng2.uniform(0.5, 1.6)   # sharpen: thin bright filaments over a dark pool
        # ── two chromatically-separated passes (dispersion) ──
        if rng2.rand() < 0.5:
            # split R and B slightly: chromatic dispersion in the caustic fringe (prismatic edge)
            dy_disp = rng2.uniform(0.6, 1.8)
            dx_disp = rng2.uniform(0.0, 0.5)
            r_field = np.roll(caustic, int(round(dy_disp)), axis=0)
            b_field = np.roll(caustic, -int(round(dy_disp)), axis=0)
            warm = rng2.uniform(0.85, 1.0) if rng2.rand() < 0.7 else rng2.uniform(0.15, 0.55)
            for ch in range(C):
                if ch == 0:
                    f = r_field
                elif ch == 2:
                    f = b_field
                else:
                    f = caustic
                boost = 1.0 + (rng2.uniform(-0.3, 0.3) if ch == 1 else 0)
                img[:, :, ch] = np.clip(img[:, :, ch] * (0.55 + 0.55 * f * boost), 0, 1)
            # warm or cool cast on the concentrated light
            if warm > 0.7:
                img[:, :, 0] = np.clip(img[:, :, 0] + caustic * 0.15, 0, 1)
                img[:, :, 2] = np.clip(img[:, :, 2] * (1 - caustic * 0.1), 0, 1)
            else:
                img[:, :, 2] = np.clip(img[:, :, 2] + caustic * 0.12, 0, 1)
        else:
            # monochrome: plain brightness modulation (pool caustics, no dispersion)
            for ch in range(C):
                img[:, :, ch] = np.clip(img[:, :, ch] * (0.5 + 0.6 * caustic), 0, 1)
        # ── vignette falloff beyond the lens radius ──
        falloff = np.clip(1 - d * 1.4, 0, 1)
        for ch in range(C):
            img[:, :, ch] = np.clip(img[:, :, ch] * (0.55 + 0.45 * falloff), 0, 1)
        # ── edge caustic ring: the lens rim itself casts a bright sharp circle ──
        rim = np.exp(-np.abs(d - 1.0) * 5.0) * 0.5 + np.exp(-np.abs(d - 1.0) * 1.8) * 0.15
        warm_rim = np.ones(C) * 1.0
        if rng2.rand() < 0.6:
            warm_rim = np.array([1.0, rng2.uniform(0.85, 1.0), rng2.uniform(0.7, 0.9)])
        for ch in range(C):
            img[:, :, ch] = np.clip(img[:, :, ch] + rim * warm_rim[ch] * rng2.uniform(0.15, 0.45), 0, 1)

    # ── 141: AERIAL / DRONE VIEW — top-down perspective: patchwork fields, roads, roofs, rivers.
    # The critical missing perspective angle — everything else is ground-level or eye-level. ──
    elif lvl == 141:
        gnd = np.array([rng2.uniform(0.08, 0.25), rng2.uniform(0.15, 0.40), rng2.uniform(0.04, 0.15)])
        tex = _perlin(rng2, 4)
        img[:] = gnd.reshape(1, 1, C) * (0.7 + 0.3 * tex.reshape(H, W, 1))
        n_patches = rng2.randint(6, 16)
        # produce a coarse voronoi-like subdivision: random seeds → nearest-neighbor fills
        seeds = [(rng2.uniform(0, W), rng2.uniform(0, H)) for _ in range(n_patches)]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        for sx, sy in seeds:
            d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
            col = gnd * rng2.uniform(0.7, 1.5)
            col = np.clip(col + rng2.randn(3).astype(np.float32) * 0.08, 0, 1)
            patch_size = rng2.uniform(6, 16)
            m = d < patch_size
            if m.any():
                for cc in range(C):
                    img[:, :, cc][m] = np.clip(img[:, :, cc][m] * 0.3 + col[cc] * 0.7, 0, 1)
        # roads: thin lines cutting across
        for _ in range(rng2.randint(1, 4)):
            rx0, ry0 = rng2.uniform(0, W), rng2.uniform(0, H)
            rx1, ry1 = rng2.uniform(0, W), rng2.uniform(0, H)
            _bresenham(img, int(rx0), int(ry0), int(rx1), int(ry1), 1, np.array([0.15, 0.13, 0.11]))
        # buildings: small bright rectangles
        for _ in range(rng2.randint(0, 7)):
            bx, by = rng2.uniform(0, W - 4), rng2.uniform(0, H - 4)
            bw, bh = rng2.uniform(1.5, 4.5), rng2.uniform(1.5, 4.5)
            m = _rect(bx, by, bx + bw, by + bh)
            roof = np.array([rng2.uniform(0.3, 0.7), rng2.uniform(0.25, 0.6), rng2.uniform(0.2, 0.5)])
            _fill(img, m, roof)
        # water body (river or pond)
        if rng2.rand() < 0.6:
            wx, wy = rng2.uniform(4, W - 4), rng2.uniform(4, H - 4)
            wr = rng2.uniform(5, 14)
            wm = _ellipse(wx, wy, wr, wr * rng2.uniform(0.3, 0.8))
            water = np.array([0.05, rng2.uniform(0.15, 0.35), rng2.uniform(0.4, 0.7)])
            _fill(img, wm, water)

    # ── 142: FOOD — plates with colored blobs, bowls, fruit shapes. Ubiquitous in real photos,
    # completely absent from the generator until now. ──
    elif lvl == 142:
        # background: table surface
        table = np.array([rng2.uniform(0.2, 0.6), rng2.uniform(0.12, 0.4), rng2.uniform(0.05, 0.2)])
        img[:] = table
        tex = _perlin(rng2, 2)
        img[:, :, :] += tex.reshape(H, W, 1) * 0.06
        # plate: large white/grey ellipse in center
        px, py = W / 2, H / 2 + rng2.uniform(-2, 3)
        plate_r = rng2.uniform(8, 12)
        plate_col = np.ones(3) * rng2.uniform(0.6, 0.95)
        a = _soft_disc(px, py, plate_r)
        for cc in range(C):
            img[:, :, cc] = np.clip(img[:, :, cc] * (1 - a) + plate_col[cc] * a, 0, 1)
        # plate rim shadow
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rim = np.clip((plate_r - 0.8 - np.sqrt((yy - py) ** 2 + (xx - px) ** 2)) * 1.5, 0, 1) * 0.15
        for cc in range(C):
            img[:, :, cc] = np.clip(img[:, :, cc] - rim, 0, 1)
        # food items on plate
        for _ in range(rng2.randint(2, 6)):
            fx = px + rng2.uniform(-plate_r * 0.55, plate_r * 0.55)
            fy = py + rng2.uniform(-plate_r * 0.45, plate_r * 0.45)
            fr = rng2.uniform(1.5, 5)
            food_col = np.array([rng2.uniform(0.3, 0.9), rng2.uniform(0.1, 0.7), rng2.uniform(0.0, 0.3)])
            a = _soft_disc(fx, fy, fr)
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] * (1 - a * 0.7) + food_col[cc] * a * 0.7, 0, 1)
        # garnish: green/yellow specks
        for _ in range(rng2.randint(0, 5)):
            gx = px + rng2.uniform(-plate_r * 0.5, plate_r * 0.5)
            gy = py + rng2.uniform(-plate_r * 0.4, plate_r * 0.4)
            gcol = np.array([rng2.uniform(0.0, 0.3), rng2.uniform(0.4, 0.9), rng2.uniform(0.0, 0.3)])
            a = _soft_disc(gx, gy, rng2.uniform(0.3, 0.9))
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] + a * gcol[cc] * 0.6, 0, 1)

    # ── 143: MRI / CT MEDICAL IMAGING — grayscale soft-tissue contrast, axial slices,
    # ring artifacts, intensity windowing. Major modality absent before. ──
    elif lvl == 143:
        # grayscale base with tissue contrast
        base = rng2.uniform(0.05, 0.15)
        img[:] = np.clip(base + _perlin(rng2, rng2.randint(3, 5)).reshape(H, W, 1) * rng2.uniform(0.3, 0.7), 0, 1)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # CT ring / band artifacts (concentric circles from detector calibration)
        if rng2.rand() < 0.5:
            cy, cx = H / 2 + rng2.uniform(-3, 3), W / 2 + rng2.uniform(-3, 3)
            for _ in range(rng2.randint(2, 6)):
                R = rng2.uniform(4, 18)
                a = np.exp(-np.abs(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) - R) * rng2.uniform(0.3, 0.9))
                img[:, :, :] += a.reshape(H, W, 1) * rng2.uniform(0.02, 0.12)
        # anatomical structure: overlapping soft blobs (organs / lesions).
        # val was 0.05-0.45 with up to 8 overlapping blobs, each clipped individually — several
        # overlaps routinely pinned to 1.0 before windowing ever ran, which is what made the
        # output look like a binary silhouette instead of graduated soft-tissue contrast.
        # shape: real organ/lesion cross-sections are irregular, not perfect circles —
        # _organic_blob's perlin-wobbled outline reused here (with the hard edge softened
        # by a small blur, since blob masks don't antialias like _soft_disc does).
        for _ in range(rng2.randint(3, 8)):
            ox, oy = rng2.uniform(4, W - 4), rng2.uniform(4, H - 4)
            rx = rng2.uniform(3, 10)
            ry = rx * rng2.uniform(0.6, 1.5)
            m, _, _, _ = _organic_blob(rng2, ox, oy, rx, ry)
            a = gaussian_filter(m.astype(np.float32), sigma=0.7)
            val = rng2.uniform(0.05, 0.28)
            img[:, :, :] = np.clip(img[:, :, :] + a.reshape(H, W, 1) * val, 0, 1)
        # bone: bright thin rings/arcs (skull, vertebrae cross-section)
        for _ in range(rng2.randint(0, 4)):
            bx, by = rng2.uniform(6, W - 6), rng2.uniform(6, H - 6)
            R = rng2.uniform(3, 7)
            a = np.exp(-np.abs(np.sqrt((yy - by) ** 2 + (xx - bx) ** 2) - R) * rng2.uniform(1.5, 3.5))
            img[:, :, :] = np.clip(img[:, :, :] + a.reshape(H, W, 1) * rng2.uniform(0.3, 0.55), 0, 1)
        # invert (CT = bone white on black background; MRI can be either)
        if rng2.rand() < 0.4:
            img[:] = 1.0 - img
        # windowing: clip to a narrower intensity range (medical display convention).
        # previous bounds (10-40 / 60-95 percentile) could leave as little as a 20-point
        # spread, which stretched the image to near-pure black/white instead of the soft
        # graduated tissue contrast real MRI/CT windowing keeps.
        lo, hi = np.percentile(img, rng2.uniform(2, 15)), np.percentile(img, rng2.uniform(80, 98))
        img = np.clip((img - lo) / max(hi - lo, 0.01), 0, 1)
        # speckle / noise (MRI has Rician noise)
        img = np.clip(img + rng2.randn(H, W, 1).astype(np.float32) * rng2.uniform(0.01, 0.06), 0, 1)
        # add crosshair / scan markers (fiducial lines)
        if rng2.rand() < 0.35:
            lx = rng2.randint(4, W - 4)
            ly = rng2.randint(4, H - 4)
            img[ly, max(0, lx - 3):min(W, lx + 3)] = rng2.uniform(0.0, 0.15)
            img[max(0, ly - 3):min(H, ly + 3), lx] = rng2.uniform(0.0, 0.15)
        # everything above is built from scalars/(H,W,1) fields, so img is exact grayscale here.
        # level 143 sits outside POST_LEVELS's exemption, so without this it would fall through
        # to the shared "camera color response" step below and get a random photographic white
        # balance tint — realistic for a photo, not for a scanner readout. Split explicitly instead:
        # most stay true grayscale (standard MRI/CT display); some get a deliberate pseudocolor
        # LUT (PET fusion / Doppler-flow style), which is a real, recognizable medical variant.
        gray = img.mean(axis=-1)
        if rng2.rand() < 0.5:
            lo_col = _photo_color(rng2, 0.0, 0.35)
            hi_col = _photo_color(rng2, 0.65, 1.0)
            for ch in range(C):
                img[:, :, ch] = lo_col[ch] + (hi_col[ch] - lo_col[ch]) * gray
        else:
            img[:] = gray.reshape(H, W, 1)

    # ── 144: CROWDS — groups of 3-20 people from a distance, interacting or walking.
    # Level 125 already does 1-4 people (occasionally 8-21); this is explicitly the
    # "crowd" regime: many small figures, far away, in a wide scene. ──
    elif lvl == 144:
        L = _rand_light(rng2)
        hz = rng2.randint(10, 24)
        _ground(rng2, img, hz, None, hi_key)
        # urban/plaza background: buildings or open sky
        if rng2.rand() < 0.5:
            sky = np.array([rng2.uniform(0.3, 0.6), rng2.uniform(0.35, 0.65), rng2.uniform(0.45, 0.8)])
            for r in range(H):
                t = min(1.0, max(0.0, (r - hz) / max(H - hz, 1)))
                img[r, :] = np.clip(img[r, :] * 0.3 + sky * (1 - t) * 0.7, 0, 1)
        n = rng2.randint(8, 26)
        # figures spread uniformly across the full width read as scattered noise; a crowd is a
        # loose cluster — bias positions toward one or two centers instead
        n_clusters = rng2.randint(1, 3)
        centers = [rng2.uniform(W * 0.15, W * 0.85) for _ in range(n_clusters)]
        for i in range(n):
            bx = np.clip(rng2.choice(centers) + rng2.uniform(-W * 0.22, W * 0.22), 1, W - 1)
            by = rng2.uniform(hz + 1, H - 1)
            depth = (by - hz) / max(H - hz, 1)
            # old size (1.0-3.5 * 0.3-1.0 depth) put the farthest figures under 1px across —
            # a crowd made of sub-pixel dots doesn't read as a crowd, just noise
            size = rng2.uniform(1.8, 4.5) * (0.45 + 0.55 * depth)
            cc = _photo_color(rng2, 0.05, 0.85)
            # simplified human: just a head + body (too small for limb detail)
            head_y = by - size * 0.35
            a = _soft_disc(bx, head_y, size * 0.28)
            for cc_i in range(C):
                img[:, :, cc_i] = np.clip(img[:, :, cc_i] * (1 - a * 0.85) + cc[cc_i] * a * 0.85, 0, 1)
            body_y0 = int(head_y + size * 0.15)
            body_y1 = int(by + size * 0.3)
            for r in range(max(0, body_y0), min(H, body_y1 + 1)):
                bw = size * 0.22 * (1 - abs(r - (body_y0 + body_y1) / 2) / max(body_y1 - body_y0, 1) * 0.8)
                x0 = int(np.clip(bx - bw, 0, W))
                x1 = int(np.clip(bx + bw, 0, W))
                img[r, x0:x1] = np.clip(img[r, x0:x1] * 0.2 + cc * 0.85, 0, 1)

    # ── 145: MACRO PHOTOGRAPHY — extreme close-up of everyday objects.
    # Coins, keys, buttons, jewelry, tools. The "face close-up" (85) equivalent for things. ──
    elif lvl == 145:
        # shallow DOF background (blurred, natural colors)
        bg = _backdrop(rng2, hi_key)
        bg = gaussian_filter(bg, sigma=(rng2.uniform(1.5, 3.5),) * 2 + (0,))
        img[:] = bg
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        L = _rand_light(rng2)
        typ = rng2.randint(0, 5)
        use_3d = rng2.rand() < 0.5
        if use_3d:  # same 5 object kinds, built as shaded 3D parts and shot from a random angle
            if typ == 0:  # coin — flat disc (short wide cylinder), edge or face depending on tilt
                metal = np.array([rng2.uniform(0.4, 0.7)] * 3) if rng2.rand() < 0.6 else np.array([rng2.uniform(0.7, 0.95), rng2.uniform(0.55, 0.8), rng2.uniform(0.2, 0.45)])
                parts = [('t', (0, 0, -0.06), (0, 0, 0.06), 0.85, 1.0, metal)]
            elif typ == 1:  # key — shaft + round bow + a couple of teeth stubs
                metal = np.array([rng2.uniform(0.55, 0.85)] * 3)
                parts = [('t', (0, 0.75, 0), (0, -0.25, 0), 0.09, 1.0, metal),
                         ('s', (0, -0.85, 0), 0.28, metal)]
                for _ in range(rng2.randint(1, 3)):
                    ty = rng2.uniform(0.15, 0.6)
                    parts.append(('t', (0, ty, 0), (0.3, ty, 0), 0.06, 1.0, metal))
            elif typ == 2:  # button — flat disc, no holes (needs boolean cut-out; plain rivet reads fine)
                btn_col = np.array([rng2.uniform(0.1, 0.5), rng2.uniform(0.1, 0.4), rng2.uniform(0.2, 0.6)])
                parts = [('t', (0, 0, -0.05), (0, 0, 0.05), 0.75, 1.0, btn_col)]
            elif typ == 3:  # jewelry / ring — torus approximated as a ring of small spheres + one gem
                metal = np.array([rng2.uniform(0.75, 1.0), rng2.uniform(0.55, 0.9), rng2.uniform(0.05, 0.3)])
                n = rng2.randint(10, 14)
                parts = [('s', (0.6 * math.cos(a), 0.6 * math.sin(a), 0), 0.16, metal * rng2.uniform(0.85, 1.15))
                         for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]
                gem = np.array([1.0, rng2.uniform(0.0, 0.5), rng2.uniform(0.0, 0.3)])
                parts.append(('s', (0.6, 0, -0.2), 0.24, gem))
            else:  # tool — handle + a wider head at one end (hammer / wrench silhouette)
                tool_col = np.array([rng2.uniform(0.15, 0.4)] * 3)
                parts = [('t', (0, 0.8, 0), (0, -0.6, 0), 0.1, 0.9, tool_col),
                         ('t', (-0.35, -0.7, 0), (0.35, -0.7, 0), 0.22, 1.0, tool_col)]
            _render_parts(img, parts, L, rng2)
        elif typ == 0:  # coin / disk — metallic gradient disc, slight tilt
            metal = np.array([rng2.uniform(0.4, 0.7)] * 3) if rng2.rand() < 0.6 else np.array([rng2.uniform(0.7, 0.95), rng2.uniform(0.55, 0.8), rng2.uniform(0.2, 0.45)])
            cx, cy = W / 2 + rng2.uniform(-3, 3), H / 2 + rng2.uniform(-2, 4)
            R = rng2.uniform(6, 13)
            a = _soft_disc(cx, cy, R)
            # slight gradient across the coin for metallic sheen
            sheen = 0.6 + 0.4 * (xx - cx) / max(R, 1)
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] * (1 - a) + metal[cc] * a * np.clip(sheen, 0, 1), 0, 1)
            # rim highlight
            rim_a = np.exp(-np.abs(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) - R) * 2.5) * 0.5
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] + rim_a * 0.35, 0, 1)
        elif typ == 1:  # key — shaft + head + teeth
            metal = np.array([rng2.uniform(0.55, 0.85)] * 3)
            kx, ky = W / 2 + rng2.uniform(-4, 4), H / 2 + rng2.uniform(-3, 5)
            ang = rng2.uniform(-0.6, 0.6)
            length = rng2.uniform(8, 18)
            ex = kx + math.cos(ang) * length
            ey = ky + math.sin(ang) * length
            _bresenham(img, int(kx), int(ky), int(ex), int(ey), int(rng2.uniform(1, 2)), metal)
            # key head: circle at one end
            a = _soft_disc(kx, ky, rng2.uniform(2.5, 5))
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] * (1 - a * 0.8) + metal[cc] * a * 0.8, 0, 1)
            # teeth: small perpendicular lines
            for _ in range(rng2.randint(1, 3)):
                tx = kx + length * rng2.uniform(0.5, 0.85)
                ty = ky
                _bresenham(img, int(tx), int(ty - 1.5), int(tx), int(ty + 1.5), 1, metal)
        elif typ == 2:  # button — round with 2 or 4 holes
            btn_col = np.array([rng2.uniform(0.1, 0.5), rng2.uniform(0.1, 0.4), rng2.uniform(0.2, 0.6)])
            bx, by = W / 2 + rng2.uniform(-2, 2), H / 2 + rng2.uniform(-2, 3)
            R = rng2.uniform(5, 11)
            a = _soft_disc(bx, by, R)
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] * (1 - a) + btn_col[cc] * a, 0, 1)
            # rim
            rim_a = np.exp(-np.abs(np.sqrt((yy - by) ** 2 + (xx - bx) ** 2) - R) * 2.0) * 0.4
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] + rim_a * 0.3, 0, 1)
            # holes
            n_holes = rng2.choice([2, 4])
            for h in range(n_holes):
                ha = h * 2 * math.pi / n_holes + rng2.uniform(-0.3, 0.3)
                hr = R * 0.4
                hx, hy = bx + math.cos(ha) * hr, by + math.sin(ha) * hr
                a = _soft_disc(hx, hy, rng2.uniform(0.5, 1.2))
                for cc in range(C):
                    img[:, :, cc] = np.clip(img[:, :, cc] * (1 - a * 0.6), 0, 1)
        elif typ == 3:  # jewelry / ring — bright metal band
            metal = np.array([rng2.uniform(0.75, 1.0), rng2.uniform(0.55, 0.9), rng2.uniform(0.05, 0.3)])
            jx, jy = W / 2 + rng2.uniform(-3, 3), H / 2 + rng2.uniform(-2, 3)
            R = rng2.uniform(4, 10)
            ring = np.exp(-np.abs(np.sqrt((yy - jy) ** 2 + (xx - jx) ** 2) - R) * rng2.uniform(1.5, 3.0))
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] + ring * metal[cc] * rng2.uniform(0.5, 1.0), 0, 1)
            # gem highlight
            a = _soft_disc(jx, jy - R * 0.35, rng2.uniform(1.5, 3.5))
            gem = np.array([1.0, rng2.uniform(0.0, 0.5), rng2.uniform(0.0, 0.3)])
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] + a * gem[cc] * 0.7, 0, 1)
        else:  # tool — wrench/hammer/screwdriver silhouette
            tool_col = np.array([rng2.uniform(0.15, 0.4)] * 3)
            tx, ty = W / 2 + rng2.uniform(-3, 3), H / 2 + rng2.uniform(-2, 3)
            ang = rng2.uniform(0, math.pi)
            length = rng2.uniform(8, 18)
            ex = tx + math.cos(ang) * length
            ey = ty + math.sin(ang) * length
            _bresenham(img, int(tx), int(ty), int(ex), int(ey), rng2.randint(1, 2), tool_col)
            # head: wider rectangle at one end
            hx0 = int(min(tx, ex) - 2)
            hx1 = int(max(tx, ex) + 2)
            hy0 = int(min(ty, ey) - rng2.randint(1, 3))
            hy1 = int(max(ty, ey) + rng2.randint(1, 3))
            for r in range(max(0, hy0), min(H, hy1 + 1)):
                img[r, max(0, hx0):min(W, hx1 + 1)] = tool_col

    # ── 146: SUNSET / SUNRISE — backlit silhouette + dramatic orange/pink/purple sky.
    # Levels 90 (silhouette) and 95 (sky/clouds) exist separately; this combines them with
    # the specific warm-horizon palette that photographs of sunsets always have. ──
    elif lvl == 146:
        # horizon biased lower than a uniform H//3-2H//3 draw: that let the flat black ground
        # silhouette eat up to 2/3 of the frame, burying the sky gradient that's the whole point
        hz = rng2.randint(H // 2, int(H * 0.8))
        sun_x = rng2.uniform(W * 0.25, W * 0.75)
        sun_y = hz + rng2.uniform(-2, 4)
        sun_R = rng2.uniform(2, 7)
        # sunset sky gradient: deep blue/purple → pink/orange → warm yellow at horizon
        top_col = np.array([rng2.uniform(0.05, 0.2), rng2.uniform(0.02, 0.1), rng2.uniform(0.3, 0.65)])
        mid_col = np.array([rng2.uniform(0.7, 0.95), rng2.uniform(0.2, 0.55), rng2.uniform(0.05, 0.3)])
        bot_col = np.array([rng2.uniform(0.9, 1.0), rng2.uniform(0.4, 0.75), rng2.uniform(0.0, 0.15)])
        for r in range(H):
            t = r / max(H - 1, 1)
            if t < 0.45:
                sky = top_col * (1 - t / 0.45) + mid_col * (t / 0.45)
            else:
                tt = (t - 0.45) / 0.55
                sky = mid_col * (1 - tt) + bot_col * tt
            img[r, :] = sky
        # sun: bright disc with glow
        for r in range(H):
            for c in range(W):
                d = math.sqrt((c - sun_x) ** 2 + (r - sun_y) ** 2)
                if d < sun_R:
                    t_in = d / sun_R
                    sun_col = np.array([1.0, 0.8 + 0.2 * (1 - t_in), 0.1 + 0.5 * (1 - t_in)])
                    img[r, c] = np.clip(img[r, c] * 0.2 + sun_col * 0.8, 0, 1)
        # sun glow (bloom + horizontal stretch — atmospheric scattering)
        glow = np.zeros((H, W), np.float32)
        for r in range(H):
            for c in range(W):
                d = math.sqrt((c - sun_x) ** 2 + ((r - sun_y) * 0.4) ** 2)
                glow[r, c] = np.exp(-d * 0.3) * np.exp(-abs(r - sun_y) * 0.02)
        glow_col = np.array([1.0, rng2.uniform(0.5, 0.85), rng2.uniform(0.0, 0.25)])
        for cc in range(C):
            img[:, :, cc] = np.clip(img[:, :, cc] + glow * glow_col[cc] * rng2.uniform(0.4, 0.8), 0, 1)
        # clouds: horizontal bands near horizon
        for _ in range(rng2.randint(2, 6)):
            cy = rng2.uniform(hz - 6, hz + 4)
            cx = rng2.uniform(3, W - 3)
            cw = rng2.uniform(3, 10)
            a = _soft_disc(cx, cy, cw)
            cloud_col = np.array([rng2.uniform(0.5, 0.9), rng2.uniform(0.2, 0.5), rng2.uniform(0.1, 0.3)])
            for cc in range(C):
                img[:, :, cc] = np.clip(img[:, :, cc] * (1 - a * 0.3) + cloud_col[cc] * a * 0.3, 0, 1)
        # ground silhouette (dark, featureless — the ground is backlit)
        gnd_col = np.array([rng2.uniform(0.0, 0.05)] * 3)
        for r in range(int(hz), H):
            t = (r - hz) / max(H - hz, 1)
            img[r, :] = gnd_col * (0.3 + 0.7 * t)
        # optional silhouette elements: trees, buildings
        if rng2.rand() < 0.6:
            for _ in range(rng2.randint(0, 3)):
                tx = rng2.uniform(0, W)
                th = rng2.uniform(4, hz * 0.7)
                _bresenham(img, int(tx), int(hz - 1), int(tx + rng2.uniform(-3, 3)), max(0, int(hz - th)), 1,
                           np.array([0.02, 0.02, 0.01]))
        # birds: tiny V marks in the sky
        if rng2.rand() < 0.4:
            for _ in range(rng2.randint(0, 4)):
                bx = rng2.uniform(4, W - 4)
                by = rng2.uniform(4, hz - 3)
                _bresenham(img, int(bx - 1), int(by + 1), int(bx), int(by), 0, np.ones(3) * 0.2)
                _bresenham(img, int(bx + 1), int(by + 1), int(bx), int(by), 0, np.ones(3) * 0.2)

    # ── 138: WOVEN / BASKET TEXTURE (interlaced over-under strands, tidy vs frayed — no
    # existing level has a two-direction crossing weave; _brushed is one direction only and
    # the indoor 'rug' patch (124) is a flat color fill, no pattern) ──
    elif lvl == 138:
        L = _rand_light(rng2)
        chaotic = rng2.rand() < 0.5               # regular grid weave vs hand-woven/frayed
        ang = rng2.choice([0.0, math.pi / 4]) + rng2.uniform(-0.08, 0.35 if chaotic else 0.08)
        ca, sa = math.cos(ang), math.sin(ang)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # camera-distance proxy, same idea as level 137: small P = many fine threads seen from
        # afar, large P = a few coarse threads filling the frame (macro/close-up weave)
        zoom = rng2.uniform(0.5, 3.0)
        if chaotic:                                # hand-woven: spacing wobbles, not a clean grid
            warp_n = (_perlin(rng2, 4) - 0.5) * rng2.uniform(1.0, 2.5) * zoom
            xx, yy = xx + warp_n, yy + warp_n
        u, v = xx * ca + yy * sa, -xx * sa + yy * ca
        P = rng2.uniform(2.0, 4.0) * zoom
        fu, fv = (u / P) % 1.0, (v / P) % 1.0
        prof_u = np.sqrt(np.clip(1 - (2 * fu - 1) ** 2, 0, 1))
        prof_v = np.sqrt(np.clip(1 - (2 * fv - 1) ** 2, 0, 1))
        over_u = (np.floor(u / P).astype(np.int32) + np.floor(v / P).astype(np.int32)) % 2 == 0
        height = np.where(over_u, prof_v, prof_u)  # checkerboard: which strand shows on top per cell
        if chaotic:
            fray = _perlin(rng2, 5) < rng2.uniform(0.05, 0.18)      # sparse frayed/broken patches
            height = np.where(fray, height * rng2.uniform(0.2, 0.5), height)
        gy, gx = np.gradient(height)
        Nz = 1.0 / np.sqrt(1 + gx ** 2 + gy ** 2)
        Nx, Ny = -gx * Nz, -gy * Nz
        col_a = _photo_color(rng2, 0.15, 0.85)
        col_b = np.clip(col_a * rng2.uniform(0.75, 1.25), 0, 1) if rng2.rand() < 0.6 else _photo_color(rng2, 0.15, 0.85)
        color = np.where(over_u[..., None], col_b, col_a)
        fiber = 0.85 + 0.3 * _perlin(rng2, 5)
        img[:] = np.clip(color * (_phong(Nx, Ny, Nz, L) * fiber)[..., None], 0, 1)

    # ── 71: CHAOS ──
    else:
        img = _1f_bg(rng2, 0.5)
        for _ in range(rng2.randint(0, 3)):
            cc = rng2.uniform(0.3, 1.0, C)
            _bresenham(img, rng2.randint(0, W), rng2.randint(0, H),
                       rng2.randint(0, W), rng2.randint(0, H), rng2.randint(0, 1), cc)
        for _ in range(rng2.randint(0, 2)):
            cx, cy = rng2.randint(6, W - 6), rng2.randint(6, H - 6)
            m = _hollow_blob(rng2, cx, cy, rng2.randint(3, 9), rng2.randint(3, 9))
            _fill(img, m, rng2.uniform(0.4, 1.0, C))
        for _ in range(rng2.randint(1, 3)):
            cx, cy = rng2.randint(4, W - 4), rng2.randint(4, H - 4)
            r = rng2.uniform(2, 6)
            m, _, _, _ = _organic_blob(rng2, cx, cy, r, r)
            _fill_alpha(img, m, rng2.uniform(0.1, 0.7, C), rng2.uniform(0.2, 0.6))
        for _ in range(rng2.randint(1, 3)):
            cx, cy = rng2.randint(5, W - 5), rng2.randint(5, H - 5)
            R = rng2.uniform(3, 7)
            t = rng2.randint(0, 4)
            L = _rand_light(rng2)
            cc = rng2.uniform(0.1, 1.0, C)
            if t == 0:
                m, Nx, Ny, Nz = _sphere(cx, cy, R)
            elif t == 1:
                m, Nx, Ny, Nz = _organic_blob(rng2, cx, cy, R, R * rng2.uniform(0.5, 2.2))
            else:
                m, Nx, Ny, Nz = _splat(rng2, cx, cy, R * rng2.uniform(0.5, 1.5))
            _draw_3d(img, m, Nx, Ny, Nz, L, cc)
        if rng2.randint(0, 3) == 0:
            _critter_2d(rng2, img, rng2.uniform(0.2, 0.9, C), rng2.randint(0, 5))
        if rng2.randint(0, 4) == 0:
            _gas_cloud(rng2, img)
        if rng2.randint(0, 2) == 0:
            img = _add_grain(img, rng2)
        if rng2.randint(0, 3) == 0:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            e = _edge_sobel(gray)
            for ch in range(C):
                img[:, :, ch] = np.where(e > 0.6, 1.0, img[:, :, ch])

    if lvl >= 82 and rng2.randint(0, 2) == 0:
        img = np.ascontiguousarray(img[:, ::-1])   # both facings exist in real data

    if lvl not in POST_LEVELS:          # post levels inherit both from their base render
        if rng2.randint(0, 3) > 0:      # camera color response: correlated RGB, moderate saturation
            img = _natural_color(img, rng2)
        if rng2.randint(0, 7) == 0:     # low-photon regime: noise scales with the signal
            img = _poisson(img, rng2)
        if rng2.randint(0, 3) == 0:     # a 32px photo is a resample: edges are never binary
            img = gaussian_filter(img, sigma=(rng2.uniform(0.3, 0.7),) * 2 + (0,))

    return np.clip(img, 0, 1).astype(np.float32)


def _validated_index(idx):
    if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, (int, np.integer)):
        raise TypeError("idx must be an integer")
    idx = int(idx)
    _level_block(idx)
    return idx


def _validated_step(step):
    if (
        isinstance(step, (bool, np.bool_))
        or not isinstance(step, (int, np.integer))
    ):
        raise TypeError("step must be an integer")
    step = int(step)
    if step < 1:
        raise ValueError("step must be positive")
    return step


def _validated_force_level(force_level):
    if force_level is None:
        return None
    if (
        isinstance(force_level, (bool, np.bool_))
        or not isinstance(force_level, (int, np.integer))
    ):
        raise TypeError("force_level must be an integer or None")
    force_level = int(force_level)
    if not 0 <= force_level < N_LEVELS:
        raise ValueError(f"force_level must be between 0 and {N_LEVELS - 1}")
    return force_level


def _validated_canvas_size(width=None, height=None, raster=None):
    """Resolve and validate the native render size for one public call."""
    explicit = width is not None or height is not None
    if width is None and height is None:
        if raster is not None:
            raster_width, raster_height = int(raster.width), int(raster.height)
            if (
                raster_width >= MIN_NATIVE_DIMENSION
                and raster_height >= MIN_NATIVE_DIMENSION
            ):
                width, height = raster_width, raster_height
            else:
                # Preserve the established tiny-raster contract: render the
                # smallest supported native canvas and reduce during format
                # conversion.
                width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
        else:
            width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    elif width is None or height is None:
        raise ValueError("width and height must be provided together")
    for name, value in (("width", width), ("height", height)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise TypeError(f"{name} must be an integer")
        if not MIN_NATIVE_DIMENSION <= int(value) <= MAX_CANVAS_DIMENSION:
            raise ValueError(
                f"{name} must be between {MIN_NATIVE_DIMENSION} and "
                f"{MAX_CANVAS_DIMENSION} for native rendering"
            )
    width, height = int(width), int(height)
    if width * height > MAX_CANVAS_PIXELS:
        raise ValueError(
            f"width * height must not exceed {MAX_CANVAS_PIXELS} pixels"
        )
    if explicit and raster is not None and (
        int(raster.width) != width or int(raster.height) != height
    ):
        raise ValueError(
            "explicit width and height must match RasterSpec.width and height"
        )
    return width, height


def _make_image_unlocked(idx, step, force_level, raster, width, height):
    """Render one already-validated index while the caller owns the state lock."""
    with _canvas_size(width, height):
        with _indexed_geometry_mode(
            width != DEFAULT_WIDTH or height != DEFAULT_HEIGHT
        ):
            image = _apply_scene(
                _render_image(idx, step, force_level), _scene(idx)
            )
    if raster is None:
        return image
    source = (
        _indexed_rgba(image, idx, force_level, raster)
        if _raster_requests_alpha(raster)
        else image
    )
    return convert_raster(source, raster)


def make_image(
    idx: int,
    step: int = 7,
    force_level: int | None = None,
    raster: RasterSpec | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    """Generate a deterministic image from one non-negative integer index.

    Args:
        idx:  Integer seed — same idx always returns the same image.
        step: Positive skip between consecutive hash-indices (default 7).
        force_level: If set, use this level instead of deriving from idx.
        raster: Optional native resolution, color mode, and bit depth.
        width: Native canvas width. Must be supplied together with ``height``.
        height: Native canvas height. Must be supplied together with ``width``.

    Returns:
        By default, a float32 array of shape (32, 32, 3) in [0, 1].
        With ``width`` and ``height``, a native ``(height, width, 3)`` render.
        When ``raster`` is provided, an array matching that specification.

    The scene attributes go on here, outside _render_image, because the level code
    recurses through _render_image for its style-filter and symmetry levels — inside,
    a level-136 image would get idx's brightness stacked on the base index's.
    """
    idx = _validated_index(idx)
    step = _validated_step(step)
    force_level = _validated_force_level(force_level)
    if raster is not None:
        _validated_raster_spec(raster)
    width, height = _validated_canvas_size(width, height, raster)
    with _RENDER_LOCK:
        return _make_image_unlocked(
            idx, step, force_level, raster, width, height
        )


BATCH_BACKENDS: tuple[str, ...] = ("auto", "serial", "process", "webgpu")
BATCH_FIDELITIES: tuple[str, ...] = ("legacy", "portable")
_TRANSFORM_LEVELS = {129, 130, 133, 135, 136}
_PRIMITIVE_LEVELS = {0, 1, 2, 3, 4, 6, 15, 16, 17, 18, 128, 142, 144}
GPU_LEVEL_FAMILIES = {
    **{level: "wave" for level in WAVE_LEVELS},
    113: "reaction_diffusion",
    **{level: "primitive_ir" for level in _PRIMITIVE_LEVELS},
    **{level: "transform" for level in _TRANSFORM_LEVELS},
}
_RENDER_GRAPH_STAGES = {
    # Start a preparation-free family first. While its command buffer is on
    # the GPU, CPU workers can prepare the dependent branches below.
    "wave": ("gpu",),
    "reaction_diffusion": ("prepare", "gpu", "finish"),
    "transform": ("prepare", "gpu", "finish"),
    "primitive_ir": ("prepare", "gpu", "finish"),
}
_AUTO_WEBGPU_MIN_IMAGES = 64
_AUTO_WEBGPU_RESULTS = {}
_AUTO_WEBGPU_LOCK = threading.Lock()
_AUTO_CPU_EXECUTORS = {}
_AUTO_CPU_EXECUTORS_LOCK = threading.Lock()
_CPU_TUNING_RESULTS = {}
_CPU_TUNING_LOCK = threading.Lock()
_CPU_AUTOTUNE_MIN_IMAGES = 256
_CPU_AUTOTUNE_TTL_SECONDS = 15 * 60
_WORKER_THREADPOOL_LIMITER = None
_CPU_WORKER_WARMED = False


def _make_image_task(arguments):
    """Picklable worker entry point used by the process backend."""
    if len(arguments) == 4:
        idx, step, force_level, raster = arguments
        width = height = None
    else:
        idx, step, force_level, raster, width, height = arguments
    return make_image(
        idx, step=step, force_level=force_level, raster=raster,
        width=width, height=height,
    )


def _worker_pid(_value):
    """Small picklable task used to start a clean persistent CPU pool."""
    return os.getpid()


def _cpu_worker_init():
    """Prevent nested native thread pools from multiplying every process."""
    global _WORKER_THREADPOOL_LIMITER
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        pass
    else:
        _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    _warm_cpu_worker()


def _cpu_worker_init_unrestricted():
    """Warm a worker while retaining the native runtime's thread policy."""
    _warm_cpu_worker()


def _warm_cpu_worker():
    """Initialize core renderer state once in each persistent worker."""
    global _CPU_WORKER_WARMED
    if _CPU_WORKER_WARMED:
        return
    if os.environ.get("SIG_CPU_WORKER_WARMUP", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        _CPU_WORKER_WARMED = True
        return
    make_image(0, force_level=0)
    _CPU_WORKER_WARMED = True


def _read_cpu_topology_value(cpu, name):
    try:
        with open(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/{name}",
            encoding="ascii",
        ) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _cgroup_cpu_limit():
    """Return an effective cgroup CPU quota, or None when unlimited."""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="ascii") as handle:
            quota, period = handle.read().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    try:
        with open(
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="ascii"
        ) as handle:
            quota = int(handle.read())
        with open(
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="ascii"
        ) as handle:
            period = int(handle.read())
        if quota > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return None


def _cpu_topology():
    logical_system = os.cpu_count() or 1
    try:
        affinity = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        affinity = tuple(range(logical_system))
    quota = _cgroup_cpu_limit()
    available_logical = len(affinity)
    if quota is not None:
        available_logical = min(available_logical, quota)
    physical_ids = set()
    for cpu in affinity:
        package = _read_cpu_topology_value(cpu, "physical_package_id")
        core = _read_cpu_topology_value(cpu, "core_id")
        if package is not None and core is not None:
            physical_ids.add((package, core))
    physical = len(physical_ids) or available_logical
    physical = max(1, min(physical, available_logical))
    model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    numa_nodes = 1
    try:
        numa_nodes = max(
            1,
            sum(
                name.startswith("node") and name[4:].isdigit()
                for name in os.listdir("/sys/devices/system/node")
            ),
        )
    except OSError:
        pass
    return {
        "model": model,
        "logical_system": logical_system,
        "affinity_cpus": affinity,
        "affinity_count": len(affinity),
        "quota_cpus": quota,
        "available_logical": max(1, available_logical),
        "physical_cores": physical,
        "smt_threads_per_core": max(
            1, math.ceil(max(1, available_logical) / physical)
        ),
        "numa_nodes": numa_nodes,
    }


def _recommended_chunksize(count, workers):
    target = max(1, math.ceil(count / max(1, workers * 8)))
    target = min(target, 32)
    lower = 1 << max(0, target.bit_length() - 1)
    upper = min(32, lower * 2)
    return lower if target - lower <= upper - target else upper


def _cpu_workload_key(levels, count, raster, topology):
    histogram = np.bincount(levels, minlength=N_LEVELS)
    quantized = np.rint(histogram * 32 / max(count, 1)).astype(np.uint8)
    distribution = hashlib.blake2b(
        quantized.tobytes(), digest_size=8
    ).hexdigest()
    scale = 1 << max(0, min(12, count.bit_length() - 1))
    raster_key = None
    if raster is not None:
        raster_key = (
            int(raster.width),
            int(raster.height),
            str(raster.mode),
            str(raster.bits_per_channel),
            str(raster.resize),
            str(raster.dither),
            bool(raster.packed),
        )
    return (
        topology["affinity_cpus"],
        topology["quota_cpus"],
        topology["physical_cores"],
        scale,
        distribution,
        raster_key,
    )


def _cpu_autotune_enabled():
    return os.environ.get("SIG_CPU_AUTOTUNE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _select_cpu_configuration(
    batch_indices,
    levels,
    *,
    step,
    force_level,
    raster,
    requested_chunksize,
    start_method,
):
    """Empirically select process count and chunk size for this CPU/workload."""
    topology = _cpu_topology()
    count = len(batch_indices)
    fallback_workers = min(count, topology["physical_cores"])
    fallback_chunk = (
        _recommended_chunksize(count, fallback_workers)
        if requested_chunksize is None
        else requested_chunksize
    )
    key = _cpu_workload_key(levels, count, raster, topology)
    now = time.monotonic()
    safe_to_measure = (
        _cpu_autotune_enabled()
        and count >= _CPU_AUTOTUNE_MIN_IMAGES
        and not multiprocessing.current_process().daemon
        and (
            start_method != "auto"
            or (
                sys.platform.startswith("linux")
                and threading.active_count() == 1
            )
        )
    )
    with _CPU_TUNING_LOCK:
        cached = _CPU_TUNING_RESULTS.get(key)
        if (
            cached is not None
            and now - cached["measured_at"] < _CPU_AUTOTUNE_TTL_SECONDS
        ):
            return (
                cached["workers"],
                cached["chunksize"],
                cached["limit_native_threads"],
            )
        if not safe_to_measure:
            compatible = [
                value
                for candidate_key, value in _CPU_TUNING_RESULTS.items()
                if candidate_key[:3] == key[:3]
            ]
            if compatible:
                recent = max(
                    compatible, key=lambda value: value["measured_at"]
                )
                return (
                    recent["workers"],
                    (
                        recent["chunksize"]
                        if requested_chunksize is None
                        else requested_chunksize
                    ),
                    recent["limit_native_threads"],
                )
            return fallback_workers, fallback_chunk, True

        sample_count = min(count, 1024)
        positions = np.linspace(
            0, count - 1, sample_count, dtype=np.int64
        )
        sample_indices = [batch_indices[int(position)] for position in positions]
        available = min(sample_count, topology["available_logical"])
        physical = min(available, topology["physical_cores"])
        candidates = {
            physical,
            min(available, physical + max(1, physical // 2)),
            available,
        }
        if count < 1024:
            candidates.add(max(1, physical // 2))
        if count < 512:
            candidates.add(1)
        candidates = sorted(value for value in candidates if value >= 1)
        repeats = 1
        timings = {}
        from concurrent.futures import ProcessPoolExecutor

        selected_start_method = _selected_process_start_method(start_method)
        context = multiprocessing.get_context(selected_start_method)

        def tasks():
            return (
                (idx, step, force_level, raster) for idx in sample_indices
            )

        def consume(executor, candidate_chunk):
            for _image in executor.map(
                _make_image_task, tasks(), chunksize=candidate_chunk
            ):
                pass

        def measure_pool(
            candidate_workers, chunk_values, limit_native_threads
        ):
            initializer = (
                _cpu_worker_init
                if limit_native_threads
                else _cpu_worker_init_unrestricted
            )
            with ProcessPoolExecutor(
                max_workers=candidate_workers,
                mp_context=context,
                initializer=initializer,
            ) as executor:
                list(
                    executor.map(
                        _worker_pid,
                        range(candidate_workers),
                        chunksize=1,
                    )
                )
                warm_chunk = next(iter(chunk_values))
                consume(executor, warm_chunk)
                for candidate_chunk in chunk_values:
                    values = []
                    for _ in range(repeats):
                        started = time.perf_counter()
                        consume(executor, candidate_chunk)
                        values.append(time.perf_counter() - started)
                    timings[
                        (
                            candidate_workers,
                            candidate_chunk,
                            limit_native_threads,
                        )
                    ] = min(values)

        for candidate_workers in candidates:
            candidate_chunk = (
                requested_chunksize
                if requested_chunksize is not None
                else _recommended_chunksize(
                    sample_count, candidate_workers
                )
            )
            measure_pool(candidate_workers, (candidate_chunk,), True)
        best_workers, best_chunk, best_limit = min(
            timings, key=timings.get
        )
        measure_pool(best_workers, (best_chunk,), False)
        best_workers, best_chunk, best_limit = min(
            timings, key=timings.get
        )
        if requested_chunksize is None and best_workers > 1:
            chunk_candidates = {
                max(1, best_chunk // 2),
                best_chunk,
                min(64, best_chunk * 2),
            }
            missing_chunks = tuple(
                candidate_chunk
                for candidate_chunk in sorted(chunk_candidates)
                if (
                    best_workers,
                    candidate_chunk,
                    best_limit,
                )
                not in timings
            )
            if missing_chunks:
                measure_pool(best_workers, missing_chunks, best_limit)
            best_workers, best_chunk, best_limit = min(
                timings, key=timings.get
            )

        selected_executor = _auto_cpu_executor(
            best_workers,
            start_method,
            limit_native_threads=best_limit,
        )
        warm_count = min(count, 2048)
        warm_positions = np.linspace(
            0, count - 1, warm_count, dtype=np.int64
        )
        warm_tasks = (
            (
                batch_indices[int(position)],
                step,
                force_level,
                raster,
            )
            for position in warm_positions
        )
        warm_batch = _collect_batch(
            selected_executor.map(
                _make_image_task, warm_tasks, chunksize=best_chunk
            ),
            warm_count,
        )
        del warm_batch

        _CPU_TUNING_RESULTS[key] = {
            "workers": int(best_workers),
            "chunksize": int(best_chunk),
            "limit_native_threads": bool(best_limit),
            "measured_at": now,
            "sample_count": sample_count,
            "candidates": {
                (
                    f"{workers}x{chunk}:"
                    f"{'native1' if limited else 'native-auto'}"
                ): seconds
                for (workers, chunk, limited), seconds in sorted(
                    timings.items()
                )
            },
        }
        return int(best_workers), int(best_chunk), bool(best_limit)


def cpu_info():
    """Describe usable CPU capacity and recent automatic tuning decisions."""
    topology = _cpu_topology()
    with _CPU_TUNING_LOCK:
        tuning = [
            {
                "workers": value["workers"],
                "chunksize": value["chunksize"],
                "native_threads": (
                    1
                    if value["limit_native_threads"]
                    else "runtime-default"
                ),
                "sample_count": value["sample_count"],
                "age_seconds": max(
                    0.0, time.monotonic() - value["measured_at"]
                ),
                "candidates": dict(value["candidates"]),
            }
            for value in _CPU_TUNING_RESULTS.values()
        ]
    try:
        from threadpoolctl import threadpool_info

        native_runtimes = [
            {
                "api": value.get("user_api"),
                "runtime": value.get("internal_api"),
                "threads": value.get("num_threads"),
                "version": value.get("version"),
                "architecture": value.get("architecture"),
            }
            for value in threadpool_info()
        ]
    except ImportError:
        native_runtimes = []
    try:
        load_average = tuple(float(value) for value in os.getloadavg())
    except (AttributeError, OSError):
        load_average = ()
    return {
        **topology,
        "autotune": _cpu_autotune_enabled(),
        "autotune_min_images": _CPU_AUTOTUNE_MIN_IMAGES,
        "native_thread_policy": "empirical",
        "native_runtimes": native_runtimes,
        "load_average": load_average,
        "tuning": tuning,
    }


def _selected_process_start_method(start_method):
    if start_method == "auto":
        safe_to_fork = (
            sys.platform.startswith("linux") and threading.active_count() == 1
        )
        return "fork" if safe_to_fork else "spawn"
    return start_method


def _auto_cpu_executor(
    workers, start_method, *, limit_native_threads=True
):
    """Return workers created before a GPU runtime starts native threads."""
    from concurrent.futures import ProcessPoolExecutor

    selected_start_method = _selected_process_start_method(start_method)
    key = (int(workers), selected_start_method, bool(limit_native_threads))
    with _AUTO_CPU_EXECUTORS_LOCK:
        executor = _AUTO_CPU_EXECUTORS.get(key)
        if executor is None:
            context = multiprocessing.get_context(selected_start_method)
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=(
                    _cpu_worker_init
                    if limit_native_threads
                    else _cpu_worker_init_unrestricted
                ),
            )
            # ProcessPoolExecutor launches all configured workers on its first
            # submission. Do that while the parent is still GPU-free.
            list(executor.map(_worker_pid, range(workers), chunksize=1))
            _AUTO_CPU_EXECUTORS[key] = executor
        return executor


def _empty_batch(raster, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    if raster is None:
        return np.empty((0, int(height), int(width), C), np.float32)
    mode, bits, _resize, _dither, packed = _validated_raster_spec(raster)
    height, width = int(raster.height), int(raster.width)
    if mode == "binary":
        if packed:
            return np.empty((0, height, math.ceil(width / 8)), np.uint8)
        return np.empty((0, height, width), np.bool_)
    if mode == "grayscale":
        return np.empty(
            (0, height, width), _unsigned_dtype(bits[0])
        )
    channels = 3 if mode == "rgb" else 4
    if packed:
        return np.empty(
            (0, height, width), _unsigned_dtype(sum(bits))
        )
    return np.empty(
        (0, height, width, channels), _unsigned_dtype(max(bits))
    )


def _collect_batch(results, count):
    """Copy ordered worker results directly into one final contiguous array."""
    iterator = iter(results)
    first = np.asarray(next(iterator))
    batch = np.empty((count, *first.shape), dtype=first.dtype)
    batch[0] = first
    for position, image in enumerate(iterator, start=1):
        value = np.asarray(image)
        if value.shape != first.shape or value.dtype != first.dtype:
            raise RuntimeError("batch backend returned inconsistent image layouts")
        batch[position] = value
    return batch


def _submit_render_graph_preparation(
    graph, executor, specifications, *, chunksize
):
    """Submit every CPU→GPU dependency before unrelated CPU render work.

    ``specifications`` maps a family to ``(callable, arguments)``. Each
    argument value is passed as the callable's single picklable argument.
    ``map`` retains process-pool chunking while its eager submission preserves
    graph priority. Keeping this policy separate makes the rule testable.
    """
    pending = {}
    for node in graph.topological_nodes():
        if node.stage != "prepare" or node.family not in specifications:
            continue
        function, arguments = specifications[node.family]
        pending[node.family] = executor.map(
            function, arguments, chunksize=chunksize
        )
    return pending


def _resolved_preparation(pending, family, fallback):
    results = pending.get(family)
    if results is None:
        return list(fallback())
    return list(results)


def _portable_webgpu_batch(
    batch_indices,
    *,
    step,
    force_level,
    raster,
    workers,
    chunksize,
    start_method,
    enabled_families=None,
    shared_executor=None,
):
    """Render homogeneous GPU groups while CPU workers handle the remainder."""
    from ._webgpu import render_wave_images

    active_families = (
        set(GPU_LEVEL_FAMILIES.values())
        if enabled_families is None
        else set(enabled_families)
    )
    levels = [
        int(force_level) if force_level is not None else _level_of(idx)
        for idx in batch_indices
    ]
    graph = RenderGraph.compile(
        levels,
        GPU_LEVEL_FAMILIES,
        _RENDER_GRAPH_STAGES,
        enabled_families=active_families,
    )
    wave_positions = list(graph.positions("wave"))
    reaction_positions = list(graph.positions("reaction_diffusion"))
    transform_positions = list(graph.positions("transform"))
    primitive_positions = list(graph.positions("primitive_ir"))
    cpu_positions = list(graph.positions("cpu"))
    reaction_indices = [
        batch_indices[position] for position in reaction_positions
    ]
    transform_indices = [
        batch_indices[position] for position in transform_positions
    ]
    transform_levels = [levels[position] for position in transform_positions]
    primitive_indices = [
        batch_indices[position] for position in primitive_positions
    ]
    primitive_levels = [levels[position] for position in primitive_positions]

    empty = _empty_batch(raster)
    batch = np.empty(
        (len(batch_indices), *empty.shape[1:]), dtype=empty.dtype
    )
    cpu_indices = [batch_indices[position] for position in cpu_positions]
    cpu_executor = shared_executor
    owns_executor = False
    cpu_iterator = None
    pending_preparation = {}
    parallel_work_count = (
        len(cpu_indices)
        + len(reaction_indices)
        + len(transform_indices)
        + len(primitive_indices)
    )
    if parallel_work_count:
        selected_workers = (
            max(
                1,
                min(
                    parallel_work_count,
                    ((os.cpu_count() or 1) + 1) // 2,
                ),
            )
            if workers is None
            else workers
        )
        use_processes = (
            parallel_work_count >= 64
            and selected_workers > 1
            and not multiprocessing.current_process().daemon
        )
        if use_processes:
            from concurrent.futures import ProcessPoolExecutor

            if cpu_executor is None:
                if start_method == "auto":
                    safe_to_fork = (
                        sys.platform.startswith("linux")
                        and threading.active_count() == 1
                    )
                    selected_start_method = (
                        "fork" if safe_to_fork else "spawn"
                    )
                else:
                    selected_start_method = start_method
                context = multiprocessing.get_context(selected_start_method)
                cpu_executor = ProcessPoolExecutor(
                    max_workers=selected_workers,
                    mp_context=context,
                    initializer=_cpu_worker_init,
                )
                owns_executor = True
            preparation_specs = {
                "reaction_diffusion": (
                    _prepare_reaction_task,
                    [(idx, step) for idx in reaction_indices],
                ),
                "transform": (
                    _prepare_transform_task,
                    [
                        (idx, level, step)
                        for idx, level in zip(
                            transform_indices, transform_levels
                        )
                    ],
                ),
                "primitive_ir": (
                    _prepare_primitive_plan_task,
                    [
                        (idx, level, step)
                        for idx, level in zip(
                            primitive_indices, primitive_levels
                        )
                    ],
                ),
            }
            pending_preparation = _submit_render_graph_preparation(
                graph,
                cpu_executor,
                preparation_specs,
                chunksize=chunksize,
            )
            if cpu_indices:
                tasks = (
                    (idx, step, force_level, raster) for idx in cpu_indices
                )
                # GPU dependencies were queued first. General CPU rendering
                # now overlaps the device without starving its next branch.
                cpu_iterator = cpu_executor.map(
                    _make_image_task, tasks, chunksize=chunksize
                )

    try:
        if wave_positions:
            gpu_indices = [
                batch_indices[position] for position in wave_positions
            ]
            gpu_levels = [levels[position] for position in wave_positions]
            gpu_images = render_wave_images(gpu_indices, gpu_levels)
            if raster is not None:
                converted = []
                for idx, image in zip(gpu_indices, gpu_images):
                    source = (
                        _indexed_rgba(image, idx, force_level, raster)
                        if _raster_requests_alpha(raster)
                        else image
                    )
                    converted.append(convert_raster(source, raster))
                gpu_images = _collect_batch(converted, len(converted))
            batch[np.asarray(wave_positions)] = gpu_images

        if reaction_positions:
            prepared_reactions = _resolved_preparation(
                pending_preparation,
                "reaction_diffusion",
                lambda: (
                    _prepare_reaction_item(idx, step)
                    for idx in reaction_indices
                ),
            )
            reaction_images = _portable_reaction_batch(
                reaction_indices,
                step=step,
                force_level=force_level,
                raster=raster,
                prepared=prepared_reactions,
                executor=cpu_executor,
                chunksize=chunksize,
            )
            batch[np.asarray(reaction_positions)] = reaction_images

        if transform_positions:
            prepared = _resolved_preparation(
                pending_preparation,
                "transform",
                lambda: (
                    _prepare_transform_item(idx, level, step)
                    for idx, level in zip(
                        transform_indices, transform_levels
                    )
                ),
            )
            transform_images = _finish_portable_transform_batch(
                transform_indices,
                prepared,
                force_level=force_level,
                raster=raster,
                executor=cpu_executor,
                chunksize=chunksize,
            )
            batch[np.asarray(transform_positions)] = transform_images

        if primitive_positions:
            prepared_plans = _resolved_preparation(
                pending_preparation,
                "primitive_ir",
                lambda: (
                    _prepare_primitive_plan(idx, level, step)
                    for idx, level in zip(
                        primitive_indices, primitive_levels
                    )
                ),
            )
            primitive_images = _finish_portable_primitive_batch(
                primitive_indices,
                prepared_plans,
                force_level=force_level,
                raster=raster,
                executor=cpu_executor,
                chunksize=chunksize,
            )
            batch[np.asarray(primitive_positions)] = primitive_images

        if cpu_indices:
            if cpu_iterator is not None:
                cpu_images = _collect_batch(cpu_iterator, len(cpu_indices))
            else:
                cpu_images = make_images(
                    cpu_indices,
                    step=step,
                    force_level=force_level,
                    raster=raster,
                    backend="serial",
                    workers=1,
                    chunksize=chunksize,
                    start_method=start_method,
                    fidelity="legacy",
                )
            batch[np.asarray(cpu_positions)] = cpu_images
        return batch
    finally:
        if owns_executor and cpu_executor is not None:
            cpu_executor.shutdown(wait=True, cancel_futures=True)


def _prepare_reaction_item(idx, step):
    """Advance the exact level-113 RNG stream up to its simulation."""
    block = _level_block(idx)
    lidx = idx % SAMPLES_PER_LEVEL
    rng2 = _rng(lidx * max(1, step) + block * 100000)
    hi_key = rng2.randint(0, 4) == 0
    if rng2.randint(0, 8) > 0:
        _backdrop(rng2, hi_key)
    if hi_key and rng2.randint(0, 2) == 0:
        rng2.uniform(0.1, 0.5)
    if rng2.randint(0, 5) == 0:
        rng2.uniform(0.15, 0.7)
    if rng2.randint(0, 12) == 0:
        rng2.uniform(0.1, 6.2)
    _rand_light_col(rng2)
    state = _reaction_diffusion_initial(rng2)
    return rng2, hi_key, state


def _prepare_reaction_task(arguments):
    """Picklable reaction-planning entry point for the RenderGraph."""
    return _prepare_reaction_item(*arguments)


def _finish_reaction_item(rng2, hi_key, field):
    """Finish level 113 after its GPU-computed simulation field."""
    c1, c2 = _photo_color(rng2, 0.05, 0.6), _photo_color(rng2, 0.3, 0.98)
    texture = field.reshape(H, W, 1)
    if rng2.randint(0, 2) == 0:
        texture = (
            field > rng2.uniform(0.35, 0.65)
        ).astype(np.float32).reshape(H, W, 1)
    image = c1.reshape(1, 1, C) * (1 - texture) + c2.reshape(1, 1, C) * texture
    if rng2.randint(0, 2) == 0:
        light = _rand_light(rng2)
        mask, nx, ny, nz = _organic_blob(
            rng2,
            W / 2 + rng2.uniform(-4, 4),
            H / 2 + rng2.uniform(-4, 4),
            rng2.uniform(8, 15),
            rng2.uniform(8, 15),
        )
        lit = _phong(nx, ny, nz, light)
        background = _backdrop(rng2, hi_key)
        body = image.copy()
        image = background
        for channel in range(C):
            image[:, :, channel][mask] = np.clip(
                body[:, :, channel][mask] * lit[mask], 0, 1
            )
    if rng2.randint(0, 2) == 0:
        image = np.ascontiguousarray(image[:, ::-1])
    if rng2.randint(0, 3) > 0:
        image = _natural_color(image, rng2)
    if rng2.randint(0, 7) == 0:
        image = _poisson(image, rng2)
    if rng2.randint(0, 3) == 0:
        image = gaussian_filter(
            image, sigma=(rng2.uniform(0.3, 0.7),) * 2 + (0,)
        )
    return np.clip(image, 0, 1).astype(np.float32)


def _finish_reaction_item_task(arguments):
    """Finish scene/raster work for one GPU-computed reaction field."""
    idx, prepared, field, force_level, raster = arguments
    rng2, hi_key, _state = prepared
    image = _apply_scene(
        _finish_reaction_item(rng2, hi_key, field), _scene(idx)
    )
    if raster is not None:
        source = (
            _indexed_rgba(image, idx, force_level, raster)
            if _raster_requests_alpha(raster)
            else image
        )
        image = convert_raster(source, raster)
    return image


def _portable_reaction_batch(
    batch_indices,
    *,
    step,
    force_level,
    raster,
    prepared=None,
    executor=None,
    chunksize=32,
):
    """Render a homogeneous level-113 batch with GPU simulation and CPU finishing."""
    from ._webgpu import render_reaction_fields

    if prepared is None:
        prepared = [
            _prepare_reaction_item(idx, step) for idx in batch_indices
        ]
    initial_u = np.stack([item[2][0] for item in prepared])
    initial_v = np.stack([item[2][1] for item in prepared])
    parameters = np.asarray(
        [
            [item[2][2], item[2][3], item[2][4], 0.0]
            for item in prepared
        ],
        np.float32,
    )
    fields = render_reaction_fields(initial_u, initial_v, parameters)
    tasks = (
        (idx, item, field, force_level, raster)
        for idx, item, field in zip(batch_indices, prepared, fields)
    )
    images = (
        executor.map(
            _finish_reaction_item_task, tasks, chunksize=chunksize
        )
        if executor is not None
        else map(_finish_reaction_item_task, tasks)
    )
    return _collect_batch(images, len(batch_indices))


def _prepare_transform_item(idx, level, step):
    """Build the CPU control-plane data for one recursive GPU transform."""
    block = _level_block(idx)
    lidx = idx % SAMPLES_PER_LEVEL
    rng2 = _rng(lidx * max(1, step) + block * 100000)
    base_level = rng2.randint(0, N_LEVELS)
    while base_level in RECURSIVE_LEVELS:
        base_level = rng2.randint(0, N_LEVELS)
    base_idx = (
        _level_start(base_level, rng2) + rng2.randint(0, SAMPLES_PER_LEVEL)
    )
    base = _render_image(base_idx, step, force_level=base_level)
    parameters = np.zeros(12, np.float32)
    displacement = np.zeros((H, W, 2), np.float32)
    if level == 129:
        parameters[0] = 0
        parameters[1] = rng2.uniform(0, math.pi)
    elif level == 130:
        parameters[0] = 1
        parameters[1] = rng2.randint(4, 11)
        parameters[2] = rng2.uniform(0, 2 * math.pi)
    elif level == 133:
        parameters[0] = 2
        parameters[1] = rng2.uniform(0, math.pi)
        parameters[2] = rng2.uniform(1.5, 4.0)
        parameters[3] = rng2.uniform(0.8, 2.2)
    elif level == 135:
        parameters[0] = 3
        strength = rng2.uniform(1.5, 4.0)
        displacement[:, :, 0] = (
            _perlin(rng2, rng2.randint(3, 5)) - 0.5
        ) * strength
        displacement[:, :, 1] = (
            _perlin(rng2, rng2.randint(3, 5)) - 0.5
        ) * strength
    elif level == 136:
        parameters[0] = 4
        light = _rand_light(rng2)
        parameters[1] = rng2.uniform(7, W - 7)
        parameters[2] = rng2.uniform(6, H - 6)
        parameters[3] = rng2.uniform(5, 11)
        parameters[4] = rng2.uniform(1.6, 3.0) * (
            -1 if rng2.rand() < 0.4 else 1
        )
        parameters[5:8] = light
    else:
        raise ValueError(f"level {level} is not a portable transform")
    return base, parameters, displacement


def _prepare_transform_task(arguments):
    """Picklable transform-planning entry point for the CPU process pool."""
    return _prepare_transform_item(*arguments)


def _portable_transform_batch(
    batch_indices,
    levels,
    *,
    step,
    force_level,
    raster,
):
    prepared = [
        _prepare_transform_item(idx, level, step)
        for idx, level in zip(batch_indices, levels)
    ]
    return _finish_portable_transform_batch(
        batch_indices,
        prepared,
        force_level=force_level,
        raster=raster,
    )


def _finish_portable_transform_batch(
    batch_indices,
    prepared,
    *,
    force_level,
    raster,
    executor=None,
    chunksize=32,
):
    """Submit prepared transforms and finish their scene/raster pipeline."""
    from ._webgpu import render_transform_images

    bases = np.stack([item[0] for item in prepared])
    parameters = np.stack([item[1] for item in prepared])
    displacement = np.stack([item[2] for item in prepared])
    transformed = render_transform_images(bases, parameters, displacement)
    tasks = (
        (idx, image, force_level, raster)
        for idx, image in zip(batch_indices, transformed)
    )
    images = (
        executor.map(
            _finish_transform_item_task, tasks, chunksize=chunksize
        )
        if executor is not None
        else map(_finish_transform_item_task, tasks)
    )
    return _collect_batch(images, len(batch_indices))


def _finish_transform_item_task(arguments):
    """Finish one transform graph branch after its GPU stage."""
    idx, image, force_level, raster = arguments
    image = _apply_scene(np.clip(image, 0, 1), _scene(idx))
    if raster is not None:
        source = (
            _indexed_rgba(image, idx, force_level, raster)
            if _raster_requests_alpha(raster)
            else image
        )
        image = convert_raster(source, raster)
    return image


def _prepare_primitive_preamble(idx, level, step):
    """Reproduce the shared legacy preamble and return its live RNG stream."""
    block = _level_block(idx)
    lidx = idx % SAMPLES_PER_LEVEL
    rng2 = _rng(lidx * max(1, step) + block * 100000)
    initial = np.zeros((H, W, C), np.float32)
    hi_key = rng2.randint(0, 4) == 0
    if level not in POST_LEVELS and rng2.randint(0, 8) > 0:
        initial += _backdrop(rng2, hi_key)
    subject_gain = (
        rng2.uniform(0.1, 0.5)
        if hi_key and rng2.randint(0, 2) == 0
        else 1.0
    )
    rim = (
        rng2.uniform(0.15, 0.7)
        if rng2.randint(0, 5) == 0
        else 0.0
    )
    iridescence = (
        rng2.uniform(0.1, 6.2)
        if rng2.randint(0, 12) == 0
        else 0.0
    )
    light_color = _rand_light_col(rng2)
    material = (subject_gain, rim, iridescence, light_color)
    return initial, rng2, hi_key, material


def _prepare_discrete_primitive_plan(idx, level, step):
    """Compile levels 0–3 into ordered rectangles and Bresenham lines."""
    initial, rng2, _hi_key, _material = _prepare_primitive_preamble(
        idx, level, step
    )
    commands = []
    if level == 0:
        for _ in range(1 + rng2.randint(0, 4)):
            px, py = rng2.randint(0, W), rng2.randint(0, H)
            color = rng2.uniform(0.3, 1.0, C)
            commands.append(
                _primitive_command(
                    PrimitiveOp.RECT,
                    x0=px,
                    y0=py,
                    p0=px,
                    p1=py,
                    color=color,
                )
            )
    elif level == 1:
        for _ in range(3 + rng2.randint(0, 10)):
            px, py = rng2.randint(0, W), rng2.randint(0, H)
            size = rng2.randint(0, 2)
            color = rng2.uniform(0.2, 1.0, C)
            commands.append(
                _primitive_command(
                    PrimitiveOp.RECT,
                    x0=px - size,
                    y0=py - size,
                    p0=px + size,
                    p1=py + size,
                    color=color,
                )
            )
    elif level == 2:
        for _ in range(1 + rng2.randint(0, 3)):
            color = rng2.uniform(0.2, 1.0, C)
            x0, y0 = rng2.randint(0, W), rng2.randint(0, H)
            x1, y1 = rng2.randint(0, W), rng2.randint(0, H)
            thickness = rng2.randint(0, 2)
            commands.append(
                _primitive_command(
                    PrimitiveOp.BRESENHAM,
                    x0=x0,
                    y0=y0,
                    p0=x1,
                    p1=y1,
                    p2=thickness,
                    color=color,
                )
            )
    elif level == 3:
        points = [
            (rng2.randint(0, W), rng2.randint(0, H))
            for _ in range(3 + rng2.randint(0, 5))
        ]
        color = rng2.uniform(0.2, 1.0, C)
        for start, end in zip(points, points[1:]):
            thickness = rng2.randint(0, 1)
            commands.append(
                _primitive_command(
                    PrimitiveOp.BRESENHAM,
                    x0=start[0],
                    y0=start[1],
                    p0=end[0],
                    p1=end[1],
                    p2=thickness,
                    color=color,
                )
            )
    else:
        raise ValueError(f"level {level} is not a discrete primitive plan")
    return initial, np.stack(commands), rng2


def _primitive_material_color(color, material):
    subject_gain, _rim, _iridescence, light_color = material
    return (
        np.asarray(color, np.float32)
        * np.float32(subject_gain)
        * np.asarray(light_color, np.float32)
    )


def _prepare_geometry_primitive_plan(idx, level, step):
    """Compile ellipse and analytic 3D levels into Primitive IR v2."""
    initial, rng2, _hi_key, material = _prepare_primitive_preamble(
        idx, level, step
    )
    commands = []
    if level == 4:
        for _ in range(1 + rng2.randint(0, 2)):
            color = rng2.uniform(0.1, 0.9, C)
            center_x = rng2.randint(5, W - 5)
            center_y = rng2.randint(5, H - 5)
            radius_x = rng2.randint(2, 11)
            radius_y = rng2.randint(2, 11)
            shape = rng2.randint(0, 3)
            angle = 0.0
            if shape == 1:
                x = rng2.randint(0, W - 10)
                y = rng2.randint(0, H - 10)
                width = rng2.randint(3, 16)
                height = rng2.randint(3, 16)
                center_x, center_y = x + width / 2, y + height / 2
                radius_x, radius_y = width / 2, height / 2
            elif shape == 2:
                radius_x = max(radius_x, radius_y)
                radius_y = radius_x * rng2.uniform(0.5, 1.8)
                angle = rng2.uniform(0, math.pi)
                cosine, sine = math.cos(angle), math.sin(angle)
                ellipses = [
                    (center_x, center_y, radius_x, radius_y)
                ]
                for _ in range(rng2.randint(2, 5)):
                    offset_x = rng2.uniform(
                        -radius_x * 0.7, radius_x * 0.7
                    )
                    offset_y = rng2.uniform(
                        -radius_x * 0.7, radius_x * 0.7
                    )
                    lobe_radius = rng2.uniform(
                        radius_x * 0.3, radius_x * 0.9
                    )
                    scale = lobe_radius / radius_x
                    ellipses.append(
                        (
                            center_x
                            + offset_x * cosine
                            - offset_y * sine,
                            center_y
                            + offset_x * sine
                            + offset_y * cosine,
                            radius_x * scale,
                            radius_y * scale,
                        )
                    )
                for ellipse in ellipses:
                    commands.append(
                        _primitive_command(
                            PrimitiveOp.ELLIPSE,
                            x0=ellipse[0],
                            y0=ellipse[1],
                            p0=ellipse[2],
                            p1=ellipse[3],
                            p2=angle,
                            color=color,
                        )
                    )
                continue
            commands.append(
                _primitive_command(
                    PrimitiveOp.ELLIPSE,
                    x0=center_x,
                    y0=center_y,
                    p0=radius_x,
                    p1=radius_y,
                    p2=angle,
                    color=color,
                )
            )
    elif level == 6:
        center_x = rng2.randint(5, W - 5)
        center_y = rng2.randint(5, H - 5)
        radius_x = rng2.randint(3, 10)
        radius_y = rng2.randint(3, 10)
        base_color = rng2.uniform(0.0, 0.4, C)
        yy, xx = np.ogrid[:H, :W]
        distance = np.sqrt(
            (xx - center_x) ** 2 / radius_x**2
            + (yy - center_y) ** 2 / radius_y**2
        )
        mask = distance <= 1.0
        maximum = float(distance[mask].max() + 1e-8)
        commands.append(
            _primitive_command(
                PrimitiveOp.ELLIPSE_GRADIENT,
                x0=center_x,
                y0=center_y,
                p0=radius_x,
                p1=radius_y,
                opacity=0.7,
                color=base_color,
                extras=(maximum, 0, 0, 0),
            )
        )
    elif level == 15:
        center_x = rng2.randint(6, W - 6)
        center_y = rng2.randint(6, H - 6)
        radius = rng2.uniform(5, 13)
        light = _rand_light(rng2)
        color = _primitive_material_color(
            rng2.uniform(0.1, 1.0, C), material
        )
        _subject_gain, rim, iridescence, _light_color = material
        commands.append(
            _primitive_command(
                PrimitiveOp.SPHERE_PHONG,
                x0=center_x,
                y0=center_y,
                p0=radius,
                p1=light[0],
                p2=light[1],
                clip_x=light[2],
                clip_y=rim,
                opacity=iridescence,
                color=color,
            )
        )
    elif level == 16:
        center_x, center_y = W / 2, H / 2
        half_size = rng2.uniform(9, 18) / 2
        light = _rand_light(rng2)
        color = rng2.uniform(0.1, 1.0, C)
        front = color * _phong(0, 0, 1, light)
        top = color * _phong(0, -1, 0, light) * 0.75
        left = color * _phong(-1, 0, 0, light) * 0.55
        bounds = (
            center_x - half_size,
            center_y - half_size,
            center_x + half_size,
            center_y + half_size,
        )
        commands.append(
            _primitive_command(
                PrimitiveOp.RECT,
                x0=bounds[0],
                y0=bounds[1],
                p0=bounds[2],
                p1=bounds[3],
                color=front,
            )
        )
        commands.append(
            _primitive_command(
                PrimitiveOp.MAX_RECT,
                x0=bounds[0],
                y0=bounds[1],
                p0=bounds[2],
                p1=center_y,
                color=top,
            )
        )
        commands.append(
            _primitive_command(
                PrimitiveOp.MAX_RECT,
                x0=bounds[0],
                y0=bounds[1],
                p0=center_x,
                p1=bounds[3],
                color=left,
            )
        )
    elif level == 17:
        center_x = rng2.randint(6, W - 6)
        center_y = rng2.randint(8, H - 8)
        radius = rng2.uniform(4, 10)
        light = _rand_light(rng2)
        color = rng2.uniform(0.1, 1.0, C)
        commands.append(
            _primitive_command(
                PrimitiveOp.CYLINDER_PHONG,
                x0=center_x,
                y0=center_y,
                p0=radius,
                p1=light[0],
                p2=light[1],
                clip_x=light[2],
                color=color,
            )
        )
        top_y = int(center_y - radius * 1.5 * 0.7)
        if top_y > 2:
            top_color = color * _phong(0, 0, 1, light)
            commands.append(
                _primitive_command(
                    PrimitiveOp.MAX_ELLIPSE,
                    x0=center_x,
                    y0=top_y,
                    p0=radius,
                    p1=radius,
                    color=top_color,
                )
            )
    elif level == 18:
        center_x = rng2.randint(10, W - 10)
        center_y = rng2.randint(10, H - 10)
        radius = rng2.uniform(6, 10)
        minor_radius = radius * rng2.uniform(0.25, 0.45)
        light = _rand_light(rng2)
        color = _primitive_material_color(
            rng2.uniform(0.1, 1.0, C), material
        )
        _subject_gain, rim, iridescence, _light_color = material
        commands.append(
            _primitive_command(
                PrimitiveOp.TORUS_PHONG,
                x0=center_x,
                y0=center_y,
                p0=radius,
                p1=minor_radius / radius,
                p2=light[0],
                clip_x=light[1],
                clip_y=light[2],
                opacity=rim,
                color=color,
                extras=(iridescence, 0, 0, 0),
            )
        )
    else:
        raise ValueError(f"level {level} is not an analytic primitive plan")
    return initial, np.stack(commands), rng2


def _prepare_food_primitive_plan(idx, step):
    """Compile level 142 soft-disc food and garnish composition."""
    initial, rng2, _hi_key, _material = _prepare_primitive_preamble(
        idx, 142, step
    )
    table = np.array(
        [
            rng2.uniform(0.2, 0.6),
            rng2.uniform(0.12, 0.4),
            rng2.uniform(0.05, 0.2),
        ]
    )
    initial[:] = table
    texture = _perlin(rng2, 2)
    initial[:, :, :] += texture.reshape(H, W, 1) * 0.06
    plate_x, plate_y = W / 2, H / 2 + rng2.uniform(-2, 3)
    plate_radius = rng2.uniform(8, 12)
    plate_color = np.ones(3) * rng2.uniform(0.6, 0.95)
    alpha = _soft_disc(plate_x, plate_y, plate_radius)
    for channel in range(C):
        initial[:, :, channel] = np.clip(
            initial[:, :, channel] * (1 - alpha)
            + plate_color[channel] * alpha,
            0,
            1,
        )
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rim = (
        np.clip(
            (
                plate_radius
                - 0.8
                - np.sqrt((yy - plate_y) ** 2 + (xx - plate_x) ** 2)
            )
            * 1.5,
            0,
            1,
        )
        * 0.15
    )
    for channel in range(C):
        initial[:, :, channel] = np.clip(
            initial[:, :, channel] - rim, 0, 1
        )

    commands = []
    for _ in range(rng2.randint(2, 6)):
        food_x = plate_x + rng2.uniform(
            -plate_radius * 0.55, plate_radius * 0.55
        )
        food_y = plate_y + rng2.uniform(
            -plate_radius * 0.45, plate_radius * 0.45
        )
        food_radius = rng2.uniform(1.5, 5)
        food_color = np.array(
            [
                rng2.uniform(0.3, 0.9),
                rng2.uniform(0.1, 0.7),
                rng2.uniform(0.0, 0.3),
            ]
        )
        commands.append(
            _primitive_command(
                PrimitiveOp.SOFT_DISC,
                x0=food_x,
                y0=food_y,
                p0=food_radius,
                opacity=0.7,
                color=food_color,
            )
        )
    for _ in range(rng2.randint(0, 5)):
        garnish_x = plate_x + rng2.uniform(
            -plate_radius * 0.5, plate_radius * 0.5
        )
        garnish_y = plate_y + rng2.uniform(
            -plate_radius * 0.4, plate_radius * 0.4
        )
        garnish_color = np.array(
            [
                rng2.uniform(0.0, 0.3),
                rng2.uniform(0.4, 0.9),
                rng2.uniform(0.0, 0.3),
            ]
        )
        commands.append(
            _primitive_command(
                PrimitiveOp.ADDITIVE_SOFT_DISC,
                x0=garnish_x,
                y0=garnish_y,
                p0=rng2.uniform(0.3, 0.9),
                opacity=0.6,
                color=garnish_color,
            )
        )
    return initial, np.stack(commands), rng2


def _prepare_crowd_primitive_plan(idx, step):
    """Compile level 144 heads and bodies into soft-disc/affine commands."""
    initial, rng2, hi_key, _material = _prepare_primitive_preamble(
        idx, 144, step
    )
    _rand_light(rng2)
    horizon = rng2.randint(10, 24)
    _ground(rng2, initial, horizon, None, hi_key)
    if rng2.rand() < 0.5:
        sky = np.array(
            [
                rng2.uniform(0.3, 0.6),
                rng2.uniform(0.35, 0.65),
                rng2.uniform(0.45, 0.8),
            ]
        )
        for row in range(H):
            amount = min(
                1.0, max(0.0, (row - horizon) / max(H - horizon, 1))
            )
            initial[row, :] = np.clip(
                initial[row, :] * 0.3 + sky * (1 - amount) * 0.7,
                0,
                1,
            )
    count = rng2.randint(8, 26)
    cluster_count = rng2.randint(1, 3)
    centers = [
        rng2.uniform(W * 0.15, W * 0.85)
        for _ in range(cluster_count)
    ]
    commands = []
    for _ in range(count):
        body_x = np.clip(
            rng2.choice(centers) + rng2.uniform(-W * 0.22, W * 0.22),
            1,
            W - 1,
        )
        body_y = rng2.uniform(horizon + 1, H - 1)
        depth = (body_y - horizon) / max(H - horizon, 1)
        size = rng2.uniform(1.8, 4.5) * (0.45 + 0.55 * depth)
        color = _photo_color(rng2, 0.05, 0.85)
        head_y = body_y - size * 0.35
        commands.append(
            _primitive_command(
                PrimitiveOp.SOFT_DISC,
                x0=body_x,
                y0=head_y,
                p0=size * 0.28,
                opacity=0.85,
                color=color,
            )
        )
        body_y0 = int(head_y + size * 0.15)
        body_y1 = int(body_y + size * 0.3)
        for row in range(max(0, body_y0), min(H, body_y1 + 1)):
            width = size * 0.22 * (
                1
                - abs(row - (body_y0 + body_y1) / 2)
                / max(body_y1 - body_y0, 1)
                * 0.8
            )
            x0 = int(np.clip(body_x - width, 0, W))
            x1 = int(np.clip(body_x + width, 0, W))
            commands.append(
                _primitive_command(
                    PrimitiveOp.AFFINE_RECT,
                    x0=x0,
                    y0=row,
                    p0=x1 - 1,
                    p1=row,
                    opacity=0.2,
                    color=color * 0.85,
                )
            )
    return initial, np.stack(commands), rng2


def _prepare_pore_plan(idx, step):
    """Compile level 128 into the shared ordered primitive IR."""
    initial, rng2, _hi_key, _material = _prepare_primitive_preamble(
        idx, 128, step
    )

    base = _photo_color(rng2, 0.25, 0.75)
    hi_head, lo_head = (1.0 - base).min(), (base - 0.02).min()
    wobble = min(0.15, 2 * hi_head, 2 * lo_head)
    initial[:] = (
        base
        + (_perlin(rng2, 4).reshape(H, W, 1) - 0.5) * wobble
    )
    bump = rng2.rand() < 0.5
    headroom = hi_head if bump else lo_head
    n_pores = rng2.randint(4, 140)
    placed = []
    attempts = 0
    while len(placed) < n_pores and attempts < n_pores * 15:
        attempts += 1
        radius = rng2.uniform(0.6, 4.5)
        cx, cy = rng2.uniform(0, W), rng2.uniform(0, H)
        if any(
            math.hypot(cx - px, cy - py) < (radius + pr) * 0.55
            for px, py, pr in placed
        ):
            continue
        placed.append((cx, cy, radius))

    commands = []
    for cx, cy, radius in placed:
        delta = min(rng2.uniform(0.2, 0.4), headroom) * (
            1 if bump else -1
        )
        shade = base + delta
        alpha_scale = rng2.uniform(0.6, 1.0)
        commands.append(
            _primitive_command(
                PrimitiveOp.SOFT_DISC,
                x0=cx,
                y0=cy,
                p0=radius,
                opacity=alpha_scale,
                color=shade,
            )
        )
        rim_color = base - delta * 0.5
        commands.append(
            _primitive_command(
                PrimitiveOp.SOFT_RING,
                x0=cx,
                y0=cy,
                p0=radius,
                p1=radius * 0.6,
                opacity=0.5,
                color=rim_color,
            )
        )
        if radius > 1.1 and rng2.rand() < 0.35:
            content_radius = radius * rng2.uniform(0.12, 0.3)
            ring_width = content_radius * rng2.uniform(0.25, 0.45)
            max_offset = max(
                0.0, radius * 0.58 - content_radius - ring_width
            )
            offset_angle = rng2.uniform(0, 2 * math.pi)
            offset_magnitude = rng2.uniform(0, max_offset)
            content_x = cx + math.cos(offset_angle) * offset_magnitude
            content_y = cy + math.sin(offset_angle) * offset_magnitude
            commands.append(
                _primitive_command(
                    PrimitiveOp.CLIPPED_SOFT_RING,
                    x0=content_x,
                    y0=content_y,
                    p0=content_radius + ring_width,
                    p1=content_radius,
                    clip_x=cx,
                    clip_y=cy,
                    clip_radius=radius,
                    color=base * 0.15,
                )
            )
            content_color = rng2.uniform(0.05, 1.0, C)
            commands.append(
                _primitive_command(
                    PrimitiveOp.CLIPPED_SOFT_DISC,
                    x0=content_x,
                    y0=content_y,
                    p0=content_radius,
                    clip_x=cx,
                    clip_y=cy,
                    clip_radius=radius,
                    color=content_color,
                )
            )
    return initial, np.stack(commands), rng2


def _prepare_primitive_plan(idx, level, step):
    if level == 128:
        return _prepare_pore_plan(idx, step)
    if level == 142:
        return _prepare_food_primitive_plan(idx, step)
    if level == 144:
        return _prepare_crowd_primitive_plan(idx, step)
    if level in {4, 6, 15, 16, 17, 18}:
        return _prepare_geometry_primitive_plan(idx, level, step)
    return _prepare_discrete_primitive_plan(idx, level, step)


def _prepare_primitive_plan_task(arguments):
    """Picklable Primitive IR compiler entry point."""
    return _prepare_primitive_plan(*arguments)


def _finish_portable_primitive_batch(
    batch_indices,
    prepared,
    *,
    force_level,
    raster,
    executor=None,
    chunksize=32,
):
    """Execute Primitive IR on GPU and preserve the legacy CPU post pipeline."""
    from ._webgpu import render_primitive_images

    initial = np.stack([item[0] for item in prepared])
    command_lists = [item[1] for item in prepared]
    rendered = render_primitive_images(initial, command_lists)
    tasks = (
        (idx, image, item[2], force_level, raster)
        for idx, image, item in zip(batch_indices, rendered, prepared)
    )
    if executor is not None:
        images = executor.map(
            _finish_primitive_item_task, tasks, chunksize=chunksize
        )
    else:
        images = map(_finish_primitive_item_task, tasks)
    return _collect_batch(images, len(batch_indices))


def _finish_primitive_item_task(arguments):
    """Finish one Primitive IR image; safe to run in a CPU worker."""
    idx, image, rng2, force_level, raster = arguments
    level = int(force_level) if force_level is not None else _level_of(idx)
    if level >= 82 and rng2.randint(0, 2) == 0:
        image = np.ascontiguousarray(image[:, ::-1])
    if level not in POST_LEVELS:
        if rng2.randint(0, 3) > 0:
            image = _natural_color(image, rng2)
        if rng2.randint(0, 7) == 0:
            image = _poisson(image, rng2)
        if rng2.randint(0, 3) == 0:
            image = gaussian_filter(
                image, sigma=(rng2.uniform(0.3, 0.7),) * 2 + (0,)
            )
    image = _apply_scene(
        np.clip(image, 0, 1).astype(np.float32), _scene(idx)
    )
    if raster is not None:
        source = (
            _indexed_rgba(image, idx, force_level, raster)
            if _raster_requests_alpha(raster)
            else image
        )
        image = convert_raster(source, raster)
    return image


def _webgpu_is_measurably_faster(
    batch_indices,
    levels,
    *,
    family,
    step,
    force_level,
    workers,
    chunksize,
    start_method,
    enabled_families=None,
    shared_executor=None,
):
    """Calibrate auto-selection once per adapter, worker count, and batch scale."""
    from ._webgpu import accelerator_info, render_wave_images

    # Large heterogeneous batches can scale differently once command buffers,
    # CPU plans, and readbacks all grow. Calibrate up to one full 4K bulk
    # dispatch rather than extrapolating a small prefix.
    sample_count = min(len(batch_indices), 4096)
    # Keep independent decisions for small and large dispatches without
    # generating an unbounded cache of exact batch lengths.
    scale = 1 << max(0, sample_count.bit_length() - 1)
    key = (
        os.environ.get("SIG_WEBGPU_ADAPTER", "").strip().casefold(),
        family,
        int(workers),
        scale,
    )
    with _AUTO_WEBGPU_LOCK:
        cached = _AUTO_WEBGPU_RESULTS.get(key)
        if cached is not None:
            return bool(cached["enabled"])

        if sample_count == len(batch_indices):
            sample_positions = np.arange(sample_count)
        else:
            sample_positions = np.linspace(
                0, len(batch_indices) - 1, sample_count, dtype=np.int64
            )
        sample_indices = [
            batch_indices[int(position)] for position in sample_positions
        ]
        sample_levels = [levels[int(position)] for position in sample_positions]
        # Initialize the shared Sobol table before timing either backend.
        make_image(sample_indices[0])
        cpu_backend = "process" if workers > 1 and sample_count >= 64 else "serial"
        started = time.perf_counter()
        if cpu_backend == "process" and shared_executor is not None:
            reference = _collect_batch(
                shared_executor.map(
                    _make_image_task,
                    (
                        (idx, step, force_level, None)
                        for idx in sample_indices
                    ),
                    chunksize=chunksize,
                ),
                sample_count,
            )
        else:
            reference = make_images(
                sample_indices,
                step=step,
                force_level=force_level,
                backend=cpu_backend,
                workers=workers,
                chunksize=chunksize,
                start_method=start_method,
                fidelity="legacy",
            )
        cpu_seconds = time.perf_counter() - started

        # Enumerate adapters only after the process-based CPU reference has
        # completed. Native GPU runtimes may start helper threads, after which
        # safely forking a CPU pool is no longer possible.
        info = accelerator_info()
        if not info["available"]:
            _AUTO_WEBGPU_RESULTS[key] = {
                "enabled": False,
                "reason": info.get("reason", "accelerator unavailable"),
                "cpu_seconds": cpu_seconds,
                "samples": sample_count,
            }
            return False

        # Adapter creation and shader compilation are cached setup costs, not
        # steady-state throughput. A tiny warm-up also validates the driver.
        warm_count = min(8, sample_count)
        families = (
            set(family.split("+"))
            if enabled_families is None
            else set(enabled_families)
        )
        sample_gpu_families = {
            GPU_LEVEL_FAMILIES[level]
            for level in sample_levels
            if level in GPU_LEVEL_FAMILIES
            and GPU_LEVEL_FAMILIES[level] in families
        }
        all_sample_gpu = len(sample_gpu_families) > 0 and all(
            level in GPU_LEVEL_FAMILIES
            and GPU_LEVEL_FAMILIES[level] in families
            for level in sample_levels
        )
        if all_sample_gpu and sample_gpu_families == {"wave"}:
            render_wave_images(
                sample_indices[:warm_count], sample_levels[:warm_count]
            )
        elif all_sample_gpu and sample_gpu_families == {"reaction_diffusion"}:
            _portable_reaction_batch(
                sample_indices[:warm_count],
                step=step,
                force_level=force_level,
                raster=None,
            )
        else:
            warm_indices = []
            for selected_family in sorted(sample_gpu_families):
                warm_indices.extend(
                    [
                        idx
                        for idx, level in zip(sample_indices, sample_levels)
                        if GPU_LEVEL_FAMILIES.get(level) == selected_family
                    ][:warm_count]
                )
            _portable_webgpu_batch(
                warm_indices,
                step=step,
                force_level=force_level,
                raster=None,
                workers=workers,
                chunksize=chunksize,
                start_method=start_method,
                enabled_families=families,
                shared_executor=shared_executor,
            )
        started = time.perf_counter()
        if all_sample_gpu and sample_gpu_families == {"wave"}:
            candidate = render_wave_images(sample_indices, sample_levels)
        elif all_sample_gpu and sample_gpu_families == {"reaction_diffusion"}:
            candidate = _portable_reaction_batch(
                sample_indices,
                step=step,
                force_level=force_level,
                raster=None,
            )
        else:
            candidate = _portable_webgpu_batch(
                sample_indices,
                step=step,
                force_level=force_level,
                raster=None,
                workers=workers,
                chunksize=chunksize,
                start_method=start_method,
                enabled_families=families,
                shared_executor=shared_executor,
            )
        gpu_seconds = time.perf_counter() - started
        maximum_error = float(np.max(np.abs(reference - candidate)))
        absolute_error = np.abs(reference - candidate)
        if sample_gpu_families & {"reaction_diffusion", "primitive_ir"}:
            # A field or blend value infinitesimally crossing a downstream
            # stochastic/threshold boundary can change a few final pixels by
            # much more than the underlying arithmetic error. Validate the
            # distribution as well as the ordinary numerical path.
            numerically_valid = bool(
                float(absolute_error.mean()) <= 1e-5
                and float(np.percentile(absolute_error, 99.0)) <= 1e-4
            )
        else:
            numerically_valid = bool(
                np.allclose(reference, candidate, rtol=1e-4, atol=1e-4)
            )
        speedup = cpu_seconds / max(gpu_seconds, 1e-12)
        enabled = numerically_valid and speedup >= 1.10
        _AUTO_WEBGPU_RESULTS[key] = {
            "enabled": enabled,
            "speedup": speedup,
            "cpu_seconds": cpu_seconds,
            "webgpu_seconds": gpu_seconds,
            "max_abs_error": maximum_error,
            "samples": sample_count,
        }
        return enabled


def make_images(
    indices: Iterable[int],
    *,
    step: int = 7,
    force_level: int | None = None,
    raster: RasterSpec | None = None,
    width: int | None = None,
    height: int | None = None,
    backend: str = "auto",
    workers: int | None = None,
    chunksize: int | None = None,
    start_method: str = "auto",
    fidelity: str = "legacy",
) -> np.ndarray:
    """Generate a deterministic batch using CPU or portable WebGPU.

    ``width`` and ``height`` select one native canvas shared by the batch.
    RasterSpec dimensions select the canvas when both axes meet the native
    minimum; smaller legacy raster outputs are safely reduced from 32×32.

    ``backend="auto"`` keeps small batches serial. With ``workers=None`` and a
    sufficiently large batch it measures physical/SMT process counts, native
    thread policy, and (when ``chunksize=None``) chunk size against the actual
    workload, then reuses a warmed pool. The process backend preserves output
    order and isolates the legacy renderer's mutable recursive state.

    ``fidelity="legacy"`` is the default byte-exact contract and never selects
    GPU arithmetic automatically. ``fidelity="portable"`` permits small
    cross-device floating-point differences and lets ``auto`` select measured
    WebGPU families only when both their local and whole-batch gates beat the
    tuned CPU. The explicit ``webgpu`` backend requires portable fidelity and
    falls back to CPU for levels without a portable kernel.
    """
    try:
        batch_indices = [_validated_index(idx) for idx in indices]
    except TypeError as error:
        if isinstance(indices, (str, bytes)) or not hasattr(indices, "__iter__"):
            raise TypeError("indices must be an iterable of integers") from error
        raise
    step = _validated_step(step)
    force_level = _validated_force_level(force_level)
    if raster is not None:
        _validated_raster_spec(raster)
    canvas_width, canvas_height = _validated_canvas_size(
        width, height, raster
    )
    nondefault_canvas = (
        canvas_width != DEFAULT_WIDTH or canvas_height != DEFAULT_HEIGHT
    )
    if not isinstance(backend, str) or backend.lower() not in BATCH_BACKENDS:
        choices = ", ".join(BATCH_BACKENDS)
        raise ValueError(f"backend must be one of: {choices}")
    backend = backend.lower()
    if (
        not isinstance(fidelity, str)
        or fidelity.lower() not in BATCH_FIDELITIES
    ):
        choices = ", ".join(BATCH_FIDELITIES)
        raise ValueError(f"fidelity must be one of: {choices}")
    fidelity = fidelity.lower()
    if backend == "webgpu" and fidelity != "portable":
        raise ValueError(
            "backend='webgpu' requires fidelity='portable' because GPU "
            "transcendentals are not byte-identical across devices"
        )
    if nondefault_canvas and backend == "webgpu":
        # Portable kernels currently encode the 32×32 legacy contract. Native
        # arbitrary-resolution batches remain exact by using the CPU renderer.
        backend = "serial"
    supported_start_methods = set(multiprocessing.get_all_start_methods())
    if (
        not isinstance(start_method, str)
        or (
            start_method.lower() != "auto"
            and start_method.lower() not in supported_start_methods
        )
    ):
        choices = ", ".join(["auto", *sorted(supported_start_methods)])
        raise ValueError(f"start_method must be one of: {choices}")
    start_method = start_method.lower()
    if chunksize is not None:
        if (
            isinstance(chunksize, (bool, np.bool_))
            or not isinstance(chunksize, (int, np.integer))
            or int(chunksize) < 1
        ):
            raise ValueError("chunksize must be a positive integer or None")
        chunksize = int(chunksize)
    if workers is not None:
        if (
            isinstance(workers, (bool, np.bool_))
            or not isinstance(workers, (int, np.integer))
            or int(workers) < 1
        ):
            raise ValueError("workers must be a positive integer or None")
        workers = int(workers)
    if not batch_indices:
        return _empty_batch(raster, canvas_width, canvas_height)

    in_daemon = multiprocessing.current_process().daemon
    batch_levels = [
        (
            int(force_level)
            if force_level is not None
            else _level_of(idx)
        )
        for idx in batch_indices
    ]
    selected_cpu_executor = None
    if workers is None and backend in {"auto", "process"}:
        (
            selected_workers,
            selected_chunksize,
            limit_native_threads,
        ) = _select_cpu_configuration(
            batch_indices,
            batch_levels,
            step=step,
            force_level=force_level,
            raster=raster,
            requested_chunksize=chunksize,
            start_method=start_method,
        )
        if (
            len(batch_indices) >= 64
            and selected_workers > 1
            and not in_daemon
        ):
            selected_cpu_executor = _auto_cpu_executor(
                selected_workers,
                start_method,
                limit_native_threads=limit_native_threads,
            )
    else:
        topology = _cpu_topology()
        selected_workers = (
            min(len(batch_indices), topology["physical_cores"])
            if workers is None
            else workers
        )
        selected_chunksize = (
            _recommended_chunksize(
                len(batch_indices), selected_workers
            )
            if chunksize is None
            else chunksize
        )
    chunksize = selected_chunksize
    auto_gpu_families = None
    auto_shared_executor = selected_cpu_executor
    if backend == "auto":
        family_positions = {}
        for position, level in enumerate(batch_levels):
            family = GPU_LEVEL_FAMILIES.get(level)
            if family is not None:
                family_positions.setdefault(family, []).append(position)
        enabled_families = set()
        if (
            fidelity == "portable"
            and raster is None
            and not nondefault_canvas
            and family_positions
        ):
            if (
                len(batch_indices) >= _AUTO_WEBGPU_MIN_IMAGES
                and selected_workers > 1
                and not in_daemon
            ):
                if auto_shared_executor is None:
                    auto_shared_executor = _auto_cpu_executor(
                        selected_workers, start_method
                    )
            for family, positions in sorted(family_positions.items()):
                if len(positions) < _AUTO_WEBGPU_MIN_IMAGES:
                    continue
                family_indices = [
                    batch_indices[position] for position in positions
                ]
                family_levels = [batch_levels[position] for position in positions]
                try:
                    if _webgpu_is_measurably_faster(
                        family_indices,
                        family_levels,
                        family=family,
                        step=step,
                        force_level=force_level,
                        workers=selected_workers,
                        chunksize=chunksize,
                        start_method=start_method,
                        shared_executor=auto_shared_executor,
                    ):
                        enabled_families.add(family)
                # Auto must remain a safe CPU fallback even when a driver
                # reports validation, device-loss, or allocation failure.
                # Explicit backend='webgpu' still surfaces the original error.
                except Exception:
                    continue
            if enabled_families:
                gpu_positions = sum(
                    len(family_positions[family])
                    for family in enabled_families
                )
                needs_hybrid_gate = (
                    len(enabled_families) > 1
                    or gpu_positions != len(batch_indices)
                )
                if needs_hybrid_gate:
                    hybrid_key = "hybrid:" + "+".join(
                        sorted(enabled_families)
                    )
                    try:
                        hybrid_enabled = _webgpu_is_measurably_faster(
                            batch_indices,
                            batch_levels,
                            family=hybrid_key,
                            step=step,
                            force_level=force_level,
                            workers=selected_workers,
                            chunksize=chunksize,
                            start_method=start_method,
                            enabled_families=enabled_families,
                            shared_executor=auto_shared_executor,
                        )
                    except Exception:
                        hybrid_enabled = False
                    if not hybrid_enabled:
                        enabled_families.clear()
        if enabled_families:
            auto_gpu_families = enabled_families
            backend = "webgpu"
        else:
            backend = (
                "process"
                if len(batch_indices) >= 64
                and selected_workers > 1
                and not in_daemon
                else "serial"
            )
    if backend == "webgpu":
        return _portable_webgpu_batch(
            batch_indices,
            step=step,
            force_level=force_level,
            raster=raster,
            workers=workers,
            chunksize=chunksize,
            start_method=start_method,
            enabled_families=auto_gpu_families,
            shared_executor=auto_shared_executor,
        )
    if backend == "process" and in_daemon:
        raise RuntimeError(
            "the process backend cannot run inside a daemon process; "
            "use backend='serial' or let the parent process own the batch pool"
        )
    if backend == "serial" or selected_workers == 1:
        images = (
            make_image(
                idx, step=step, force_level=force_level, raster=raster,
                width=width, height=height,
            )
            for idx in batch_indices
        )
        return _collect_batch(images, len(batch_indices))
    elif auto_shared_executor is not None:
        images = auto_shared_executor.map(
            _make_image_task,
            (
                (
                    idx, step, force_level, raster,
                    width, height,
                )
                for idx in batch_indices
            ),
            chunksize=chunksize,
        )
        return _collect_batch(images, len(batch_indices))
    else:
        from concurrent.futures import ProcessPoolExecutor

        tasks = (
            (
                idx, step, force_level, raster,
                width, height,
            )
            for idx in batch_indices
        )
        # Fork is substantially cheaper for this SciPy-heavy package, but only
        # use it from a single-threaded Linux parent. Otherwise spawn avoids
        # copying locks or third-party worker threads into the renderer.
        if start_method == "auto":
            safe_to_fork = (
                sys.platform.startswith("linux") and threading.active_count() == 1
            )
            selected_start_method = "fork" if safe_to_fork else "spawn"
        else:
            selected_start_method = start_method
        process_context = multiprocessing.get_context(selected_start_method)
        with ProcessPoolExecutor(
            max_workers=selected_workers,
            mp_context=process_context,
            initializer=_cpu_worker_init,
        ) as executor:
            images = executor.map(_make_image_task, tasks, chunksize=chunksize)
            return _collect_batch(images, len(batch_indices))


# ═══════════════════════════════════════════════════════════════
# Symmetry post-processing filters (mirror fold, kaleidoscope)
# ═══════════════════════════════════════════════════════════════

def _mirror_fold(base, theta):
    """Reflect the image across a line through its center at angle `theta`."""
    cx, cy = (W - 1) / 2, (H - 1) / 2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    c, s = math.cos(theta), math.sin(theta)
    u = dx * c + dy * s               # local axis along the fold line
    v = np.abs(-dx * s + dy * c)      # local axis across it, folded onto the positive side
    src_x = cx + u * c - v * s
    src_y = cy + u * s + v * c
    out = np.empty_like(base)
    for ch in range(C):
        out[:, :, ch] = map_coordinates(base[:, :, ch], [src_y, src_x], order=1, mode='nearest')
    return out

def _kaleidoscope(base, n_wedges, rot_offset):
    """Repeat one angular wedge of the image radially around the center, mirrored each step —
    the classic kaleidoscope look (N mirror axes through the center, not just one)."""
    cx, cy = (W - 1) / 2, (H - 1) / 2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    wedge = 2 * math.pi / n_wedges
    a = np.mod(np.arctan2(dy, dx) - rot_offset, wedge)
    a = np.minimum(a, wedge - a) + rot_offset      # fold each wedge onto one reference slice
    src_x, src_y = cx + r * np.cos(a), cy + r * np.sin(a)
    out = np.empty_like(base)
    for ch in range(C):
        out[:, :, ch] = map_coordinates(base[:, :, ch], [src_y, src_x], order=1, mode='nearest')
    return out

def _diffraction_2d(base, angle, freq, disp):
    """Diffraction grating over the whole frame: R/G/B separate along the grating axis (spectral
    fringing, like light through a prism/CD) modulated by a periodic transmission band."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    proj = xx * math.cos(angle) + yy * math.sin(angle)
    out = np.empty_like(base)
    for ch in range(C):
        shift = (ch - 1) * disp          # R and B shift opposite ways, G stays put: spectral spread
        sx = xx + shift * math.cos(angle)
        sy = yy + shift * math.sin(angle)
        out[:, :, ch] = map_coordinates(base[:, :, ch], [sy, sx], order=1, mode='nearest')
    fringe = 0.6 + 0.4 * np.sin(proj * freq)
    return np.clip(out * fringe.reshape(H, W, 1), 0, 1)

def _refraction_2d(base, dx, dy):
    """Continuous refractive warp of the whole frame — heat haze, rippled glass, water surface.
    dx/dy are per-pixel (H,W) displacement fields, not a single global shift."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.empty_like(base)
    for ch in range(C):
        out[:, :, ch] = map_coordinates(base[:, :, ch], [yy + dy, xx + dx], order=1, mode='nearest')
    return out

def _refraction_3d(base, cx, cy, R, mag, L):
    """True lens/sphere refraction: inside the circle, resample the base image at 1/mag the
    radius — a magnifying (mag>1) or diverging/inverted (mag<0, like a water droplet) lens,
    unlike `_glass()`'s flat alpha-blend which doesn't bend what's behind it at all."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m, Nx, Ny, Nz = _sphere(cx, cy, R)
    dxp, dyp = xx - cx, yy - cy
    sx, sy = cx + dxp / mag, cy + dyp / mag
    out = base.copy()
    for ch in range(C):
        warped = map_coordinates(base[:, :, ch], [sy, sx], order=1, mode='nearest')
        out[:, :, ch] = np.where(m, warped, out[:, :, ch])
    lit = _phong(Nx, Ny, Nz, L)
    edge = (1.0 - np.abs(Nz)) ** 3                    # rim brightening near the silhouette
    for ch in range(C):
        out[:, :, ch][m] = np.clip(out[:, :, ch][m] * (0.7 + 0.3 * lit[m]) + edge[m] * 0.25, 0, 1)
    return np.clip(out, 0, 1)

# ═══════════════════════════════════════════════════════════════
# Style post-processing filters
# ═══════════════════════════════════════════════════════════════

def _style_edge_map(img):
    gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    gx = np.abs(np.diff(gray, axis=1, append=gray[:,-1:]))
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:,:]))
    return np.sqrt(gx**2 + gy**2)


def _apply_style(img, style_idx, rng):
    """Apply a style filter to an already-generated image.

    Styles:
      0 = raw (passthrough)
      1 = realistic (camera grain + mild blur + chromatic shift)
      2 = cartoon (posterize + outline)
      3 = sketch (grayscale edges on paper)
      4 = watercolor (heavy blur + desaturate + paper)
      5 = neon (boost sat + edge glow)
      6 = vintage (sepia + vignette + grain)
      7 = pixel (downscale + nearest upscale + posterize)
    """
    if style_idx == 0:
        return img
    elif style_idx == 1:
        out = gaussian_filter(img, sigma=(rng.uniform(0.2,0.5), rng.uniform(0.2,0.5), 0))
        noise = rng.randn(*out.shape).astype(np.float32) * rng.uniform(0.01, 0.03)
        out = np.clip(out + noise, 0, 1)
        out[:,:,0] = np.clip(out[:,:,0] + rng.uniform(-0.02, 0.02), 0, 1)
        out[:,:,2] = np.clip(out[:,:,2] + rng.uniform(-0.02, 0.02), 0, 1)
        return out
    elif style_idx == 2:
        levels = int(rng.uniform(3, 6))
        out = np.floor(img * levels) / max(levels-1, 1)
        edges = _style_edge_map(img)
        edge_mask = edges > rng.uniform(0.15, 0.35)
        for cc in range(C):
            out[:,:,cc][edge_mask] = 0.05
        return np.clip(out, 0, 1)
    elif style_idx == 3:
        edges = _style_edge_map(img)
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
        sketch = 1.0 - np.clip(edges * rng.uniform(1.5, 3.0), 0, 1)
        paper = 1.0 - rng.rand(*edges.shape).astype(np.float32) * 0.04
        out = np.stack([sketch * paper] * C, axis=-1)
        return np.clip(out, 0, 1)
    elif style_idx == 4:
        out = gaussian_filter(img, sigma=(rng.uniform(0.8, 1.5), rng.uniform(0.8, 1.5), 0))
        gray = out.mean(axis=-1, keepdims=True)
        alpha = rng.uniform(0.3, 0.6)
        out = out * (1 - alpha) + gray * alpha
        tex = rng.rand(*out.shape[:2]).astype(np.float32) * 0.04
        out = np.clip(out + tex.reshape(out.shape[0], out.shape[1], 1), 0, 1)
        return out
    elif style_idx == 5:
        # neon = dark background + saturated glowing edges, not "brighten everything": the old
        # version multiplied the whole (often already-bright) frame by up to 1.6x and regularly
        # clipped straight to solid white, losing every bit of the effect
        out = np.clip(img * rng.uniform(0.15, 0.35), 0, 1)
        edges = _style_edge_map(img)
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
        glow = rng.uniform(0.3, 1.0, C)
        glow = glow / max(glow.max(), 1e-8)             # keep one saturated channel at full strength
        for cc in range(C):
            out[:,:,cc] = np.clip(out[:,:,cc] + edges * glow[cc] * rng.uniform(0.8, 1.15), 0, 1)
        return out
    elif style_idx == 6:
        gray = img.mean(axis=-1)
        # stretch to the frame's own contrast range first: without it, frames whose source
        # content was already dark/flat collapsed to near-black after the 0.55x blue-channel cut
        lo, hi = np.percentile(gray, 3), np.percentile(gray, 97)
        gray = np.clip((gray - lo) / max(hi - lo, 1e-3), 0, 1)
        out = np.stack([np.clip(gray*1.1,0,1), np.clip(gray*0.85,0,1), np.clip(gray*0.55,0,1)], axis=-1)
        yy, xx = np.ogrid[:H, :W]
        d = np.sqrt((xx-H/2)**2 + (yy-W/2)**2) / 22
        vig = np.clip(1.0 - d, 0.3, 1.0)
        out = out * vig.reshape(H, W, 1)
        out = np.clip(out + rng.randn(*out.shape).astype(np.float32)*0.03, 0, 1)
        return out
    else:  # style_idx == 7: pixel art
        scale = int(rng.uniform(3, 7))
        low_h, low_w = max(1, H//scale), max(1, W//scale)
        from scipy.ndimage import zoom
        small = zoom(img, (low_h/H, low_w/W, 1), order=0)
        out = zoom(small, (H/low_h, W/low_w, 1), order=0)
        out = out[:H, :W, :C]
        levels = int(rng.uniform(4, 10))
        out = np.floor(out * levels) / max(levels-1, 1)
        return np.clip(out, 0, 1)


# ═══════════════════════════════════════════════════════════════
# Raster output: resolution, color mode, and bits per channel
# ═══════════════════════════════════════════════════════════════

_RASTER_MODE_ALIASES = {
    "rgb": "rgb",
    "rgba": "rgba",
    "rgba2222": "rgba",
    "gray": "grayscale",
    "grey": "grayscale",
    "grayscale": "grayscale",
    "greyscale": "grayscale",
    "binary": "binary",
    "bitmap": "binary",
    "black_and_white": "binary",
    "black-white": "binary",
    "bw": "binary",
    "mono": "binary",
    "monochrome": "binary",
}
_RASTER_RESIZE_MODES = {"nearest": 0, "bilinear": 1, "bicubic": 3}
_RASTER_DITHER_MODES = {"none", "ordered", "floyd_steinberg"}
_RASTER_ALPHA_MODES = {"auto", "opaque", "background", "luminance"}
_INDEXED_SPRITE_LEVELS = (
    set(range(0, 7))
    | set(range(15, 21))
    | {39, 40, 45}
    | set(range(48, 64))
    | set(range(82, 86))
    | {90, 92}
    | set(range(100, 112))
    | set(range(120, 126))
    | {131, 134, 137, 139, 142, 145}
)


def _validated_raster_spec(spec: RasterSpec):
    if not isinstance(spec, RasterSpec):
        raise TypeError("raster must be a RasterSpec")
    if (
        isinstance(spec.width, (bool, np.bool_))
        or isinstance(spec.height, (bool, np.bool_))
        or not isinstance(spec.width, (int, np.integer))
        or not isinstance(spec.height, (int, np.integer))
        or spec.width <= 0
        or spec.height <= 0
    ):
        raise ValueError("RasterSpec.width and height must be positive integers")
    if (
        int(spec.width) > MAX_CANVAS_DIMENSION
        or int(spec.height) > MAX_CANVAS_DIMENSION
        or int(spec.width) * int(spec.height) > MAX_CANVAS_PIXELS
    ):
        raise ValueError(
            f"RasterSpec dimensions must not exceed {MAX_CANVAS_DIMENSION} "
            f"per axis or {MAX_CANVAS_PIXELS} total pixels"
        )
    if not isinstance(spec.mode, str) or spec.mode.lower() not in _RASTER_MODE_ALIASES:
        choices = ", ".join(sorted(_RASTER_MODE_ALIASES))
        raise ValueError(f"unsupported raster mode {spec.mode!r}; expected one of: {choices}")
    mode = _RASTER_MODE_ALIASES[spec.mode.lower()]
    rgba2222 = spec.mode.lower() == "rgba2222"
    if not isinstance(spec.resize, str) or spec.resize.lower() not in _RASTER_RESIZE_MODES:
        raise ValueError("RasterSpec.resize must be nearest, bilinear, or bicubic")
    if not isinstance(spec.dither, str) or spec.dither.lower() not in _RASTER_DITHER_MODES:
        raise ValueError("RasterSpec.dither must be none, ordered, or floyd_steinberg")
    if (
        not isinstance(spec.alpha_mode, str)
        or spec.alpha_mode.lower() not in _RASTER_ALPHA_MODES
    ):
        raise ValueError(
            "RasterSpec.alpha_mode must be auto, opaque, background, or luminance"
        )
    if not isinstance(spec.packed, (bool, np.bool_)):
        raise ValueError("RasterSpec.packed must be a boolean")
    if (
        isinstance(spec.threshold, (bool, np.bool_))
        or not isinstance(spec.threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(spec.threshold)
        or not 0.0 <= spec.threshold <= 1.0
    ):
        raise ValueError("RasterSpec.threshold must be between 0 and 1")
    if (
        isinstance(spec.alpha, (bool, np.bool_))
        or not isinstance(spec.alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(spec.alpha)
        or not 0.0 <= spec.alpha <= 1.0
    ):
        raise ValueError("RasterSpec.alpha must be between 0 and 1")

    raw_bits = spec.bits_per_channel
    if rgba2222:
        bits = (2, 2, 2, 2)
        packed = True
    elif mode in {"rgb", "rgba"}:
        channel_count = 3 if mode == "rgb" else 4
        if isinstance(raw_bits, (int, np.integer)) and not isinstance(raw_bits, (bool, np.bool_)):
            bits = (int(raw_bits),) * channel_count
        elif isinstance(raw_bits, tuple) and len(raw_bits) == channel_count:
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in raw_bits
            ):
                raise ValueError(
                    f"{mode.upper()} bits_per_channel must contain "
                    f"{channel_count} integers"
                )
            bits = tuple(int(value) for value in raw_bits)
        else:
            channels = "(R, G, B)" if mode == "rgb" else "(R, G, B, A)"
            raise ValueError(
                f"{mode.upper()} bits_per_channel must be an integer or a {channels} tuple"
            )
        packed = bool(spec.packed)
    else:
        if (
            isinstance(raw_bits, (bool, np.bool_))
            or not isinstance(raw_bits, (int, np.integer))
        ):
            raise ValueError(f"{mode} bits_per_channel must be one integer")
        bits = (int(raw_bits),)
        packed = bool(spec.packed)
    if any(value < 1 or value > 16 for value in bits):
        raise ValueError("bits_per_channel values must be between 1 and 16")
    if mode == "binary" and bits != (1,):
        raise ValueError("binary output requires bits_per_channel=1")
    if mode == "grayscale" and packed:
        raise ValueError(
            "packed grayscale output is not supported; use mode='binary' "
            "for one-bit packed pixels"
        )
    return mode, bits, spec.resize.lower(), spec.dither.lower(), packed


def _raster_requests_alpha(spec):
    return (
        isinstance(spec, RasterSpec)
        and isinstance(spec.mode, str)
        and _RASTER_MODE_ALIASES.get(spec.mode.lower()) == "rgba"
    )


def _background_matte(image):
    """Estimate a soft foreground matte from colors connected to the frame border."""
    rgb = np.clip(np.asarray(image, np.float32)[:, :, :3], 0, 1)
    height, width = rgb.shape[:2]
    border = np.concatenate(
        [rgb[0], rgb[-1], rgb[1:-1, 0], rgb[1:-1, -1]], axis=0
    )
    median = np.median(border, axis=0)
    deviation = np.linalg.norm(border - median, axis=1)
    cutoff = np.percentile(deviation, 72)
    palette = border[deviation <= cutoff + 1e-6]
    if len(palette) == 0:
        palette = border
    if len(palette) > 96:
        palette = palette[np.linspace(0, len(palette) - 1, 96).astype(int)]
    color_distance = np.min(
        np.linalg.norm(rgb[:, :, None, :] - palette[None, None, :, :], axis=3),
        axis=2,
    ) / math.sqrt(3.0)

    top = gaussian_filter(rgb[0], sigma=(1.2, 0))
    bottom = gaussian_filter(rgb[-1], sigma=(1.2, 0))
    left = gaussian_filter(rgb[:, 0], sigma=(1.2, 0))
    right = gaussian_filter(rgb[:, -1], sigma=(1.2, 0))
    ty = np.linspace(0, 1, height, dtype=np.float32).reshape(height, 1, 1)
    tx = np.linspace(0, 1, width, dtype=np.float32).reshape(1, width, 1)
    vertical = top.reshape(1, width, 3) * (1 - ty) + bottom.reshape(1, width, 3) * ty
    horizontal = left.reshape(height, 1, 3) * (1 - tx) + right.reshape(height, 1, 3) * tx
    estimated_background = (vertical + horizontal) * 0.5
    spatial_distance = (
        np.linalg.norm(rgb - estimated_background, axis=2) / math.sqrt(3.0)
    )
    score = color_distance * 0.72 + spatial_distance * 0.28
    matte = np.clip((score - 0.025) / 0.11, 0, 1)
    matte = gaussian_filter(matte.astype(np.float32), sigma=0.55)
    if matte.max() < 0.25 and rgb.std() > 0.01:
        smooth = gaussian_filter(rgb, sigma=(2.5, 2.5, 0))
        local_contrast = (
            np.linalg.norm(rgb - smooth, axis=2) / math.sqrt(3.0)
        )
        local_contrast = gaussian_filter(local_contrast, sigma=0.6)
        matte = np.maximum(
            matte, np.clip((local_contrast - 0.004) / 0.025, 0, 1)
        )
    matte[matte < 0.035] = 0.0
    return np.clip(matte, 0, 1).astype(np.float32)


def extract_alpha(image, alpha_mode="background", level=None):
    """Create a float32 alpha plane for RGB content.

    ``auto`` removes the background only for object/sprite-oriented indexed
    levels. Full-frame procedural and scene levels remain opaque.
    """
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] not in (3, 4):
        raise ValueError("extract_alpha expects an H×W×3 RGB or H×W×4 RGBA array")
    mode = str(alpha_mode).lower()
    if mode not in _RASTER_ALPHA_MODES:
        raise ValueError("alpha_mode must be auto, opaque, background, or luminance")
    if np.issubdtype(source.dtype, np.integer):
        scale = float(np.iinfo(source.dtype).max)
        rgb = source.astype(np.float32) / scale
    else:
        rgb = source.astype(np.float32, copy=False)
    rgb = np.clip(rgb, 0, 1)
    if rgb.shape[2] == 4:
        alpha = rgb[:, :, 3]
        return np.clip(alpha, 0, 1)
    if mode == "opaque" or (
        mode == "auto" and (level is None or int(level) not in _INDEXED_SPRITE_LEVELS)
    ):
        return np.ones(rgb.shape[:2], np.float32)
    if mode == "luminance":
        return np.clip(np.max(rgb[:, :, :3], axis=2), 0, 1).astype(np.float32)
    return _background_matte(rgb)


def _indexed_rgba(image, idx, force_level, raster):
    level = int(force_level) if force_level is not None else _level_of(idx)
    alpha = extract_alpha(image, raster.alpha_mode, level=level)
    return np.concatenate([image, alpha[:, :, None]], axis=2).astype(np.float32)


@lru_cache(maxsize=32)
def _resize_coordinates(source_h, source_w, height, width):
    target_y = (
        np.array([(source_h - 1) * 0.5], np.float32)
        if height == 1 else np.linspace(0, source_h - 1, height, dtype=np.float32)
    )
    target_x = (
        np.array([(source_w - 1) * 0.5], np.float32)
        if width == 1 else np.linspace(0, source_w - 1, width, dtype=np.float32)
    )
    yy, xx = np.meshgrid(target_y, target_x, indexing="ij")
    yy.setflags(write=False)
    xx.setflags(write=False)
    return yy, xx


def _resize_float_image(image, width, height, resize):
    """Resize an RGB float image to exactly ``height × width`` before quantizing."""
    source_h, source_w = image.shape[:2]
    if (source_w, source_h) == (width, height):
        return image.copy()
    order = _RASTER_RESIZE_MODES[resize]
    working = image
    if order > 0 and (width < source_w or height < source_h):
        sigma_y = max(0.0, 0.5 * (source_h / height - 1.0))
        sigma_x = max(0.0, 0.5 * (source_w / width - 1.0))
        if sigma_x or sigma_y:
            working = gaussian_filter(image, sigma=(sigma_y, sigma_x, 0))
    yy, xx = _resize_coordinates(source_h, source_w, height, width)
    result = np.empty((height, width, working.shape[2]), np.float32)
    for channel in range(working.shape[2]):
        result[:, :, channel] = map_coordinates(
            working[:, :, channel], [yy, xx], order=order, mode="nearest",
            prefilter=order > 1,
        )
    return np.clip(result, 0, 1)


@lru_cache(maxsize=32)
def _ordered_offsets(height, width):
    bayer = np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        np.float32,
    )
    tiled = np.tile(bayer, (math.ceil(height / 4), math.ceil(width / 4)))
    offsets = (tiled[:height, :width] + 0.5) / 16.0 - 0.5
    offsets.setflags(write=False)
    return offsets


def _quantize_float_channels(values, levels, dither):
    """Quantize an H×W×C float array to integer channel levels."""
    work = np.clip(np.asarray(values, np.float32), 0, 1)
    level_array = np.asarray(levels, np.float32).reshape(1, 1, -1)
    if dither == "none":
        return np.rint(work * level_array).astype(np.uint64)
    if dither == "ordered":
        offsets = _ordered_offsets(work.shape[0], work.shape[1]).reshape(
            work.shape[0], work.shape[1], 1
        )
        return np.clip(
            np.floor(work * level_array + 0.5 + offsets), 0, level_array
        ).astype(np.uint64)

    diffused = work.copy()
    quantized = np.empty(diffused.shape, np.uint64)
    for y in range(diffused.shape[0]):
        for x in range(diffused.shape[1]):
            old = np.clip(diffused[y, x], 0, 1)
            integer = np.rint(old * level_array[0, 0])
            new = integer / level_array[0, 0]
            quantized[y, x] = integer.astype(np.uint64)
            error = old - new
            if x + 1 < diffused.shape[1]:
                diffused[y, x + 1] += error * (7.0 / 16.0)
            if y + 1 < diffused.shape[0]:
                if x > 0:
                    diffused[y + 1, x - 1] += error * (3.0 / 16.0)
                diffused[y + 1, x] += error * (5.0 / 16.0)
                if x + 1 < diffused.shape[1]:
                    diffused[y + 1, x + 1] += error * (1.0 / 16.0)
    return quantized


def _quantize_binary(gray, threshold, dither):
    values = np.clip(np.asarray(gray, np.float32), 0, 1)
    if dither == "none":
        return values >= threshold
    if dither == "ordered":
        return values >= np.clip(
            threshold + _ordered_offsets(values.shape[0], values.shape[1]), 0, 1
        )
    diffused = values.copy()
    result = np.empty(values.shape, bool)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            old = float(np.clip(diffused[y, x], 0, 1))
            new = 1.0 if old >= threshold else 0.0
            result[y, x] = bool(new)
            error = old - new
            if x + 1 < values.shape[1]:
                diffused[y, x + 1] += error * (7.0 / 16.0)
            if y + 1 < values.shape[0]:
                if x > 0:
                    diffused[y + 1, x - 1] += error * (3.0 / 16.0)
                diffused[y + 1, x] += error * (5.0 / 16.0)
                if x + 1 < values.shape[1]:
                    diffused[y + 1, x + 1] += error * (1.0 / 16.0)
    return result


def _unsigned_dtype(bits):
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    return np.uint64


def convert_raster(image: np.ndarray, spec: RasterSpec) -> np.ndarray:
    """Convert canonical RGB/RGBA float output to a raster representation.

    Resizing and color conversion happen in float space. Quantization is always
    last, so every requested bit is meaningful. Binary output is ``bool`` unless
    ``packed=True``, in which case eight horizontal pixels are packed per byte.
    ``mode='rgba2222'`` always packs R7-6, G5-4, B3-2, and A1-0 in one uint8.
    """
    mode, bits, resize, dither, packed_output = _validated_raster_spec(spec)
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] not in (3, 4):
        raise ValueError("convert_raster expects an H×W×3 RGB or H×W×4 RGBA array")
    if not np.all(np.isfinite(source)):
        raise ValueError("convert_raster input must contain only finite values")
    if np.issubdtype(source.dtype, np.integer):
        source = source.astype(np.float32) / np.iinfo(source.dtype).max
    else:
        source = source.astype(np.float32, copy=False)
    source = np.clip(source, 0, 1)
    if source.shape[2] == 4:
        premultiplied = source.copy()
        premultiplied[:, :, :3] *= premultiplied[:, :, 3:4]
        resized_premultiplied = _resize_float_image(
            premultiplied, int(spec.width), int(spec.height), resize
        )
        resized_alpha = resized_premultiplied[:, :, 3:4]
        resized_rgb = np.zeros_like(resized_premultiplied[:, :, :3])
        visible = resized_alpha[:, :, 0] > 1e-8
        resized_rgb[visible] = (
            resized_premultiplied[:, :, :3][visible]
            / resized_alpha[visible]
        )
        resized = np.concatenate([resized_rgb, resized_alpha], axis=2)
    else:
        resized = _resize_float_image(
            source, int(spec.width), int(spec.height), resize
        )

    if mode in {"rgb", "rgba"}:
        color = resized[:, :, :3]
        if mode == "rgba":
            if resized.shape[2] == 4:
                alpha = resized[:, :, 3:4] * float(spec.alpha)
            else:
                alpha = np.full(
                    (spec.height, spec.width, 1), float(spec.alpha), np.float32
                )
            color = np.concatenate([color, alpha], axis=2)
        levels = tuple((1 << value) - 1 for value in bits)
        quantized = _quantize_float_channels(color, levels, dither)
        if packed_output:
            total_bits = sum(bits)
            packed = np.zeros((spec.height, spec.width), np.uint64)
            shift = total_bits
            for channel, channel_bits in enumerate(bits):
                shift -= channel_bits
                packed |= quantized[:, :, channel] << shift
            return np.ascontiguousarray(packed.astype(_unsigned_dtype(total_bits)))
        return np.ascontiguousarray(quantized.astype(_unsigned_dtype(max(bits))))

    gray = (
        resized[:, :, 0] * 0.2126
        + resized[:, :, 1] * 0.7152
        + resized[:, :, 2] * 0.0722
    )
    if mode == "binary":
        binary = _quantize_binary(gray, float(spec.threshold), dither)
        if packed_output:
            return np.ascontiguousarray(np.packbits(binary, axis=1, bitorder="big"))
        return np.ascontiguousarray(binary)

    levels = ((1 << bits[0]) - 1,)
    quantized = _quantize_float_channels(gray[:, :, None], levels, dither)[:, :, 0]
    if packed_output and bits[0] == 1:
        return np.ascontiguousarray(
            np.packbits(quantized.astype(bool), axis=1, bitorder="big")
        )
    return np.ascontiguousarray(quantized.astype(_unsigned_dtype(bits[0])))


def make_image_raster(
    idx: int,
    raster: RasterSpec,
    step: int = 7,
    force_level: int | None = None,
) -> np.ndarray:
    """Generate an indexed image directly in a requested raster representation."""
    return make_image(idx, step=step, force_level=force_level, raster=raster)


# ═══════════════════════════════════════════════════════════════
# Structured scene API
# ═══════════════════════════════════════════════════════════════

_SCENE_CATALOG_KINDS = {
    "tree", "conifer", "palm", "bush", "flower", "plane", "insect",
    "spider", "blob", "human",
}
_SCENE_MACRO_KINDS = {
    "macro_coin", "macro_key", "macro_button", "macro_jewelry", "macro_tool",
}
_SCENE_OBJECT_KINDS = (
    _SCENE_CATALOG_KINDS
    | _SCENE_MACRO_KINDS
    | {
        "sphere_3d", "box_3d", "tube", "blob_3d", "building_3d", "car_3d",
        "wireframe_cube", "wireframe_pyramid", "wireframe_grid", "disc",
        "disc_with_rim", "particle_field", "food_plate",
    }
)


def _scene_color(color, rng, lo=0.08, hi=0.95):
    """Return one clipped RGB color, using a photographic random default."""
    if color is None:
        return _photo_color(rng, lo, hi)
    value = np.asarray(color, np.float32)
    if value.shape != (C,) or not np.all(np.isfinite(value)):
        raise ValueError("scene colors must contain exactly three finite values")
    return np.clip(value, 0.0, 1.0)


def _scene_material(obj, color):
    """Evaluate an optional procedural material for one scene object."""
    if obj.material is None:
        return color
    extent = max(
        float(obj.radius) * 2.0,
        float(obj.width or 0.0),
        float(obj.height or 0.0),
        float(obj.size),
        1.0,
    )
    return evaluate_material(
        obj.material,
        color,
        H,
        W,
        (obj.resolved_x, obj.resolved_y),
        extent,
    )


def _render_background(img: np.ndarray, bg: Background, rng: np.random.RandomState):
    """Render ``bg`` onto ``img`` in place."""
    if not isinstance(bg, Background):
        raise TypeError("SceneSpec.background must be a Background")
    kind = bg.kind.lower()
    if kind in {"none", "transparent"}:
        return
    if kind == "solid":
        img[:] = _scene_color(bg.color, rng, 0.04, 0.55).reshape(1, 1, C)
        return
    if kind == "gradient_sky":
        if bg.horizon is None:
            horizon = H
        else:
            if not np.isfinite(bg.horizon) or not 0.0 <= bg.horizon <= 1.0:
                raise ValueError("Background.horizon must be between 0 and 1")
            horizon = max(1, int(round(float(bg.horizon) * H)))
        _sky_gradient(rng, img, hz=horizon, sun=True)
        if bg.color is not None:
            tint = _scene_color(bg.color, rng)
            img[:] = np.clip(img * (0.55 + 0.45 * tint.reshape(1, 1, C)), 0, 1)
        return
    if kind == "perlin":
        noise = _perlin(rng, 4).reshape(H, W, 1)
        color = _scene_color(
            bg.color, rng, 0.62 if bg.hi_key else 0.04, 1.0 if bg.hi_key else 0.65
        )
        floor = 0.72 if bg.hi_key else 0.25
        img[:] = np.clip(
            color.reshape(1, 1, C) * (floor + (1.0 - floor) * noise), 0, 1
        )
        return
    raise ValueError(
        f"unsupported background kind {bg.kind!r}; expected solid, gradient_sky, "
        "perlin, none, or transparent"
    )


def _parts_with_color(parts, color):
    """Override generated catalogue colors while retaining its geometry."""
    if color is None:
        return parts
    result = []
    for part in parts:
        if part[0] == "s":
            result.append((part[0], part[1], part[2], color))
        else:
            result.append((part[0], part[1], part[2], part[3], part[4], color))
    return result


def _coverage_over(base, layer):
    """Source-over union for two coverage/alpha planes."""
    return base + layer * (1.0 - base)


def _render_wireframe(img, obj, color, rng):
    """Project a cube, pyramid, or ground grid and draw its explicit edges."""
    kind = obj.kind.lower()
    if kind == "wireframe_grid":
        vertices, edges = [], []
        for coordinate in np.linspace(-1.0, 1.0, 6):
            first = len(vertices)
            vertices.extend([(-1.0, 0.0, coordinate), (1.0, 0.0, coordinate)])
            edges.append((first, first + 1))
            first = len(vertices)
            vertices.extend([(coordinate, 0.0, -1.0), (coordinate, 0.0, 1.0)])
            edges.append((first, first + 1))
    elif kind == "wireframe_pyramid":
        vertices = [
            (-0.6, 0.5, -0.6), (0.6, 0.5, -0.6), (0.6, 0.5, 0.6),
            (-0.6, 0.5, 0.6), (0.0, -0.7, 0.0),
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
    else:
        vertices = [
            (x, y, z)
            for z in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for x in (-0.5, 0.5)
        ]
        edges = [
            (i, j)
            for i in range(8)
            for j in range(i + 1, 8)
            if sum(a != b for a, b in zip(vertices[i], vertices[j])) == 1
        ]
    points = _rot3(
        np.asarray(vertices, np.float32),
        rng.uniform(-math.pi, math.pi) if obj.yaw is None else float(obj.yaw),
        rng.uniform(-0.55, 0.55) if obj.pitch is None else float(obj.pitch),
    )
    distance = 3.0 + 2.0 * float(np.clip(obj.depth, 0, 1))
    scale = max(0.1, float(obj.size))
    perspective = 1.0 / np.maximum(points[:, 2] + distance, 0.3)
    u = obj.resolved_x + points[:, 0] * perspective * scale * distance
    v = obj.resolved_y + points[:, 1] * perspective * scale * distance
    thickness = 1 if obj.thick else 0
    coverage_image = np.zeros((H, W, C), np.float32)
    for start, end in edges:
        coordinates = (
            int(round(u[start])), int(round(v[start])),
            int(round(u[end])), int(round(v[end])),
        )
        _bresenham(img, *coordinates, thickness, color)
        _bresenham(coverage_image, *coordinates, thickness, np.ones(C, np.float32))
    return coverage_image[:, :, 0]


def _render_particle_field(img, obj, color, rng):
    count = int(np.clip(round(obj.size * 3), 8, 96))
    spread = max(1.0, obj.width if obj.width is not None else obj.size)
    height = max(1.0, obj.height if obj.height is not None else obj.size)
    coverage = np.zeros((H, W), np.float32)
    for _ in range(count):
        x = obj.resolved_x + rng.normal(0, spread * 0.45)
        y = obj.resolved_y + rng.normal(0, height * 0.45)
        radius = max(0.25, obj.radius * rng.uniform(0.08, 0.28))
        particle_color = np.clip(color * rng.uniform(0.65, 1.25), 0, 1)
        disc = _soft_disc(x, y, radius)
        img[:] = _blend(img, disc, particle_color)
        coverage = _coverage_over(coverage, disc)
    return coverage


def _render_macro(img, obj, color, L):
    x, y, radius = obj.resolved_x, obj.resolved_y, max(0.5, float(obj.radius))
    kind = obj.kind.lower()
    if kind in {"macro_coin", "macro_button", "macro_jewelry"}:
        outer = _soft_disc(x, y, radius)
        inner_radius = radius * (0.62 if kind != "macro_jewelry" else 0.48)
        inner = _soft_disc(x, y, inner_radius)
        img[:] = _blend(img, outer, np.clip(color * 0.65, 0, 1))
        if kind == "macro_jewelry":
            img[:] = _blend(img, inner, np.zeros(C, np.float32))
            coverage = np.clip(outer - inner, 0, 1)
        else:
            img[:] = _blend(img, inner, color)
            coverage = outer
        if kind == "macro_button":
            hole = max(0.35, radius * 0.1)
            for dx in (-radius * 0.22, radius * 0.22):
                for dy in (-radius * 0.22, radius * 0.22):
                    img[:] = _blend(img, _soft_disc(x + dx, y + dy, hole), color * 0.18)
        return coverage
    length = max(2.0, obj.width if obj.width is not None else obj.size * 1.7)
    angle = -0.35 if obj.yaw is None else float(obj.yaw)
    dx, dy = math.cos(angle) * length * 0.5, math.sin(angle) * length * 0.5
    coverage = _tube(
        img, x - dx, y - dy, x + dx, y + dy,
        max(0.5, radius * 0.22), color, L,
    ).astype(np.float32)
    if kind == "macro_key":
        ring = np.clip(
            _soft_disc(x - dx, y - dy, radius)
            - _soft_disc(x - dx, y - dy, radius * 0.55),
            0, 1,
        )
        img[:] = _blend(img, ring, color)
        coverage = _coverage_over(coverage, ring)
    else:
        head = _rect(x - dx - radius * 0.7, y - dy - radius * 0.7,
                     x - dx + radius * 0.7, y - dy + radius * 0.7)
        _fill(img, head, np.clip(color * 0.8, 0, 1))
        coverage = _coverage_over(coverage, head.astype(np.float32))
    return coverage


def _render_food_plate(img, obj, color, rng):
    x, y, radius = obj.resolved_x, obj.resolved_y, max(1.5, float(obj.radius))
    coverage = _soft_disc(x, y, radius)
    img[:] = _blend(img, coverage, np.clip(color * 0.75 + 0.2, 0, 1))
    img[:] = _blend(img, _soft_disc(x, y, radius * 0.78), np.clip(color + 0.12, 0, 1))
    for _ in range(max(3, int(obj.size // 2))):
        angle = rng.uniform(0, 2 * math.pi)
        distance = rng.uniform(0, radius * 0.5)
        food_color = _photo_color(rng, 0.12, 0.95)
        item_radius = rng.uniform(radius * 0.08, radius * 0.22)
        img[:] = _blend(
            img,
            _soft_disc(x + math.cos(angle) * distance, y + math.sin(angle) * distance,
                       item_radius),
            food_color,
        )
    return coverage


def _render_object(img: np.ndarray, obj: ObjectSpec, light, rng: np.random.RandomState):
    """Render one object and return its source-over coverage plane."""
    if not isinstance(obj, ObjectSpec):
        raise TypeError("SceneSpec.objects must contain ObjectSpec instances")
    kind = obj.kind.lower()
    if kind not in _SCENE_OBJECT_KINDS:
        choices = ", ".join(sorted(_SCENE_OBJECT_KINDS))
        raise ValueError(f"unsupported object kind {obj.kind!r}; expected one of: {choices}")
    x, y = obj.resolved_x, obj.resolved_y
    if not np.all(np.isfinite([x, y, obj.depth, obj.radius, obj.size, obj.opacity])):
        raise ValueError("object position, depth, radius, size, and opacity must be finite")
    if not 0.0 <= obj.opacity <= 1.0:
        raise ValueError("ObjectSpec.opacity must be between 0 and 1")
    color = _scene_color(obj.color, rng)
    material_color = _scene_material(obj, color)
    coverage = np.zeros((H, W), np.float32)
    before = img.copy() if obj.opacity < 1.0 else None

    if kind == "sphere_3d":
        mask, nx, ny, nz = _sphere(x, y, max(0.1, float(obj.radius)))
        _draw_3d(img, mask, nx, ny, nz, light, material_color)
        coverage = mask.astype(np.float32)
    elif kind == "blob_3d":
        radius = max(0.1, float(obj.radius))
        shape_type = 1 if obj.shape_type is None else int(obj.shape_type)
        if shape_type == 0:
            mask, nx, ny, nz = _sphere(x, y, radius)
        elif shape_type == 1:
            ry = max(0.1, obj.height * 0.5 if obj.height is not None else radius)
            mask, nx, ny, nz = _organic_blob(rng, x, y, radius, ry)
        elif shape_type == 2:
            mask, nx, ny, nz = _splat(rng, x, y, radius)
        else:
            raise ValueError("ObjectSpec.shape_type must be 0, 1, or 2")
        _draw_3d(img, mask, nx, ny, nz, light, material_color)
        coverage = mask.astype(np.float32)
    elif kind == "tube":
        length = max(0.2, obj.width if obj.width is not None else obj.size)
        angle = (rng.uniform(-math.pi, math.pi) if obj.yaw is None else float(obj.yaw))
        dx = math.cos(angle) * length * 0.5
        dy = (
            float(obj.height) * 0.5
            if obj.height is not None
            else math.sin(angle) * length * 0.5
        )
        coverage = _tube(
            img, x - dx, y - dy, x + dx, y + dy,
            max(0.1, obj.radius), material_color, light,
        ).astype(np.float32)
    elif kind in {"box_3d", "building_3d", "car_3d"}:
        width = max(0.2, obj.width if obj.width is not None else obj.size)
        default_height = obj.size if kind != "car_3d" else obj.size * 0.45
        height = max(0.2, obj.height if obj.height is not None else default_height)
        bottom = y if obj.ground_y is not None or kind != "box_3d" else y + height * 0.5
        top = bottom - height
        angle = rng.uniform(0.35, 1.15) if obj.yaw is None else float(obj.yaw)
        projection = (
            obj.depth_3d if obj.depth_3d is not None else
            width * (0.22 if kind != "car_3d" else 0.16)
        )
        dx, dy = math.cos(angle) * projection, -abs(math.sin(angle) * projection)
        faces = _box3d(
            img, x - width * 0.5, top, x + width * 0.5, bottom,
            dx, dy, material_color, light,
        )
        for face in faces:
            coverage = _coverage_over(coverage, face.astype(np.float32))
        if kind == "building_3d":
            window_color = np.clip(1.0 - color * 0.55, 0.08, 1.0)
            rows = max(1, int(height // 4))
            columns = max(1, int(width // 4))
            for row in range(rows):
                for column in range(columns):
                    wx = x - width * 0.38 + (column + 0.5) * width * 0.76 / columns
                    wy = top + (row + 0.7) * height * 0.82 / rows
                    _fill(img, _rect(wx - 0.55, wy - 0.7, wx + 0.55, wy + 0.7),
                          window_color)
        elif kind == "car_3d":
            cabin_width = width * 0.48
            cabin_faces = _box3d(
                img, x - cabin_width * 0.5, top - height * 0.48,
                x + cabin_width * 0.5, top, dx * 0.7, dy * 0.7,
                np.clip(color * 0.9, 0, 1), light,
            )
            for face in cabin_faces:
                coverage = _coverage_over(coverage, face.astype(np.float32))
            wheel_color = np.full(C, 0.06, np.float32)
            for wheel_x in (x - width * 0.32, x + width * 0.32):
                wheel = _soft_disc(wheel_x, bottom, height * 0.28)
                img[:] = _blend(img, wheel, wheel_color)
                coverage = _coverage_over(coverage, wheel)
    elif kind in _SCENE_CATALOG_KINDS:
        parts = _parts_with_color(
            _object_parts(rng, kind), color if obj.color is not None else None
        )
        distance = 3.0 + 2.0 * float(np.clip(obj.depth, 0, 1))
        masks = _render_parts(
            img, parts, light, rng, yaw=obj.yaw, pitch=obj.pitch,
            dist=distance, size=max(0.1, float(obj.size)), anchor=(0, x, y),
            allow_closeup=False,
        )
        for mask in masks:
            coverage = _coverage_over(coverage, mask.astype(np.float32))
    elif kind.startswith("wireframe_"):
        coverage = _render_wireframe(img, obj, color, rng)
    elif kind in {"disc", "disc_with_rim"}:
        radius = max(0.1, float(obj.radius))
        disc = _soft_disc(x, y, radius)
        img[:] = _blend(img, disc, material_color)
        coverage = disc
        if kind == "disc_with_rim":
            rim = np.clip(
                _soft_disc(x, y, radius)
                - _soft_disc(x, y, max(0.1, radius - max(0.8, radius * 0.16))),
                0, 1,
            )
            img[:] = _blend(img, rim, np.clip(material_color * 0.45, 0, 1))
    elif kind == "particle_field":
        coverage = _render_particle_field(img, obj, color, rng)
    elif kind in _SCENE_MACRO_KINDS:
        coverage = _render_macro(img, obj, color, light)
    elif kind == "food_plate":
        coverage = _render_food_plate(img, obj, color, rng)

    if before is not None:
        img[:] = before + (img - before) * float(obj.opacity)
        coverage *= float(obj.opacity)
    for child in sorted(obj.children, key=lambda item: -item.depth):
        child_coverage = _render_object(img, child, light, rng)
        coverage = _coverage_over(coverage, child_coverage)
    return np.clip(coverage, 0, 1)


def _apply_post(img: np.ndarray, post: PostSpec, rng: np.random.RandomState):
    """Apply explicit post-processing without changing indexed-image behavior."""
    if not isinstance(post, PostSpec):
        raise TypeError("SceneSpec.post must be a PostSpec")
    style_names = {
        None: 0, "realistic": 1, "cartoon": 2, "sketch": 3,
        "watercolor": 4, "neon": 5, "vintage": 6, "pixel_art": 7, "pixel": 7,
    }
    style = None if post.style is None else post.style.lower()
    if style not in style_names:
        raise ValueError(
            "PostSpec.style must be realistic, cartoon, sketch, watercolor, neon, "
            "vintage, pixel_art, or None"
        )
    if not np.all(np.isfinite([post.grain, post.blur, post.vignette])):
        raise ValueError("post-processing strengths must be finite")
    if post.grain < 0 or post.blur < 0 or post.vignette < 0:
        raise ValueError("post-processing strengths cannot be negative")
    out = _apply_style(img, style_names[style], rng)
    if post.grain:
        out = np.clip(
            out + rng.randn(*out.shape).astype(np.float32) * float(post.grain) * 0.08,
            0, 1,
        )
    if post.blur:
        out = gaussian_filter(out, sigma=(float(post.blur), float(post.blur), 0))
    if post.vignette:
        out = _vignette(out, rng, strength=min(1.0, float(post.vignette)))
    return np.clip(out, 0, 1)


def _apply_post_alpha(alpha, post):
    """Apply only coverage-changing post effects to an alpha plane."""
    sigma = float(post.blur)
    style = None if post.style is None else post.style.lower()
    if style == "realistic":
        sigma = max(sigma, 0.35)
    elif style == "watercolor":
        sigma = max(sigma, 1.0)
    elif style == "neon":
        sigma = max(sigma, 0.45)
    if sigma > 0:
        alpha = gaussian_filter(alpha.astype(np.float32), sigma=sigma)
    return np.clip(alpha, 0, 1).astype(np.float32)


def _validated_scene_seed(seed):
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
    ):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    return seed


def _validate_scene_object(obj, active_ids):
    if not isinstance(obj, ObjectSpec):
        raise TypeError("SceneSpec.objects and children must contain ObjectSpec instances")
    if not isinstance(obj.kind, str):
        raise TypeError("ObjectSpec.kind must be a string")
    if (
        isinstance(obj.depth, (bool, np.bool_))
        or not isinstance(obj.depth, (int, float, np.integer, np.floating))
        or not np.isfinite(obj.depth)
    ):
        raise ValueError("ObjectSpec.depth must be a finite number")
    if not isinstance(obj.children, (list, tuple)):
        raise TypeError("ObjectSpec.children must be a list or tuple")
    identity = id(obj)
    if identity in active_ids:
        raise ValueError("ObjectSpec.children cannot contain a reference cycle")
    active_ids.add(identity)
    try:
        for child in obj.children:
            _validate_scene_object(child, active_ids)
    finally:
        active_ids.remove(identity)


def _validate_scene_spec(spec):
    if not isinstance(spec, SceneSpec):
        raise TypeError("spec must be a SceneSpec")
    if not isinstance(spec.background, Background):
        raise TypeError("SceneSpec.background must be a Background")
    if not isinstance(spec.background.kind, str):
        raise TypeError("Background.kind must be a string")
    if not isinstance(spec.light, LightSpec):
        raise TypeError("SceneSpec.light must be a LightSpec")
    if not isinstance(spec.post, PostSpec):
        raise TypeError("SceneSpec.post must be a PostSpec")
    if spec.layout is not None:
        if not isinstance(spec.layout, LayoutSpec):
            raise TypeError("SceneSpec.layout must be a LayoutSpec or None")
        allowed_relations = {"left_of", "right_of", "above", "below", "behind", "in_front_of"}
        for relation in spec.layout.relations:
            if not isinstance(relation, LayoutRelation):
                raise TypeError("LayoutSpec.relations must contain LayoutRelation instances")
            if relation.relation.lower() not in allowed_relations:
                raise ValueError(f"unsupported layout relation {relation.relation!r}")
            if not np.isfinite(relation.gap):
                raise ValueError("LayoutRelation.gap must be finite")
    if spec.post.style is not None and not isinstance(spec.post.style, str):
        raise TypeError("PostSpec.style must be a string or None")
    if not isinstance(spec.objects, (list, tuple)):
        raise TypeError("SceneSpec.objects must be a list or tuple")
    for obj in spec.objects:
        _validate_scene_object(obj, set())


def _scale_scene_object(obj, scale_x, scale_y):
    """Scale one canonical 32×32 object tree to the active native canvas."""
    scale = min(scale_x, scale_y)
    return replace(
        obj,
        x=float(obj.x) * scale_x,
        y=float(obj.y) * scale_y,
        radius=float(obj.radius) * scale,
        width=None if obj.width is None else float(obj.width) * scale_x,
        height=None if obj.height is None else float(obj.height) * scale_y,
        depth_3d=(
            None if obj.depth_3d is None else float(obj.depth_3d) * scale
        ),
        size=float(obj.size) * scale,
        cx=None if obj.cx is None else float(obj.cx) * scale_x,
        cy=None if obj.cy is None else float(obj.cy) * scale_y,
        ground_y=(
            None if obj.ground_y is None else float(obj.ground_y) * scale_y
        ),
        children=[
            _scale_scene_object(child, scale_x, scale_y)
            for child in obj.children
        ],
    )


def _scale_scene_spec(spec, width, height):
    """Return a non-mutating native-resolution view of a canonical scene."""
    if width == DEFAULT_WIDTH and height == DEFAULT_HEIGHT:
        return spec
    scale_x = width / DEFAULT_WIDTH
    scale_y = height / DEFAULT_HEIGHT
    return replace(
        spec,
        objects=[
            _scale_scene_object(obj, scale_x, scale_y)
            for obj in spec.objects
        ],
    )


def _render_scene_current_canvas(
    spec: SceneSpec,
    seed: int = 0,
    raster: RasterSpec | None = None,
) -> np.ndarray:
    """Render a structured scene, optionally converting its raster representation.

    Objects are painted from far to near according to ``depth``.  Randomized
    defaults are local to ``seed`` and isolated from ``make_image`` state. With
    no ``raster``, the historical 32×32 RGB float32 contract is preserved.
    """
    if not isinstance(spec, SceneSpec):
        raise TypeError("spec must be a SceneSpec")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not isinstance(spec.light, LightSpec):
        raise TypeError("SceneSpec.light must be a LightSpec")

    rng = np.random.RandomState(int(seed))
    img = np.zeros((H, W, C), np.float32)
    _render_background(img, spec.background, rng)
    wants_alpha = raster is not None and _raster_requests_alpha(raster)
    transparent_background = spec.background.kind.lower() in {"none", "transparent"}
    alpha = (
        np.zeros((H, W), np.float32)
        if wants_alpha and transparent_background
        else np.ones((H, W), np.float32)
    )
    if spec.light.direction is None:
        light = _rand_light(rng)
    else:
        light = np.asarray(spec.light.direction, np.float32)
        if light.shape != (3,) or not np.all(np.isfinite(light)):
            raise ValueError("LightSpec.direction must contain exactly three finite values")
        norm = float(np.linalg.norm(light))
        if norm < 1e-8:
            raise ValueError("LightSpec.direction cannot be the zero vector")
        light = light / norm

    global _RIM, _IRID, _SUBJECT_GAIN, _LIGHT_COL
    # Copy the tint array: render helpers must not leak mutable global state
    # between a high-resolution camera render and the legacy indexed contract.
    previous_render_state = (_RIM, _IRID, _SUBJECT_GAIN, np.array(_LIGHT_COL, copy=True))
    _RIM, _IRID, _SUBJECT_GAIN = 0.0, 0.0, 1.0
    _LIGHT_COL = np.ones(C, np.float32)
    try:
        for obj in sorted(spec.objects, key=lambda item: -item.depth):
            coverage = _render_object(img, obj, light, rng)
            if wants_alpha:
                alpha = _coverage_over(alpha, coverage)
        if wants_alpha and transparent_background:
            visible = alpha > 1e-6
            img[visible] /= alpha[visible, None]
            img[~visible] = 0.0
        img = _apply_post(img, spec.post, rng)
        if wants_alpha:
            alpha = _apply_post_alpha(alpha, spec.post)
    finally:
        _RIM, _IRID, _SUBJECT_GAIN, _LIGHT_COL = previous_render_state
    image = np.clip(img, 0, 1).astype(np.float32)
    if wants_alpha:
        image = np.concatenate([image, alpha[:, :, None]], axis=2)
    return image if raster is None else convert_raster(image, raster)


def _make_scene_unlocked(spec, seed, raster, width, height):
    """Render a validated scene while the caller owns the render-state lock."""
    with _canvas_size(width, height):
        native_spec = _scale_scene_spec(spec, width, height)
        return _render_scene_current_canvas(native_spec, seed, raster)


def make_scene(
    spec: SceneSpec,
    seed: int = 0,
    raster: RasterSpec | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    """Render one deterministic structured scene.

    Objects are painted from far to near according to ``depth``. Randomized
    defaults are local to ``seed``. Without dimensions the output is a
    32×32×3 float32 RGB array. ``width`` and ``height`` select a native canvas;
    otherwise a RasterSpec of at least 32 pixels per axis selects it.
    Public calls are safe across concurrent threads.
    """
    _validate_scene_spec(spec)
    spec = resolve_layout(spec)
    seed = _validated_scene_seed(seed)
    if raster is not None:
        _validated_raster_spec(raster)
    width, height = _validated_canvas_size(width, height, raster)
    with _RENDER_LOCK:
        return _make_scene_unlocked(
            spec, seed=seed, raster=raster, width=width, height=height
        )


def make_scene_raster(
    spec: SceneSpec,
    raster: RasterSpec,
    seed: int = 0,
) -> np.ndarray:
    """Render a structured scene directly in a requested raster representation."""
    return make_scene(spec, seed=seed, raster=raster)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def _cli_demo():
    """Render a complete level preview grid to disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels_v8.png")

    N = N_LEVELS
    fig, axes = plt.subplots(N, 5, figsize=(12, N * 2.1))
    for li in range(N):
        for si in range(5):
            idx = _level_start(li) + si * (SAMPLES_PER_LEVEL // 5)
            img = make_image(idx, 7)
            axes[li, si].imshow(img, vmin=0, vmax=1)
            axes[li, si].set_title(
                f"{li}:{LEVEL_NAMES[li]} {idx}", fontsize=3.8
            )
            axes[li, si].axis("off")
    plt.suptitle(f"{N_LEVELS} levels — progressive visual generator", fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved -> {out_path}")
    plt.close()


def _cli_single(idx: int):
    """Print stats for a single image."""
    img = make_image(idx, 7)
    print(f"idx={idx}  shape={img.shape}  dtype={img.dtype}")
    print(f"min={img.min():.4f}  max={img.max():.4f}  mean={img.mean():.4f}")
    block = _level_block(idx)
    print(f"level={_level_of(idx)}  block={block}")


def _cli_check():
    """Every level: right shape/range, deterministic, non-degenerate, and fast enough."""
    import time
    for lvl in range(N_LEVELS):
        t0 = time.time()
        imgs = [make_image(_level_start(lvl) + k, 7) for k in range(4)]
        dt = (time.time() - t0) / 4 * 1000
        for im in imgs:
            assert im.shape == (H, W, C) and im.dtype == np.float32, f"lvl {lvl}: bad shape/dtype"
            assert 0.0 <= im.min() and im.max() <= 1.0, f"lvl {lvl}: out of [0,1]"
        assert max(im.std() for im in imgs) > 0.01, f"lvl {lvl}: blank output"
        again = make_image(_level_start(lvl), 7)
        assert np.array_equal(again, imgs[0]), f"lvl {lvl}: not deterministic"
        assert H == 32 and W == 32, f"lvl {lvl}: leaked canvas resolution"
        flag = "  SLOW" if dt > 5 else ""
        print(f"lvl {lvl:3d}  {dt:5.2f} ms  std={np.mean([im.std() for im in imgs]):.3f}{flag}")
    print(f"OK — {N_LEVELS} levels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"Synthetic Image Generator — {N_LEVELS} levels, deterministic")
    ap.add_argument("--demo", action="store_true", help=f"Render {N_LEVELS}×5 preview grid")
    ap.add_argument("--single", type=int, default=None, help="Generate a single image by index")
    ap.add_argument("--check", action="store_true", help="Self-check every level")
    args = ap.parse_args()

    if args.check:
        _cli_check()
    elif args.single is not None:
        _cli_single(args.single)
    elif args.demo or len(sys.argv) == 1:
        _cli_demo()
