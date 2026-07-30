"""Small deterministic procedural materials for structured scene objects.

Materials deliberately depend only on NumPy.  They produce an RGB field in
canvas coordinates, which lets the existing shape masks and lighting code
reuse the result without introducing a texture/image dependency.
"""

from __future__ import annotations

import hashlib
import numpy as np

try:  # package import
    from .scene_spec import MaterialSpec
except ImportError:  # direct ``python generators/synthetic_image_generator.py`` mode
    from scene_spec import MaterialSpec


MATERIAL_NAMES = (
    "flat", "wood", "brushed_metal", "marble", "stone", "concrete", "fabric", "water", "emissive", "glass",
)


def _seed(spec: MaterialSpec) -> int:
    payload = f"{spec.name.lower()}:{spec.seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def evaluate_material(
    spec: MaterialSpec,
    base_color: np.ndarray,
    height: int,
    width: int,
    center: tuple[float, float],
    extent: float,
) -> np.ndarray:
    """Return an ``(height, width, 3)`` deterministic albedo field."""

    if not isinstance(spec, MaterialSpec):
        raise TypeError("material must be a MaterialSpec")
    name = spec.name.lower()
    if name not in MATERIAL_NAMES:
        raise ValueError(f"unknown material {spec.name!r}; expected one of: {', '.join(MATERIAL_NAMES)}")
    if not np.isfinite([spec.scale, spec.rotation, spec.strength]).all():
        raise ValueError("material scale, rotation, and strength must be finite")
    if spec.scale <= 0 or spec.strength < 0:
        raise ValueError("material scale must be positive and strength cannot be negative")

    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    angle = float(spec.rotation) * np.pi / 180.0
    ca, sa = np.cos(angle), np.sin(angle)
    extent = max(float(extent), 1.0)
    u = ((xx - float(center[0])) * ca + (yy - float(center[1])) * sa) / extent
    v = (-(xx - float(center[0])) * sa + (yy - float(center[1])) * ca) / extent
    frequency = 4.0 / float(spec.scale)
    rng = np.random.RandomState(_seed(spec))
    noise = rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32)

    if name == "flat":
        pattern = np.zeros((height, width), np.float32)
    elif name == "wood":
        rings = np.sin(np.sqrt((u * 1.4 + noise * 0.08) ** 2 + (v * 0.35) ** 2) * frequency * 7.0)
        pattern = rings * 0.5 + noise * 0.08
    elif name == "brushed_metal":
        pattern = np.sin(v * frequency * 22.0) * 0.45 + noise * 0.06
    elif name == "marble":
        veins = np.sin((u + np.sin(v * 8.0) * 0.18 + noise * 0.12) * frequency * 9.0)
        pattern = np.sign(veins) * np.abs(veins) ** 3 * 0.5 + noise * 0.08
    elif name == "stone":
        pattern = np.sin(u * frequency * 11.0) * np.sin(v * frequency * 7.0) * 0.35 + noise * 0.18
    elif name == "concrete":
        pattern = noise * 0.22 + np.sin((u + v) * frequency * 5.0) * 0.12
    elif name == "fabric":
        pattern = (np.sin(u * frequency * 30.0) + np.sin(v * frequency * 30.0)) * 0.22 + noise * 0.06
    elif name == "water":
        pattern = np.sin((u + v * 0.35) * frequency * 18.0 + noise * 0.8) * 0.35
    elif name == "glass":
        pattern = np.sin((u + v) * frequency * 10.0) * 0.08 + noise * 0.03
    else:  # emissive: a quiet base pattern; brightness is handled by the caller/style.
        pattern = np.sin((u - v) * frequency * 8.0) * 0.12

    modulation = 1.0 + float(spec.strength) * pattern
    return np.clip(np.asarray(base_color, np.float32).reshape(1, 1, 3) * modulation[..., None], 0.0, 1.0)
