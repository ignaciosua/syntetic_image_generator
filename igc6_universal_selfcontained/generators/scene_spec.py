"""Structured scene descriptions for the synthetic image generator.

The dataclasses in this module contain data only.  Rendering lives in
``synthetic_image_generator.py`` so it can reuse the generator's internal
drawing helpers without making them public API.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


Color = tuple[float, float, float]


class CollisionShape(enum.Enum):
    """Shape used by :mod:`generators.scene_graph` collision checks."""

    NONE = "none"
    AABB = "aabb"
    CIRCLE = "circle"
    CAPSULE = "capsule"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in scene coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def overlaps(self, other: "BoundingBox") -> bool:
        return self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1

    def contains(self, px: float, py: float) -> bool:
        return self.x0 <= px <= self.x1 and self.y0 <= py <= self.y1

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


@dataclass(frozen=True)
class RasterSpec:
    """Requested spatial resolution, color mode, and channel precision.

    ``width`` is the X resolution and ``height`` is the Y resolution.
    ``bits_per_channel`` may be one integer or a channel tuple such as
    ``(5, 6, 5)``. ``alpha_mode`` controls how RGB-only indexed images acquire
    transparency when an RGBA output is requested.
    """

    width: int = 32
    height: int = 32
    mode: str = "rgb"
    bits_per_channel: int | tuple[int, ...] = 8
    resize: str = "bilinear"
    dither: str = "none"
    packed: bool = False
    threshold: float = 0.5
    alpha: float = 1.0
    alpha_mode: str = "auto"


@dataclass
class LightSpec:
    """Directional light used by all shaded objects in a scene."""

    direction: tuple[float, float, float] | None = (0.3, -0.7, 0.6)


@dataclass
class Background:
    """Scene background.

    Supported kinds are ``solid``, ``gradient_sky``, ``perlin``, ``none``, and
    ``transparent``. ``horizon`` is expressed as a fraction of image height.
    """

    kind: str = "solid"
    color: Color | None = None
    horizon: float | None = None
    hi_key: bool = False


@dataclass(frozen=True)
class AtlasRegion:
    """One named rectangle within an :class:`AtlasSpec` texture."""

    name: str
    x: int
    y: int
    width: int
    height: int
    pivot_x: float = 0.5
    pivot_y: float = 0.5


@dataclass(frozen=True)
class AtlasSpec:
    """A texture atlas: one image plus named sub-regions."""

    image: np.ndarray
    regions: tuple[AtlasRegion, ...]
    default_region: str = ""

    def get(self, name: str) -> AtlasRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"no atlas region named {name!r}")


@dataclass(frozen=True)
class FlipbookClip:
    """An ordered sequence of atlas regions played back as an animation."""

    frames: tuple[AtlasRegion, ...]
    frame_duration_ms: int
    loop: bool = True


@dataclass
class ObjectSpec:
    """One independently positioned scene object.

    ``x``/``y`` are the canonical coordinates.  ``cx``/``cy`` and
    ``ground_y`` are accepted aliases because the plan's public examples use
    those spellings.  When supplied, aliases take precedence.
    """

    kind: str
    x: float = 16.0
    y: float = 16.0
    depth: float = 0.5
    radius: float = 5.0
    width: float | None = None
    height: float | None = None
    depth_3d: float | None = None
    size: float = 8.0
    color: Color | None = None
    yaw: float | None = None
    pitch: float | None = None
    shape_type: int | None = None
    thick: bool = False
    opacity: float = 1.0
    children: list["ObjectSpec"] = field(default_factory=list)
    cx: float | None = None
    cy: float | None = None
    ground_y: float | None = None
    tags: set[str] = field(default_factory=set)
    layer: int = 0
    visible: bool = True
    name: str | None = None
    collision_shape: CollisionShape = CollisionShape.NONE
    sprite: AtlasRegion | None = None
    texture_scale: float = 1.0
    flip_x: bool = False
    flip_y: bool = False
    flipbook: FlipbookClip | None = None
    frame: int = 0
    elapsed_ms: float = 0.0

    @property
    def resolved_x(self) -> float:
        return float(self.x if self.cx is None else self.cx)

    @property
    def resolved_y(self) -> float:
        if self.ground_y is not None:
            return float(self.ground_y)
        return float(self.y if self.cy is None else self.cy)


@dataclass
class PostSpec:
    """Optional whole-frame styling and camera effects."""

    style: str | None = None
    grain: float = 0.0
    blur: float = 0.0
    vignette: float = 0.0


@dataclass
class Layer:
    """A named render layer. ``order`` matches ``ObjectSpec.layer``."""

    name: str
    order: int = 0
    parallax: float = 1.0
    visible: bool = True


@dataclass
class SceneSpec:
    """A complete composable scene."""

    background: Background = field(default_factory=Background)
    objects: list[ObjectSpec] = field(default_factory=list)
    light: LightSpec = field(default_factory=LightSpec)
    post: PostSpec = field(default_factory=PostSpec)
    layers: dict[str, Layer] = field(default_factory=dict)


if __name__ == "__main__":
    raster = RasterSpec(width=64, height=48, mode="grayscale", bits_per_channel=16)
    assert raster.width == 64 and raster.height == 48
    scene = SceneSpec()
    assert scene.objects == []
    aliased = SceneSpec(
        objects=[ObjectSpec(kind="sphere_3d", cx=10, cy=20, radius=5)]
    )
    assert aliased.objects[0].resolved_x == 10
    assert aliased.objects[0].resolved_y == 20

    default_obj = ObjectSpec(kind="tree")
    assert default_obj.tags == set() and default_obj.layer == 0
    assert default_obj.visible is True and default_obj.name is None
    assert default_obj.collision_shape is CollisionShape.NONE

    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 15, 15)
    c = BoundingBox(20, 20, 30, 30)
    assert a.overlaps(b) and not a.overlaps(c)
    assert a.contains(5, 5) and not a.contains(11, 5)
    assert a.area() == 100.0

    print("OK — scene_spec importable and valid")
