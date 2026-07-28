"""Structured scene descriptions for the synthetic image generator.

The dataclasses in this module contain data only.  Rendering lives in
``synthetic_image_generator.py`` so it can reuse the generator's internal
drawing helpers without making them public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field


Color = tuple[float, float, float]


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
class SceneSpec:
    """A complete composable scene."""

    background: Background = field(default_factory=Background)
    objects: list[ObjectSpec] = field(default_factory=list)
    light: LightSpec = field(default_factory=LightSpec)
    post: PostSpec = field(default_factory=PostSpec)


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
    print("OK — scene_spec importable and valid")
