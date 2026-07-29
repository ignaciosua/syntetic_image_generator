"""Property tweens and keyframe tracks over ``ObjectSpec`` attributes.

Tweens only mutate plain numeric fields on ``ObjectSpec`` objects already in
the scene (found via :class:`~generators.scene_graph.SceneQuery`) — they
never touch the renderer, so they compose with ``make_scene``/``SceneGraph``
exactly like any other object mutation between frames.

ponytail: only object properties are tweenable in this pass (not camera
zoom/rotation) — extend ``_resolve_targets``/``apply`` if a later use case
needs to tween the camera too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .scene_graph import SceneQuery
from .scene_spec import ObjectSpec, SceneSpec


class Updatable(Protocol):
    """Anything ``SceneGraph.animations`` can drive: ``AnimationTrack`` is
    one implementation; ``generators.behaviors`` adds others (chase/orbit)
    that don't fit the fixed from/to shape of a ``Tween``."""

    def update(self, dt: float, scene: SceneSpec) -> None: ...


def _ease(kind: str, t: float) -> float:
    t = max(0.0, min(1.0, t))
    if kind == "linear":
        return t
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return 1 - (1 - t) * (1 - t)
    if kind == "ease_in_out":
        return 3 * t * t - 2 * t * t * t
    if kind == "bounce":
        if t < 1 / 2.75:
            return 7.5625 * t * t
        if t < 2 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        if t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375
    if kind == "elastic":
        if t in (0.0, 1.0):
            return t
        return -(2 ** (10 * (t - 1))) * math.sin((t - 1.075) * (2 * math.pi) / 0.3)
    raise ValueError(f"unknown easing {kind!r}")


@dataclass
class Tween:
    target: str
    property: str
    from_value: float
    to_value: float
    duration: float
    easing: str = "linear"
    delay: float = 0.0
    loop: bool = False
    ping_pong: bool = False
    elapsed: float = 0.0
    _reversed: bool = field(default=False, repr=False)

    @property
    def finished(self) -> bool:
        return not self.loop and self.elapsed >= self.delay + self.duration

    def value(self) -> float:
        t = 0.0
        if self.duration > 0:
            t = (self.elapsed - self.delay) / self.duration
        t = max(0.0, min(1.0, t))
        if self._reversed:
            t = 1.0 - t
        eased = _ease(self.easing, t)
        return self.from_value + (self.to_value - self.from_value) * eased

    def advance(self, dt: float) -> None:
        if self.finished:
            return
        self.elapsed += dt
        span = self.delay + self.duration
        if self.elapsed < span:
            return
        if self.ping_pong:
            self._reversed = not self._reversed
            self.elapsed = self.delay + (self.elapsed - span)
        elif self.loop:
            self.elapsed = self.delay + (self.elapsed - span)


def _resolve_targets(scene: SceneSpec, name_or_tag: str) -> list[ObjectSpec]:
    query = SceneQuery(scene)
    named = query.named(name_or_tag)
    if named is not None:
        return [named]
    return query.tagged(name_or_tag)


class AnimationTrack:
    def __init__(self, tweens: list[Tween] | None = None):
        self.tweens = tweens or []

    def update(self, dt: float, scene: SceneSpec) -> None:
        for tween in self.tweens:
            tween.advance(dt)
            for obj in _resolve_targets(scene, tween.target):
                setattr(obj, tween.property, tween.value())

    @property
    def finished(self) -> bool:
        return all(t.finished for t in self.tweens)


@dataclass
class Keyframe:
    time: float
    properties: dict[str, float]


class KeyframeTrack:
    def __init__(self, target: str, keyframes: list[Keyframe]):
        self.target = target
        self.keyframes = sorted(keyframes, key=lambda k: k.time)

    def sample(self, time: float) -> dict[str, float]:
        if not self.keyframes:
            return {}
        if time <= self.keyframes[0].time:
            return dict(self.keyframes[0].properties)
        if time >= self.keyframes[-1].time:
            return dict(self.keyframes[-1].properties)

        for a, b in zip(self.keyframes, self.keyframes[1:]):
            if a.time <= time <= b.time:
                span = b.time - a.time
                t = (time - a.time) / span if span > 0 else 0.0
                result: dict[str, float] = {}
                for key in set(a.properties) | set(b.properties):
                    av = a.properties.get(key, b.properties.get(key, 0.0))
                    bv = b.properties.get(key, av)
                    result[key] = av + (bv - av) * t
                return result
        return dict(self.keyframes[-1].properties)


if __name__ == "__main__":
    assert _ease("linear", 0.5) == 0.5
    assert _ease("ease_in", 0.5) == 0.25
    assert abs(_ease("ease_out", 0.5) - 0.75) < 1e-9

    obj = ObjectSpec(kind="sphere_3d", name="hero", x=0)
    scene = SceneSpec(objects=[obj])
    track = AnimationTrack([Tween(target="hero", property="x", from_value=0, to_value=100, duration=1.0)])
    track.update(0.5, scene)
    assert obj.x == 50.0
    track.update(0.5, scene)
    assert obj.x == 100.0 and track.finished

    kf = KeyframeTrack("hero", [Keyframe(0, {"y": 0}), Keyframe(2, {"y": 10})])
    assert kf.sample(1)["y"] == 5.0
    assert kf.sample(-1)["y"] == 0.0
    assert kf.sample(5)["y"] == 10.0

    print("OK — tweens interpolate/ease and keyframe tracks sample correctly")
