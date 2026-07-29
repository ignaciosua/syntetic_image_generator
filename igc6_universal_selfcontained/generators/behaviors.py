"""The behavior registry: composable dynamics attached to an already-built
``SceneGraph`` from ``scene_levels.py``. Behaviors are pure composition on
top of the v0.10-v0.24 engine (``PhysicsWorld``, ``Tween``/``AnimationTrack``,
``ParticlePool``) — no new engine code, only orchestration.

``compatible_archetypes`` (``None`` = any) exists because a behavior needs
specific tags to find a target (e.g. ``orbit`` needs a ``"center"``-tagged
object) — attaching it to an archetype without that tag is a silent no-op,
so the schedule only draws from the compatible subset to avoid wasting a
sample's variation budget on a behavior that does nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .game_loop import SceneGraph
from .input_system import InputState
from .particles import ParticleEmitterSpec, ParticlePool
from .physics import PhysicsWorld, RigidBodySpec
from .scene_graph import SceneQuery
from .scene_spec import ObjectSpec, SceneSpec
from .tween import AnimationTrack, Tween


def _rand_color(rng: np.random.RandomState):
    return tuple(float(c) for c in rng.uniform(0.15, 0.95, size=3))


def _pick_tagged(scene: SceneSpec, tag: str, rng: np.random.RandomState) -> ObjectSpec | None:
    candidates = SceneQuery(scene).tagged(tag)
    if not candidates:
        return None
    return candidates[int(rng.randint(0, len(candidates)))]


def _ensure_name(obj: ObjectSpec) -> str:
    if obj.name is None:
        obj.name = f"_anon_{id(obj)}"
    return obj.name


def _attach_patrol(graph: SceneGraph, rng: np.random.RandomState) -> None:
    target = _pick_tagged(graph.scene_spec, "movable", rng)
    if target is None:
        return
    name = _ensure_name(target)
    axis = "x" if rng.uniform() < 0.5 else "y"
    current = target.x if axis == "x" else target.y
    offset = float(rng.uniform(3, 8))
    duration = float(rng.uniform(0.8, 2.5))
    tween = Tween(
        target=name, property=axis, from_value=current - offset, to_value=current + offset,
        duration=duration, easing="ease_in_out", loop=True, ping_pong=True,
    )
    graph.animations.append(AnimationTrack([tween]))


@dataclass
class _ChaseController:
    """Duck-typed like ``AnimationTrack`` (``.update(dt, scene)``) but steers
    a named object toward (or away from) another every frame — a
    fixed-endpoint ``Tween`` can't do this since the target keeps moving."""

    chaser_name: str
    target_name: str
    speed: float
    flee: bool = False

    def update(self, dt: float, scene: SceneSpec) -> None:
        query = SceneQuery(scene)
        chaser, target = query.named(self.chaser_name), query.named(self.target_name)
        if chaser is None or target is None:
            return
        dx = target.resolved_x - chaser.resolved_x
        dy = target.resolved_y - chaser.resolved_y
        dist = math.hypot(dx, dy) or 1e-6
        sign = -1.0 if self.flee else 1.0
        chaser.x = chaser.resolved_x + sign * (dx / dist) * self.speed * dt
        chaser.y = chaser.resolved_y + sign * (dy / dist) * self.speed * dt
        chaser.cx = chaser.cy = chaser.ground_y = None


def _attach_chase_or_flee(graph: SceneGraph, rng: np.random.RandomState, flee: bool) -> None:
    player = _pick_tagged(graph.scene_spec, "player", rng)
    chaser = _pick_tagged(graph.scene_spec, "enemy", rng) or _pick_tagged(graph.scene_spec, "movable", rng)
    if player is None or chaser is None or chaser is player:
        return
    speed = float(rng.uniform(4.0, 12.0))
    graph.animations.append(_ChaseController(_ensure_name(chaser), _ensure_name(player), speed, flee=flee))


def _attach_chase(graph: SceneGraph, rng: np.random.RandomState) -> None:
    _attach_chase_or_flee(graph, rng, flee=False)


def _attach_flee(graph: SceneGraph, rng: np.random.RandomState) -> None:
    _attach_chase_or_flee(graph, rng, flee=True)


@dataclass
class _OrbitController:
    body_name: str
    center_x: float
    center_y: float
    radius: float
    angular_speed: float  # radians/sec
    angle: float

    def update(self, dt: float, scene: SceneSpec) -> None:
        body = SceneQuery(scene).named(self.body_name)
        if body is None:
            return
        self.angle += self.angular_speed * dt
        body.x = self.center_x + self.radius * math.cos(self.angle)
        body.y = self.center_y + self.radius * math.sin(self.angle)
        body.cx = body.cy = body.ground_y = None


def _attach_orbit(graph: SceneGraph, rng: np.random.RandomState) -> None:
    center = _pick_tagged(graph.scene_spec, "center", rng)
    bodies = SceneQuery(graph.scene_spec).tagged("body")
    if center is None or not bodies:
        return
    for body in bodies:
        dx, dy = body.resolved_x - center.resolved_x, body.resolved_y - center.resolved_y
        radius = math.hypot(dx, dy) or 1.0
        angle = math.atan2(dy, dx)
        angular_speed = float(rng.uniform(0.5, 2.0)) * (1 if rng.uniform() < 0.5 else -1)
        graph.animations.append(
            _OrbitController(_ensure_name(body), center.resolved_x, center.resolved_y, radius, angular_speed, angle)
        )


def _attach_gravity_drop(graph: SceneGraph, rng: np.random.RandomState) -> None:
    if graph.physics is None:
        graph.physics = PhysicsWorld(gravity=(0.0, float(rng.uniform(60, 140))))
    ground = _pick_tagged(graph.scene_spec, "ground", rng)
    if ground is not None and not graph.physics.has_body(ground):
        graph.physics.add_body(ground, RigidBodySpec(is_static=True))
    movers = SceneQuery(graph.scene_spec).tagged("ball") or SceneQuery(graph.scene_spec).tagged("box")
    for obj in movers:
        if graph.physics.has_body(obj):
            continue
        graph.physics.add_body(
            obj, RigidBodySpec(mass=float(rng.uniform(0.5, 3.0)), restitution=float(rng.uniform(0.1, 0.6)))
        )


def _attach_bounce_particles(graph: SceneGraph, rng: np.random.RandomState) -> None:
    target = _pick_tagged(graph.scene_spec, "ball", rng) or _pick_tagged(graph.scene_spec, "box", rng)
    if target is None:
        return
    name = _ensure_name(target)
    color = _rand_color(rng)
    emitter = ParticleEmitterSpec(
        rate=float(rng.uniform(10, 30)), lifetime=(0.2, 0.6), speed=(2.0, 10.0),
        start_color=color, end_color=(*color, 0.0), start_size=2.0, end_size=0.0, max_particles=60,
    )
    graph.particles.append(ParticlePool(emitter, origin=name))


@dataclass(frozen=True)
class Behavior:
    name: str
    compatible_archetypes: frozenset[str] | None
    attach: Callable[[SceneGraph, np.random.RandomState], None]


BEHAVIORS: tuple[Behavior, ...] = (
    Behavior("patrol", None, _attach_patrol),
    Behavior("chase", frozenset({"platformer", "top_down_arena", "crowd"}), _attach_chase),
    Behavior("flee", frozenset({"platformer", "top_down_arena", "crowd"}), _attach_flee),
    Behavior("orbit", frozenset({"orbiting_bodies"}), _attach_orbit),
    Behavior("gravity_drop", frozenset({"physics_playground", "stacked_boxes"}), _attach_gravity_drop),
    Behavior("bounce_particles", frozenset({"physics_playground", "stacked_boxes"}), _attach_bounce_particles),
)
_BEHAVIORS_BY_NAME = {b.name: b for b in BEHAVIORS}


def compatible_behaviors(archetype_name: str) -> list[Behavior]:
    return [b for b in BEHAVIORS if b.compatible_archetypes is None or archetype_name in b.compatible_archetypes]


if __name__ == "__main__":
    from .scene_levels import _build_orbiting_bodies, _build_physics_playground, _build_top_down_arena

    rng = np.random.RandomState(0)
    arena = _build_top_down_arena(rng)
    _attach_chase(arena, rng)
    chaser = SceneQuery(arena.scene_spec).named("guard")
    player = SceneQuery(arena.scene_spec).named("player")
    d0 = math.hypot(chaser.resolved_x - player.resolved_x, chaser.resolved_y - player.resolved_y)
    for _ in range(30):
        arena.update(1 / 60, InputState())
    d1 = math.hypot(chaser.resolved_x - player.resolved_x, chaser.resolved_y - player.resolved_y)
    assert d1 < d0  # chase actually closed the distance

    orbit_graph = _build_orbiting_bodies(np.random.RandomState(1))
    _attach_orbit(orbit_graph, np.random.RandomState(1))
    body = SceneQuery(orbit_graph.scene_spec).tagged("body")[0]
    center = SceneQuery(orbit_graph.scene_spec).named("center")
    r0 = math.hypot(body.resolved_x - center.resolved_x, body.resolved_y - center.resolved_y)
    for _ in range(10):
        orbit_graph.update(1 / 60, InputState())
    r1 = math.hypot(body.resolved_x - center.resolved_x, body.resolved_y - center.resolved_y)
    assert abs(r1 - r0) < 1e-6  # orbit keeps a constant radius

    playground = _build_physics_playground(np.random.RandomState(2))
    ball = SceneQuery(playground.scene_spec).tagged("ball")[0]
    y0 = ball.resolved_y
    _attach_gravity_drop(playground, np.random.RandomState(2))
    assert playground.physics is not None and playground.physics.has_body(ball)
    for _ in range(10):
        playground.update(1 / 60, InputState())
    assert ball.resolved_y > y0  # gravity moved it (world +y is down, matches ObjectSpec convention)

    assert {b.name for b in compatible_behaviors("orbiting_bodies")} == {"patrol", "orbit"}

    print("OK — behaviors close distance (chase), keep radius (orbit), and add physics (gravity_drop)")
