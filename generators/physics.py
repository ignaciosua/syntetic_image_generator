"""Lightweight game-physics: semi-implicit Euler integration, AABB/circle
collision resolution, raycasts. Not a conservation-grade simulator.

Simulation reads/writes an ``ObjectSpec``'s canonical ``x``/``y`` fields, not
the ``cx``/``cy``/``ground_y`` aliases (whichever is set wins in
``resolved_x``/``resolved_y``, but physics never touches those). To avoid
silently simulating a body whose position never visibly moves,
``add_body`` snapshots ``resolved_x``/``resolved_y`` into ``x``/``y`` and
clears the aliases once, up front.

``y`` follows the renderer's row convention (0 at the top, increasing
downward — confirmed by ``_render_object``'s ``box_3d`` handling, where a
taller box's top is ``y - height``). A **positive** ``gravity[1]`` therefore
pulls bodies down the image, not negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .scene_graph import check_collision, object_aabb
from .scene_spec import BoundingBox, CollisionShape, ObjectSpec


def _exact_half_extents(obj: ObjectSpec) -> tuple[float, float]:
    """The object's *own* collision half-width/half-height — radius for a
    circle, width/height for a box. Unlike ``object_aabb`` (a deliberately
    generous, query-oriented box that floors on ``radius``/``size``
    defaults — see its docstring), this never lets an unset ``size``
    (default 8.0) or ``radius`` (default 5.0) inflate a shape that never
    asked for one; penetration math needs the real geometry, not padding.
    """

    if obj.collision_shape is CollisionShape.CIRCLE:
        return obj.radius, obj.radius
    return 0.5 * (obj.width if obj.width is not None else obj.size), 0.5 * (obj.height if obj.height is not None else obj.size)


def _contact_normal_and_penetration(
    obj_a: ObjectSpec, obj_b: ObjectSpec, dx: float, dy: float
) -> tuple[float, float, float]:
    """Contact normal (pointing a -> b) and penetration depth, using each
    object's exact shape rather than a padded generic bounding box.

    Center-to-center direction is only a valid normal for circle-circle
    pairs. Any pair involving a box needs the axis of minimum penetration
    (standard SAT-style separation) instead — using center-to-center there
    lets a wide/thin box (e.g. flat ground under an off-center ball) yield
    a mostly-sideways normal, which was launching bodies horizontally on
    deep/fast penetration instead of pushing them back out vertically.
    """

    if obj_a.collision_shape is CollisionShape.CIRCLE and obj_b.collision_shape is CollisionShape.CIRCLE:
        dist = math.hypot(dx, dy) or 1e-6
        return dx / dist, dy / dist, max(0.0, obj_a.radius + obj_b.radius - dist)

    half_wa, half_ha = _exact_half_extents(obj_a)
    half_wb, half_hb = _exact_half_extents(obj_b)
    overlap_x = (half_wa + half_wb) - abs(dx)
    overlap_y = (half_ha + half_hb) - abs(dy)
    if overlap_x < overlap_y:
        return (1.0 if dx >= 0 else -1.0), 0.0, max(0.0, overlap_x)
    return 0.0, (1.0 if dy >= 0 else -1.0), max(0.0, overlap_y)


@dataclass
class RigidBodySpec:
    mass: float = 1.0
    is_static: bool = False
    is_kinematic: bool = False
    velocity: tuple[float, float] = (0.0, 0.0)
    angular_velocity: float = 0.0
    friction: float = 0.5
    restitution: float = 0.3
    linear_damping: float = 0.01
    angular_damping: float = 0.05
    gravity_scale: float = 1.0


@dataclass
class Contact:
    body_a: ObjectSpec
    body_b: ObjectSpec
    point: tuple[float, float]
    normal: tuple[float, float]
    penetration: float


class ContactHandler(Protocol):
    def on_collision_enter(self, contact: Contact) -> None: ...
    def on_collision_exit(self, a: ObjectSpec, b: ObjectSpec) -> None: ...


@dataclass
class RayCastHit:
    obj: ObjectSpec
    point: tuple[float, float]
    distance: float


class PhysicsWorld:
    def __init__(self, gravity: tuple[float, float] = (0.0, 98.0)):
        self.gravity = gravity
        self._entries: list[tuple[ObjectSpec, RigidBodySpec]] = []
        self._velocities: dict[int, list[float]] = {}
        self._handlers: list[ContactHandler] = []
        self._active_pairs: set[tuple[int, int]] = set()

    @property
    def bodies(self) -> list[RigidBodySpec]:
        return [body for _, body in self._entries]

    def has_body(self, obj: ObjectSpec) -> bool:
        return any(o is obj for o, _ in self._entries)

    def entries(self) -> list[tuple[ObjectSpec, RigidBodySpec]]:
        return list(self._entries)

    def add_body(self, obj: ObjectSpec, body: RigidBodySpec) -> None:
        obj.x, obj.y = obj.resolved_x, obj.resolved_y
        obj.cx = obj.cy = obj.ground_y = None
        self._entries.append((obj, body))
        self._velocities[id(obj)] = list(body.velocity)

    def remove_body(self, body: RigidBodySpec) -> None:
        self._entries = [(o, b) for o, b in self._entries if b is not body]

    def set_velocity(self, obj: ObjectSpec, vx: float, vy: float) -> None:
        """Drive ``obj`` by intent (a desired velocity) instead of
        setting its position directly — lets a controller (chase/flee)
        express "move this way" while ``step()``'s collision resolution
        still gets the final say (e.g. a wall blocking it)."""

        self._velocities[id(obj)] = [vx, vy]

    def add_handler(self, handler: ContactHandler) -> None:
        self._handlers.append(handler)

    def step(self, dt: float) -> list[Contact]:
        for obj, body in self._entries:
            if body.is_static or body.is_kinematic:
                continue
            vx, vy = self._velocities[id(obj)]
            vx += self.gravity[0] * body.gravity_scale * dt
            vy += self.gravity[1] * body.gravity_scale * dt
            vx *= max(0.0, 1.0 - body.linear_damping)
            vy *= max(0.0, 1.0 - body.linear_damping)
            obj.x += vx * dt
            obj.y += vy * dt
            self._velocities[id(obj)] = [vx, vy]

        contacts = self._resolve_collisions()
        self._dispatch_events(contacts)
        return contacts

    def _resolve_collisions(self) -> list[Contact]:
        contacts: list[Contact] = []
        n = len(self._entries)
        for i in range(n):
            obj_a, body_a = self._entries[i]
            if obj_a.collision_shape is CollisionShape.NONE:
                continue
            for j in range(i + 1, n):
                obj_b, body_b = self._entries[j]
                if obj_b.collision_shape is CollisionShape.NONE:
                    continue
                if not check_collision(obj_a, obj_b):
                    continue

                dx = obj_b.resolved_x - obj_a.resolved_x
                dy = obj_b.resolved_y - obj_a.resolved_y
                nx, ny, penetration = _contact_normal_and_penetration(obj_a, obj_b, dx, dy)

                self._separate(obj_a, body_a, obj_b, body_b, nx, ny, penetration)
                self._apply_impulse(obj_a, body_a, obj_b, body_b, nx, ny)

                point = ((obj_a.resolved_x + obj_b.resolved_x) / 2, (obj_a.resolved_y + obj_b.resolved_y) / 2)
                contacts.append(Contact(obj_a, obj_b, point, (nx, ny), penetration))
        return contacts

    def _separate(self, a, ba: RigidBodySpec, b, bb: RigidBodySpec, nx, ny, penetration):
        if ba.is_static and bb.is_static:
            return
        if ba.is_static:
            b.x += nx * penetration
            b.y += ny * penetration
        elif bb.is_static:
            a.x -= nx * penetration
            a.y -= ny * penetration
        else:
            a.x -= nx * penetration / 2
            a.y -= ny * penetration / 2
            b.x += nx * penetration / 2
            b.y += ny * penetration / 2

    def _apply_impulse(self, a, ba: RigidBodySpec, b, bb: RigidBodySpec, nx, ny):
        va = self._velocities.get(id(a), [0.0, 0.0])
        vb = self._velocities.get(id(b), [0.0, 0.0])
        rvx, rvy = vb[0] - va[0], vb[1] - va[1]
        vel_along_normal = rvx * nx + rvy * ny
        if vel_along_normal > 0:
            return  # separating already

        inv_mass_a = 0.0 if ba.is_static else 1.0 / max(ba.mass, 1e-6)
        inv_mass_b = 0.0 if bb.is_static else 1.0 / max(bb.mass, 1e-6)
        if inv_mass_a + inv_mass_b == 0:
            return

        e = min(ba.restitution, bb.restitution)
        j = -(1 + e) * vel_along_normal / (inv_mass_a + inv_mass_b)

        if not ba.is_static:
            va[0] -= j * inv_mass_a * nx
            va[1] -= j * inv_mass_a * ny
            self._velocities[id(a)] = va
        if not bb.is_static:
            vb[0] += j * inv_mass_b * nx
            vb[1] += j * inv_mass_b * ny
            self._velocities[id(b)] = vb

    def _dispatch_events(self, contacts: list[Contact]) -> None:
        current = {(id(c.body_a), id(c.body_b)) for c in contacts}
        for c in contacts:
            key = (id(c.body_a), id(c.body_b))
            if key not in self._active_pairs:
                for h in self._handlers:
                    h.on_collision_enter(c)
        for key in self._active_pairs - current:
            id_a, id_b = key
            obj_a = next((o for o, _ in self._entries if id(o) == id_a), None)
            obj_b = next((o for o, _ in self._entries if id(o) == id_b), None)
            if obj_a is not None and obj_b is not None:
                for h in self._handlers:
                    h.on_collision_exit(obj_a, obj_b)
        self._active_pairs = current

    def ray_cast(
        self, origin: tuple[float, float], direction: tuple[float, float], max_distance: float
    ) -> RayCastHit | None:
        """Nearest AABB hit along the ray, if any (slab method)."""

        ox, oy = origin
        dlen = math.hypot(*direction) or 1e-6
        dx, dy = direction[0] / dlen, direction[1] / dlen

        best: RayCastHit | None = None
        for obj, _ in self._entries:
            if obj.collision_shape is CollisionShape.NONE:
                continue
            box = object_aabb(obj)
            t = _ray_aabb(ox, oy, dx, dy, box, max_distance)
            if t is not None and (best is None or t < best.distance):
                best = RayCastHit(obj, (ox + dx * t, oy + dy * t), t)
        return best


def _ray_aabb(ox: float, oy: float, dx: float, dy: float, box: BoundingBox, max_distance: float) -> float | None:
    t0, t1 = 0.0, max_distance
    for o, d, lo, hi in ((ox, dx, box.x0, box.x1), (oy, dy, box.y0, box.y1)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
            continue
        ta, tb = (lo - o) / d, (hi - o) / d
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return None
    return t0 if t0 > 0 else (t1 if t1 > 0 else None)


if __name__ == "__main__":
    world = PhysicsWorld(gravity=(0, 100))
    falling = ObjectSpec(kind="sphere_3d", x=0, y=100, radius=1, collision_shape=CollisionShape.NONE)
    world.add_body(falling, RigidBodySpec(mass=1.0))
    world.step(1.0)
    assert falling.y > 100  # gravity pulled it down the image (y grows downward)
    assert world.has_body(falling)
    assert not world.has_body(ObjectSpec(kind="sphere_3d"))

    # cx/cy-built objects get normalized into x/y on add_body, or physics would
    # silently move a field resolved_x/y never reads.
    aliased = ObjectSpec(kind="sphere_3d", cx=0, cy=100, radius=1, collision_shape=CollisionShape.NONE)
    alias_world = PhysicsWorld(gravity=(0, 100))
    alias_world.add_body(aliased, RigidBodySpec(mass=1.0))
    assert aliased.cx is None and aliased.x == 0.0
    alias_world.step(1.0)
    assert aliased.resolved_y > 100

    ground = ObjectSpec(kind="box_3d", x=0, y=0, size=20, collision_shape=CollisionShape.AABB)
    ball = ObjectSpec(kind="sphere_3d", x=0, y=5, radius=2, collision_shape=CollisionShape.CIRCLE)
    static_world = PhysicsWorld(gravity=(0, 0))
    static_world.add_body(ground, RigidBodySpec(is_static=True))
    static_world.add_body(ball, RigidBodySpec(mass=1.0, restitution=0.5))

    events: list[str] = []

    class Logger:
        def on_collision_enter(self, contact):
            events.append("enter")

        def on_collision_exit(self, a, b):
            events.append("exit")

    static_world.add_handler(Logger())

    hit = static_world.ray_cast((0, 50), (0, -1), 100)
    assert hit is not None and hit.obj is ground  # ground's near edge (t=40) is closer than the ball's (t=41)

    contacts = static_world.step(0.016)
    assert len(contacts) == 1 and events == ["enter"]

    print("OK — physics gravity/collision/impulse/raycast behave sanely")
