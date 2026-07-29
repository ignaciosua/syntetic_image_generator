"""Lightweight game-physics: semi-implicit Euler integration, AABB/circle
collision resolution, raycasts. Not a conservation-grade simulator.

Mutates the ``x``/``y`` fields of the ``ObjectSpec`` each body is attached
to directly (not ``cx``/``cy``/``ground_y``) — bodies are expected to use the
canonical coordinate fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .scene_graph import check_collision, object_aabb
from .scene_spec import BoundingBox, CollisionShape, ObjectSpec


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
    def __init__(self, gravity: tuple[float, float] = (0.0, -98.0)):
        self.gravity = gravity
        self._entries: list[tuple[ObjectSpec, RigidBodySpec]] = []
        self._velocities: dict[int, list[float]] = {}
        self._handlers: list[ContactHandler] = []
        self._active_pairs: set[tuple[int, int]] = set()

    @property
    def bodies(self) -> list[RigidBodySpec]:
        return [body for _, body in self._entries]

    def add_body(self, obj: ObjectSpec, body: RigidBodySpec) -> None:
        self._entries.append((obj, body))
        self._velocities[id(obj)] = list(body.velocity)

    def remove_body(self, body: RigidBodySpec) -> None:
        self._entries = [(o, b) for o, b in self._entries if b is not body]

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
                dist = math.hypot(dx, dy) or 1e-6
                nx, ny = dx / dist, dy / dist

                box_a, box_b = object_aabb(obj_a), object_aabb(obj_b)
                overlap_x = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
                overlap_y = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
                penetration = max(0.0, min(overlap_x, overlap_y))

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
    world = PhysicsWorld(gravity=(0, -100))
    falling = ObjectSpec(kind="sphere_3d", x=0, y=100, radius=1, collision_shape=CollisionShape.NONE)
    world.add_body(falling, RigidBodySpec(mass=1.0))
    world.step(1.0)
    assert falling.y < 100  # gravity pulled it down

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
