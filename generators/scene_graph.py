"""Spatial queries and collision checks over a ``SceneSpec``.

This module is read-only with respect to rendering: it never imports from
``synthetic_image_generator.py`` and never influences ``make_scene``'s pixel
output. It only inspects ``ObjectSpec``/``SceneSpec`` data.
"""

from __future__ import annotations

import math

from .scene_spec import BoundingBox, CollisionShape, ObjectSpec, SceneSpec


def object_aabb(obj: ObjectSpec) -> BoundingBox:
    """A conservative bounding box for ``obj``.

    ponytail: this approximates each kind with a generous square/box built
    from size/radius/width/height rather than mirroring every per-kind
    constant used by ``_render_object`` (those constants live only inside the
    renderer and duplicating them exactly would risk drifting from it).
    Tighten per-kind only if a real use case (e.g. exact detection-box
    labels) needs pixel-accurate boxes.
    """

    cx, cy = obj.resolved_x, obj.resolved_y
    half_w = 0.5 * (obj.width if obj.width is not None else obj.size)
    half_h = 0.5 * (obj.height if obj.height is not None else obj.size)
    half_w = max(half_w, obj.radius)
    half_h = max(half_h, obj.radius)
    return BoundingBox(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _flatten(objects: list[ObjectSpec]) -> list[ObjectSpec]:
    flat: list[ObjectSpec] = []
    for obj in objects:
        flat.append(obj)
        if obj.children:
            flat.extend(_flatten(obj.children))
    return flat


class SceneQuery:
    """Spatial/tag lookups over a snapshot of a scene's objects."""

    def __init__(self, scene: SceneSpec):
        self._objects = _flatten(list(scene.objects))

    def at_point(self, x: float, y: float) -> list[ObjectSpec]:
        return [o for o in self._objects if object_aabb(o).contains(x, y)]

    def in_rect(self, x0: float, y0: float, x1: float, y1: float) -> list[ObjectSpec]:
        rect = BoundingBox(x0, y0, x1, y1)
        return [o for o in self._objects if object_aabb(o).overlaps(rect)]

    def tagged(self, tag: str) -> list[ObjectSpec]:
        return [o for o in self._objects if tag in o.tags]

    def named(self, name: str) -> ObjectSpec | None:
        for o in self._objects:
            if o.name == name:
                return o
        return None


def _circle_overlap(a: ObjectSpec, b: ObjectSpec) -> bool:
    dx = a.resolved_x - b.resolved_x
    dy = a.resolved_y - b.resolved_y
    return math.hypot(dx, dy) <= (a.radius + b.radius)


def _aabb_circle_overlap(box_obj: ObjectSpec, circle_obj: ObjectSpec) -> bool:
    box = object_aabb(box_obj)
    cx, cy = circle_obj.resolved_x, circle_obj.resolved_y
    nearest_x = min(max(cx, box.x0), box.x1)
    nearest_y = min(max(cy, box.y0), box.y1)
    return math.hypot(cx - nearest_x, cy - nearest_y) <= circle_obj.radius


def check_collision(a: ObjectSpec, b: ObjectSpec) -> bool:
    """True if ``a`` and ``b`` overlap given their ``collision_shape``.

    ponytail: only AABB/CIRCLE pairs are implemented. CAPSULE support is
    deferred until a phase that actually needs it (e.g. physics bodies)
    since segment-vs-shape distance math has no caller yet.
    """

    shapes = {a.collision_shape, b.collision_shape}
    if CollisionShape.NONE in (a.collision_shape, b.collision_shape):
        return False
    if CollisionShape.CAPSULE in shapes:
        raise NotImplementedError("CAPSULE collision checks are not implemented yet")

    if shapes == {CollisionShape.CIRCLE}:
        return _circle_overlap(a, b)
    if shapes == {CollisionShape.AABB}:
        return object_aabb(a).overlaps(object_aabb(b))
    # one AABB, one CIRCLE
    box_obj, circle_obj = (a, b) if a.collision_shape is CollisionShape.AABB else (b, a)
    return _aabb_circle_overlap(box_obj, circle_obj)


def find_collisions(objects: list[ObjectSpec]) -> list[tuple[ObjectSpec, ObjectSpec]]:
    """All colliding pairs among ``objects``.

    ponytail: O(n^2) pairwise scan, fine at expected scene sizes (tens of
    objects). Upgrade to a spatial hash / sweep-and-prune if a later phase
    needs hundreds of dynamic bodies.
    """

    active = [o for o in objects if o.collision_shape is not CollisionShape.NONE]
    pairs: list[tuple[ObjectSpec, ObjectSpec]] = []
    for i, a in enumerate(active):
        for b in active[i + 1 :]:
            if check_collision(a, b):
                pairs.append((a, b))
    return pairs


if __name__ == "__main__":
    scene = SceneSpec(
        objects=[
            ObjectSpec(kind="sphere_3d", x=10, y=10, radius=4, tags={"player"}, name="hero"),
            ObjectSpec(kind="sphere_3d", x=12, y=10, radius=4, tags={"enemy"}),
            ObjectSpec(kind="box_3d", x=100, y=100, size=8, name="wall"),
        ]
    )
    q = SceneQuery(scene)
    assert q.named("hero") is scene.objects[0]
    assert q.named("missing") is None
    assert {o.name for o in q.tagged("enemy")} == {None}
    assert scene.objects[1] in q.in_rect(0, 0, 20, 20)
    assert scene.objects[2] not in q.in_rect(0, 0, 20, 20)
    assert scene.objects[0] in q.at_point(10, 10)

    a, b = scene.objects[0], scene.objects[1]
    a.collision_shape = CollisionShape.CIRCLE
    b.collision_shape = CollisionShape.CIRCLE
    assert check_collision(a, b)  # centers 2 apart, radii 4+4

    box = ObjectSpec(kind="box_3d", x=0, y=0, size=10, collision_shape=CollisionShape.AABB)
    circ = ObjectSpec(kind="sphere_3d", x=6, y=0, radius=2, collision_shape=CollisionShape.CIRCLE)
    far_circ = ObjectSpec(kind="sphere_3d", x=100, y=100, radius=2, collision_shape=CollisionShape.CIRCLE)
    assert check_collision(box, circ)
    assert not check_collision(box, far_circ)

    pairs = find_collisions([a, b, box, circ, far_circ])
    assert (a, b) in pairs or (b, a) in pairs
    print("OK — scene_graph queries and collisions behave as expected")
