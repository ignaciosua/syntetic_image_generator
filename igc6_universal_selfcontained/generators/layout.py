"""Minimal semantic layout resolver for structured scenes."""
from __future__ import annotations
from dataclasses import replace
try:
    from .scene_spec import ObjectSpec, SceneSpec
except ImportError:
    from scene_spec import ObjectSpec, SceneSpec


def _extent(obj: ObjectSpec) -> tuple[float, float]:
    r = max(float(obj.radius), 0.5)
    return (float(obj.width or obj.size or r * 2), float(obj.height or obj.size or r * 2))


def resolve_layout(scene: SceneSpec) -> SceneSpec:
    """Return a copy with named relation constraints applied deterministically."""
    layout = scene.layout
    if layout is None or not layout.relations:
        return scene
    objects = list(scene.objects)
    by_name = {obj.name: i for i, obj in enumerate(objects) if obj.name}
    for _ in range(max(1, int(layout.iterations))):
        for rel in layout.relations:
            if rel.source not in by_name or rel.target not in by_name:
                continue
            si, ti = by_name[rel.source], by_name[rel.target]
            source, target = objects[si], objects[ti]
            sw, sh = _extent(source); tw, th = _extent(target)
            gap = float(rel.gap); x, y = source.resolved_x, source.resolved_y
            tx, ty = target.resolved_x, target.resolved_y
            kind = rel.relation.lower()
            if kind == "left_of": x = tx - (sw + tw) * 0.5 - gap
            elif kind == "right_of": x = tx + (sw + tw) * 0.5 + gap
            elif kind == "above": y = ty - (sh + th) * 0.5 - gap
            elif kind == "below": y = ty + (sh + th) * 0.5 + gap
            elif kind in {"behind", "in_front_of"}:
                d = abs(gap) if gap else 0.1
                objects[si] = replace(source, depth=target.depth - d if kind == "behind" else target.depth + d)
                continue
            objects[si] = replace(source, x=x, y=y, cx=None, cy=None, ground_y=None)
    return replace(scene, objects=objects)
