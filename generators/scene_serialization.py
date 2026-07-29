"""Stable JSON-friendly serialization for declarative scenes."""
from __future__ import annotations

from dataclasses import fields
from enum import Enum
from typing import Any

from .scene_spec import (Background, Layer, LayoutRelation, LayoutSpec, LightSpec,
                         MaterialSpec, ObjectSpec, PostSpec, SceneSpec)


def _value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_value(v) for v in value]
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, list):
        return [_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {f.name: _value(getattr(value, f.name)) for f in fields(value)
                if f.name not in {"sprite", "flipbook"}}
    return value


def scene_to_dict(scene: SceneSpec) -> dict[str, Any]:
    """Convert a SceneSpec into JSON-compatible declarative data."""
    if not isinstance(scene, SceneSpec):
        raise TypeError("scene must be a SceneSpec")
    return _value(scene)


def _object(data: dict[str, Any]) -> ObjectSpec:
    children = [_object(child) for child in data.pop("children", [])]
    material = data.get("material")
    if isinstance(material, dict):
        data["material"] = MaterialSpec(**material)
    data["children"] = children
    return ObjectSpec(**data)


def scene_from_dict(data: dict[str, Any]) -> SceneSpec:
    """Rebuild a SceneSpec from :func:`scene_to_dict` output."""
    if not isinstance(data, dict):
        raise TypeError("scene data must be a mapping")
    raw = dict(data)
    raw["background"] = Background(**raw.get("background", {}))
    raw["light"] = LightSpec(**raw.get("light", {}))
    raw["post"] = PostSpec(**raw.get("post", {}))
    raw["objects"] = [_object(dict(obj)) for obj in raw.get("objects", [])]
    raw["layers"] = {name: Layer(**value) for name, value in raw.get("layers", {}).items()}
    layout = raw.get("layout")
    if isinstance(layout, dict):
        raw["layout"] = LayoutSpec(
            relations=tuple(LayoutRelation(**r) for r in layout.get("relations", [])),
            iterations=layout.get("iterations", 2),
        )
    return SceneSpec(**raw)
