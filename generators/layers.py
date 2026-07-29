"""Named render layers with parallax, rendered via the public ``make_scene`` API.

ponytail: this only calls the public ``make_scene``/``make_scene_raster``
functions once per visible layer — it never reaches into the renderer's
internals, so it cannot change what any single ``make_scene`` call produces.
"""

from __future__ import annotations

import dataclasses

from .camera import CameraSpec, world_to_screen
from .scene_spec import Background, Layer, ObjectSpec, RasterSpec, SceneSpec


class LayerManager:
    def __init__(self) -> None:
        self._layers: dict[str, Layer] = {}

    def add_layer(self, name: str, order: int, parallax: float = 1.0, visible: bool = True) -> Layer:
        layer = Layer(name=name, order=order, parallax=parallax, visible=visible)
        self._layers[name] = layer
        return layer

    def get_layer(self, name: str) -> Layer:
        return self._layers[name]

    @property
    def layers(self) -> list[Layer]:
        return sorted(self._layers.values(), key=lambda l: l.order)

    def sort_objects(self, objects: list[ObjectSpec]) -> list[ObjectSpec]:
        """Painter's-algorithm order: layer order first, then depth (far to near)."""

        return sorted(objects, key=lambda o: (o.layer, -o.depth))

    def render_layers(
        self, scene: SceneSpec, camera: CameraSpec, make_scene_fn
    ) -> dict[str, "object"]:
        """Render each visible layer to its own RGBA array.

        ``make_scene_fn`` is the public ``make_scene``/``make_scene_raster``
        callable (dependency-injected so this module never imports the
        renderer directly). Returns ``{layer_name: rgba_array}``; caller
        composites in ``self.layers`` order.
        """

        bottom_order = min((l.order for l in self._layers.values()), default=0)
        out: dict[str, "object"] = {}
        raster = RasterSpec(width=camera.viewport_width, height=camera.viewport_height, mode="rgba")

        for layer in self.layers:
            if not layer.visible:
                continue
            layer_cam = CameraSpec(
                viewport_width=camera.viewport_width,
                viewport_height=camera.viewport_height,
                world_x=camera.world_x * layer.parallax,
                world_y=camera.world_y * layer.parallax,
                zoom=camera.zoom,
                rotation=camera.rotation,
            )
            transformed = []
            for obj in scene.objects:
                if obj.layer != layer.order or not obj.visible:
                    continue
                sx, sy = world_to_screen(obj.resolved_x, obj.resolved_y, layer_cam)
                transformed.append(dataclasses.replace(obj, x=sx, y=sy, cx=None, cy=None, ground_y=None))

            background = scene.background if layer.order == bottom_order else Background(kind="none")
            layer_scene = SceneSpec(background=background, objects=transformed, light=scene.light, post=scene.post)
            out[layer.name] = make_scene_fn(layer_scene, raster=raster)
        return out


if __name__ == "__main__":
    mgr = LayerManager()
    mgr.add_layer("bg", order=0, parallax=0.5)
    mgr.add_layer("world", order=1, parallax=1.0)
    mgr.add_layer("hud", order=999, parallax=0.0, visible=True)
    assert [l.name for l in mgr.layers] == ["bg", "world", "hud"]

    objs = [
        ObjectSpec(kind="tree", x=0, y=0, depth=0.2, layer=1),
        ObjectSpec(kind="disc", x=0, y=0, depth=0.9, layer=1),
        ObjectSpec(kind="tree", x=0, y=0, depth=0.5, layer=0),
    ]
    ordered = mgr.sort_objects(objs)
    assert ordered[0].layer == 0  # layer 0 painted before layer 1
    assert ordered[1].depth == 0.9 and ordered[2].depth == 0.2  # far-to-near within layer 1

    print("OK — layers sort by (layer, depth) and render_layers composes CameraSpecs per layer")
