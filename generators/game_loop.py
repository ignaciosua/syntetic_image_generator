"""Fixed-timestep clock and the ``SceneGraph`` that ties camera, physics,
particles, layers, tilemaps, sprites, and a HUD together into one
renderable game state.

``SceneGraph.render()`` composes, bottom to top: tilemaps -> the scene
(``LayerManager``-composited if any layers are registered, otherwise one
plain ``make_scene`` call) -> sprite-tagged objects -> particles -> HUD.
Every layer/tile/sprite pass is either a call to the public ``make_scene``
or a plain numpy alpha-composite onto the accumulated frame — nothing here
reaches into renderer internals.

``SceneGraph.update()`` also drives the two systems that previously needed
manual wiring: ``animations`` (a list of ``AnimationTrack``, advanced every
step against ``scene_spec``) and any ``ParticlePool`` in ``particles`` whose
``origin`` is set (auto-emits at ``emitter.rate`` using a per-pool
deterministic RNG seeded from ``SceneGraph.seed``).

An ``ObjectSpec`` with ``sprite`` or ``flipbook`` set is drawn *only* via
the atlas sprite pass (excluded from the ``make_scene`` call) — set one or
the other, not both a renderer ``kind`` appearance and a sprite. Only
top-level objects are checked (children of a sprite-bearing object aren't
sprite-excluded independently).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .atlas import AtlasSpec, flipbook_region, stamp_sprite
from .camera import CameraSpec, world_to_screen
from .input_system import InputState
from .layers import LayerManager
from .particles import ParticlePool
from .physics import PhysicsWorld
from .scene_graph import SceneQuery
from .scene_spec import ObjectSpec, RasterSpec, SceneSpec
from .synthetic_image_generator import make_scene
from .tilemap import TilemapRenderer, TilemapSpec
from .tween import AnimationTrack, Updatable
from .ui import HUD


@dataclass
class GameClock:
    target_fps: int = 60
    fixed_timestep: float = 1 / 60
    max_frame_time: float = 0.25
    time_scale: float = 1.0
    elapsed: float = 0.0
    frame_count: int = 0
    accumulator: float = 0.0

    def tick(self, real_dt: float) -> int:
        """Advance the clock by ``real_dt`` (capped, time-scaled) and return
        how many ``fixed_timestep`` updates should run this frame."""

        capped = min(real_dt, self.max_frame_time) * self.time_scale
        self.accumulator += capped
        steps = 0
        while self.accumulator >= self.fixed_timestep:
            self.accumulator -= self.fixed_timestep
            self.elapsed += self.fixed_timestep
            self.frame_count += 1
            steps += 1
        return steps


def _alpha_over(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Standard RGBA uint8 source-over: ``top`` composited above ``base``."""

    base_f = base.astype(np.float32)
    top_f = top.astype(np.float32)
    top_a = top_f[..., 3:4] / 255.0
    out = np.empty_like(base)
    out[..., :3] = (top_f[..., :3] * top_a + base_f[..., :3] * (1 - top_a)).clip(0, 255).astype(np.uint8)
    out[..., 3:4] = (top_f[..., 3:4] + base_f[..., 3:4] * (1 - top_a)).clip(0, 255).astype(np.uint8)
    return out


def _split_sprite_objects(objects: list[ObjectSpec]) -> tuple[list[ObjectSpec], list[ObjectSpec]]:
    renderer_objects, sprite_objects = [], []
    for obj in objects:
        (sprite_objects if (obj.sprite is not None or obj.flipbook is not None) else renderer_objects).append(obj)
    return renderer_objects, sprite_objects


@dataclass
class SceneGraph:
    scene_spec: SceneSpec = field(default_factory=SceneSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    physics: PhysicsWorld | None = None
    particles: list[ParticlePool] = field(default_factory=list)
    layers: LayerManager = field(default_factory=LayerManager)
    tilemaps: list[TilemapSpec] = field(default_factory=list)
    atlas: AtlasSpec | None = None
    hud: HUD | None = None
    animations: list[Updatable] = field(default_factory=list)
    clock: GameClock = field(default_factory=GameClock)
    seed: int = 0
    last_hud_events: list[str] = field(default_factory=list, init=False, repr=False)
    last_contacts: list = field(default_factory=list, init=False, repr=False)
    _particle_rngs: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def objects(self) -> list[ObjectSpec]:
        flat: list[ObjectSpec] = []
        for obj in self.scene_spec.objects:
            flat.append(obj)
            flat.extend(obj.children)
        return flat

    @property
    def root(self) -> list[ObjectSpec]:
        return self.scene_spec.objects

    def update(self, dt: float, input_state: InputState) -> None:
        for track in self.animations:
            track.update(dt, self.scene_spec)

        self.last_contacts = self.physics.step(dt) if self.physics is not None else []

        for obj in self.objects:
            if obj.flipbook is not None:
                obj.elapsed_ms += dt * 1000.0

        self._update_particles(dt)

        self.last_hud_events = self.hud.handle_input(input_state) if self.hud is not None else []

    def _update_particles(self, dt: float) -> None:
        query = None
        for index, pool in enumerate(self.particles):
            pool.update(dt)
            if pool.origin is None:
                continue
            if isinstance(pool.origin, str):
                query = query or SceneQuery(self.scene_spec)
                target = query.named(pool.origin)
                if target is None:
                    continue
                origin_pos = (target.resolved_x, target.resolved_y)
            else:
                origin_pos = pool.origin
            rng = self._particle_rngs.setdefault(id(pool), np.random.RandomState(self.seed + index))
            pool.emit_continuous(dt, origin_pos, rng)

    def render(self) -> np.ndarray:
        raster = RasterSpec(width=self.camera.viewport_width, height=self.camera.viewport_height, mode="rgba")
        renderer_objects, sprite_objects = _split_sprite_objects(self.scene_spec.objects)
        base_scene = dataclasses.replace(self.scene_spec, objects=renderer_objects)

        if self.layers.layers:
            layer_images = self.layers.render_layers(base_scene, self.camera, make_scene)
            frame = None
            for layer in self.layers.layers:  # sorted bottom (lowest order) to top
                img = layer_images.get(layer.name)
                if img is None:
                    continue
                frame = img if frame is None else _alpha_over(frame, img)
            if frame is None:
                frame = np.zeros((self.camera.viewport_height, self.camera.viewport_width, 4), np.uint8)
        else:
            frame = make_scene(base_scene, seed=self.seed, raster=raster)

        for tilemap in self.tilemaps:
            frame = _alpha_over(TilemapRenderer.render(tilemap, self.camera), frame)

        if self.atlas is not None:
            for obj in sprite_objects:
                region = flipbook_region(obj.flipbook, obj.elapsed_ms) if obj.flipbook is not None else obj.sprite
                if region is None:
                    continue
                sx, sy = world_to_screen(obj.resolved_x, obj.resolved_y, self.camera)
                stamp_sprite(frame, self.atlas, region, sx, sy, scale=obj.texture_scale, flip_x=obj.flip_x, flip_y=obj.flip_y)

        for pool in self.particles:
            pool.render(frame, self.camera)

        if self.hud is not None:
            self.hud.render(frame)

        return frame

    def tick(self, real_dt: float, input_state: InputState) -> np.ndarray:
        """One complete frame: run all due fixed updates, then render once."""

        for _ in range(self.clock.tick(real_dt)):
            self.update(self.clock.fixed_timestep, input_state)
        return self.render()


if __name__ == "__main__":
    clock = GameClock(fixed_timestep=1 / 60)
    steps = clock.tick(1 / 30)  # two 60Hz steps in one 1/30s frame
    assert steps == 2 and clock.frame_count == 2

    huge = clock.tick(10.0)  # spiral-of-death guard: capped at max_frame_time
    assert huge == round(0.25 / (1 / 60))

    scene = SceneSpec(objects=[ObjectSpec(kind="sphere_3d", x=16, y=16, radius=6)])
    graph = SceneGraph(scene_spec=scene, camera=CameraSpec(viewport_width=32, viewport_height=32))
    frame = graph.tick(1 / 60, InputState())
    assert frame.shape == (32, 32, 4) and frame.dtype == np.uint8

    # animations wired into update(): a Tween moves an object without any
    # manual AnimationTrack.update() call from the caller.
    from .tween import Tween

    hero = ObjectSpec(kind="sphere_3d", name="hero", x=0, y=16, radius=4)
    animated_scene = SceneSpec(objects=[hero])
    animated_graph = SceneGraph(
        scene_spec=animated_scene,
        camera=CameraSpec(viewport_width=32, viewport_height=32),
        animations=[AnimationTrack([Tween(target="hero", property="x", from_value=0, to_value=32, duration=1.0)])],
    )
    animated_graph.update(0.5, InputState())
    assert hero.x == 16.0

    # particles wired into update(): a pool with an origin auto-emits.
    from .particles import ParticleEmitterSpec

    emitter = ParticleEmitterSpec(rate=100.0, lifetime=(1.0, 1.0), speed=(0, 0), max_particles=8)
    pool = ParticlePool(emitter, origin=(16.0, 16.0))
    auto_graph = SceneGraph(scene_spec=SceneSpec(), camera=CameraSpec(viewport_width=32, viewport_height=32), particles=[pool])
    auto_graph.update(0.5, InputState())
    assert pool.alive.sum() > 0  # emitted without any manual pool.emit*() call

    # same seed -> same emission draws, deterministically
    pool_a = ParticlePool(ParticleEmitterSpec(rate=50.0, max_particles=16), origin=(0.0, 0.0))
    pool_b = ParticlePool(ParticleEmitterSpec(rate=50.0, max_particles=16), origin=(0.0, 0.0))
    graph_a = SceneGraph(scene_spec=SceneSpec(), particles=[pool_a], seed=5)
    graph_b = SceneGraph(scene_spec=SceneSpec(), particles=[pool_b], seed=5)
    graph_a.update(0.5, InputState())
    graph_b.update(0.5, InputState())
    assert np.array_equal(pool_a.positions, pool_b.positions)

    print("OK — GameClock steps deterministically; SceneGraph wires animations/particle auto-emission")
