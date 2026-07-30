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
from .camera import (update_camera_target, update_camera_shake, apply_camera_effects,
                     trigger_camera_shake, transition_camera)
from .input_system import InputState
from .layers import LayerManager
from .particles import ParticlePool
from .physics import PhysicsWorld
from .scene_graph import SceneQuery, flatten_objects
from .scene_spec import ObjectSpec, RasterSpec, SceneSpec
from .synthetic_image_generator import make_scene
from .tilemap import TilemapRenderer, TilemapSpec
from .tween import AnimationTrack, Updatable
from .scene_serialization import scene_from_dict, scene_to_dict
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


def _camera_object(obj: ObjectSpec, camera: CameraSpec) -> ObjectSpec:
    """Project world coordinates into the renderer's logical 32x32 canvas."""
    logical_w, logical_h = 32.0, 32.0
    sx = (obj.resolved_x - camera.world_x) * camera.zoom + camera.viewport_width / 2.0
    sy = (obj.resolved_y - camera.world_y) * camera.zoom + camera.viewport_height / 2.0
    scale = min(logical_w / camera.viewport_width, logical_h / camera.viewport_height) * camera.zoom
    projected = dataclasses.replace(
        obj, x=sx * logical_w / camera.viewport_width, y=sy * logical_h / camera.viewport_height,
        cx=None, cy=None, ground_y=None,
        radius=obj.radius * scale, size=obj.size * scale,
        width=None if obj.width is None else obj.width * scale,
        height=None if obj.height is None else obj.height * scale,
        children=[_camera_object(child, camera) for child in obj.children],
    )
    return projected


def _camera_scene(scene: SceneSpec, camera: CameraSpec) -> SceneSpec:
    if camera.viewport_width == 32 and camera.viewport_height == 32 and camera.world_x == 0 and camera.world_y == 0 and camera.zoom == 1:
        return scene
    return dataclasses.replace(scene, objects=[_camera_object(obj, camera) for obj in scene.objects])


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
    _camera_transition: tuple | None = field(default=None, init=False, repr=False)

    @property
    def objects(self) -> list[ObjectSpec]:
        return flatten_objects(list(self.scene_spec.objects))

    @property
    def root(self) -> list[ObjectSpec]:
        return self.scene_spec.objects

    def update(self, dt: float, input_state: InputState) -> None:
        for track in self.animations:
            track.update(dt, self.scene_spec)

        self.last_contacts = self.physics.step(dt) if self.physics is not None else []
        if self.last_contacts and "impact_shake" in self.camera.effects:
            strength = min(8.0, 1.5 + 0.5 * len(self.last_contacts))
            self.trigger_camera_shake(strength, 0.18, 24.0)

        for obj in self.objects:
            if obj.flipbook is not None:
                obj.elapsed_ms += dt * 1000.0

        self._update_particles(dt)

        update_camera_target(self.camera, self.objects, dt)
        update_camera_shake(self.camera, dt, self.clock.elapsed)
        if self._camera_transition is not None:
            start, end, elapsed, duration, easing = self._camera_transition
            elapsed = min(duration, elapsed + max(dt, 0.0))
            self.camera = transition_camera(start, end, elapsed / max(duration, 1e-9), easing)
            self._camera_transition = None if elapsed >= duration else (start, end, elapsed, duration, easing)

    def trigger_camera_shake(self, intensity: float = 4.0, duration: float = 0.25,
                             frequency: float = 18.0) -> None:
        """Trigger a deterministic impact impulse on this graph's camera."""
        trigger_camera_shake(self.camera, intensity, duration, frequency)

    def transition_to_camera(self, target: CameraSpec, duration: float = 0.5,
                             easing: str = "smoothstep") -> None:
        """Schedule a deterministic cinematic transition to ``target``."""
        self._camera_transition = (dataclasses.replace(self.camera), dataclasses.replace(target),
                                   0.0, max(float(duration), 1e-9), easing)

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
        base_scene = _camera_scene(dataclasses.replace(self.scene_spec, objects=renderer_objects), self.camera)

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
        return apply_camera_effects(frame, self.camera.effects)

    def snapshot(self) -> dict:
        """Return a JSON-friendly deterministic checkpoint of the simulation."""
        bodies = []
        if self.physics is not None:
            for obj, body in self.physics.entries():
                bodies.append({"name": obj.name, "x": obj.x, "y": obj.y,
                               "velocity": list(self.physics.velocity(obj)),
                               "is_static": body.is_static})
        tweens = []
        for track in self.animations:
            for tween in getattr(track, "tweens", ()):
                tweens.append({"target": tween.target, "property": tween.property,
                               "elapsed": tween.elapsed, "reversed": tween.reversed})
        particles = []
        for pool in self.particles:
            particles.append({
                "alive": pool.alive.astype(bool).tolist(),
                "positions": pool.positions.tolist(), "velocities": pool.velocities.tolist(),
                "ages": pool.ages.tolist(), "lifetimes": pool.lifetimes.tolist(),
                "emit_carry": pool._emit_carry,
            })
        return {"scene": scene_to_dict(self.scene_spec), "bodies": bodies,
                "tweens": tweens, "particles": particles,
                "camera": {"viewport_width": self.camera.viewport_width, "viewport_height": self.camera.viewport_height,
                            "world_x": self.camera.world_x, "world_y": self.camera.world_y,
                            "zoom": self.camera.zoom, "rotation": self.camera.rotation},
                "clock": {"elapsed": self.clock.elapsed,
                "frame_count": self.clock.frame_count, "accumulator": self.clock.accumulator}}

    def restore_snapshot(self, snapshot: dict) -> None:
        """Restore mutable simulation state from :meth:`snapshot`."""
        self.scene_spec = scene_from_dict(snapshot["scene"])
        by_name = {obj.name: obj for obj in self.objects if obj.name}
        for entry in snapshot.get("bodies", ()):
            obj = by_name.get(entry.get("name"))
            if obj is not None:
                obj.x, obj.y = entry["x"], entry["y"]
                if self.physics is not None and self.physics.has_body(obj):
                    self.physics.set_velocity(obj, *entry["velocity"])
        for entry, tween_state in zip(
            (tween for track in self.animations for tween in getattr(track, "tweens", ())),
            snapshot.get("tweens", ()),
        ):
            entry.elapsed = tween_state.get("elapsed", entry.elapsed)
            entry._reversed = bool(tween_state.get("reversed", entry.reversed))
        for pool, state in zip(self.particles, snapshot.get("particles", ())):
            pool.alive[:] = np.asarray(state["alive"], dtype=bool)
            pool.positions[:] = np.asarray(state["positions"], dtype=np.float32)
            pool.velocities[:] = np.asarray(state["velocities"], dtype=np.float32)
            pool.ages[:] = np.asarray(state["ages"], dtype=np.float32)
            pool.lifetimes[:] = np.asarray(state["lifetimes"], dtype=np.float32)
            pool._emit_carry = float(state.get("emit_carry", 0.0))
        camera = snapshot.get("camera", {})
        for key in ("viewport_width", "viewport_height", "world_x", "world_y", "zoom", "rotation"):
            if key in camera:
                setattr(self.camera, key, camera[key])
        state = snapshot.get("clock", {})
        self.clock.elapsed = state.get("elapsed", 0.0)
        self.clock.frame_count = state.get("frame_count", 0)
        self.clock.accumulator = state.get("accumulator", 0.0)

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
