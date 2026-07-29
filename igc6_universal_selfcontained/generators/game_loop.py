"""Fixed-timestep clock and the ``SceneGraph`` that ties camera, physics,
particles, layers, and a scene together into one renderable game state.

``SceneGraph.render()`` calls the public ``make_scene`` exactly like any
other caller would — it never touches renderer internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import CameraSpec
from .input_system import InputState
from .layers import LayerManager
from .particles import ParticlePool
from .physics import PhysicsWorld
from .scene_spec import ObjectSpec, RasterSpec, SceneSpec
from .synthetic_image_generator import make_scene


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


@dataclass
class SceneGraph:
    scene_spec: SceneSpec = field(default_factory=SceneSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    physics: PhysicsWorld | None = None
    particles: list[ParticlePool] = field(default_factory=list)
    layers: LayerManager = field(default_factory=LayerManager)
    clock: GameClock = field(default_factory=GameClock)
    seed: int = 0

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
        if self.physics is not None:
            self.physics.step(dt)
        for pool in self.particles:
            pool.update(dt)

    def render(self) -> np.ndarray:
        raster = RasterSpec(width=self.camera.viewport_width, height=self.camera.viewport_height, mode="rgba")
        frame = make_scene(self.scene_spec, seed=self.seed, raster=raster)
        for pool in self.particles:
            pool.render(frame, self.camera)
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

    print("OK — GameClock steps deterministically and SceneGraph.tick renders a frame")
