"""The scene-catalog archetype registry: `SCENE_GRAPH_PLAN.md`'s primitives
composed into `idx -> deterministic SceneGraph` the same way `levels.py`
composes the renderer's drawing routines into `idx -> deterministic image`.

Each `_build_*` function takes a seeded ``np.random.RandomState`` and
returns a fully-wired ``SceneGraph`` with no physics/particles-as-dynamics
attached (that's `generators/behaviors.py`'s job in Phase B) — archetypes
are pure *composition*: object kinds, positions, sizes, colors, tags.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .camera import CameraSpec
from .game_loop import SceneGraph
from .particles import ParticleEmitterSpec, ParticlePool
from .scene_spec import AtlasRegion, AtlasSpec, Background, CollisionShape, ObjectSpec, SceneSpec
from .tilemap import TilemapSpec

_VIEWPORT = 32


def _rand_color(rng: np.random.RandomState, base: tuple[float, float, float] | None = None, spread: float = 0.25):
    if base is None:
        return tuple(float(c) for c in rng.uniform(0.15, 0.95, size=3))
    return tuple(float(np.clip(b + rng.uniform(-spread, spread), 0.05, 0.98)) for b in base)


def _build_platformer(rng: np.random.RandomState) -> SceneGraph:
    ground = ObjectSpec(
        kind="box_3d", x=16, y=30, width=32, height=4,
        color=_rand_color(rng, base=(0.3, 0.3, 0.35)), collision_shape=CollisionShape.AABB, tags={"ground"},
    )
    platforms = []
    for _ in range(int(rng.randint(1, 4))):
        platforms.append(
            ObjectSpec(
                kind="box_3d", x=float(rng.uniform(4, 28)), y=float(rng.uniform(10, 24)),
                width=float(rng.uniform(6, 12)), height=2,
                color=_rand_color(rng, base=(0.5, 0.4, 0.25)), collision_shape=CollisionShape.AABB, tags={"platform"},
            )
        )
    player = ObjectSpec(
        kind="human", x=float(rng.uniform(2, 6)), y=26, size=float(rng.uniform(3, 5)), radius=2,
        color=(0.2, 0.9, 0.3), collision_shape=CollisionShape.CIRCLE, tags={"player", "movable"}, name="player",
    )
    enemy = ObjectSpec(
        kind="spider", x=float(rng.uniform(20, 30)), y=float(rng.uniform(8, 20)), size=float(rng.uniform(2, 4)),
        radius=1.5, color=(0.8, 0.2, 0.2), collision_shape=CollisionShape.CIRCLE, tags={"enemy", "movable"}, name="enemy",
    )
    scene = SceneSpec(background=Background(kind="gradient_sky", horizon=0.6), objects=[ground, *platforms, player, enemy])
    return SceneGraph(scene_spec=scene, camera=CameraSpec(viewport_width=_VIEWPORT, viewport_height=_VIEWPORT))


def _build_top_down_arena(rng: np.random.RandomState) -> SceneGraph:
    wall_color = (0.25, 0.25, 0.3)
    walls = [
        ObjectSpec(kind="box_3d", x=16, y=1, width=32, height=2, color=wall_color, collision_shape=CollisionShape.AABB, tags={"wall"}),
        ObjectSpec(kind="box_3d", x=16, y=31, width=32, height=2, color=wall_color, collision_shape=CollisionShape.AABB, tags={"wall"}),
        ObjectSpec(kind="box_3d", x=1, y=16, width=2, height=32, color=wall_color, collision_shape=CollisionShape.AABB, tags={"wall"}),
        ObjectSpec(kind="box_3d", x=31, y=16, width=2, height=32, color=wall_color, collision_shape=CollisionShape.AABB, tags={"wall"}),
    ]
    player = ObjectSpec(
        kind="human", x=6, y=6, size=3, radius=2, color=(0.2, 0.9, 0.3),
        collision_shape=CollisionShape.CIRCLE, tags={"player", "movable"}, name="player",
    )
    pickups = [
        ObjectSpec(
            kind="macro_coin", x=float(rng.uniform(4, 28)), y=float(rng.uniform(4, 28)),
            size=float(rng.uniform(1, 2)), color=(1.0, 0.85, 0.2), tags={"pickup"},
        )
        for _ in range(int(rng.randint(2, 5)))
    ]
    guard = ObjectSpec(
        kind="spider", x=float(rng.uniform(20, 28)), y=float(rng.uniform(20, 28)), size=2.5, radius=1.5,
        color=(0.8, 0.2, 0.2), collision_shape=CollisionShape.CIRCLE, tags={"enemy", "movable"}, name="guard",
    )
    scene = SceneSpec(background=Background(kind="solid", color=(0.1, 0.1, 0.12)), objects=[*walls, player, *pickups, guard])
    return SceneGraph(scene_spec=scene, camera=CameraSpec(viewport_width=_VIEWPORT, viewport_height=_VIEWPORT))


def _build_physics_playground(rng: np.random.RandomState) -> SceneGraph:
    ground = ObjectSpec(
        kind="box_3d", x=16, y=30, width=32, height=4, color=(0.3, 0.3, 0.35),
        collision_shape=CollisionShape.AABB, tags={"ground"},
    )
    balls = [
        ObjectSpec(
            kind="sphere_3d", x=float(rng.uniform(4, 28)), y=float(rng.uniform(4, 20)), radius=float(rng.uniform(1.5, 3.5)),
            color=_rand_color(rng), collision_shape=CollisionShape.CIRCLE, tags={"ball", "movable"}, name=f"ball_{i}",
        )
        for i in range(int(rng.randint(3, 6)))
    ]
    scene = SceneSpec(objects=[ground, *balls])
    return SceneGraph(scene_spec=scene, camera=CameraSpec(viewport_width=_VIEWPORT, viewport_height=_VIEWPORT))


def _build_tilemap_maze(rng: np.random.RandomState) -> SceneGraph:
    tile_size, grid = 4, 8
    tile_img = np.zeros((4, 8, 4), np.uint8)
    tile_img[:, 0:4] = (60, 60, 70, 255)   # floor
    tile_img[:, 4:8] = (90, 60, 40, 255)   # wall
    regions = (
        AtlasRegion(name="floor", x=0, y=0, width=4, height=4),
        AtlasRegion(name="wall", x=4, y=0, width=4, height=4),
    )
    tileset = AtlasSpec(image=tile_img, regions=regions, default_region="floor")

    tiles = np.ones((grid, grid), np.uint16)   # 1 = floor everywhere
    tiles[0, :] = tiles[-1, :] = tiles[:, 0] = tiles[:, -1] = 2  # border walls
    for _ in range(int(rng.randint(3, 8))):
        gx, gy = int(rng.randint(2, grid - 2)), int(rng.randint(2, grid - 2))
        tiles[gy, gx] = 2
    tilemap = TilemapSpec(width=grid, height=grid, tile_size=tile_size, tiles=tiles, tileset=tileset)

    player = ObjectSpec(
        kind="sphere_3d", x=16, y=16, radius=1.5, color=(0.2, 0.9, 0.3),
        collision_shape=CollisionShape.CIRCLE, tags={"player", "movable"}, name="player",
    )
    scene = SceneSpec(background=Background(kind="none"), objects=[player])
    return SceneGraph(
        scene_spec=scene,
        camera=CameraSpec(viewport_width=_VIEWPORT, viewport_height=_VIEWPORT, world_x=16, world_y=16),
        tilemaps=[tilemap],
    )


def _build_particle_burst(rng: np.random.RandomState) -> SceneGraph:
    color = _rand_color(rng)
    emitter_spec = ParticleEmitterSpec(
        lifetime=(4.0, 6.0), speed=(5.0, 20.0), direction=(0.0, -1.0), spread=360.0,
        start_color=color, end_color=(*color, 0.0), start_size=3.0, end_size=0.5, max_particles=40,
    )
    pool = ParticlePool(emitter_spec)
    pool.emit(30, (16.0, 16.0), rng)
    emitter_obj = ObjectSpec(kind="disc", x=16, y=16, radius=1.5, color=color, tags={"emitter", "movable"}, name="emitter")
    scene = SceneSpec(background=Background(kind="solid", color=(0.05, 0.05, 0.08)), objects=[emitter_obj])
    return SceneGraph(
        scene_spec=scene, camera=CameraSpec(viewport_width=_VIEWPORT, viewport_height=_VIEWPORT), particles=[pool]
    )


def _build_crowd(rng: np.random.RandomState) -> SceneGraph:
    kinds = ["human", "insect", "blob"]
    npcs = [
        ObjectSpec(
            kind=str(rng.choice(kinds)), x=float(rng.uniform(2, 30)), y=float(rng.uniform(2, 30)),
            size=float(rng.uniform(1.5, 3)), color=_rand_color(rng), tags={"npc", "movable"}, name=f"npc_{i}",
        )
        for i in range(int(rng.randint(6, 10)))
    ]
    player = ObjectSpec(kind="human", x=16, y=16, size=3, color=(0.2, 0.9, 0.3), tags={"player"}, name="player")
    scene = SceneSpec(objects=[*npcs, player])
    return SceneGraph(scene_spec=scene, camera=CameraSpec(viewport_width=_VIEWPORT, viewport_height=_VIEWPORT))


def _build_orbiting_bodies(rng: np.random.RandomState) -> SceneGraph:
    center = ObjectSpec(kind="sphere_3d", x=16, y=16, radius=4, color=(1.0, 0.85, 0.3), tags={"center"}, name="center")
    bodies = []
    for i in range(int(rng.randint(2, 5))):
        angle = float(rng.uniform(0, 2 * np.pi))
        radius = float(rng.uniform(6, 14))
        bodies.append(
            ObjectSpec(
                kind="sphere_3d", x=16 + radius * np.cos(angle), y=16 + radius * np.sin(angle),
                radius=float(rng.uniform(1, 2.5)), color=_rand_color(rng), tags={"body", "movable"}, name=f"body_{i}",
            )
        )
    scene = SceneSpec(background=Background(kind="solid", color=(0.03, 0.03, 0.08)), objects=[center, *bodies])
    return SceneGraph(scene_spec=scene, camera=CameraSpec(viewport_width=_VIEWPORT, viewport_height=_VIEWPORT))


def _build_stacked_boxes(rng: np.random.RandomState) -> SceneGraph:
    ground = ObjectSpec(
        kind="box_3d", x=16, y=30, width=32, height=4, color=(0.3, 0.3, 0.35),
        collision_shape=CollisionShape.AABB, tags={"ground"},
    )
    boxes = []
    x, y = float(rng.uniform(12, 20)), 27.0
    for i in range(int(rng.randint(3, 6))):
        size = float(rng.uniform(3, 5))
        y -= size
        boxes.append(
            ObjectSpec(
                kind="box_3d", x=x, y=y, width=size, height=size,
                color=_rand_color(rng, base=(0.6, 0.4, 0.2)), collision_shape=CollisionShape.AABB,
                tags={"box", "movable"}, name=f"box_{i}",
            )
        )
    scene = SceneSpec(objects=[ground, *boxes])
    return SceneGraph(scene_spec=scene, camera=CameraSpec(viewport_width=_VIEWPORT, viewport_height=_VIEWPORT))


@dataclass(frozen=True)
class SceneArchetype:
    name: str
    build: Callable[[np.random.RandomState], SceneGraph]


SCENE_ARCHETYPES: tuple[SceneArchetype, ...] = (
    SceneArchetype("platformer", _build_platformer),
    SceneArchetype("top_down_arena", _build_top_down_arena),
    SceneArchetype("physics_playground", _build_physics_playground),
    SceneArchetype("tilemap_maze", _build_tilemap_maze),
    SceneArchetype("particle_burst", _build_particle_burst),
    SceneArchetype("crowd", _build_crowd),
    SceneArchetype("orbiting_bodies", _build_orbiting_bodies),
    SceneArchetype("stacked_boxes", _build_stacked_boxes),
)
_ARCHETYPES_BY_NAME = {a.name: a for a in SCENE_ARCHETYPES}

SAMPLES_PER_ARCHETYPE = 325
N_ARCHETYPES = len(SCENE_ARCHETYPES)
SCENE_CATALOG_CYCLE_SIZE = SAMPLES_PER_ARCHETYPE * N_ARCHETYPES
SCENE_CATALOG_CONTRACT_VERSION = "sgc-8-cycle-v1"


def scene_archetype_of(idx: int) -> SceneArchetype:
    if not isinstance(idx, int) or isinstance(idx, bool):
        raise TypeError("idx must be an int")
    if idx < 0:
        raise ValueError("idx must be non-negative")
    return SCENE_ARCHETYPES[(idx // SAMPLES_PER_ARCHETYPE) % N_ARCHETYPES]


def scene_seed_of(idx: int) -> int:
    return idx


# One exact sample per archetype (idx = archetype_index * SAMPLES_PER_ARCHETYPE),
# hashed over the rendered frame bytes. Computed by scripts/print_scene_catalog_hash.py;
# any change touching archetype geometry/RNG draw order must update this alongside
# a version bump, the same rule LEGACY_LEVEL_SAMPLE_SHA256 already follows.
SCENE_CATALOG_SAMPLE_SHA256 = "5d7307b4afe0d1b4584db8c416c3546b3eb63ae7cdc853079e11331cb26c7041"


if __name__ == "__main__":
    assert N_ARCHETYPES == 8
    a = scene_archetype_of(0)
    assert a.name == "platformer"
    b = scene_archetype_of(SAMPLES_PER_ARCHETYPE)
    assert b.name == "top_down_arena"
    c = scene_archetype_of(SCENE_CATALOG_CYCLE_SIZE)
    assert c.name == "platformer"  # wraps after one full cycle

    graph1 = a.build(np.random.RandomState(scene_seed_of(0)))
    graph2 = a.build(np.random.RandomState(scene_seed_of(0)))
    frame1, frame2 = graph1.render(), graph2.render()
    assert np.array_equal(frame1, frame2)  # deterministic given the same seed

    for archetype in SCENE_ARCHETYPES:
        graph = archetype.build(np.random.RandomState(1))
        frame = graph.render()
        assert frame.shape == (_VIEWPORT, _VIEWPORT, 4), archetype.name

    print("OK — scene archetypes build deterministically and cover the full registry")
