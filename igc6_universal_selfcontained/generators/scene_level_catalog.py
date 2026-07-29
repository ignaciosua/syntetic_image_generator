"""Deterministic 154-level scene catalog built from reusable archetypes."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable
import numpy as np
import time

from .camera import CameraSpec, clamp_camera
from .game_loop import SceneGraph
from .scene_spec import Background, MaterialSpec, ObjectSpec, SceneSpec
from .scene_levels import SCENE_ARCHETYPES as LEGACY_ARCHETYPES
from .input_system import NullInputProvider
from .synthetic_image_generator import make_scene
from .physics import PhysicsWorld, RigidBodySpec
from .particles import ParticleEmitterSpec, ParticlePool
from .tween import AnimationTrack, Tween
from .scene_spec import CollisionShape
from .scene_spec import AtlasRegion, AtlasSpec
from .tilemap import TilemapSpec
from .scene_graph import check_collision, object_aabb

SCENE_LEVEL_CONTRACT_VERSION = "slc-154-v1"
SCENE_LEVEL_SAMPLE_SHA256 = "99cbc0faf44f4d30559fe0c465b4dc814809f8c2b2492c54a7603414d59aa2ee"
SCENE_LEVEL_TRAJECTORY_SAMPLE_SHA256 = "e16a26f89e1dee98529b138514b939e8e3c49d4455d4d60f227e9708c44e4c8b"
SAMPLES_PER_SCENE_LEVEL = 325

ARCHETYPE_NAMES = (
    "empty_minimal", "gradient_environment", "abstract_shapes", "geometric_stack", "symmetry_scene", "pattern_grid",
    "single_sphere", "single_box", "single_tube", "single_blob", "single_emissive", "single_transparent",
    "object_row", "object_grid", "object_stack", "object_cluster", "object_scatter", "object_pile",
    "grass_field", "forest", "tree_cluster", "flower_field", "mountain_landscape", "underwater_scene",
    "room", "building_block", "city_block", "bridge_scene", "warehouse", "street_scene",
    "single_character", "character_pair", "character_group", "animal_group", "predator_prey", "crowd_scene",
    "physics_playground", "stacked_objects", "falling_objects", "bouncing_objects", "orbiting_bodies", "pendulum_or_chain",
    "particle_emitter", "particle_burst", "fluid_like_field", "reaction_diffusion", "tilemap_maze", "platformer_level",
)
STYLES = ("natural", "realistic", "cartoon", "sketch", "neon", "pixel", "monochrome")
SIMULATIONS = ("static", "tweened", "physics", "particles", "orbit", "patrol", "bounce", "fall", "chase")
COMPOSITIONS = ("centered", "left_weighted", "right_weighted", "foreground_background", "occlusion", "depth_layers")
CATEGORY_BY_RANGE = ((range(0, 6), "visual"), (range(6, 18), "visual"), (range(18, 24), "environment"),
                     (range(24, 30), "environment"), (range(30, 36), "character"), (range(36, 42), "physics"),
                     (range(42, 48), "system"))


@dataclass(frozen=True)
class SceneLevelSpec:
    name: str
    archetype: str
    variant: str
    style: str
    simulation: str
    composition: str = "centered"
    camera: CameraSpec = field(default_factory=lambda: CameraSpec(viewport_width=32, viewport_height=32))
    seed_offset: int = 0
    frame_count: int = 1
    fixed_dt: float = 1 / 60
    world_width: float = 32.0
    world_height: float = 32.0


@dataclass(frozen=True)
class SceneLevelArchetype:
    name: str
    build: Callable[[np.random.RandomState, SceneLevelSpec], SceneGraph]
    supported_styles: frozenset[str] = frozenset(STYLES)
    supported_simulations: frozenset[str] = frozenset(SIMULATIONS)


FRAME_POLICY = {"static": 1, "tweened": 120, "particles": 180, "orbit": 240,
                "physics": 240, "patrol": 180, "bounce": 180, "fall": 120, "chase": 180}


def scene_level_frame_count(idx: int, quality: str = "training") -> int:
    """Return recommended temporal length for dataset generation."""
    level = scene_level_spec(idx)
    if quality == "preview": return min(12, FRAME_POLICY[level.simulation])
    if quality == "fast": return min(60, FRAME_POLICY[level.simulation])
    if quality != "training": raise ValueError("quality must be preview, fast, or training")
    return FRAME_POLICY[level.simulation]


def _generic_build(rng: np.random.RandomState, level: SceneLevelSpec) -> SceneGraph:
    n = 1 if level.archetype.startswith("single_") else int(rng.randint(2, 7))
    if level.variant == "sparse": n = max(1, n // 2)
    if level.variant == "dense": n += 3
    kind = "sphere_3d" if "sphere" in level.archetype else "box_3d"
    if "character" in level.archetype or "crowd" in level.archetype or "animal" in level.archetype:
        kind = "human"
    objects = []
    for i in range(n):
        size = float(rng.uniform(2, 7))
        x = float(rng.uniform(3, 29)); y = float(rng.uniform(3, 29))
        if level.composition == "centered": x, y = 16.0 + i * 0.5, 16.0
        elif level.composition == "left_weighted": x = float(rng.uniform(2, 16))
        elif level.composition == "right_weighted": x = float(rng.uniform(16, 30))
        elif level.composition == "foreground_background": y = 25.0 if i % 2 else 10.0
        elif level.composition == "depth_layers": y = float(rng.uniform(6, 26))
        color = (0.08, 0.9, 0.95) if level.style == "neon" else tuple(float(v) for v in rng.uniform(0.15, 0.9, 3))
        material_name = {"natural": "wood", "realistic": "stone", "cartoon": "flat", "sketch": "concrete",
                         "neon": "emissive", "pixel": "flat", "monochrome": "brushed_metal"}[level.style]
        tags = {level.archetype}
        if level.simulation in {"patrol", "chase", "orbit"}: tags.add("movable")
        objects.append(ObjectSpec(kind=kind, name=f"object_{i}", x=x, y=y, size=size,
                                  radius=size / 2, color=color, material=MaterialSpec(material_name, seed=i), tags=tags))
    if level.archetype in {"gradient_environment", "forest", "mountain_landscape", "underwater_scene"}:
        background = Background(kind="gradient_sky", horizon=float(rng.uniform(.35, .7)))
    else:
        background = Background(kind="solid", color=tuple(float(v) for v in rng.uniform(.02, .15, 3)))
    return SceneGraph(scene_spec=SceneSpec(background=background, objects=objects), camera=level.camera)


def _priority_build(rng: np.random.RandomState, level: SceneLevelSpec) -> SceneGraph | None:
    """Purpose-built compositions for the first production archetypes."""
    if level.archetype == "forest":
        ground = ObjectSpec(kind="box_3d", name="ground", x=16, y=30, width=32, height=5,
                            color=(.18, .28, .12), material=MaterialSpec("stone", seed=1),
                            collision_shape=CollisionShape.AABB, tags={"ground"})
        trees = [ObjectSpec(kind="tree", name=f"tree_{i}", x=4 + i * 6 + rng.uniform(-1, 1),
                            y=rng.uniform(12, 22), size=rng.uniform(5, 9), radius=3,
                            color=(.12, .45 + rng.uniform(0, .2), .12), material=MaterialSpec("wood", seed=i),
                            depth=.2 + i * .1, tags={"tree", "nature"})
                 for i in range(5 if level.variant != "sparse" else 3)]
        return SceneGraph(SceneSpec(background=Background(kind="gradient_sky", horizon=.55), objects=[ground, *trees]), level.camera)
    if level.archetype == "empty_minimal":
        marker = ObjectSpec(kind="disc", name="marker", x=16, y=16, radius=1.0, color=(.5, .5, .5), tags={"minimal"})
        return SceneGraph(SceneSpec(background=Background(kind="solid", color=(.04, .04, .05)), objects=[marker]), level.camera)
    if level.archetype == "gradient_environment":
        horizon = ObjectSpec(kind="plane", name="horizon", x=16, y=22, width=32, height=20, color=(.3, .55, .8), depth=.1, tags={"environment"})
        return SceneGraph(SceneSpec(background=Background(kind="gradient_sky", horizon=.48), objects=[horizon]), level.camera)
    if level.archetype == "tree_cluster":
        trees = [ObjectSpec(kind="tree", name=f"tree_{i}", x=5 + i * 7, y=18 + rng.uniform(-2, 2), size=7, radius=3, color=(.1, .5, .15), material=MaterialSpec("wood", seed=i), tags={"tree"}) for i in range(4)]
        return SceneGraph(SceneSpec(background=Background(kind="gradient_sky", horizon=.6), objects=trees), level.camera)
    if level.archetype == "room":
        walls = [ObjectSpec(kind="box_3d", name="floor", x=16, y=29, width=32, height=6, color=(.28, .2, .14), material=MaterialSpec("wood"), tags={"ground"}),
                 ObjectSpec(kind="box_3d", name="back_wall", x=16, y=4, width=32, height=3, color=(.35, .4, .48), material=MaterialSpec("concrete"), depth=.1)]
        furniture = [ObjectSpec(kind="box_3d", name="table", x=16, y=22, width=12, height=3, color=(.5, .25, .1), material=MaterialSpec("wood"), depth=.4),
                     ObjectSpec(kind="sphere_3d", name="lamp", x=16, y=14, size=3, radius=2, color=(1., .75, .2), material=MaterialSpec("emissive"), depth=.5)]
        return SceneGraph(SceneSpec(background=Background(kind="solid", color=(.08, .1, .14)), objects=[*walls, *furniture]), level.camera)
    if level.archetype == "single_character":
        hero = ObjectSpec(kind="human", name="hero", x=16, y=20, size=7, radius=3, color=(.2, .8, .4), tags={"player", "character"})
        shadow = ObjectSpec(kind="disc", name="shadow", x=16, y=25, radius=4, color=(.04, .04, .04), opacity=.35, depth=.1)
        return SceneGraph(SceneSpec(background=Background(kind="gradient_sky", horizon=.65), objects=[shadow, hero]), level.camera)
    if level.archetype == "particle_emitter":
        emitter = ObjectSpec(kind="sphere_3d", name="emitter", x=16, y=19, size=4, radius=2, color=(1., .3, .1), material=MaterialSpec("emissive"), tags={"emitter"})
        return SceneGraph(SceneSpec(background=Background(kind="solid", color=(.02, .02, .06)), objects=[emitter]), level.camera)
    if level.archetype == "physics_playground":
        from .scene_levels import _build_physics_playground
        return _build_physics_playground(rng)
    if level.archetype == "platformer_level":
        from .scene_levels import _build_platformer
        return _build_platformer(rng)
    if level.archetype == "orbiting_bodies":
        center = ObjectSpec(kind="sphere_3d", name="center", x=16, y=16, radius=3, color=(1., .75, .2), tags={"center"})
        bodies = [ObjectSpec(kind="sphere_3d", name=f"body_{i}", x=16 + (i + 1) * 4, y=16, radius=1.5,
                             color=tuple(float(v) for v in rng.uniform(.2, .9, 3)), tags={"body", "movable"}) for i in range(3)]
        return SceneGraph(SceneSpec(background=Background(kind="solid", color=(.02, .02, .06)), objects=[center, *bodies]), level.camera)
    if level.archetype in {"grass_field", "flower_field", "mountain_landscape", "underwater_scene"}:
        is_water = level.archetype == "underwater_scene"
        bg = Background(kind="solid", color=(.03, .16, .3) if is_water else (.35, .65, .9))
        ground = ObjectSpec(kind="plane", name="ground", x=16, y=28, width=32, height=8,
                            color=(.06, .25, .12) if not is_water else (.04, .12, .2), depth=.1,
                            tags={"ground"})
        things = []
        count = 8 if level.variant == "dense" else 4
        for i in range(count):
            kind = "flower" if level.archetype == "flower_field" else ("particle_field" if is_water else "bush")
            things.append(ObjectSpec(kind=kind, name=f"nature_{i}", x=float(rng.uniform(2, 30)),
                                     y=float(rng.uniform(10, 27)), size=float(rng.uniform(2, 5)),
                                     radius=2, color=tuple(float(v) for v in rng.uniform(.2, .9, 3)),
                                     depth=.2 + i / 20, tags={"nature"}))
        if level.archetype == "mountain_landscape":
            things.extend(ObjectSpec(kind="plane", name=f"mountain_{i}", x=6 + i * 10, y=12 + rng.uniform(-2, 2), width=14, height=14, color=(.25, .3, .4), depth=.1 + i / 20, tags={"mountain"}) for i in range(3))
        return SceneGraph(SceneSpec(background=bg, objects=[ground, *things]), level.camera)
    if level.archetype == "tilemap_maze":
        image = np.zeros((8, 16, 4), dtype=np.uint8)
        image[:, :8] = (65, 75, 90, 255); image[:, 8:] = (150, 90, 55, 255)
        atlas = AtlasSpec(image=image, regions=(AtlasRegion("floor", 0, 0, 8, 8), AtlasRegion("wall", 8, 0, 8, 8)), default_region="floor")
        tiles = np.ones((8, 8), dtype=np.uint16); tiles[0, :] = tiles[-1, :] = tiles[:, 0] = tiles[:, -1] = 2
        for _ in range(8): tiles[int(rng.randint(1, 7)), int(rng.randint(1, 7))] = 2
        tilemap = TilemapSpec(width=8, height=8, tile_size=4, tiles=tiles, tileset=atlas)
        player = ObjectSpec(kind="sphere_3d", name="player", x=16, y=16, radius=1.5, color=(.2, .9, .3), tags={"player"})
        return SceneGraph(SceneSpec(background=Background(kind="none"), objects=[player]), level.camera, tilemaps=[tilemap], atlas=atlas)
    if level.archetype in {"city_block", "street_scene", "building_block", "bridge_scene", "warehouse"}:
        bg = Background(kind="gradient_sky", horizon=.5)
        objects = [ObjectSpec(kind="plane", name="street", x=16, y=29, width=32, height=6, color=(.12, .13, .16), depth=.1, tags={"ground"})]
        if level.archetype == "bridge_scene":
            objects.append(ObjectSpec(kind="box_3d", name="bridge", x=16, y=18, width=28, height=3, color=(.35, .35, .4), material=MaterialSpec("concrete"), tags={"platform", "ground"}))
        count = 3 if level.archetype in {"city_block", "warehouse"} else 2
        for i in range(count):
            kind = "building_3d" if level.archetype != "warehouse" else "box_3d"
            objects.append(ObjectSpec(kind=kind, name=f"building_{i}", x=5 + i * 11, y=17, width=8, height=20 if i % 2 else 14,
                                       size=8, color=(.25 + i * .08, .3, .38), material=MaterialSpec("concrete", seed=i), depth=.2 + i / 10, tags={"building"}))
        if level.archetype in {"street_scene", "bridge_scene"}:
            objects.append(ObjectSpec(kind="human", name="walker", x=8, y=24, size=4, radius=2, color=(.2, .8, .3), tags={"movable", "player"}))
        if level.archetype == "warehouse":
            objects.extend(ObjectSpec(kind="box_3d", name=f"crate_{i}", x=10 + (i % 3) * 6, y=24 - (i // 3) * 4, size=3, color=(.55, .3, .12), material=MaterialSpec("wood", seed=i), tags={"box", "movable"}) for i in range(6))
        return SceneGraph(SceneSpec(background=bg, objects=objects), level.camera)
    if level.archetype in {"character_pair", "character_group", "animal_group", "predator_prey", "crowd_scene"}:
        count = {"character_pair": 2, "character_group": 4, "animal_group": 5, "predator_prey": 2, "crowd_scene": 7}[level.archetype]
        objects = []
        for i in range(count):
            kind = "human" if "character" in level.archetype or level.archetype == "crowd_scene" else ("spider" if i == 0 and level.archetype == "predator_prey" else "insect")
            tags = {"movable", "character"};
            if level.archetype == "predator_prey": tags.add("enemy" if i == 0 else "player")
            objects.append(ObjectSpec(kind=kind, name=f"agent_{i}", x=6 + i * 4, y=20 + rng.uniform(-2, 2), size=3, radius=1.5, color=tuple(float(v) for v in rng.uniform(.2, .9, 3)), tags=tags))
        return SceneGraph(SceneSpec(background=Background(kind="gradient_sky", horizon=.6), objects=objects), level.camera)
    if level.archetype in {"stacked_objects", "falling_objects", "bouncing_objects", "pendulum_or_chain"}:
        ground = ObjectSpec(kind="box_3d", name="ground", x=16, y=30, width=32, height=4, color=(.3, .3, .35), collision_shape=CollisionShape.AABB, tags={"ground"})
        objects = [ground]
        for i in range(4):
            x = 16 if level.archetype == "stacked_objects" else 7 + i * 6
            y = 26 - i * 4 if level.archetype != "falling_objects" else 5 + i * 3
            objects.append(ObjectSpec(kind="box_3d", name=f"body_{i}", x=x, y=y, size=3, color=tuple(float(v) for v in rng.uniform(.2, .8, 3)), collision_shape=CollisionShape.AABB, tags={"box", "movable", "body"}))
        return SceneGraph(SceneSpec(objects=objects), level.camera)
    if level.archetype == "particle_burst":
        emitter = ObjectSpec(kind="sphere_3d", name="emitter", x=16, y=18, radius=2, size=4, color=(1., .35, .1), material=MaterialSpec("emissive"), tags={"emitter"})
        return SceneGraph(SceneSpec(background=Background(kind="solid", color=(.02, .02, .06)), objects=[emitter]), level.camera)
    if level.archetype == "fluid_like_field":
        blobs = []
        for i in range(10):
            angle = i * np.pi * 2 / 10
            blobs.append(ObjectSpec(kind="blob_3d", name=f"flow_{i}", x=16 + np.cos(angle) * (4 + i * .6),
                                    y=16 + np.sin(angle) * (4 + i * .6), radius=2.0 + (i % 3), size=3 + i % 2,
                                    color=(.08, .35 + i / 20, .8), material=MaterialSpec("water", seed=i),
                                    tags={"fluid", "movable"}, depth=i / 20))
        return SceneGraph(SceneSpec(background=Background(kind="solid", color=(.01, .04, .12)), objects=blobs), level.camera)
    if level.archetype == "reaction_diffusion":
        blobs = []
        for y in range(4, 29, 5):
            for x in range(4, 29, 5):
                value = .5 + .5 * np.sin(x * .7 + y * .31)
                blobs.append(ObjectSpec(kind="blob", name=f"cell_{x}_{y}", x=x, y=y, radius=1.5 + value * 2,
                                        color=(.15 + value * .7, .08, .3 + value * .5), material=MaterialSpec("marble", seed=x + y),
                                        tags={"reaction"}))
        return SceneGraph(SceneSpec(background=Background(kind="solid", color=(.02, .01, .04)), objects=blobs), level.camera)
    if level.archetype in {"abstract_shapes", "geometric_stack", "symmetry_scene", "pattern_grid"}:
        shapes = []
        for i in range(9):
            if level.archetype == "geometric_stack":
                x, y = 16 + rng.uniform(-2, 2), 27 - i * 2.6
            elif level.archetype == "pattern_grid":
                x, y = 5 + (i % 3) * 11, 5 + (i // 3) * 11
            elif level.archetype == "symmetry_scene":
                x, y = 16 + (i % 3 - 1) * 7, 8 + (i // 3) * 8
            else:
                x, y = float(rng.uniform(4, 28)), float(rng.uniform(4, 28))
            shapes.append(ObjectSpec(kind="sphere_3d" if i % 2 else "box_3d", name=f"shape_{i}", x=x, y=y,
                                     size=4, radius=2, color=(.2 + i / 12, .3, .8 - i / 15),
                                     material=MaterialSpec("emissive" if level.style == "neon" else "flat", seed=i), tags={"abstract"}, depth=i / 20))
        return SceneGraph(SceneSpec(background=Background(kind="solid", color=(.03, .03, .05)), objects=shapes), level.camera)
    if level.archetype in {"single_sphere", "single_box", "single_tube", "single_blob", "single_emissive", "single_transparent"}:
        kind = {"single_sphere": "sphere_3d", "single_box": "box_3d", "single_tube": "tube", "single_blob": "blob_3d", "single_emissive": "sphere_3d", "single_transparent": "sphere_3d"}[level.archetype]
        material = "emissive" if level.archetype == "single_emissive" else ("glass" if level.archetype == "single_transparent" else "flat")
        obj = ObjectSpec(kind=kind, name="subject", x=16, y=16, size=9, radius=4.5, width=9, height=9,
                         color=(.1, .8, .95) if material == "emissive" else (.35, .5, .75),
                         opacity=.55 if material == "glass" else 1.0, material=MaterialSpec(material), tags={"subject"})
        return SceneGraph(SceneSpec(background=Background(kind="solid", color=(.04, .05, .08)), objects=[obj]), level.camera)
    if level.archetype in {"object_row", "object_grid", "object_stack", "object_cluster", "object_scatter", "object_pile"}:
        objects = []
        count = 6 if level.archetype != "object_stack" else 5
        for i in range(count):
            if level.archetype == "object_row": x, y = 4 + i * 5, 16
            elif level.archetype == "object_grid": x, y = 8 + (i % 3) * 8, 8 + (i // 3) * 12
            elif level.archetype == "object_stack": x, y = 16 + rng.uniform(-1, 1), 26 - i * 4
            elif level.archetype == "object_pile": x, y = 16 + rng.uniform(-5, 5), 22 + rng.uniform(-5, 5)
            else: x, y = float(rng.uniform(4, 28)), float(rng.uniform(5, 27))
            objects.append(ObjectSpec(kind="box_3d" if i % 2 else "sphere_3d", name=f"item_{i}", x=x, y=y, size=4,
                                      radius=2, color=tuple(float(v) for v in rng.uniform(.2, .85, 3)), material=MaterialSpec("stone", seed=i), tags={"item", "movable"}))
        return SceneGraph(SceneSpec(background=Background(kind="solid", color=(.06, .07, .1)), objects=objects), level.camera)
    return None


def _legacy_build(index: int):
    legacy = LEGACY_ARCHETYPES[index % len(LEGACY_ARCHETYPES)]
    return legacy.build


SCENE_ARCHETYPES = tuple(
    SceneLevelArchetype(name, (lambda rng, level, i=i: _legacy_build(i)(rng)) if i < len(LEGACY_ARCHETYPES) else _generic_build)
    for i, name in enumerate(ARCHETYPE_NAMES)
)
_BY_NAME = {a.name: a for a in SCENE_ARCHETYPES}

_VARIANTS = ("small", "medium", "large", "sparse", "dense", "symmetrical", "irregular")
SCENE_LEVELS = tuple(
    SceneLevelSpec(
        name=f"{ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)]}_{_VARIANTS[(i // len(ARCHETYPE_NAMES)) % len(_VARIANTS)]}",
        archetype=ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)],
        variant=_VARIANTS[(i // len(ARCHETYPE_NAMES)) % len(_VARIANTS)],
        composition=COMPOSITIONS[(i // len(ARCHETYPE_NAMES)) % len(COMPOSITIONS)],
        style=STYLES[(i // 7) % len(STYLES)],
        simulation=("static" if i % 4 else SIMULATIONS[(i // 4) % len(SIMULATIONS)]),
        frame_count=60 if i % 4 == 0 else 1,
        world_width=(512.0 if ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)] in {
            "forest", "tree_cluster", "city_block", "street_scene", "building_block",
            "bridge_scene", "warehouse", "tilemap_maze", "platformer_level",
            "mountain_landscape", "underwater_scene", "room"
        } else 32.0),
        world_height=(512.0 if ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)] in {
            "forest", "tree_cluster", "city_block", "street_scene", "building_block",
            "bridge_scene", "warehouse", "tilemap_maze", "platformer_level",
            "mountain_landscape", "underwater_scene", "room"
        } else 32.0),
        camera=CameraSpec(
            viewport_width=(512 if ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)] in {
                "forest", "tree_cluster", "city_block", "street_scene", "building_block",
                "bridge_scene", "warehouse", "tilemap_maze", "platformer_level",
                "mountain_landscape", "underwater_scene", "room"
            } else 32),
            viewport_height=(512 if ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)] in {
                "forest", "tree_cluster", "city_block", "street_scene", "building_block",
                "bridge_scene", "warehouse", "tilemap_maze", "platformer_level",
                "mountain_landscape", "underwater_scene", "room"
            } else 32),
            world_x=(256.0 if ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)] in {
                "forest", "tree_cluster", "city_block", "street_scene", "building_block",
                "bridge_scene", "warehouse", "tilemap_maze", "platformer_level",
                "mountain_landscape", "underwater_scene", "room"
            } else 16.0),
            world_y=(256.0 if ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)] in {
                "forest", "tree_cluster", "city_block", "street_scene", "building_block",
                "bridge_scene", "warehouse", "tilemap_maze", "platformer_level",
                "mountain_landscape", "underwater_scene", "room"
            } else 16.0),
        ),
    ) for i in range(154)
)
SCENE_LEVEL_NAMES = tuple(level.name for level in SCENE_LEVELS)
SCENE_N_LEVELS = len(SCENE_LEVELS)
SCENE_LEVEL_CYCLE_SIZE = SCENE_N_LEVELS * SAMPLES_PER_SCENE_LEVEL


def scene_level_of(idx: int) -> int:
    if not isinstance(idx, (int, np.integer)) or isinstance(idx, bool): raise TypeError("idx must be an integer")
    if idx < 0: raise ValueError("idx must be non-negative")
    return (int(idx) // SAMPLES_PER_SCENE_LEVEL) % SCENE_N_LEVELS


def scene_level_start(level: int, cycle: int = 0) -> int:
    if not 0 <= int(level) < SCENE_N_LEVELS: raise ValueError("level out of range")
    if cycle < 0: raise ValueError("cycle must be non-negative")
    return (int(cycle) * SCENE_N_LEVELS + int(level)) * SAMPLES_PER_SCENE_LEVEL


def scene_level_spec(idx: int) -> SceneLevelSpec:
    return SCENE_LEVELS[scene_level_of(idx)]


def scene_level_by_name(name: str) -> SceneLevelSpec:
    for level in SCENE_LEVELS:
        if level.name == name:
            return level
    raise KeyError(f"unknown scene level {name!r}")


def validate_scene_level_catalog() -> None:
    if len(SCENE_LEVELS) != 154 or len(set(SCENE_LEVEL_NAMES)) != 154: raise AssertionError("invalid scene level catalog")
    for level in SCENE_LEVELS:
        if level.archetype not in _BY_NAME or level.style not in STYLES or level.simulation not in SIMULATIONS: raise AssertionError(level)


def validate_scene_compatibility() -> None:
    """Ensure every declared level can build the systems it requests."""
    for level in SCENE_LEVELS:
        graph = build_scene_level(level, scene_level_start(SCENE_LEVELS.index(level)))
        if level.simulation != "static" and not graph.objects:
            raise AssertionError(f"dynamic level has no objects: {level.name}")
        if level.simulation == "particles" and not graph.objects:
            raise AssertionError(f"particle level has no emitter target: {level.name}")


def scene_archetype_category(name: str) -> str:
    try:
        index = ARCHETYPE_NAMES.index(name)
    except ValueError as error:
        raise KeyError(name) from error
    for indices, category in CATEGORY_BY_RANGE:
        if index in indices:
            return category
    raise AssertionError(name)


def validate_scene_geometry() -> list[dict]:
    """Check initial bounds and report actual overlaps for every level."""
    reports = []
    for level_index in range(SCENE_N_LEVELS):
        graph = make_scene_level_graph(scene_level_start(level_index))
        out = [obj.name for obj in graph.objects if not (0 <= obj.resolved_x <= graph.camera.viewport_width and 0 <= obj.resolved_y <= graph.camera.viewport_height)]
        overlaps = []
        for i, left in enumerate(graph.objects):
            for right in graph.objects[i + 1:]:
                if check_collision(left, right): overlaps.append((left.name, right.name))
        reports.append({"level": level_index, "out_of_bounds": out, "overlaps": overlaps})
    return reports


def scene_family_hashes() -> dict[str, str]:
    """Return independent render hashes for each six-archetype family."""
    import hashlib
    families = {"composition": range(0, 6), "objects": range(6, 18), "nature": range(18, 24),
                "built": range(24, 30), "characters": range(30, 36), "physics": range(36, 42),
                "systems": range(42, 48)}
    result = {}
    for name, archetype_indices in families.items():
        digest = hashlib.sha256()
        for archetype_index in archetype_indices:
            for level_index, level in enumerate(SCENE_LEVELS):
                if ARCHETYPE_NAMES.index(level.archetype) == archetype_index:
                    digest.update(make_scene_level(scene_level_start(level_index)).tobytes())
        result[name] = digest.hexdigest()
    return result


def make_scene_level_graph(idx: int) -> SceneGraph:
    level = scene_level_spec(idx)
    seed = int(idx) + level.seed_offset
    graph = build_scene_level(level, seed)
    return build_scene_simulation(graph, level, seed)


def build_scene_level(level: SceneLevelSpec, seed: int) -> SceneGraph:
    """Build composition only; simulation is attached separately."""
    if level.archetype not in _BY_NAME:
        raise ValueError(f"unknown scene archetype {level.archetype!r}")
    rng = np.random.RandomState(int(seed))
    specialized = _priority_build(rng, level)
    graph = specialized if specialized is not None else _BY_NAME[level.archetype].build(rng, level)
    # World archetypes are authored in the canonical 32-unit design space;
    # expand them into a real large world while the camera selects a 512-unit
    # region.  This keeps the same composition reusable at thumbnail scale.
    if level.world_width > 32.0:
        sx, sy = level.world_width / 32.0, level.world_height / 32.0
        for obj in graph.objects:
            obj.x = (obj.resolved_x - 16.0) * sx + level.world_width / 2.0
            obj.y = (obj.resolved_y - 16.0) * sy + level.world_height / 2.0
            obj.size *= min(sx, sy)
            obj.radius *= min(sx, sy)
            if obj.width is not None: obj.width *= sx
            if obj.height is not None: obj.height *= sy
    # Camera state is runtime-mutated; never let viewport trajectories mutate
    # the immutable catalog entry shared by subsequent renders.
    graph.camera = replace(level.camera)
    if level.world_width > 32.0:
        graph.camera.bounds = (0.0, 0.0, level.world_width, level.world_height)
    clamp_camera(graph.camera)
    return graph


def build_scene_simulation(graph: SceneGraph, level: SceneLevelSpec, seed: int) -> SceneGraph:
    """Attach deterministic runtime systems selected by ``level.simulation``."""
    rng = np.random.RandomState(int(seed) ^ 0x9E3779B9)
    movers = graph.objects
    mode = level.simulation
    if mode in {"physics", "fall", "bounce"}:
        graph.physics = PhysicsWorld(gravity=(0.0, 80.0 if mode != "bounce" else 0.0))
        for obj in movers:
            if obj.collision_shape is CollisionShape.NONE:
                obj.collision_shape = CollisionShape.CIRCLE
            graph.physics.add_body(obj, RigidBodySpec(
                restitution=float(rng.uniform(.2, .8)), velocity=(float(rng.uniform(-8, 8)), float(rng.uniform(-4, 4)))
            ))
    if mode == "tweened" and movers:
        obj = movers[0]
        graph.animations.append(AnimationTrack([Tween(obj.name or "object_0", "x", obj.resolved_x - 4, obj.resolved_x + 4, max(level.fixed_dt, 1.0), loop=True, ping_pong=True)]))
    if mode == "patrol":
        from .behaviors import _attach_patrol
        _attach_patrol(graph, rng)
    elif mode == "orbit":
        from .behaviors import _attach_orbit
        if len(movers) >= 2:
            movers[0].tags.add("center")
            for body in movers[1:]: body.tags.add("body")
            _attach_orbit(graph, rng)
        elif movers:
            obj = movers[0]
            graph.animations.append(AnimationTrack([Tween(obj.name or "object_0", "y", obj.resolved_y - 2, obj.resolved_y + 2, 1.0, loop=True, ping_pong=True)]))
    elif mode == "chase" and len(movers) >= 2:
        from .behaviors import _ChaseController
        graph.animations.append(_ChaseController(movers[0].name or "object_0", movers[1].name or "object_1", speed=8.0, physics=graph.physics))
    elif mode == "chase" and movers:
        obj = movers[0]
        graph.animations.append(AnimationTrack([Tween(obj.name or "object_0", "x", obj.resolved_x - 3, obj.resolved_x + 3, 1.0, loop=True, ping_pong=True)]))
    elif mode == "orbit" and movers:
        obj = movers[0]
        graph.animations.append(AnimationTrack([Tween(obj.name or "object_0", "y", obj.resolved_y - 2, obj.resolved_y + 2, 1.0, loop=True, ping_pong=True)]))
    if mode == "particles" or "particle" in level.archetype:
        target = movers[0] if movers else None
        if target is not None:
            target.name = target.name or "emitter"
            color = target.color or (1.0, .5, .2)
            graph.particles.append(ParticlePool(ParticleEmitterSpec(
                rate=18.0, lifetime=(.4, 1.2), speed=(4.0, 16.0), spread=360.0,
                start_color=color, end_color=(*color, 0.0), max_particles=80,
            ), origin=target.name))
    return graph


def make_scene_level(idx: int, *, raster=None) -> np.ndarray:
    graph = make_scene_level_graph(idx)
    if raster is None:
        return graph.render()
    return make_scene(graph.scene_spec, seed=int(idx), raster=raster)


def make_scene_trajectory_level(idx: int, n_frames: int = 60, fixed_dt: float = 1 / 60):
    if n_frames < 1 or fixed_dt <= 0: raise ValueError("n_frames and fixed_dt must be positive")
    graph = make_scene_level_graph(idx)
    frames, states, events = [], [], []
    provider = NullInputProvider()
    for _ in range(int(n_frames)):
        graph.update(fixed_dt, provider.poll())
        frames.append(graph.render())
        object_states = {}
        for i, obj in enumerate(graph.objects):
            state = {"x": obj.resolved_x, "y": obj.resolved_y, "depth": obj.depth,
                     "tags": sorted(obj.tags), "layer": obj.layer,
                     "bbox": {"x0": obj.resolved_x - obj.radius, "y0": obj.resolved_y - obj.radius,
                              "x1": obj.resolved_x + obj.radius, "y1": obj.resolved_y + obj.radius}}
            if obj.material is not None:
                state["material"] = {"name": obj.material.name, "scale": obj.material.scale,
                                      "rotation": obj.material.rotation, "strength": obj.material.strength,
                                      "seed": obj.material.seed}
            if graph.physics is not None and graph.physics.has_body(obj):
                state["velocity"] = list(graph.physics.velocity(obj))
            object_states[obj.name or str(i)] = state
        states.append(object_states)
        events.append({"contacts": [{"a": c.body_a.name, "b": c.body_b.name} for c in graph.last_contacts],
                       "hud": list(graph.last_hud_events), "camera": {"world_x": graph.camera.world_x,
                       "world_y": graph.camera.world_y, "zoom": graph.camera.zoom, "rotation": graph.camera.rotation,
                       "mode": graph.camera.mode, "target": graph.camera.target,
                       "bounds": graph.camera.bounds, "dead_zone": graph.camera.dead_zone,
                       "look_ahead": graph.camera.look_ahead, "smoothing": graph.camera.smoothing,
                       "effects": list(graph.camera.effects)}})
    from .scene_catalog import SceneSample
    level = scene_level_spec(idx)
    metadata = {
        "level": scene_level_of(idx), "name": level.name, "archetype": level.archetype,
        "variant": level.variant, "composition": level.composition, "style": level.style, "simulation": level.simulation,
        "seed": int(idx) + level.seed_offset,
        "camera": {"viewport_width": level.camera.viewport_width, "viewport_height": level.camera.viewport_height,
                    "world_x": level.camera.world_x, "world_y": level.camera.world_y, "zoom": level.camera.zoom},
    }
    return SceneSample(idx=int(idx), archetype=level.archetype, behaviors=[], seed=int(idx), frames=frames, objects_state=states, events=events, metadata=metadata)


def make_scene_training_trajectory(idx: int, *, quality: str = "training", fixed_dt: float = 1 / 60):
    """Generate the recommended full sequence for training or preview."""
    return make_scene_trajectory_level(idx, n_frames=scene_level_frame_count(idx, quality), fixed_dt=fixed_dt)


def analyze_scene_trajectory(sample) -> dict:
    """Classify a trajectory as static, looping, evolving, or divergent."""
    import json, hashlib
    signatures = [hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest() for state in sample.objects_state]
    first_repeat = None
    seen = {}
    for index, signature in enumerate(signatures):
        if signature in seen:
            first_repeat = (seen[signature], index)
            break
        seen[signature] = index
    displacements = []
    if sample.objects_state:
        initial = sample.objects_state[0]
        for state in sample.objects_state:
            total = 0.0
            for name, value in state.items():
                if name in initial:
                    dx = value["x"] - initial[name]["x"]; dy = value["y"] - initial[name]["y"]
                    total += float((dx * dx + dy * dy) ** .5)
            displacements.append(total)
    maximum = max(displacements, default=0.0)
    return {"frames": len(sample.frames), "first_repeat": first_repeat, "max_displacement": maximum,
            "classification": "static" if maximum == 0 else ("loop" if first_repeat else "evolving")}


def make_scene_camera_trajectory(idx: int, n_frames: int = 120, viewport=(512, 512), mode: str = "pan"):
    """Render a deterministic camera path over the same scene state."""
    valid_modes = {"static", "top_down", "top_down_follow", "pan", "zoom", "diagonal", "orbit", "dolly",
                   "first_person", "third_person", "follow", "lead", "rail", "shake", "rotate"}
    if mode not in valid_modes:
        raise ValueError("unsupported camera mode")
    template = make_scene_level_graph(idx)
    origin_x, origin_y = template.camera.world_x, template.camera.world_y
    frames = []
    for frame_index in range(int(n_frames)):
        phase = frame_index / max(1, n_frames - 1)
        graph = make_scene_level_graph(idx)
        graph.camera.viewport_width, graph.camera.viewport_height = viewport
        angle = phase * np.pi * 2
        target = template.objects[0] if template.objects else None
        if mode in {"follow", "top_down_follow", "first_person", "third_person", "lead"} and target is not None:
            # Deterministic subject patrol so follow cameras visibly track it.
            target.x = target.resolved_x + np.sin(angle) * 2.5
            target.y = target.resolved_y + np.cos(angle * 0.7) * 1.5
        if mode in {"static", "top_down"}:
            dx, dy, zoom = 0.0, 0.0, 1.0
        elif mode == "pan":
            dx, dy, zoom = np.sin(angle) * 128, np.cos(angle) * 64, 1.0
        elif mode == "zoom":
            dx, dy, zoom = 0.0, 0.0, 0.65 + 0.7 * (0.5 + 0.5 * np.sin(angle))
        elif mode == "diagonal":
            dx, dy, zoom = (phase - .5) * 240, (phase - .5) * 180, 1.0
        elif mode == "orbit":
            dx, dy, zoom = np.cos(angle) * 150, np.sin(angle) * 150, 1.0 + .15 * np.sin(angle * 2)
        else:  # dolly: move forward while gently reframing
            dx, dy, zoom = (phase - .5) * 180, np.sin(angle) * 50, 0.75 + phase * .8
        if mode == "top_down_follow" and target is not None:
            dx, dy, zoom = target.resolved_x - origin_x, target.resolved_y - origin_y, 1.0
        elif mode == "first_person" and target is not None:
            dx, dy, zoom = target.resolved_x - origin_x, target.resolved_y - origin_y, 1.45
        elif mode == "third_person" and target is not None:
            dx, dy, zoom = target.resolved_x - origin_x - 70, target.resolved_y - origin_y - 45, 0.9
        elif mode == "follow" and target is not None:
            dx, dy, zoom = target.resolved_x - origin_x + np.sin(angle) * 18, target.resolved_y - origin_y + np.cos(angle) * 12, 1.0
        elif mode == "lead" and target is not None:
            dx, dy, zoom = target.resolved_x - origin_x + 80, target.resolved_y - origin_y, 1.0
        elif mode == "rail":
            dx, dy, zoom = (phase - .5) * 360, np.sin(angle) * 24, 1.0
        elif mode == "shake":
            dx, dy, zoom = np.sin(angle * 17) * 8, np.cos(angle * 23) * 6, 1.0
        elif mode == "rotate":
            dx, dy, zoom = 0.0, 0.0, 1.0
        graph.camera.world_x = round(float(origin_x + dx), 6)
        graph.camera.world_y = round(float(origin_y + dy), 6)
        graph.camera.zoom = round(float(zoom), 6)
        if mode == "rotate":
            graph.camera.rotation = round(float(np.sin(angle) * 12), 6)
        frames.append(graph.render())
    return frames


def make_scene_camera_sample(idx: int, n_frames: int = 120, viewport=(512, 512), mode: str = "pan") -> dict:
    """Return camera trajectory frames plus deterministic per-frame metadata."""
    template = make_scene_level_graph(idx)
    frames = make_scene_camera_trajectory(idx, n_frames=n_frames, viewport=viewport, mode=mode)
    origin_x, origin_y = template.camera.world_x, template.camera.world_y
    states = []
    for frame_index in range(int(n_frames)):
        phase = frame_index / max(1, n_frames - 1)
        angle = phase * np.pi * 2
        states.append({"frame": frame_index, "phase": phase, "mode": mode,
                       "world_x": round(float(origin_x + np.sin(angle) * 128), 6),
                       "world_y": round(float(origin_y + np.cos(angle) * 64), 6),
                       "viewport": [int(viewport[0]), int(viewport[1])]})
    return {"frames": frames, "camera_states": states, "mode": mode, "idx": int(idx)}


def make_scene_level_batch(indices, *, raster=None) -> list[np.ndarray]:
    return [make_scene_level(int(index), raster=raster) for index in indices]


def scene_quality_report(idx: int) -> dict:
    """Return lightweight deterministic quality diagnostics for one level."""
    graph = make_scene_level_graph(idx)
    frame = graph.render()
    rgb = frame[..., :3]
    collisions = 0
    for left_index, left in enumerate(graph.objects):
        for right in graph.objects[left_index + 1:]:
            if check_collision(left, right):
                collisions += 1
    out_of_view = sum(1 for obj in graph.objects if not (0 <= obj.resolved_x <= graph.camera.viewport_width and 0 <= obj.resolved_y <= graph.camera.viewport_height))
    return {
        "idx": int(idx), "objects": len(graph.objects),
        "visible_pixels": int(np.count_nonzero(frame[..., 3])),
        "finite": bool(np.isfinite(frame).all()),
        "mean": float(rgb.mean()), "std": float(rgb.std()),
        "overlaps": collisions, "out_of_view_centers": out_of_view,
        "camera": {"width": graph.camera.viewport_width, "height": graph.camera.viewport_height},
    }


def validate_scene_quality(indices=None) -> list[dict]:
    selected = range(SCENE_N_LEVELS) if indices is None else indices
    reports = [scene_quality_report(scene_level_start(int(level))) for level in selected]
    for report in reports:
        if not report["finite"] or report["objects"] == 0 or report["visible_pixels"] == 0:
            raise AssertionError(f"invalid scene quality: {report}")
    return reports


def scene_performance_report(indices=None) -> dict:
    selected = list(range(SCENE_N_LEVELS)) if indices is None else [int(i) for i in indices]
    start = time.perf_counter()
    for level in selected:
        make_scene_level(scene_level_start(level))
    elapsed = time.perf_counter() - start
    return {"levels": len(selected), "seconds": elapsed, "seconds_per_level": elapsed / max(1, len(selected))}


def scene_diversity_report() -> dict:
    """Measure color spread, visible density, depth and object-count diversity."""
    means, stds, densities, depths, counts = [], [], [], [], []
    for level in range(SCENE_N_LEVELS):
        graph = make_scene_level_graph(scene_level_start(level))
        frame = graph.render()
        means.append(float(frame[..., :3].mean())); stds.append(float(frame[..., :3].std()))
        densities.append(float(np.mean(frame[..., 3] > 0))); counts.append(len(graph.objects))
        depths.append(float(np.std([obj.depth for obj in graph.objects])) if graph.objects else 0.0)
    return {"color_mean_range": (min(means), max(means)), "color_std_range": (min(stds), max(stds)),
            "density_range": (min(densities), max(densities)), "depth_std_range": (min(depths), max(depths)),
            "object_count_range": (min(counts), max(counts))}


def validate_scene_dynamics(indices=None) -> list[dict]:
    """Validate that declared dynamic levels evolve finite object state."""
    selected = range(SCENE_N_LEVELS) if indices is None else [int(i) for i in indices]
    reports = []
    for level_index in selected:
        spec = scene_level_spec(scene_level_start(level_index))
        graph = make_scene_level_graph(scene_level_start(level_index))
        before = [(float(obj.resolved_x), float(obj.resolved_y)) for obj in graph.objects]
        particles_before = [(int(pool.alive.sum()), float(pool.ages.sum())) for pool in graph.particles]
        graph.update(spec.fixed_dt, NullInputProvider().poll())
        after = [(float(obj.resolved_x), float(obj.resolved_y)) for obj in graph.objects]
        finite = all(np.isfinite(point).all() for point in after)
        changed = any(a != b for a, b in zip(before, after)) or any(
            (int(pool.alive.sum()), float(pool.ages.sum())) != previous for pool, previous in zip(graph.particles, particles_before)
        )
        dynamic = spec.simulation != "static"
        if dynamic and not finite:
            raise AssertionError(f"non-finite dynamic scene: {spec.name}")
        reports.append({"level": level_index, "name": spec.name, "simulation": spec.simulation,
                        "changed": changed, "finite": finite, "objects": len(graph.objects)})
    return reports


validate_scene_level_catalog()
