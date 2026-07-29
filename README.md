# Synthetic Image Generator

`synthetic-image-generator` is a deterministic procedural image library with
154 indexed content levels, native arbitrary-resolution rendering, composable
scenes with procedural materials and semantic layout, 16 camera post-effects,
transparency, resizing, quantization, dithering, packed pixel formats, and
ordered batch generation. An optional scene-graph layer adds camera/viewport
transforms with tracking and shake, physics, particles, tilemaps, HUD, tweens,
session replay, and single-file HTML export. A 154-level indexed scene catalog
applies the same `idx → deterministic sample` schedule to whole scenes and
their dynamics — for training data that needs more than one static composition.

The canonical indexed contract returns a NumPy `float32` RGB image with shape
`(32, 32, 3)` and values in `[0, 1]`. A non-negative index always maps to the
same level and image for a fixed contract version.

## Installation

```bash
python -m pip install synthetic-image-generator
```

From a repository checkout with dev dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Portable GPU acceleration is optional and does not require CUDA:

```bash
python -m pip install -e ".[gpu]"
```

Atlas rendering needs Pillow and Matplotlib:

```bash
python -m pip install -e ".[visualization]"
python scripts/render_generator_atlas.py
python scripts/render_raster_atlas.py
```

## Quick start

```python
import synthetic_image_generator as sig

# Indexed image — 32×32 float32 RGB (default, byte-exact contract)
image = sig.make_image(42)
assert image.shape == (32, 32, 3)

# Native resolution — renders at the requested size, not upscaled
native = sig.make_image(42, width=192, height=108)
assert native.shape == (108, 192, 3)

# Raster conversion with quantization and dithering
rgb565 = sig.make_image(42, raster=sig.RasterSpec(
    width=64, height=64, mode="rgb", bits_per_channel=(5, 6, 5), packed=True,
))
assert rgb565.shape == (64, 64)
```

Indices are unbounded and cycle through the 154-level catalog:

```python
level = sig.level_of(10_000_000)
start = sig.level_start(level, cycle=3)
```

The level schedule is `sig-154-cycle-v1`. Geometry changes between catalog
cycles because the absolute block remains part of the deterministic seed.

## Batch generation

`make_images()` preserves input order and produces exactly the same images as
repeated `make_image()` calls:

```python
batch = sig.make_images(range(10_000), backend="auto")
assert batch.shape == (10_000, 32, 32, 3)

# Native-resolution batch
batch = sig.make_images(range(100), width=96, height=64, backend="serial")
assert batch.shape == (100, 64, 96, 3)
```

Available backends:

- `serial`: lowest startup cost; preferred for small batches.
- `process`: parallel CPU workers with isolated legacy render state.
- `webgpu`: vendor-neutral compute through Vulkan, Metal, or DirectX 12.
- `auto`: serial below 64 images, otherwise process-based parallelism under
  legacy fidelity; portable fidelity may select a hardware WebGPU adapter.

The default `fidelity="legacy"` remains byte exact. GPU execution is explicit
because transcendental operations can differ across drivers:

```python
info = sig.accelerator_info()
if info["available"]:
    batch = sig.make_images(indices, backend="webgpu", fidelity="portable")
```

CPU autotuning (`workers=None`, `chunksize=None`, batches ≥256 images) respects
process affinity, cgroup quotas, physical/SMT cores, and NUMA topology. Inspect
the detected system with `sig.cpu_info()`.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SIG_WEBGPU_ALLOW_SOFTWARE` | unset (off) | Allow software adapters (e.g. llvmpipe) for validation. |
| `SIG_WEBGPU_ADAPTER` | unset | Vendor/device substring to pick a specific GPU. |
| `SIG_CPU_AUTOTUNE` | `1` (on) | Disable empirical CPU worker/chunksize tuning. |
| `SIG_CPU_WORKER_WARMUP` | `1` (on) | Skip warming a render inside each new worker process. |

## Structured scenes

```python
from synthetic_image_generator import (
    Background, LightSpec, MaterialSpec, ObjectSpec,
    LayoutRelation, LayoutSpec, SceneSpec, make_scene,
)

table = ObjectSpec(
    kind="box_3d", name="table", x=16, y=22, width=14, height=3,
    color=(0.45, 0.25, 0.1), material=MaterialSpec("wood", seed=3),
)
lamp = ObjectSpec(
    kind="sphere_3d", name="lamp", x=16, y=15, size=4, radius=2,
    color=(1.0, 0.75, 0.2), material=MaterialSpec("emissive"),
)

scene = SceneSpec(
    background=Background(kind="gradient_sky", horizon=0.5),
    objects=[table, lamp],
    light=LightSpec(direction=(0.3, -0.6, 0.7)),
    layout=LayoutSpec(relations=(
        LayoutRelation("above", "lamp", "table", gap=1),
    )),
)

image = make_scene(scene, seed=7)
assert image.shape == (32, 32, 3)

# High-resolution scene render
hd = make_scene(scene, width=512, height=512)
```

Objects are composed from far to near according to `depth`. Child reference
cycles, invalid seeds, non-finite depths, unsupported objects, and malformed
scene components are rejected before rendering.

### Procedural materials

Nine deterministic materials, evaluated from NumPy without image assets:

| Name | Look |
|---|---|
| `flat` | Uniform albedo (default) |
| `wood` | Ring grain |
| `brushed_metal` | Linear striations |
| `marble` | Swirling veins |
| `stone` | Irregular granular noise |
| `concrete` | Speckled grey |
| `fabric` | Fine weave |
| `water` | Rippling caustics |
| `emissive` | Glowing self-illumination |
| `glass`  | Subtle refraction noise + opacity |

```python
obj = ObjectSpec(
    kind="sphere_3d", x=16, y=16, radius=8,
    color=(0.2, 0.5, 0.3),
    material=MaterialSpec(name="marble", scale=1.5, rotation=30, seed=7),
)
```

### Semantic layout

`LayoutSpec` applies deterministic constraints between named objects without
mutating the caller's spec:

```python
scene = SceneSpec(
    objects=[left_box, right_box],
    layout=LayoutSpec(relations=(
        LayoutRelation("left_of", "left", "right", gap=2),
    )),
)
```

Supported relations: `left_of`, `right_of`, `above`, `below`, `behind`,
`in_front_of`.

### Serialization

`scene_to_dict()` and `scene_from_dict()` provide a JSON-compatible round trip
for the complete declarative scene, including nested objects, materials, layers,
and layout:

```python
data = sig.scene_to_dict(scene)
restored = sig.scene_from_dict(data)
assert np.array_equal(sig.make_scene(scene), sig.make_scene(restored))
```

## Camera system

The camera module provides tracking, cinematic transitions, screen shake, and
16 post-effects — all deterministic and reusable outside the scene graph:

```python
from synthetic_image_generator import (
    CameraSpec, update_camera_target, trigger_camera_shake,
    transition_camera, apply_camera_effects,
)

cam = CameraSpec(
    viewport_width=512, viewport_height=512,
    world_x=300, world_y=220, zoom=1.0,
    target="player", dead_zone=(8, 8), smoothing=0.8,
    effects=("vignette", "film_grain"),
)

# Track a target with dead-zone and smoothing
cam = update_camera_target(cam, objects, dt=1/60)

# Trigger impact shake
cam = trigger_camera_shake(cam, intensity=4.0, duration=0.25)

# Cinematic transition
cam = transition_camera(start_cam, end_cam, t=0.5, easing="smoothstep")
```

Post-effects (all 16): `vignette`, `grayscale`, `scanlines`, `flash`,
`motion_blur`, `chromatic_aberration`, `pixelate`, `film_grain`,
`depth_of_field`, `fog`, `underwater`, `night_vision`, `thermal`, `damage`,
`rain`, `dust`.

```bash
python scripts/render_camera_effect_demos.py  # HTML gallery of all 16 effects
```

## Scene graph / mini game engine

An additive layer on top of `make_scene` adds camera/viewport transforms with
tracking/shake/transitions, named layers with parallax, sprite atlases, a
particle system, lightweight 2D physics, an input abstraction (with
deterministic replay), a fixed-timestep game loop, a HUD/widget system, tweens,
tilemaps, session record/replay, and a single-file HTML export. Every piece
calls only the public `make_scene`/`make_scene_raster` functions; the indexed
contract is unaffected.

```python
from synthetic_image_generator import (
    CameraSpec, CollisionShape, InputState, ObjectSpec, PhysicsWorld,
    RigidBodySpec, SceneGraph, SceneSpec,
)

ground = ObjectSpec(kind="box_3d", x=16, y=28, size=4, collision_shape=CollisionShape.AABB)
ball = ObjectSpec(kind="sphere_3d", name="ball", x=16, y=4, radius=3,
                  collision_shape=CollisionShape.CIRCLE)

physics = PhysicsWorld(gravity=(0, 50))
physics.add_body(ground, RigidBodySpec(is_static=True))
physics.add_body(ball, RigidBodySpec(mass=1.0, restitution=0.4))

graph = SceneGraph(
    scene_spec=SceneSpec(objects=[ground, ball]),
    camera=CameraSpec(viewport_width=512, viewport_height=512,
                      world_x=16, world_y=16, effects=("vignette",)),
    physics=physics, seed=7,
)

frame = graph.tick(1/60, InputState())
assert frame.shape == (512, 512, 4)
```

`SceneGraph.update()` drives physics, animations, particles, flipbook playback,
camera tracking, shake decay, and cinematic transitions.
`SceneGraph.render()` composites: tilemaps → scene → sprites → particles → HUD.

### Snapshot and restore

Capture and restore mutable simulation state:

```python
checkpoint = graph.snapshot()
# ... simulate many steps ...
graph.restore_snapshot(checkpoint)
```

Covers: object positions, physics velocities, tween progress, particle state,
camera pose, and the fixed-step clock.

### HTML export

```python
sig.export_live_html(graph, "demo.html", width=512, height=512)
```

Draws flat-shaded 2D proxy shapes with an interactive JS loop — not a port of
the Phong-shaded 3D renderer, but sufficient for prototyping and demos.

```bash
python scripts/render_scene_graph_demos.py  # → media/scene_graph_demos/
```

## Scene level catalog

A 154-entry deterministic catalog built from 48 reusable archetypes × 7
variants × 7 visual styles × 9 simulation modes:

| Category | Archetypes | Count |
|---|---|---|
| Visual | abstract_shapes, geometric_stack, symmetry_scene, pattern_grid, empty_minimal, gradient_environment | 6 |
| Objects | single_sphere/box/tube/blob/emissive/transparent, object_row/grid/stack/cluster/scatter/pile | 12 |
| Environment | grass_field, forest, tree_cluster, flower_field, mountain_landscape, underwater_scene | 6 |
| Built | room, building_block, city_block, bridge_scene, warehouse, street_scene | 6 |
| Character | single_character, character_pair, character_group, animal_group, predator_prey, crowd_scene | 6 |
| Physics | physics_playground, stacked_objects, falling_objects, bouncing_objects, orbiting_bodies, pendulum_or_chain | 6 |
| System | particle_emitter, particle_burst, fluid_like_field, reaction_diffusion, tilemap_maze, platformer_level | 6 |

Simulation modes: `static`, `tweened`, `physics`, `particles`, `orbit`,
`patrol`, `bounce`, `fall`, `chase`.

```python
# Single frame
frame = sig.make_scene_level(sig.scene_level_start(42))
assert frame.shape[2] == 4  # RGBA

# Full trajectory
sample = sig.make_scene_trajectory_level(
    sig.scene_level_start(42), n_frames=60,
)

# Training trajectory (policy: 1–240 frames depending on simulation)
sample = sig.make_scene_training_trajectory(
    sig.scene_level_start(42), quality="training",
)

# Camera trajectory (14 modes: pan, zoom, orbit, follow, dolly, rail, …)
frames = sig.make_scene_camera_trajectory(
    sig.scene_level_start(18), n_frames=120, mode="pan",
)
```

Inspect the catalog:

```python
spec = sig.scene_level_spec(idx)             # SceneLevelSpec
name = sig.scene_level_by_name("forest_small")
category = sig.scene_archetype_category("forest")  # "environment"
```

Quality and diversity diagnostics:

```python
report = sig.scene_quality_report(sig.scene_level_start(0))
diversity = sig.scene_diversity_report()
reports = sig.validate_scene_quality()
```

Demos and datasets:

```bash
python scripts/render_scene_archetype_demos.py    # Interactive HTML (48 archetypes)
python scripts/render_scene_level_gallery.py      # Grid gallery (154 levels)
python scripts/export_scene_training_dataset.py   # .npz trajectories + manifest
```

The scene level catalog has its own contract version
(`SCENE_LEVEL_CONTRACT_VERSION`) and regression hashes
(`SCENE_LEVEL_SAMPLE_SHA256`, `SCENE_LEVEL_TRAJECTORY_SAMPLE_SHA256`)
independent from `make_image`'s indexed contract.

### Legacy scene catalog

The older `make_scene_catalog(idx)` and `make_scene_trajectory(idx)` (from
`scene_catalog.py`) remain available. The scene level catalog above is the
recommended replacement.

## Raster output

Native canvases support 32–4096 pixels per axis and at most 8,388,608 total
pixels. A `RasterSpec` with both axes ≥32 px drives the native render canvas
before color conversion and quantization. Outputs smaller than 32 px on either
axis are safely reduced from 32×32.

Supported modes: `rgb`, `rgba`, `rgba2222` (packed), `grayscale`, `binary`.

Resize methods: `nearest`, `bilinear`, `bicubic`. Dithering: `none`, `ordered`,
`floyd_steinberg`. RGB/RGBA channel tuples can be packed, e.g. `(5, 6, 5)` for
RGB565.

Transparent scenes use premultiplied-alpha resizing to avoid dark sprite edges:

```python
scene = SceneSpec(
    background=Background(kind="transparent"),
    objects=[ObjectSpec(kind="disc_with_rim", x=16, y=16, radius=7,
                        color=(1.0, 0.2, 0.1), opacity=0.75)],
)
sprite = make_scene(scene, raster=RasterSpec(width=64, height=64, mode="rgba"))
assert sprite.shape == (64, 64, 4)
```

## Determinism and concurrency

The legacy renderer contains recursive supersampling and post-processing
levels. Public single-image and scene calls serialize that legacy mutable
render state, making concurrent thread calls correct. High-throughput
generation should use `make_images(..., backend="process")`, where every worker
owns independent state.

The regression suite verifies one byte-exact sample from every level against
`LEGACY_LEVEL_SAMPLE_SHA256`, alongside thread, process, raster, scene, native
resolution, and scene-level-catalog tests (~100 tests).

## Source layout

`pyproject.toml` maps the physical `generators/` directory directly to the
installed `synthetic_image_generator` namespace. There is no duplicated package
source. The distribution includes a `py.typed` marker.

| Module | Purpose |
|---|---|
| `synthetic_image_generator.py` | Indexed image renderer (154 levels) + `make_image`/`make_scene` entry points |
| `levels.py` | Canonical level catalog and index schedule |
| `scene_spec.py` | Declarative scene types (`SceneSpec`, `ObjectSpec`, `RasterSpec`, `MaterialSpec`, `LayoutSpec`, …) |
| `materials.py` | Nine procedural materials (wood, marble, metal, …) |
| `layout.py` | Semantic layout constraint resolver |
| `camera.py` | Camera tracking, shake, transitions, 16 post-effects |
| `scene_graph.py` | Spatial queries and bounding-box helpers |
| `game_loop.py` | `SceneGraph` — fixed-step game loop, snapshot/restore |
| `scene_level_catalog.py` | 154-entry scene level catalog with simulations and camera trajectories |
| `scene_serialization.py` | JSON round-trip for `SceneSpec` |
| `scene_catalog.py` | Legacy scene catalog (pre-0.26.0) |
| `_primitive_ir.py` / `_rendergraph.py` | WebGPU RenderGraph IR and compiler |
| `wave.py` | Plane-wave cross-modal levels (148–153) |
| `physics.py` / `tween.py` / `particles.py` / `tilemap.py` / `ui.py` / `layers.py` | Satellite game-loop modules |
| `_webgpu.py` | WebGPU acceleration backend |
| `web_export.py` | Single-file HTML export |
| `input_system.py` / `replay.py` | Input abstraction and deterministic replay |

Generated atlases:

- [`media/synthetic_generator_universal_atlas.png`](media/synthetic_generator_universal_atlas.png)
- [`media/synthetic_generator_raster_atlas.png`](media/synthetic_generator_raster_atlas.png)

See [`SOURCE_OF_TRUTH.md`](SOURCE_OF_TRUTH.md) for the canonical-source policy,
[`CHANGELOG.md`](CHANGELOG.md) for release notes, and
[`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md) for the WebGPU roadmap.
