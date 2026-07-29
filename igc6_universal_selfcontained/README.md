# Synthetic Image Generator

`synthetic-image-generator` is a deterministic procedural image library with
154 indexed content levels, composable scenes, transparency, resizing,
quantization, dithering, packed pixel formats, and ordered batch generation.
An optional scene-graph layer adds camera/physics/particles/tilemaps/HUD/
tweens on top, for simple 2D games and interactive demos.

The canonical indexed contract returns a NumPy `float32` RGB image with shape
`(32, 32, 3)` and values in `[0, 1]`. A non-negative index always maps to the
same level and image for a fixed contract version.

## Installation

```bash
python -m pip install -e .
```

Development dependencies and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Portable GPU acceleration is optional and does not require CUDA:

```bash
python -m pip install -e ".[gpu]"
```

From a repository checkout, atlas rendering additionally needs Pillow and
Matplotlib:

```bash
python -m pip install -e ".[visualization]"
python scripts/render_generator_atlas.py
python scripts/render_raster_atlas.py
```

## Indexed images

```python
import synthetic_image_generator as sig

image = sig.make_image(42)
assert image.shape == (32, 32, 3)
assert image.dtype.name == "float32"
```

Indices are unbounded and cycle through the 154-level catalog:

```python
level = sig.level_of(10_000_000)
start = sig.level_start(level, cycle=3)
```

The level schedule is `sig-154-cycle-v1`. The geometry changes between catalog
cycles because the absolute block remains part of the deterministic seed.

## Batch generation

`make_images()` preserves input order and produces exactly the same images as
repeated `make_image()` calls:

```python
indices = range(10_000)
batch = sig.make_images(indices, backend="auto")
assert batch.shape == (10_000, 32, 32, 3)
```

Available backends:

- `serial`: lowest startup cost; preferred for small batches.
- `process`: parallel CPU workers with isolated legacy render state.
- `webgpu`: vendor-neutral compute through Vulkan, Metal, or DirectX 12.
- `auto`: serial below 64 images, otherwise process-based parallelism under
  legacy fidelity; portable fidelity may select a hardware WebGPU adapter.

The default `fidelity="legacy"` remains byte exact and does not silently switch
to GPU arithmetic. GPU execution is explicit because transcendental operations
can differ slightly across drivers:

```python
info = sig.accelerator_info()
if info["available"]:
    batch = sig.make_images(
        indices,
        backend="webgpu",
        fidelity="portable",
    )
```

The portable output is deterministic for a fixed adapter/driver. Portable
WebGPU currently covers 25 levels in four reusable families:

- Gray-Scott reaction-diffusion at level 113.
- Primitive Render IR at levels 0–4, 6, 15–18, 128, 142, and 144. CPU planners
  emit ordered rectangles, Bresenham lines, rotated ellipses, soft discs,
  rings, clipping, additive/affine composition, and analytic Phong surfaces
  for one generic WGSL interpreter.
- Mirror, kaleidoscope, diffraction, displacement, and lens transforms at
  levels 129, 130, 133, 135, and 136.
- Plane waves at levels 148–153.

Reaction-diffusion preserves the legacy RNG stream and offloads all 200–340
simulation iterations. Primitive IR keeps random control flow on CPU and sends
ordered pixel work to one generic WGSL interpreter—the same separation of
control and parallel data work can be extended to more primitive-heavy levels.
Mixed batches are grouped by family and restored to their original input order.
Numerical validation uses `1e-4` absolute/relative tolerance on continuous
paths and mean/percentile bounds where a tiny upstream difference can cross a
later stochastic or hard threshold.

Unsupported groups retain the CPU renderer. Heterogeneous batches select GPU
families independently, submit remaining work to a persistent process pool
created before driver initialization, and overlap both devices. Internally,
RenderGraph v1 compiles each family into explicit `prepare → GPU → finish`
dependencies. CPU preparation needed to feed the GPU is queued before unrelated
CPU rendering, and independent branches are restored through one ordered
assembly node. A second whole-batch performance gate prevents individually
fast families from forming a slower combined schedule. `GPU_LEVEL_FAMILIES`
exposes the current level-to-kernel registry.

`accelerator_info()` never requires CUDA and reports the selected adapter,
backend, and whether it is real hardware. Software adapters such as llvmpipe are
rejected for normal acceleration. They can be enabled only for validation with
`SIG_WEBGPU_ALLOW_SOFTWARE=1`. On a multi-GPU system,
`SIG_WEBGPU_ADAPTER=amd` (or another case-insensitive vendor/device substring)
selects a particular adapter; otherwise discrete and then integrated hardware
are tried in priority order, with device-creation fallback between them.

The process backend accepts `workers`, `chunksize`, and `start_method`.
Leaving both `workers` and `chunksize` as `None` enables empirical CPU tuning
for batches of at least 256 images. The tuner:

- respects process affinity and cgroup CPU quotas;
- detects physical cores, SMT capacity, and NUMA topology;
- measures physical-core, partial-SMT, and full-SMT process counts;
- measures chunk sizes and both one-thread and runtime-default BLAS policies;
- samples the actual level/raster distribution rather than a synthetic loop;
- caches the result for 15 minutes and reuses a warmed persistent process pool.

Inspect the detected system and previous decisions with `sig.cpu_info()`.
`SIG_CPU_AUTOTUNE=0` disables measurement, while explicit `workers=` or
`chunksize=` values override that dimension. `SIG_CPU_WORKER_WARMUP=0` disables
worker warmup for applications that prefer minimum startup work.

On a single-threaded Linux parent, `start_method="auto"` uses `fork` to avoid
reimporting the SciPy-heavy package; other environments use `spawn`. When
called from a program that uses the multiprocessing `spawn` method, invoke it
below the usual `if __name__ == "__main__":` guard. Nested daemon workers
should use the serial backend and let their parent own the process pool.

```python
if __name__ == "__main__":
    batch = sig.make_images(
        range(50_000),
        backend="process",
        workers=None,
        chunksize=None,
    )
    print(sig.cpu_info())
```

The GPU dependency remains optional; importing the package never initializes a
graphics driver. If `backend="auto", fidelity="portable"` cannot see a hardware
adapter, it safely retains the CPU path. For bulk batches, `auto` groups the
real family distribution and performs adapter- and batch-scale-specific
calibration against the configured CPU worker backend. It samples across the
full input (up to 4,096 images), enables only families that pass numerical
checks and a 1.10× speed gate, then applies the same gate to the complete
hybrid schedule. The full 154-level catalog can therefore use GPU and CPU
together when that exact distribution wins, while a different machine or
batch safely remains process-only.

Benchmark serial and process output, including a byte-exact equality check:

```bash
python scripts/benchmark_batch.py --count 3080
```

Benchmark the portable kernel separately:

```bash
python scripts/benchmark_batch.py \
    --dataset accelerated --count 4096 --webgpu
```

Measured on this workspace with warmed pipelines:

| Workload / adapter | WebGPU | CPU reference | Speedup |
|---|---:|---:|---:|
| 4,096 wave, NVIDIA Vulkan | 59,876 img/s | 17,938 img/s | 3.34× |
| 4,096 wave, AMD Vulkan | 34,069 img/s | 17,980 img/s | 1.89× |
| 4,096 wave, llvmpipe CPU | 10,034 img/s | 18,231 img/s | 0.55× |
| 1,024 discrete IR, llvmpipe CPU | 1,317 img/s | 3,947 img/s | 0.33× |
| 600 dense IR, llvmpipe CPU | 889 img/s | 1,298 img/s | 0.68× |
| 1,950 analytic IR, llvmpipe CPU | 1,963 img/s | 5,686 img/s | 0.35× |
| 1,024 reaction, NVIDIA Vulkan | 2,103 img/s | 1,060 img/s | 1.98× |
| 4,096 wave+reaction mix, NVIDIA Vulkan | 12,380 img/s | 5,510 img/s | 2.25× |
| 4,096 wave+reaction mix, AMD Vulkan | 10,141 img/s | 5,545 img/s | 1.83× |
| 1,024 level-128 plans, NVIDIA Vulkan | 991 img/s | 825 img/s | 1.20× |
| 1,024 level-128 plans, AMD Vulkan | 987 img/s | 855 img/s | 1.15× |
| 4,096 previous 13-level mix, NVIDIA Vulkan | 10,738 img/s | 7,041 img/s | 1.53× |

CPU autotuning on the workspace's Ryzen 9 7950X3D selected 16–24 processes
and chunks of 4–8 depending on the measured run. Sustained full-catalog
throughput reached 8,800–9,522 img/s, versus about 7,549 img/s for the previous
fixed 16-process/chunk-32 policy. After that improvement, the full-catalog
hybrid GPU schedule measured only 1.01× and was correctly rejected by the
global 1.10× gate; `auto` retained the faster tuned CPU route.

The software result is why normal discovery rejects CPU adapters and why
portable `auto` uses a measured performance gate. Initial adapter creation,
shader compilation, and automatic calibration are one-time costs; the table
reports subsequent batch throughput.

### Environment variables

Quick reference for the tuning knobs described above:

| Variable | Default | Effect |
|---|---|---|
| `SIG_WEBGPU_ALLOW_SOFTWARE` | unset (off) | `1`/`true`/`yes`/`on` allows software adapters (e.g. llvmpipe) for validation; normal acceleration still rejects them. |
| `SIG_WEBGPU_ADAPTER` | unset | Case-insensitive vendor/device substring (e.g. `amd`) to pick a specific adapter on multi-GPU systems. |
| `SIG_CPU_AUTOTUNE` | `1` (on) | `0`/`false`/`no`/`off` disables empirical CPU worker/chunksize tuning; explicit `workers=`/`chunksize=` still override. |
| `SIG_CPU_WORKER_WARMUP` | `1` (on) | `0`/`false`/`no`/`off` skips warming a render inside each new persistent worker process, for faster startup. |

## Structured scenes

```python
from synthetic_image_generator import (
    Background,
    LightSpec,
    ObjectSpec,
    PostSpec,
    SceneSpec,
    make_scene,
)

scene = SceneSpec(
    background=Background(kind="gradient_sky", horizon=0.45),
    objects=[
        ObjectSpec(
            kind="sphere_3d",
            cx=16,
            cy=18,
            radius=5,
            color=(0.2, 0.6, 1.0),
        ),
        ObjectSpec(
            kind="building_3d",
            x=8,
            ground_y=24,
            width=10,
            height=15,
            color=(0.8, 0.7, 0.5),
        ),
    ],
    light=LightSpec(direction=(0.3, -0.6, 0.7)),
    post=PostSpec(style="realistic", grain=0.1),
)

image = make_scene(scene, seed=7)
assert image.shape == (32, 32, 3)
```

Objects are composed from far to near according to `depth`. Child reference
cycles, invalid seeds, non-finite depths, unsupported objects, and malformed
scene components are rejected before rendering.

## Scene graph / mini game engine

An additive layer on top of `make_scene` adds camera/viewport transforms,
named layers with parallax, sprite atlases, a particle system, lightweight
2D physics, an input abstraction (with deterministic replay), a
fixed-timestep game loop, a HUD/widget system, tweens, tilemaps, session
record/replay, and a single-file HTML export. Every piece is a satellite
module that either never imports the renderer or only calls the public
`make_scene`/`make_scene_raster` functions like any other caller, so the
indexed contract and existing `make_scene` calls are unaffected.

```python
from synthetic_image_generator import (
    CameraSpec, CollisionShape, InputState, ObjectSpec, PhysicsWorld,
    RigidBodySpec, SceneGraph, SceneSpec,
)

ground = ObjectSpec(kind="box_3d", x=16, y=28, size=32, collision_shape=CollisionShape.AABB)
ball = ObjectSpec(kind="sphere_3d", x=16, y=4, radius=3, collision_shape=CollisionShape.CIRCLE, name="ball")

physics = PhysicsWorld(gravity=(0, 50))  # positive y falls down the image
physics.add_body(ground, RigidBodySpec(is_static=True))
physics.add_body(ball, RigidBodySpec(mass=1.0, restitution=0.4))

graph = SceneGraph(
    scene_spec=SceneSpec(objects=[ground, ball]),
    camera=CameraSpec(viewport_width=64, viewport_height=64),
    physics=physics,
    seed=7,
)

frame = graph.tick(1 / 60, InputState())
assert frame.shape == (64, 64, 4)
```

`SceneGraph.update()` (called once per fixed step inside `tick()`) drives
physics, `animations` (`Tween`/`AnimationTrack`), particle auto-emission for
any `ParticlePool` with an `origin` set, flipbook playback, and HUD click
dispatch. `SceneGraph.render()` composites, bottom to top: `tilemaps` ->
the scene (per-`layer` via `LayerManager` if any are registered, otherwise
one `make_scene` call) -> sprite-tagged objects (`ObjectSpec.sprite`/
`flipbook`) -> particles -> `hud`, into one RGBA frame.

This layer targets game-loop convenience, not conservation-grade physics or
pixel-identical browser rendering — `export_html` draws flat-shaded 2D proxy
shapes with a small interactive JS loop, not a port of the Phong-shaded 3D
renderer. See [`SCENE_GRAPH_PLAN.md`](SCENE_GRAPH_PLAN.md) for the design
rationale and [`CHANGELOG.md`](CHANGELOG.md) (0.10.0 onward) for what each
phase added. It sits outside the byte-exact policy in
[`SOURCE_OF_TRUTH.md`](SOURCE_OF_TRUTH.md), which covers `make_image`'s
indexed contract only.

## Scene catalog

The same `idx -> deterministic sample` schedule `make_image`/`LEVEL_TABLE`
use for indexed images, applied to whole scenes and their dynamics:

```python
from synthetic_image_generator import make_scene_catalog, make_scene_trajectory

sample = make_scene_catalog(42)  # one composed scene, like make_image(idx)
assert sample.frames[0].shape == (32, 32, 4)
assert sample.archetype and sample.seed == 42

trajectory = make_scene_trajectory(42, n_frames=60)  # 60 deterministic frames
assert len(trajectory.frames) == 60
assert trajectory.behaviors  # e.g. ["patrol", "gravity_drop"]
```

`make_scene_catalog(idx)` resolves to one of `SCENE_ARCHETYPES` (composition
only: object kinds/positions/tags, no physics/behaviors attached) and
renders one settled frame. `make_scene_trajectory(idx, n_frames=...)` also
attaches 1-3 behaviors from `BEHAVIORS` compatible with that archetype
(`patrol`, `chase`, `flee`, `orbit`, `gravity_drop`, `bounce_particles`) and
ticks the scene forward, recording every frame plus per-object labels
(`SceneSample.objects_state`, tags + bounding box) and events
(`SceneSample.events`, physics contacts + HUD clicks) — the same
`SceneSample` shape works as a detection-label source, an RL trajectory, or
a world-model replay target, so no consumer has to be picked in advance.
Both functions are regression-gated (`SCENE_CATALOG_SAMPLE_SHA256`/
`SCENE_TRAJECTORY_SAMPLE_SHA256`) the same way the indexed image catalog is,
under an independent contract (`SCENE_CATALOG_CONTRACT_VERSION`) that never
touches `make_image`'s. See [`SCENE_CATALOG_PLAN.md`](SCENE_CATALOG_PLAN.md)
for the full design and `scripts/export_scene_dataset.py` for writing a
batch of samples to disk.

## Raster output

`RasterSpec.width` is the X resolution and `height` is the Y resolution.

```python
from synthetic_image_generator import RasterSpec, make_image

rgb8 = make_image(
    42,
    raster=RasterSpec(
        width=128,
        height=64,
        mode="rgb",
        bits_per_channel=8,
    ),
)
assert rgb8.shape == (64, 128, 3)
assert rgb8.dtype.name == "uint8"
```

Supported modes:

- `rgb` and `rgba`, with 1–16 bits per channel.
- `rgba2222`, packed as `RR GG BB AA`.
- `grayscale`.
- `binary`, optionally packed eight horizontal pixels per byte.

Resize methods are `nearest`, `bilinear`, and `bicubic`. Dithering methods are
`none`, `ordered`, and `floyd_steinberg`. RGB/RGBA channel tuples can be packed,
for example `(5, 6, 5)` for RGB565.

```python
rgb565 = make_image(
    42,
    raster=RasterSpec(
        width=64,
        height=64,
        mode="rgb",
        bits_per_channel=(5, 6, 5),
        packed=True,
    ),
)
assert rgb565.shape == (64, 64)
assert rgb565.dtype.name == "uint16"
```

Transparent scenes use premultiplied-alpha resizing to avoid dark sprite edges:

```python
sprite_scene = SceneSpec(
    background=Background(kind="transparent"),
    objects=[
        ObjectSpec(
            kind="disc_with_rim",
            x=16,
            y=16,
            radius=7,
            color=(1.0, 0.2, 0.1),
            opacity=0.75,
        )
    ],
)
sprite = make_scene(
    sprite_scene,
    raster=RasterSpec(width=64, height=64, mode="rgba"),
)
assert sprite.shape == (64, 64, 4)
```

## Determinism and concurrency

The legacy renderer contains recursive supersampling and post-processing
levels. Public single-image and scene calls serialize that legacy mutable
render state, making concurrent thread calls correct. High-throughput
generation should use `make_images(..., backend="process")`, where every worker
owns independent state.

The regression suite verifies one byte-exact sample from every level against
`LEGACY_LEVEL_SAMPLE_SHA256`, alongside thread, process, raster, scene, and API
tests.

## Source layout

`pyproject.toml` maps the physical `generators/` directory directly to the
installed `synthetic_image_generator` namespace. There is no duplicated package
source. The distribution includes a `py.typed` marker for type-aware tooling.

The generated atlases are available in:

- [`media/synthetic_generator_universal_atlas.png`](media/synthetic_generator_universal_atlas.png)
- [`media/synthetic_generator_raster_atlas.png`](media/synthetic_generator_raster_atlas.png)

See [`SOURCE_OF_TRUTH.md`](SOURCE_OF_TRUTH.md) for the canonical-source and
contract policy, [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md) for the
CPU/WebGPU RenderGraph roadmap, and [`CHANGELOG.md`](CHANGELOG.md) for release
changes.
