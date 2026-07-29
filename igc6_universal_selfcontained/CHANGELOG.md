# Changelog

## 0.25.4

**Wall/platform collision for chase & flee (`platformer`/`top_down_arena`/
`crowd`):** these archetypes tag their walls/ground/platforms
`collision_shape=AABB`, but nothing ever attached a `PhysicsWorld` for
them — only `gravity_drop` did, and it's only compatible with
`physics_playground`/`stacked_boxes`. So `chase`/`flee` steered straight
through walls with zero regard for them (confirmed in the real engine, not
just the HTML export). Fixed:
- `PhysicsWorld.set_velocity()` (new): drive a body by intent (a desired
  velocity) instead of writing its position directly, so `step()`'s
  collision resolution still gets the final say.
- `generators/behaviors.py`: new `_attach_solid_walls()` gives
  `wall`/`ground`/`platform`-tagged objects a static body (creating a
  zero-gravity `PhysicsWorld` if needed; a no-op for archetypes with none,
  e.g. `crowd`). `_ChaseController` now sets velocity via physics when the
  chaser has a body — walls actually block the chase/flee instead of being
  walked through. Falls back to the old direct-position path otherwise
  (unchanged behavior for `crowd`, which has no walls).
- `export_live_html`: the keyboard-driven player (client-side only —
  Python has no input system driving it) gets a synthetic kinematic body
  when the archetype has walls, purely for the export, so it collides with
  them too instead of walking straight through.
- JS engine synced to match (`contactNormalAndPenetration`/
  `exactHalfExtents` mirror `physics.py`'s `_contact_normal_and_penetration`/
  `_exact_half_extents`; chase sets a physics body's velocity when one
  exists). Verified against Python frame-by-frame via the Node harness.
- `SCENE_TRAJECTORY_SAMPLE_SHA256` updated (chase geometry changed for any
  sample that draws `chase`/`flee`); `SCENE_CATALOG_SAMPLE_SHA256`
  unchanged (`make_scene_catalog` never attaches behaviors).

## 0.25.3

**Collision penetration/normal fix (affects v0.10–v0.25.2 `PhysicsWorld`):**
`_resolve_collisions` computed penetration depth via the generic,
deliberately-padded `object_aabb()` (floors half-extent at `ObjectSpec`'s
`radius=5.0`/`size=8.0` defaults even for shapes that never set them),
and always took the contact normal as the center-to-center direction —
only valid for circle-circle pairs. Together these made any physics body
without an explicit `width`/`height` matching its true size (e.g. a ball
with only `radius` set) get resolved against a phantom, oversized box, and
any AABB-involving contact (ground, walls, stacked boxes) could get a
diagonal normal instead of axis-aligned — compounding into runaway
velocity within a few dozen frames (`gravity_drop`-driven scenes exploding
out of world bounds instead of settling). Fixed by computing penetration
and normal from each object's own exact geometry (radius for circles,
width/height for boxes) via the new `_exact_half_extents`/
`_contact_normal_and_penetration` helpers, with box-involving pairs using
the axis of minimum penetration (standard SAT-style separation) for the
normal instead of center-to-center. `stacked_boxes`/`physics_playground`
now settle into resting stacks/bounces instead of diverging.
`SCENE_TRAJECTORY_SAMPLE_SHA256` updated (`make_scene_trajectory` output
changes for any sample that ends up running `gravity_drop`);
`SCENE_CATALOG_SAMPLE_SHA256` is unchanged (`make_scene_catalog` never
attaches behaviors, so no physics ever ran there).

**Live HTML export (`export_live_html`, `generators/web_export.py`):**
new exporter alongside `export_html` — ports `PhysicsWorld` and the
`generators.behaviors` controllers (patrol/chase/flee/orbit) to a JS
engine driven by the exact deterministic parameters the Python side
computed (positions, tween endpoints, chase speeds, gravity, restitution,
...), so the exported page runs a real simulation instead of a static
snapshot. No client-side RNG, no need to port numpy's RNG to match a
Python `idx` bit-for-bit. Verified against the Python engine by running
the emitted `<script>` under Node with DOM calls stubbed (gravity,
collision, chase, and tween all reproduce the same outcome as the
Python side). `bounce_particles` isn't ported (no particle rendering
client-side yet — physics/collisions still run correctly).
`scripts/render_scene_graph_demos.py` now builds all 8 scene-catalog
archetypes through `export_live_html` (previously used hand-written
pixel-space demos disconnected from the real archetype/behavior registry,
and before that, a broken variant that plugged 32-unit world coordinates
directly into a much larger canvas with no scaling).

## 0.25.0

Implements `SCENE_CATALOG_PLAN.md` in full: an indexed, deterministic
`idx -> SceneSample` catalog on top of the v0.10–v0.24 engine — the scene
equivalent of `LEVEL_TABLE`/`make_image(idx)` — plus a correctness fix to
gravity direction that predates this release.

**Gravity sign fix (affects v0.15–v0.24 physics/particle gravity):**
`_render_object`'s `box_3d` handling (`top = bottom - height`) confirms the
renderer's `y` grows downward, but `PhysicsWorld`'s and
`ParticleEmitterSpec`'s default `gravity` was `(0, -98)` — pulling bodies
*up* the image, not down. Defaults are now `(0, 98)`; every constructed
example (this file's own text doesn't encode direction, but `README.md`,
`SCENE_GRAPH_PLAN.md`, and the test suite did and are corrected). Any code
that explicitly passed a negative `gravity[1]` expecting "down" needs its
sign flipped.

**Scene catalog (Phase A — composition):**
- `generators/scene_levels.py`: `SceneArchetype`, an 8-archetype registry
  (`platformer`, `top_down_arena`, `physics_playground`, `tilemap_maze`,
  `particle_burst`, `crowd`, `orbiting_bodies`, `stacked_boxes`) — each a
  `RandomState -> SceneGraph` builder with zero physics/particles-as-dynamics
  attached (composition only). `scene_archetype_of`/`scene_seed_of` mirror
  `level_of`/`level_start`'s indexed schedule (`SAMPLES_PER_ARCHETYPE=325`,
  contract `sgc-8-cycle-v1`), independent of the image catalog's index space.
- `make_scene_catalog(idx)`: builds the archetype, steps physics a fixed 30
  settle steps, renders once. Regression-gated by
  `SCENE_CATALOG_SAMPLE_SHA256` (one sample per archetype), same pattern as
  `LEGACY_LEVEL_SAMPLE_SHA256`.

**Scene catalog (Phase B — behavior/dynamics):**
- `generators/behaviors.py`: a 6-behavior registry (`patrol`, `chase`,
  `flee`, `orbit`, `gravity_drop`, `bounce_particles`), each an
  `attach(SceneGraph, RandomState)` composing existing `Tween`/
  `PhysicsWorld`/`ParticlePool` primitives onto an already-built archetype.
  `chase`/`flee`/`orbit` needed a duck-typed `Updatable` (added to
  `tween.py`) since a fixed-endpoint `Tween` can't steer toward a moving
  target — `SceneGraph.animations` now accepts either.
- `make_scene_trajectory(idx, n_frames=60)`: draws 1-3 compatible behaviors
  deterministically from the same seed, ticks forward, records every frame
  plus `objects_state` (now includes `tags`/`bbox`, extended in
  `replay.serialize_objects`) and `events` (physics contacts + HUD clicks,
  exposed via new `SceneGraph.last_contacts`). Regression-gated by
  `SCENE_TRAJECTORY_SAMPLE_SHA256`.
- `SceneSample` is the shared, consumer-agnostic result type for both
  functions: frames + labels (vision), + events/`input_log` slot (RL), +
  `seed` (world-model replay) in one structure.

**Scene catalog (Phase C — batch/export):**
- `make_scene_catalog_batch`/`make_scene_trajectory_batch`: ordered,
  sequential batch helpers (no process-pool autotuning like `make_images()`
  — deferred until a real export is measured slow enough to need it).
- `scripts/export_scene_dataset.py`: writes samples to disk as `.npy`
  frames + a JSON metadata sidecar per index.

Kept deliberately at 8 archetypes × 6 behaviors for this first pass (not
154-scale) — see `SCENE_CATALOG_PLAN.md`'s scope note. Full suite (55
tests, including both new regression hashes and the untouched
`make_image` byte-exact regression) passes.

## 0.24.0

Wires the two remaining standalone systems into `SceneGraph.update()`, so
tweens and particle emission no longer need manual per-frame driving from
the caller.

- `SceneGraph` gains `animations: list[AnimationTrack]`; `update()` calls
  `track.update(dt, scene_spec)` for each, before physics steps (so a
  Tween-driven kinematic object's new position is what physics sees that
  frame).
- `ParticlePool` gains an optional `origin: tuple[float, float] | str |
  None`. `SceneGraph.update()` auto-calls `emit_continuous` for any pool
  with an origin set — a fixed world point, or an `ObjectSpec.name` to
  follow every frame via `SceneQuery`. Emission is deterministic: each pool
  gets its own `RandomState` seeded from `SceneGraph.seed` plus its index in
  `particles`, built once and reused every update. Pools without an
  `origin` are unaffected (still driven manually, as before — fully
  backward compatible with 0.22.0/0.23.0 usage).
- Added tests covering both: a Tween moving an object with zero manual
  wiring, two independently-seeded pools producing identical draws, and a
  pool following a named object's live position.

## 0.23.0

Closes the gap left by 0.22.0: `SceneGraph.render()` previously only called
`make_scene` and drew particles, leaving `LayerManager`, `TilemapRenderer`,
sprite atlases, and `HUD` as disconnected utilities the caller had to
composite by hand.

- `SceneGraph` gains `atlas`/`tilemaps`/`hud` fields. `render()` now composes
  bottom to top: tilemaps -> the scene (per-layer composited via
  `LayerManager` if any layers are registered, else one plain `make_scene`
  call, unchanged from 0.22.0) -> sprite-tagged objects (`ObjectSpec.sprite`/
  `flipbook`, stamped via the atlas and excluded from the `make_scene` pass)
  -> particles -> HUD. Every step is still either the public `make_scene` or
  a plain numpy alpha-composite; nothing new touches renderer internals.
  `update()` now also advances `flipbook` playback time and dispatches HUD
  click events (`SceneGraph.last_hud_events`).
- Fixed a footgun in `PhysicsWorld.add_body`: bodies built with the `cx`/
  `cy`/`ground_y` aliases (which `resolved_x`/`resolved_y` prefer over `x`/
  `y`) were simulated but never visibly moved, because physics only wrote
  `x`/`y`. `add_body` now normalizes the aliases into `x`/`y` once, up front.
- Added an end-to-end test: two independently constructed `SceneGraph`s with
  physics, the same seed, and the same input sequence produce byte-identical
  frames over 5 ticks, and a second test proves the full tilemap/sprite/HUD
  compositing order in one frame.

## 0.22.0

Phases 2-13 of the scene-graph/mini-game-engine roadmap (`SCENE_GRAPH_PLAN.md`),
landed together as one increment following the module/versioning pattern
established by 0.10.0. Every new module is a pure-data or pure-logic satellite
(`generators/camera.py`, `layers.py`, `atlas.py`, `particles.py`, `physics.py`,
`input_system.py`, `game_loop.py`, `ui.py`, `tween.py`, `tilemap.py`,
`replay.py`, `web_export.py`) that either never imports the renderer or only
calls the public `make_scene`/`make_scene_raster` functions exactly like any
other caller — none of it can change `make_image` or existing `make_scene`
output. Full suite (43 tests, including the byte-exact `make_image` regression
test) passes unchanged.

- **v0.11 — Camera & Viewport**: `CameraSpec`, `world_to_screen`/
  `screen_to_world`, `aabb_in_view` for culling.
- **v0.12 — Layers**: `Layer`/`LayerManager`, `SceneSpec.layers`, painter's-
  order `sort_objects`, per-layer parallax rendering via the public
  `make_scene` API.
- **v0.13 — Sprites/Atlas**: `AtlasRegion`/`AtlasSpec`/`FlipbookClip`,
  `stamp_sprite` (alpha-composites onto any RGBA canvas), `flipbook_region`.
  `ObjectSpec` gains `sprite`/`texture_scale`/`flip_x`/`flip_y`/`flipbook`/
  `frame`/`elapsed_ms` (all optional).
- **v0.14 — Particles**: `ParticleEmitterSpec`/`ParticlePool` — deterministic
  emission (caller-supplied `RandomState`), vectorized integration, disc
  rendering onto an RGBA canvas.
- **v0.15 — Physics**: `RigidBodySpec`/`PhysicsWorld` — semi-implicit Euler,
  AABB/circle separation + impulse resolution, `Contact`/`ContactHandler`
  enter/exit events, slab-method `ray_cast`. Game-physics fidelity, not
  conservation-grade.
- **v0.16 — Input**: `InputState`/`InputProvider` with `ReplayInputProvider`
  (deterministic), `WebInputProvider` (externally-pushed events),
  `NullInputProvider`, and an optional `PygameInputProvider` (lazy-imports
  `pygame`; not a hard dependency).
- **v0.17 — Game loop**: `GameClock` (fixed-timestep accumulator with a
  spiral-of-death cap) and `SceneGraph`, which composes camera/physics/
  particles/layers/clock and renders via the public `make_scene`.
- **v0.18 — UI/HUD**: `WidgetSpec`/`HUD`, reusing the generator's existing
  5x7 bitmap font (`_stamp_glyph`) for label text instead of a new font
  system. Click events are edge-detected (press, not hold) and returned as
  plain event-name strings — no embedded callbacks.
- **v0.19 — Tweens**: `Tween`/`AnimationTrack` (linear/ease-in/ease-out/
  ease-in-out/bounce/elastic easings, delay/loop/ping-pong) and
  `Keyframe`/`KeyframeTrack`, resolving targets by name or tag via
  `SceneQuery`.
- **v0.20 — Tilemaps**: `TilemapSpec` (tile storage, `collision_mask`) and
  `TilemapRenderer.render`, camera-culled numpy slice-and-stamp.
- **v0.21 — Replay**: `FrameRecord`/`SessionRecorder`/`SessionPlayer` —
  JSON-serializable input logs; replay determinism comes from the caller's
  update logic being deterministic (no RNG inside `SceneGraph.update`).
- **v0.22 — Web export**: `export_html` writes one self-contained HTML file
  with objects drawn as flat-shaded 2D proxy shapes and a small inline JS
  loop (arrow-key/WASD movement for `"player"`-tagged objects). This is an
  interactive layout/interaction demo, not a JS port of the Phong-shaded 3D
  renderer — use server-side `make_scene` frame sequences if pixel-identical
  visuals are required.

## 0.10.0

- Added scene-graph object metadata: `ObjectSpec` gains `tags`, `layer`,
  `visible`, `name`, and `collision_shape` fields (all optional, defaulted,
  additive-only).
- Added `BoundingBox` and `CollisionShape` to `generators.scene_spec`, and a
  new `generators.scene_graph` module with `object_aabb`, `SceneQuery`
  (`at_point`/`in_rect`/`tagged`/`named`), `check_collision`, and
  `find_collisions` (AABB/CIRCLE pairs; CAPSULE deferred).
- `scene_graph.py` only reads `ObjectSpec`/`SceneSpec` data and never imports
  from the renderer, so it cannot affect `make_image` or `make_scene` output.
  Verified: `make_image`'s byte-exact regression test is untouched, and a new
  test proves `make_scene` renders identically whether or not the new
  metadata fields are set on an object.
- First phase of the scene-graph/mini-game-engine roadmap in
  `SCENE_GRAPH_PLAN.md`; remaining phases (camera, layers, sprites,
  particles, physics, input, game loop, UI, tweens, tilemaps, replay, web
  export) follow in later versions.

## 0.9.0

- Added Primitive Render IR v2 operations for rotated ellipses, ellipse
  gradients, maximum composition, Phong spheres/cylinders/tori, rim light, and
  iridescence while retaining the single generic WGSL interpreter.
- Migrated levels 4, 6, and 15–18, expanding portable WebGPU coverage from 19
  to 25 of 154 levels without adding per-level shaders.
- Decomposed the legacy irregular splat into an equivalent ordered union of
  rotated ellipse commands instead of transferring per-pixel masks.
- Exhaustively validated all 325 samples of all six migrated levels. Mean
  errors stayed below `4e-8`; one level-18 boundary pixel crossed its hard
  float32 mask while p99 stayed below `2e-7`.
- Measured analytic IR through llvmpipe at 1,963 img/s versus 5,686 img/s on
  eight CPU workers; automatic routing correctly retains CPU on that adapter.

## 0.8.0

- Generalized the level-128 pore shader into Primitive Render IR v1: one
  validated 12-float command layout and one WGSL interpreter now handle soft
  discs, rings, clipped discs/rings, additive discs, affine rectangles, and
  exact Bresenham lines.
- Migrated levels 0–3, 142 (food), and 144 (crowds) to the shared IR without
  changing their legacy preamble, MT19937 draw order, post-processing, scene
  application, or CPU fallback.
- Expanded portable WebGPU coverage from 13 to 19 of 154 levels without adding
  a level-specific shader.
- Renamed the accelerated family from `render_plan` to `primitive_ir`; the
  benchmark keeps `--dataset render_plan` as a compatibility alias.
- Added CPU control-plane regression tests for IR compilation and WebGPU
  equivalence coverage for all migrated discrete levels.
- Exhaustively checked all 325 samples of every newly migrated level:
  levels 0–3 are byte exact and levels 142/144 stay below `9e-7` maximum
  absolute error on the validation adapter.
- Validated the complete IR through llvmpipe: discrete plans reached
  1,317 img/s and dense plans 889 img/s, both correctly remaining below the
  automatic acceleration gate versus the tuned CPU.

## 0.7.0

- Added RenderGraph v1 for heterogeneous bulk generation. The four current
  WebGPU families now have validated preparation, device, finishing, and
  ordered-assembly dependencies instead of an implicit manual schedule.
- Prioritized all CPU→GPU preparation before general CPU fallback work and
  parallelized reaction/transform finishing through the shared process pool,
  reducing accelerator starvation in mixed batches.
- Added empirical CPU autotuning for `workers=None`: it detects affinity,
  cgroup quotas, physical cores, SMT, NUMA, and the loaded native math runtime.
- Benchmarks physical, partial-SMT, and full-SMT process counts against the
  actual level/raster distribution, then refines process chunks and tests
  one-thread versus runtime-default native thread policies.
- Added warmed persistent CPU pools and a 15-minute workload/topology cache.
  Explicit worker/chunk settings remain authoritative, and
  `SIG_CPU_AUTOTUNE=0` disables empirical measurement.
- Added public `cpu_info()` diagnostics with topology, load, native runtimes,
  candidate timings, and the selected configuration.
- Changed the default `chunksize` to `None`, allowing automatic tuning when the
  caller does not specify it.
- Added `threadpoolctl` to control and inspect native pools without assuming
  that limiting them is faster; both policies are measured.
- On the Ryzen 9 7950X3D test system, the tuner selected 16–24 workers with
  chunks of 4–8 and sustained 8,800–9,522 full-catalog img/s, up from about
  7,549 img/s under the previous fixed policy.
- Recalibrated CPU/GPU routing against the faster CPU. The full catalog now
  stays CPU-only because hybrid measured below the 1.10x gate, while the
  13-level accelerated mix still reached 10,738 img/s versus 7,041 CPU.

## 0.6.0

- Added the first reusable GPU render-plan interpreter. Level 128 now compiles
  its variable pore geometry into ordered disc, ring, occlusion, and blend
  commands; one WGSL dispatch evaluates the plan in parallel over every pixel.
- Parallelized render-plan preparation and finishing through the existing CPU
  worker pool while the GPU owns raster composition.
- Added per-family calibration followed by a whole-schedule gate, with
  stratified samples of up to 4,096 images. Fast GPU families can now overlap
  unsupported CPU levels without allowing a locally fast kernel to slow the
  complete batch.
- Added persistent CPU planners created before WebGPU starts native driver
  threads, retaining cheap Linux fork workers safely across calibrated calls.
- Added one mixed portable transform kernel for mirror, kaleidoscope,
  diffraction, displacement refraction, and spherical-lens refraction at
  levels 129, 130, 133, 135, and 136.
- Expanded portable WebGPU coverage from 7 to 13 of 154 levels. Automatic
  selection still requires a measured 1.10x win for the actual workload, so
  portable-but-slower transform batches remain on CPU.
- Measured level-128 render plans at 991 img/s on NVIDIA Vulkan (1.20x over
  eight CPU processes) and 987 img/s on AMD Vulkan (1.15x).
- Measured the warmed 13-level hybrid mix at 7,137 img/s on NVIDIA and
  6,894 img/s on integrated AMD; measured the full 154-level catalog at
  5,568 img/s on NVIDIA (1.15x over the calibrated CPU schedule).
- Vectorized the legacy 2D flame's inner sample loop without changing its
  regression digest, reducing representative level-65 latency from 33.8 to
  6.9 ms/image and level-67 latency from 18.4 to 2.8 ms/image.
- Added transform and render-plan benchmark datasets plus cross-family
  numerical validation.

## 0.5.0

- Reworked portable generation around homogeneous bulk families rather than
  per-image accelerator decisions.
- Added a WebGPU Gray-Scott reaction-diffusion kernel: one 256-lane workgroup
  owns a complete 32x32 image and runs all 200–340 simulation steps using
  portable 16 KiB workgroup storage.
- Preserved the exact legacy RNG stream around GPU reaction simulation, then
  reused the existing CPU color, scene, and post pipeline.
- Added a stable `GPU_LEVEL_FAMILIES` registry and ordered multilevel dispatch
  for reaction-diffusion plus all six wave levels.
- Changed automatic calibration to benchmark the actual mixed family
  distribution as one bulk workload before selecting WebGPU.
- Started unsupported CPU process work before GPU dispatch so heterogeneous
  explicit batches can overlap CPU and GPU execution.
- Added `wave`, `reaction`, `accelerated`, and full `catalog` benchmark modes.
- Measured 12,380 img/s for the seven-level accelerated mix on NVIDIA Vulkan
  (2.25x over eight CPU processes) and 10,141 img/s on AMD Vulkan (1.83x).

## 0.4.0

- Added an optional CUDA-free WebGPU backend implemented with WGSL compute
  shaders over wgpu's Vulkan, Metal, and DirectX 12 adapters.
- Added a fused three-stage plane-wave pipeline: per-pixel synthesis, parallel
  min/max reduction, and color/scene mapping.
- Vectorized the legacy-compatible MT19937 scene-attribute preparation used by
  GPU batches, removing thousands of per-image RandomState constructions.
- Added hybrid scheduling so unsupported generator levels retain CPU coverage
  while compatible levels execute as one GPU batch in input order.
- Added explicit `legacy` and `portable` fidelity contracts; legacy output and
  its regression digest remain byte exact.
- Added lazy hardware discovery and `accelerator_info()`. CPU/software adapters
  are rejected from automatic acceleration but remain opt-in for validation.
- Made portable `auto` selection performance-gated: it calibrates against the
  configured CPU backend and requires at least 1.10x measured speedup.
- Verified the same WGSL kernels through Vulkan on NVIDIA discrete and AMD
  integrated GPUs, measuring 3.34x and 1.89x over the eight-process CPU path.
- Added GPU equivalence, validation, fallback, packaging, and benchmark
  coverage. The optional runtime is installed through the `gpu` extra.

## 0.3.0

- Added ordered `make_images()` batches with `serial`, `process`, and `auto`
  CPU backends.
- Streamed ordered worker results directly into the final contiguous batch to
  avoid retaining a list plus a second stacked copy.
- Made public indexed and structured-scene rendering safe across concurrent
  thread callers while preserving process-parallel throughput.
- Preserved the `sig-154-cycle-v1` byte-exact legacy sample digest.
- Optimized reaction–diffusion periodic stencils without changing numerical
  output.
- Reused immutable geometry, resize, and dithering coordinate grids.
- Reduced Sobol initialization peak memory with chunked float32 storage.
- Fixed alpha extraction from integer RGB inputs.
- Added strict validation for `step`, scene seeds, scene graphs, and unsupported
  packed grayscale output.
- Added process/serial benchmarking and expanded thread, process, validation,
  raster, and scene regression tests.
- Fixed the built-in atlas demo to use all 154 canonical level names.
- Moved the distribution version to one metadata source and added `py.typed`.
- Replaced experiment-specific documentation with an accurate library README
  and compatibility policy.
