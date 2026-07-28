# Changelog

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
