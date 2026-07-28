# Universal optimization plan

This plan adapts the useful execution ideas behind Bend—explicit independent
work, dependency reduction, and bulk parallelism—without adopting HVM,
interaction nets, CUDA, or a vendor-specific runtime.

## Current baseline

- The byte-exact legacy backend covers all 154 levels.
- CPU batches detect effective topology and empirically tune worker count,
  chunk size, SMT use, and native thread-pool policy.
- Portable WebGPU covers 25 levels in four families.
- Automatic routing requires numerical validity and at least 1.10× measured
  speedup for both each family and the complete hybrid schedule.
- RenderGraph v1 represents heterogeneous batches as explicit preparation,
  GPU, finishing, CPU fallback, and ordered assembly nodes.

## Where the graph model applies

| Work shape | Existing examples | Best execution model |
|---|---|---|
| Closed-form independent pixels | Levels 148–153 | Fused WGSL kernel |
| Iterative local stencil | Level 113 | One workgroup per image |
| Ordered primitive composition | Level 128 | CPU plan + GPU interpreter |
| Image-space resampling | Levels 129, 130, 133, 135, 136 | Batched transform kernel |
| Irregular scene construction | Most remaining levels | CPU control graph feeding reusable GPU primitives |
| Small or branch-heavy work | Many 32×32 legacy levels | Tuned CPU unless bulk measurement proves otherwise |

## Delivery phases

### 1. Dependency-aware scheduler — complete

- Compile batch positions into validated graph nodes.
- Queue every CPU→GPU dependency before unrelated CPU fallback work.
- Run CPU finishing for reaction and transform branches through the shared
  process pool.
- Preserve input order through one assembly node.

Acceptance evidence:

- 24 tests pass, including the 154-level legacy regression contract.
- A controlled workload with 256 CPU renders and 24 accelerator preparations
  on eight workers reduced median accelerator-feed latency from 126.1 ms to
  22.8 ms.
- The complete portable path passes on llvmpipe; hardware timing must still be
  repeated when a physical adapter is visible.

### 2. Primitive Render IR — foundation complete

Represent common operations as compact typed commands:

- rectangle, Bresenham line, rotated ellipse, gradient ellipse, disc, ring,
  clipped soft mask, sphere, cylinder, and torus are implemented;
- levels 0–4, 6, 15–18, 128, 142, and 144 share one command layout and one
  WGSL interpreter;
- ordered alpha blend, additive blend, affine composition, and exact discrete
  overwrite are implemented;
- capsule and palette operations remain candidates for later expansion;
- per-image command spans with stable ordering.

Software-adapter validation measured 1,317 img/s for discrete IR and 889 img/s
for dense IR. Both remain below the tuned CPU reference, so automatic routing
correctly rejects them. Physical NVIDIA/AMD timing remains required.
Exhaustive validation across all 325 indexed samples of each newly migrated
level found byte-exact output for levels 0–3 and maximum absolute errors below
`9e-7` for levels 142 and 144.
Analytic IR validation covered every sample of levels 4, 6, and 15–18. Its
mean errors remain below `4e-8`; one torus boundary pixel crossed a hard mask
because WGSL uses float32, while the 99th percentile remained below `2e-7`.
LlvmPipe reached 1,963 img/s versus 5,686 img/s on tuned CPU and was correctly
rejected.

Migrate levels by operation coverage rather than writing one shader per level.
Start with levels dominated by `_soft_disc`, `_blend`, `_ellipse`, `_tube`, and
`_bresenham`. Keep control flow and legacy MT19937 planning on CPU initially.

### 3. Fusion and GPU-resident intermediates

- Fuse adjacent color, mask, blend, and scene operations into one dispatch.
- Reuse buffer arenas across warmed calls.
- Keep intermediate images on the selected adapter until their terminal graph
  node, performing one final readback per output batch.
- Bucket compatible command streams to reduce dispatch count.

### 4. Filters and deterministic parallel randomness

- Add reusable blur, Sobel, displacement, morphology, and reduction nodes.
- Introduce an opt-in counter-based portable RNG keyed by
  `(index, node, pixel, draw)` for new portable recipes.
- Keep the legacy RNG path unchanged; portable RNG equivalence is statistical,
  not byte exact.

### 5. Adaptive devices

- Extend the cost cache from family-level decisions to graph-node decisions.
- Consider multiple physical adapters only when each is independently valid
  and its transfer cost is included in the whole-schedule measurement.
- Keep software adapters validation-only.

## Invariants

1. Default `fidelity="legacy"` remains byte exact.
2. Unsupported or slower graph branches fall back to CPU.
3. No CUDA, Bend, HVM, or vendor SDK becomes a required dependency.
4. No accelerator is selected from theoretical capability alone; the actual
   workload must pass numerical checks and the 1.10× performance gate.
5. A graph optimization must reduce whole-batch time, not merely make an
   isolated kernel faster.
