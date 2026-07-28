# Canonical source and compatibility policy

## Active distribution

- Distribution: `synthetic-image-generator`
- Import namespace: `synthetic_image_generator`
- Version source: `generators/_version.py`
- Physical package source: `generators/`
- Indexed schedule contract: `sig-154-cycle-v1`

`pyproject.toml` maps `generators/` directly to the public namespace. A build
must contain one package tree and must not copy the renderer into a second
source directory.

## Indexed compatibility

The canonical output of `make_image(idx)` is RGB `float32`, shape
`(32, 32, 3)`, range `[0, 1]`. The schedule is:

```text
level = (idx // 325) % 154
```

Indices are non-negative and unbounded. The absolute block remains part of the
seed, so later catalog cycles preserve their level label without repeating the
first cycle's geometry.

`LEGACY_LEVEL_SAMPLE_SHA256` covers one exact sample from every level. Any
optimization touching numerical order must pass that digest before it can
enter the legacy backend.

## Parallel execution policy

The recursive legacy renderer temporarily changes process-local render state
for supersampling and lighting effects. Public thread calls are protected by a
reentrant lock. Parallel throughput is provided by `make_images()` using
isolated worker processes; ordered batch output must equal a serial stack byte
for byte.

When worker count is not explicit, batches of at least 256 images may
empirically calibrate against their actual level/raster distribution. The
candidate space must respect affinity and cgroup quotas and may include
physical cores, SMT, chunk size, and native thread-pool policy. Results are
topology/workload scoped and time limited. Calibration may change execution
strategy only; it must not change indexed output. Persistent pools must be
created before native GPU driver threads when CPU/GPU overlap is possible.

Accelerator backends may be added behind the same batch API. They must declare
whether they provide:

1. Byte-exact legacy compatibility.
2. Backend-local deterministic output.
3. Fast, numerically approximate output.

No accelerator may silently claim the legacy contract without passing its
regression corpus.

The WebGPU backend is classified as backend-local deterministic / numerically
portable. It is available only with `fidelity="portable"`. Wave output is
validated against legacy at `rtol=atol=1e-4`; reaction-diffusion uses mean and
99th-percentile error bounds because its optional hard threshold is
discontinuous; ordered Primitive IR uses the same distributional check where
a later stochastic operation can amplify one sub-ULP blend difference. Default
and explicit legacy fidelity continue to use only byte-exact CPU
implementations. Mixed portable batches calibrate each family and then the
whole CPU/GPU schedule, may overlap compatible GPU levels with CPU fallback,
and must not change input order. Heterogeneous execution is represented as a
validated dependency graph. Preparation nodes that feed accelerator work take
submission priority over unrelated CPU render nodes; every terminal branch
must precede the ordered batch-assembly node.

## Release checks

Run from this directory:

```bash
python -m pytest -q
python -m pip wheel . --no-deps --wheel-dir /tmp/sig-dist
```

Then install the wheel in an isolated environment and smoke-test:

```bash
python -c "import synthetic_image_generator as s; print(s.__version__, s.make_image(42).shape)"
```

The README, package metadata, `__version__`, wheel filename, and installed
metadata must report the same version.
