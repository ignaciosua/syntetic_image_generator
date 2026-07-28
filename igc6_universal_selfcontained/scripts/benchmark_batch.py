"""Reproducible CPU/WebGPU throughput benchmark for the public batch API."""

import argparse
import os
import time

import numpy as np

import synthetic_image_generator as sig


def benchmark(indices, backend, workers, chunksize):
    started = time.perf_counter()
    images = sig.make_images(
        indices,
        backend=backend,
        workers=workers,
        chunksize=chunksize,
        fidelity="portable" if backend == "webgpu" else "legacy",
    )
    elapsed = time.perf_counter() - started
    return images, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3080)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="worker chunk size; omit to use empirical CPU autotuning",
    )
    parser.add_argument(
        "--dataset",
        choices=(
            "catalog",
            "wave",
            "accelerated",
            "reaction",
            "render_plan",
            "primitive_ir",
            "primitive_discrete",
            "primitive_dense",
            "primitive_analytic",
            "transform",
        ),
        default="catalog",
        help=(
            "catalog cycles through all levels; accelerated uses every GPU "
            "family; primitive_* datasets isolate IR workload shapes"
        ),
    )
    parser.add_argument(
        "--webgpu",
        action="store_true",
        help="also benchmark the optional vendor-neutral WebGPU backend",
    )
    parser.add_argument(
        "--allow-software",
        action="store_true",
        help="allow a CPU WebGPU adapter such as llvmpipe for validation",
    )
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")

    if args.allow_software:
        os.environ["SIG_WEBGPU_ALLOW_SOFTWARE"] = "1"
    if args.dataset == "wave":
        levels = tuple(sorted(sig.WAVE_LEVELS))
    elif args.dataset == "reaction":
        levels = (113,)
    elif args.dataset == "primitive_discrete":
        levels = (0, 1, 2, 3)
    elif args.dataset == "primitive_dense":
        levels = (128, 142, 144)
    elif args.dataset == "primitive_analytic":
        levels = (4, 6, 15, 16, 17, 18)
    elif args.dataset in {"render_plan", "primitive_ir"}:
        levels = tuple(
            level
            for level, family in sorted(sig.GPU_LEVEL_FAMILIES.items())
            if family == "primitive_ir"
        )
    elif args.dataset == "transform":
        levels = tuple(
            level
            for level, family in sorted(sig.GPU_LEVEL_FAMILIES.items())
            if family == "transform"
        )
    elif args.dataset == "accelerated":
        levels = tuple(sorted(sig.GPU_LEVEL_FAMILIES))
    else:
        levels = tuple(range(sig.N_LEVELS))
    indices = [
        sig.level_start(levels[index % len(levels)])
        + (index // len(levels)) % sig.SAMPLES_PER_LEVEL
        for index in range(args.count)
    ]
    serial, serial_time = benchmark(indices, "serial", 1, args.chunksize)
    parallel, parallel_time = benchmark(
        indices, "process", args.workers, args.chunksize
    )
    if not np.array_equal(serial, parallel):
        raise RuntimeError("process output differs from serial output")

    print(
        f"serial:  {len(indices) / serial_time:,.1f} images/s "
        f"({serial_time:.3f}s)"
    )
    print(
        f"process: {len(indices) / parallel_time:,.1f} images/s "
        f"({parallel_time:.3f}s)"
    )
    print(f"speedup: {serial_time / parallel_time:.2f}x")
    if args.workers is None:
        cpu = sig.cpu_info()
        if cpu["tuning"]:
            selected = cpu["tuning"][-1]
            print(
                "cpu auto: "
                f"{selected['workers']} workers, "
                f"chunk {selected['chunksize']}, "
                f"native threads {selected['native_threads']}"
            )
    if args.webgpu:
        try:
            # Adapter creation and shader compilation are one-time costs. Keep
            # them out of steady-state throughput while still surfacing adapter
            # failures before the timed batch.
            warm_indices = []
            seen_families = set()
            for idx in indices:
                family = sig.GPU_LEVEL_FAMILIES.get(sig.level_of(idx))
                if family is not None and family not in seen_families:
                    seen_families.add(family)
                    warm_indices.append(idx)
            benchmark(
                warm_indices or indices[: min(8, len(indices))],
                "webgpu",
                1,
                args.chunksize,
            )
            portable, portable_time = benchmark(
                indices, "webgpu", args.workers, args.chunksize
            )
        except RuntimeError as error:
            print(f"webgpu: unavailable ({error})")
        else:
            absolute_error = np.abs(serial - portable)
            if args.dataset in (
                "reaction",
                "render_plan",
                "primitive_ir",
                "primitive_discrete",
                "primitive_dense",
                "primitive_analytic",
                "accelerated",
            ):
                equivalent = (
                    float(absolute_error.mean()) <= 1e-5
                    and float(np.percentile(absolute_error, 99.0)) <= 1e-4
                )
            else:
                equivalent = np.allclose(
                    serial,
                    portable,
                    rtol=1e-4,
                    atol=1e-4,
                    equal_nan=False,
                )
            if not equivalent:
                maximum = float(absolute_error.max())
                raise RuntimeError(
                    f"WebGPU output exceeds tolerance; max abs error {maximum}"
                )
            fastest_cpu = min(serial_time, parallel_time)
            info = sig.accelerator_info()
            print(
                f"webgpu: {len(indices) / portable_time:,.1f} images/s "
                f"({portable_time:.3f}s, "
                f"{fastest_cpu / portable_time:.2f}x vs fastest CPU)"
            )
            print(
                "adapter: "
                f"{info.get('device', 'unknown')} / "
                f"{info.get('backend', 'unknown')} / "
                f"{info.get('adapter_type', 'unknown')}"
            )


if __name__ == "__main__":
    main()
