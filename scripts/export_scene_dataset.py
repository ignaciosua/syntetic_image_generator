"""Write scene-catalog samples to disk: one directory per idx with each
frame as a .npy array and a JSON metadata sidecar (archetype, behaviors,
seed, per-frame object state/events).

ponytail: no packed array format (.npz/HDF5) yet, and no process-pool
parallelism — see SCENE_CATALOG_PLAN.md Phase C: the right storage layout
depends on which consumer (vision/RL/world-model) shows up first, and
guessing now risks building the wrong export path. This is the minimum
that actually gets samples onto disk in a format any of them can read.

Usage:
    python scripts/export_scene_dataset.py --count 16 --mode catalog
    python scripts/export_scene_dataset.py --count 4 --mode trajectory --n-frames 30
"""

import argparse
import json
from pathlib import Path

import numpy as np

from synthetic_image_generator import make_scene_catalog, make_scene_trajectory


def _write_sample(sample, out_dir: Path) -> None:
    sample_dir = out_dir / f"{sample.idx:08d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(sample.frames):
        np.save(sample_dir / f"frame_{i:04d}.npy", frame)
    meta = {
        "idx": sample.idx,
        "archetype": sample.archetype,
        "behaviors": sample.behaviors,
        "seed": sample.seed,
        "objects_state": sample.objects_state,
        "events": sample.events,
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=0, help="first idx to export")
    parser.add_argument("--count", type=int, default=8, help="number of samples to export")
    parser.add_argument("--mode", choices=["catalog", "trajectory"], default="catalog")
    parser.add_argument("--n-frames", type=int, default=60, help="trajectory length (mode=trajectory only)")
    parser.add_argument("--out", type=Path, default=Path("scene_dataset"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for idx in range(args.start, args.start + args.count):
        sample = (
            make_scene_catalog(idx)
            if args.mode == "catalog"
            else make_scene_trajectory(idx, n_frames=args.n_frames)
        )
        _write_sample(sample, args.out)

    print(f"Wrote {args.count} {args.mode} samples to {args.out}")


if __name__ == "__main__":
    main()
