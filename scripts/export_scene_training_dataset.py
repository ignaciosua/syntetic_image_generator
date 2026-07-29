"""Export long, exact renderer trajectories for scene-model training."""
import argparse
from pathlib import Path
import json
import numpy as np
from synthetic_image_generator import make_scene_training_trajectory, scene_level_start, SCENE_N_LEVELS, analyze_scene_trajectory


def export(output="media/scene_training", quality="training", levels=None):
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    selected = range(SCENE_N_LEVELS) if levels is None else levels
    manifest = []
    for level in selected:
        sample = make_scene_training_trajectory(scene_level_start(int(level)), quality=quality)
        path = root / f"level_{int(level):03d}.npz"
        np.savez_compressed(path, frames=np.stack(sample.frames), objects=np.array(sample.objects_state, dtype=object), events=np.array(sample.events, dtype=object))
        manifest.append({"level": int(level), "file": path.name, "frames": len(sample.frames), "analysis": analyze_scene_trajectory(sample)})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="media/scene_training")
    parser.add_argument("--quality", choices=("preview", "fast", "training"), default="training")
    parser.add_argument("--levels", nargs="*", type=int)
    args = parser.parse_args()
    print(export(args.output, args.quality, args.levels))
