"""Print the regression hashes for the scene catalog (Phase A) and scene
trajectories (Phase B) — one exact sample per archetype, same pattern as
the indexed image catalog's LEGACY_LEVEL_SAMPLE_SHA256.

Run after any change to archetype/behavior geometry or RNG draw order, then
paste the printed values into scene_levels.SCENE_CATALOG_SAMPLE_SHA256 and
scene_catalog.SCENE_TRAJECTORY_SAMPLE_SHA256.
"""

import hashlib

from synthetic_image_generator import (
    N_ARCHETYPES,
    SAMPLES_PER_ARCHETYPE,
    make_scene_catalog,
    make_scene_trajectory,
)


def main():
    catalog_hash = hashlib.sha256()
    for i in range(N_ARCHETYPES):
        sample = make_scene_catalog(i * SAMPLES_PER_ARCHETYPE)
        catalog_hash.update(sample.frames[0].tobytes())
    print("SCENE_CATALOG_SAMPLE_SHA256 =", repr(catalog_hash.hexdigest()))

    trajectory_hash = hashlib.sha256()
    for i in range(N_ARCHETYPES):
        sample = make_scene_trajectory(i * SAMPLES_PER_ARCHETYPE, n_frames=60)
        for frame in sample.frames:
            trajectory_hash.update(frame.tobytes())
    print("SCENE_TRAJECTORY_SAMPLE_SHA256 =", repr(trajectory_hash.hexdigest()))


if __name__ == "__main__":
    main()
