import hashlib

import synthetic_image_generator as sig


def test_one_exact_scene_sample_per_archetype():
    digest = hashlib.sha256()
    for i in range(sig.N_ARCHETYPES):
        sample = sig.make_scene_catalog(i * sig.SAMPLES_PER_ARCHETYPE)
        digest.update(sample.frames[0].tobytes())

    assert digest.hexdigest() == sig.SCENE_CATALOG_SAMPLE_SHA256


def test_one_exact_scene_trajectory_per_archetype():
    digest = hashlib.sha256()
    for i in range(sig.N_ARCHETYPES):
        sample = sig.make_scene_trajectory(i * sig.SAMPLES_PER_ARCHETYPE, n_frames=60)
        for frame in sample.frames:
            digest.update(frame.tobytes())

    assert digest.hexdigest() == sig.SCENE_TRAJECTORY_SAMPLE_SHA256
