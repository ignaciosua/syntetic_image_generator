import hashlib
import synthetic_image_generator as sig


def test_scene_level_trajectory_regression_hash():
    digest = hashlib.sha256()
    for level in range(sig.SCENE_N_LEVELS):
        index = sig.scene_level_start(level)
        for frame in sig.make_scene_trajectory_level(index, n_frames=3).frames:
            digest.update(frame.tobytes())
    assert digest.hexdigest() == sig.SCENE_LEVEL_TRAJECTORY_SAMPLE_SHA256
