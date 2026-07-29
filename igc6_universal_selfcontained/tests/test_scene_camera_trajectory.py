import numpy as np
import synthetic_image_generator as sig


def test_camera_trajectory_is_high_resolution_and_reproducible():
    first = sig.make_scene_camera_trajectory(sig.scene_level_start(18), n_frames=4)
    assert first[0].shape == (512, 512, 4)
    assert any(not np.array_equal(first[0], frame) for frame in first[1:])
