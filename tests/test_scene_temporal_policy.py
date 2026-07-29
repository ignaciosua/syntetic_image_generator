import synthetic_image_generator as sig


def test_temporal_policy_scales_sequences_by_simulation():
    assert sig.scene_level_frame_count(sig.scene_level_start(0), "preview") <= 12
    counts = [sig.scene_level_frame_count(sig.scene_level_start(i)) for i in range(sig.SCENE_N_LEVELS)]
    assert max(counts) >= 180
    assert min(counts) == 1
