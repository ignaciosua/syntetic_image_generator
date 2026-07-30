import synthetic_image_generator as sig


def test_training_trajectory_has_analysis_metadata():
    sample = sig.make_scene_training_trajectory(sig.scene_level_start(4), quality="fast")
    analysis = sig.analyze_scene_trajectory(sample)
    assert analysis["frames"] == len(sample.frames)
    assert analysis["classification"] in {"static", "loop", "evolving"}
