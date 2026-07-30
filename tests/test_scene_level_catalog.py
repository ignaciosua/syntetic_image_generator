import numpy as np
import synthetic_image_generator as sig


def test_scene_level_catalog_has_154_deterministic_entries():
    assert sig.SCENE_N_LEVELS == 154
    assert len(sig.SCENE_LEVEL_NAMES) == 154
    assert len(set(sig.SCENE_LEVEL_NAMES)) == 154
    assert sig.scene_level_of(0) == 0
    assert sig.scene_level_of(sig.SAMPLES_PER_SCENE_LEVEL) == 1
    assert sig.scene_level_of(sig.SCENE_LEVEL_CYCLE_SIZE) == 0


def test_scene_level_render_and_trajectory_are_reproducible():
    first = sig.make_scene_level(17)
    second = sig.make_scene_level(17)
    assert np.array_equal(first, second)
    trajectory = sig.make_scene_trajectory_level(17, n_frames=3)
    assert len(trajectory.frames) == 3
    assert trajectory.archetype == sig.scene_level_spec(17).archetype
    sample = sig.make_scene_trajectory_level(sig.scene_level_start(17), n_frames=2)
    first_state = next(iter(sample.objects_state[0].values()))
    assert {"x", "y", "depth", "tags", "layer", "bbox"} <= set(first_state)
    assert {"camera", "contacts", "hud"} <= set(sample.events[0])


def test_all_scene_levels_build_and_serialize():
    for level in range(sig.SCENE_N_LEVELS):
        idx = sig.scene_level_start(level)
        graph = sig.make_scene_level_graph(idx)
        restored = sig.scene_from_dict(sig.scene_to_dict(graph.scene_spec))
        assert restored.objects is not None


def test_scene_level_cycles_change_seeded_geometry():
    first = sig.make_scene_level(sig.scene_level_start(20, cycle=0))
    later = sig.make_scene_level(sig.scene_level_start(20, cycle=1))
    assert not np.array_equal(first, later)
