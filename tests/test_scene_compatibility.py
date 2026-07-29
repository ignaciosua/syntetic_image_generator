import json
import synthetic_image_generator as sig


def test_scene_catalog_compatibility_and_json_metadata():
    sig.validate_scene_compatibility()
    sample = sig.make_scene_trajectory_level(sig.scene_level_start(44), n_frames=2)
    json.dumps(sample.metadata)
    json.dumps(sample.objects_state)
    json.dumps(sample.events)
