import hashlib
import synthetic_image_generator as sig


def test_scene_level_catalog_regression_hash():
    digest = hashlib.sha256()
    for level in range(sig.SCENE_N_LEVELS):
        digest.update(sig.make_scene_level(sig.scene_level_start(level)).tobytes())
    assert digest.hexdigest() == sig.SCENE_LEVEL_SAMPLE_SHA256
