import synthetic_image_generator as sig


def test_scene_family_hashes_cover_all_families():
    hashes = sig.scene_family_hashes()
    assert set(hashes) == {"composition", "objects", "nature", "built", "characters", "physics", "systems"}
    assert all(len(value) == 64 for value in hashes.values())
