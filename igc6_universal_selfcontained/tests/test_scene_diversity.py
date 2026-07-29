import synthetic_image_generator as sig


def test_scene_diversity_report_has_nontrivial_spread():
    report = sig.scene_diversity_report()
    assert report["color_mean_range"][1] > report["color_mean_range"][0]
    assert report["object_count_range"][1] >= report["object_count_range"][0]
