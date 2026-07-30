import synthetic_image_generator as sig


def test_dynamic_scene_catalog_states_remain_finite():
    reports = sig.validate_scene_dynamics()
    assert len(reports) == sig.SCENE_N_LEVELS
    dynamic = [report for report in reports if report["simulation"] != "static"]
    assert dynamic and all(report["finite"] for report in dynamic)
