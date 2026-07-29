import synthetic_image_generator as sig


def test_scene_quality_reports_geometry_diagnostics():
    report = sig.scene_quality_report(sig.scene_level_start(0))
    assert {"overlaps", "out_of_view_centers", "visible_pixels"} <= set(report)
    reports = sig.validate_scene_quality(indices=[0, 20, 40, 80, 120])
    assert all(report["finite"] for report in reports)
