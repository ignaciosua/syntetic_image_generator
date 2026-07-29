import subprocess
import sys
import synthetic_image_generator as sig


def test_all_archetypes_have_categories_and_geometry_audit_runs():
    for name in sig.ARCHETYPE_NAMES:
        assert sig.scene_archetype_category(name) in {"visual", "environment", "character", "physics", "system"}
    reports = sig.validate_scene_geometry()
    assert len(reports) == sig.SCENE_N_LEVELS


def test_catalog_is_deterministic_in_a_fresh_process():
    code = "import synthetic_image_generator as s; print(s.make_scene_level(s.scene_level_start(37)).tobytes().hex())"
    first = subprocess.check_output([sys.executable, "-c", code], text=True)
    second = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert first == second
