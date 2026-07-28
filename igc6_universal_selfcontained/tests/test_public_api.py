import numpy as np

import synthetic_image_generator as sig


def test_public_indexed_contract():
    first = sig.make_image(42)
    second = sig.make_image(42)

    assert sig.__version__ == "0.2.2"
    assert sig.N_LEVELS == 154
    assert (sig.H, sig.W, sig.C) == (32, 32, 3)
    assert len(sig.LEVEL_NAMES) == sig.N_LEVELS
    assert first.shape == (32, 32, 3)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0


def test_public_scene_rgba_has_real_transparency_and_xy_resolution():
    scene = sig.SceneSpec(
        background=sig.Background(kind="transparent"),
        objects=[
            sig.ObjectSpec(
                kind="sphere_3d",
                x=16,
                y=16,
                radius=8,
                color=(1.0, 0.2, 0.1),
                opacity=0.6,
            )
        ],
    )
    rgba = sig.make_scene(
        scene,
        raster=sig.RasterSpec(width=96, height=48, mode="rgba"),
    )

    assert rgba.shape == (48, 96, 4)
    assert rgba.dtype == np.uint8
    assert rgba[..., 3].min() == 0
    assert rgba[..., 3].max() > 0
    assert np.any((rgba[..., 3] > 0) & (rgba[..., 3] < 255))


def test_public_raster_modes_and_bit_depths():
    image = sig.make_image(314)

    gray16 = sig.convert_raster(
        image,
        sig.RasterSpec(width=37, height=61, mode="grayscale", bits_per_channel=16),
    )
    binary = sig.convert_raster(
        image,
        sig.RasterSpec(width=35, height=21, mode="binary", bits_per_channel=1),
    )
    rgb565 = sig.convert_raster(
        image,
        sig.RasterSpec(
            width=47,
            height=23,
            mode="rgb",
            bits_per_channel=(5, 6, 5),
            packed=True,
        ),
    )
    rgba2222 = sig.make_image(
        314,
        raster=sig.RasterSpec(width=41, height=27, mode="rgba2222"),
    )

    assert gray16.shape == (61, 37) and gray16.dtype == np.uint16
    assert binary.shape == (21, 35) and binary.dtype == np.bool_
    assert rgb565.shape == (23, 47) and rgb565.dtype == np.uint16
    assert rgba2222.shape == (27, 41) and rgba2222.dtype == np.uint8


def test_public_exports_are_declared():
    expected = {
        "Background",
        "LightSpec",
        "ObjectSpec",
        "PostSpec",
        "RasterSpec",
        "SceneSpec",
        "convert_raster",
        "extract_alpha",
        "level_block",
        "level_of",
        "level_start",
        "make_image",
        "make_image_raster",
        "make_scene",
        "make_scene_raster",
    }

    assert expected <= set(sig.__all__)


def test_schedule_cycles_forever_without_repeating_geometry():
    cycle = sig.LEVEL_CYCLE_SIZE

    assert sig.level_of(0) == 0
    assert sig.level_of(cycle - 1) == sig.N_LEVELS - 1
    assert sig.level_of(cycle) == 0
    assert sig.level_of(10_000_000) == (
        (10_000_000 // sig.SAMPLES_PER_LEVEL) % sig.N_LEVELS
    )
    assert sig.level_start(37, cycle=11) == 11 * cycle + 37 * sig.SAMPLES_PER_LEVEL

    first = sig.make_image(sig.level_start(0), force_level=0)
    later = sig.make_image(sig.level_start(0, cycle=1000), force_level=0)
    assert not np.array_equal(first, later)


def test_index_and_forced_level_validation():
    import pytest

    with pytest.raises(ValueError):
        sig.make_image(-1)
    with pytest.raises(TypeError):
        sig.make_image(1.5)
    with pytest.raises(ValueError):
        sig.make_image(42, force_level=sig.N_LEVELS)
    with pytest.raises(TypeError):
        sig.make_image(42, force_level=3.5)
