from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import synthetic_image_generator as sig
import synthetic_image_generator.synthetic_image_generator as implementation


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (32, 32),
        (64, 64),
        (127, 65),
        (320, 180),
    ],
)
def test_every_indexed_level_renders_at_native_resolution(width, height):
    for level in range(sig.N_LEVELS):
        image = sig.make_image(
            sig.level_start(level),
            force_level=level,
            width=width,
            height=height,
        )
        assert image.shape == (height, width, 3), level
        assert image.dtype == np.float32, level
        assert np.isfinite(image).all(), level
        assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0, level
        again = sig.make_image(
            sig.level_start(level),
            force_level=level,
            width=width,
            height=height,
        )
        assert np.array_equal(image, again), level


def test_native_resolution_is_deterministic_and_not_a_32px_upscale():
    index = sig.level_start(113) + 17
    first = sig.make_image(index, force_level=113, width=95, height=57)
    second = sig.make_image(index, force_level=113, width=95, height=57)
    legacy = sig.make_image(index, force_level=113)
    resized_legacy = implementation._resize_float_image(
        legacy, 95, 57, "bilinear"
    )

    assert np.array_equal(first, second)
    assert not np.array_equal(first, resized_legacy)
    assert float(np.mean(np.abs(first - resized_legacy))) > 1e-3


def test_indexed_geometry_helpers_keep_relative_size_on_native_canvas():
    with implementation._canvas_size(32, 32):
        with implementation._indexed_geometry_mode(False):
            small_mask = implementation._sphere(16, 16, 4)[0]
    with implementation._canvas_size(128, 128):
        with implementation._indexed_geometry_mode(True):
            large_mask = implementation._sphere(64, 64, 4)[0]

    assert large_mask.mean() == pytest.approx(small_mask.mean(), abs=0.01)


def test_raster_dimensions_drive_the_native_canvas(monkeypatch):
    original_resize = implementation._resize_float_image

    def reject_spatial_resize(image, width, height, resize):
        assert image.shape[:2] == (height, width)
        return original_resize(image, width, height, resize)

    monkeypatch.setattr(
        implementation, "_resize_float_image", reject_spatial_resize
    )
    raster = sig.RasterSpec(width=61, height=37, mode="rgba")
    image = sig.make_image_raster(314, raster)
    assert image.shape == (37, 61, 4)
    assert image.dtype == np.uint8


def test_raster_output_is_quantized_from_the_same_native_float_render():
    raster = sig.RasterSpec(
        width=83, height=49, mode="rgb", bits_per_channel=16
    )
    native = sig.make_image(808, width=83, height=49)
    converted = sig.convert_raster(native, raster)
    direct = sig.make_image(808, raster=raster)
    assert np.array_equal(direct, converted)


def test_structured_scene_scales_canonical_geometry_without_mutation():
    obj = sig.ObjectSpec(
        kind="disc", x=8, y=16, radius=4, color=(1.0, 1.0, 1.0)
    )
    scene = sig.SceneSpec(
        background=sig.Background(kind="solid", color=(0.0, 0.0, 0.0)),
        objects=[obj],
    )
    small = sig.make_scene(scene)
    large = sig.make_scene(scene, width=96, height=64)

    small_y, small_x = np.argwhere(small[..., 0] > 0.5).mean(axis=0)
    large_y, large_x = np.argwhere(large[..., 0] > 0.5).mean(axis=0)
    assert small_x / 32 == pytest.approx(large_x / 96, abs=0.015)
    assert small_y / 32 == pytest.approx(large_y / 64, abs=0.015)
    assert (obj.x, obj.y, obj.radius) == (8, 16, 4)


def test_every_structured_object_kind_renders_on_rectangular_canvas():
    for kind in sorted(implementation._SCENE_OBJECT_KINDS):
        scene = sig.SceneSpec(
            objects=[
                sig.ObjectSpec(
                    kind=kind,
                    x=16,
                    y=20,
                    radius=5,
                    width=8,
                    height=8,
                    size=8,
                )
            ]
        )
        image = sig.make_scene(scene, width=127, height=65)
        assert image.shape == (65, 127, 3), kind
        assert image.dtype == np.float32, kind
        assert np.isfinite(image).all(), kind


def test_batch_and_concurrent_calls_keep_each_requested_canvas_isolated():
    batch = sig.make_images(
        [0, 1, 2, 3], width=73, height=41, backend="serial"
    )
    assert batch.shape == (4, 41, 73, 3)

    sizes = [(32, 32), (47, 35), (96, 64), (65, 127)] * 3
    with ThreadPoolExecutor(max_workers=4) as executor:
        images = list(
            executor.map(
                lambda size: sig.make_image(
                    99, width=size[0], height=size[1]
                ),
                sizes,
            )
        )
    assert [
        image.shape for image in images
    ] == [(height, width, 3) for width, height in sizes]
    assert (implementation.W, implementation.H) == (32, 32)


def test_native_batch_uses_exact_cpu_fallback_for_fixed_size_gpu_kernels():
    indices = [0, sig.level_start(113), sig.level_start(129)]
    expected = sig.make_images(
        indices, width=73, height=41, backend="serial"
    )
    portable = sig.make_images(
        indices,
        width=73,
        height=41,
        backend="webgpu",
        fidelity="portable",
    )
    assert np.array_equal(portable, expected)
    with pytest.raises(ValueError, match="portable"):
        sig.make_images(
            indices, width=73, height=41, backend="webgpu"
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"width": 64}, ValueError),
        ({"width": 31, "height": 32}, ValueError),
        ({"width": True, "height": 32}, TypeError),
        ({"width": 4097, "height": 32}, ValueError),
        ({"width": 4096, "height": 4096}, ValueError),
    ],
)
def test_native_resolution_validation(kwargs, error):
    with pytest.raises(error):
        sig.make_image(0, **kwargs)


def test_explicit_dimensions_must_match_raster_dimensions():
    raster = sig.RasterSpec(width=64, height=48)
    with pytest.raises(ValueError, match="must match"):
        sig.make_image(0, raster=raster, width=64, height=64)


def test_tiny_legacy_rasters_remain_supported_by_safe_reduction():
    raster = sig.RasterSpec(width=11, height=7, mode="grayscale")
    image = sig.make_image_raster(7, raster)
    assert image.shape == (7, 11)
    assert image.dtype == np.uint8
