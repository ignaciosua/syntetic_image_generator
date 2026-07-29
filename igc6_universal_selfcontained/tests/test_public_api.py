from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import synthetic_image_generator as sig


def test_public_indexed_contract():
    first = sig.make_image(42)
    second = sig.make_image(42)

    assert sig.__version__ == "0.9.0"
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
        "cpu_info",
        "accelerator_info",
        "extract_alpha",
        "level_block",
        "level_of",
        "level_start",
        "make_image",
        "make_images",
        "make_image_raster",
        "make_scene",
        "make_scene_raster",
    }

    assert expected <= set(sig.__all__)


def test_cpu_info_reports_effective_capacity_safely():
    info = sig.cpu_info()

    assert info["logical_system"] >= 1
    assert info["affinity_count"] >= info["available_logical"] >= 1
    assert 1 <= info["physical_cores"] <= info["available_logical"]
    assert info["smt_threads_per_core"] >= 1
    assert info["numa_nodes"] >= 1
    assert isinstance(info["autotune"], bool)
    assert info["native_thread_policy"] == "empirical"
    assert isinstance(info["native_runtimes"], list)
    assert isinstance(info["tuning"], list)


def test_webgpu_contract_validation_and_small_auto_portable_fallback():
    indices = [sig.level_start(level) + 2 for level in (148, 149, 150)]

    with pytest.raises(ValueError, match="fidelity='portable'"):
        sig.make_images(indices, backend="webgpu")
    with pytest.raises(ValueError, match="fidelity"):
        sig.make_images(indices, fidelity="approximate")

    expected = sig.make_images(indices, backend="serial")
    automatic = sig.make_images(
        indices, backend="auto", fidelity="portable", workers=1
    )
    assert np.array_equal(automatic, expected)


def test_accelerator_info_is_safe_without_optional_runtime():
    info = sig.accelerator_info()

    assert info["api"] == "webgpu"
    assert isinstance(info["available"], bool)
    if not info["available"]:
        assert info["reason"]


def test_wgpu_adapter_type_normalizes_native_camel_case():
    from synthetic_image_generator._webgpu import _adapter_type

    assert _adapter_type({"adapter_type": "DiscreteGPU"}) == "DISCRETE_GPU"
    assert _adapter_type({"adapter_type": "integrated-gpu"}) == "INTEGRATED_GPU"


def test_batched_scene_parameters_match_scalar_randomstate():
    from synthetic_image_generator.scene import scene, scene_values_batch

    indices = [0, 1, 42, 1234, 2**31, 10_000_000]
    expected = np.asarray(
        [
            [
                scene(idx)["energy"],
                scene(idx)["warmth"],
                scene(idx)["contrast"],
            ]
            for idx in indices
        ],
        np.float32,
    )
    assert np.array_equal(scene_values_batch(indices), expected)


def test_webgpu_wave_kernel_when_an_adapter_is_available(monkeypatch):
    pytest.importorskip("wgpu")
    monkeypatch.setenv("SIG_WEBGPU_ALLOW_SOFTWARE", "1")
    indices = [sig.level_start(level) + 7 for level in range(148, 154)]
    expected = sig.make_images(indices, backend="serial")
    try:
        portable = sig.make_images(
            indices, backend="webgpu", fidelity="portable"
        )
    except RuntimeError as error:
        pytest.skip(f"WebGPU adapter unavailable: {error}")

    assert portable.shape == expected.shape
    assert portable.dtype == np.float32
    assert np.allclose(portable, expected, rtol=1e-4, atol=1e-4)

    mixed_indices = [
        sig.level_start(0) + 3,
        sig.level_start(148) + 3,
        sig.level_start(67) + 3,
    ]
    mixed_expected = sig.make_images(mixed_indices, backend="serial")
    mixed = sig.make_images(
        mixed_indices, backend="webgpu", fidelity="portable"
    )
    assert np.array_equal(mixed[2], mixed_expected[2])
    assert np.allclose(mixed[:2], mixed_expected[:2], rtol=1e-4, atol=1e-4)

    bulk_indices = []
    for sample in range(4):
        for level in (113, 148, 150, 153):
            bulk_indices.append(sig.level_start(level) + sample)
    order = [7, 0, 13, 2, 15, 4, 10, 1, 8, 3, 14, 5, 12, 6, 11, 9]
    shuffled = [bulk_indices[position] for position in order]
    bulk_expected = sig.make_images(shuffled, backend="serial")
    bulk_portable = sig.make_images(
        shuffled, backend="webgpu", fidelity="portable"
    )
    difference = np.abs(bulk_expected - bulk_portable)
    assert float(difference.mean()) <= 1e-5
    assert float(np.percentile(difference, 99.0)) <= 1e-4

    planned_indices = [
        sig.level_start(level) + sample
        for sample in range(2)
        for level in (
            0,
            1,
            2,
            3,
            4,
            6,
            15,
            16,
            17,
            18,
            128,
            129,
            130,
            133,
            135,
            136,
            142,
            144,
        )
    ]
    planned_expected = sig.make_images(planned_indices, backend="serial")
    planned_portable = sig.make_images(
        planned_indices, backend="webgpu", fidelity="portable"
    )
    planned_difference = np.abs(planned_expected - planned_portable)
    assert float(planned_difference.mean()) <= 1e-5
    assert float(np.percentile(planned_difference, 99.0)) <= 1e-4


def test_auto_portable_routes_only_after_positive_measurement(monkeypatch):
    import synthetic_image_generator.synthetic_image_generator as implementation
    import synthetic_image_generator._webgpu as webgpu

    indices = [sig.level_start(148) + (index % 325) for index in range(64)]
    sentinel = np.full((64, sig.H, sig.W, sig.C), 0.25, np.float32)
    monkeypatch.setattr(
        webgpu,
        "accelerator_info",
        lambda: {"available": True, "device": "test", "backend": "test"},
    )
    monkeypatch.setattr(
        implementation, "_webgpu_is_measurably_faster", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        implementation, "_portable_webgpu_batch", lambda *args, **kwargs: sentinel
    )

    result = sig.make_images(
        indices,
        backend="auto",
        fidelity="portable",
        workers=1,
    )
    assert result is sentinel


def test_auto_portable_gates_partial_gpu_schedule_as_a_whole(monkeypatch):
    import synthetic_image_generator.synthetic_image_generator as implementation

    wave = [
        sig.level_start(148) + (index % sig.SAMPLES_PER_LEVEL)
        for index in range(64)
    ]
    cpu = [
        sig.level_start(67) + (index % sig.SAMPLES_PER_LEVEL)
        for index in range(64)
    ]
    indices = [value for pair in zip(wave, cpu) for value in pair]
    sentinel = np.full((len(indices), sig.H, sig.W, sig.C), 0.5, np.float32)
    calls = []

    def measured(*args, family, **kwargs):
        calls.append((family, kwargs.get("enabled_families")))
        return True

    captured = {}

    def portable(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        implementation, "_webgpu_is_measurably_faster", measured
    )
    monkeypatch.setattr(implementation, "_portable_webgpu_batch", portable)

    result = sig.make_images(
        indices,
        backend="auto",
        fidelity="portable",
        workers=1,
    )

    assert result is sentinel
    assert calls == [("wave", None), ("hybrid:wave", {"wave"})]
    assert captured["enabled_families"] == {"wave"}


def test_gpu_level_registry_groups_bulk_families():
    assert sig.GPU_LEVEL_FAMILIES[113] == "reaction_diffusion"
    assert {
        sig.GPU_LEVEL_FAMILIES[level]
        for level in (
            0,
            1,
            2,
            3,
            4,
            6,
            15,
            16,
            17,
            18,
            128,
            142,
            144,
        )
    } == {"primitive_ir"}
    assert {
        sig.GPU_LEVEL_FAMILIES[level]
        for level in (129, 130, 133, 135, 136)
    } == {"transform"}
    assert {
        sig.GPU_LEVEL_FAMILIES[level] for level in range(148, 154)
    } == {"wave"}


def test_render_graph_compiles_batch_dependencies_and_positions():
    from synthetic_image_generator._rendergraph import RenderGraph
    import synthetic_image_generator.synthetic_image_generator as implementation

    levels = [148, 113, 0, 128, 129, 67, 153]
    graph = RenderGraph.compile(
        levels,
        implementation.GPU_LEVEL_FAMILIES,
        implementation._RENDER_GRAPH_STAGES,
    )

    assert graph.positions("wave") == (0, 6)
    assert graph.positions("reaction_diffusion") == (1,)
    assert graph.positions("primitive_ir") == (2, 3)
    assert graph.positions("transform") == (4,)
    assert graph.positions("cpu") == (5,)
    ordered = graph.topological_nodes()
    order = {node.key: position for position, node in enumerate(ordered)}
    for node in ordered:
        assert all(order[dependency] < order[node.key] for dependency in node.dependencies)
    assert set(graph.node("batch:assemble").dependencies) == {
        "wave:gpu",
        "reaction_diffusion:finish",
        "transform:finish",
        "primitive_ir:finish",
        "cpu:render",
    }


def test_render_graph_submits_gpu_preparation_in_pipeline_order():
    from synthetic_image_generator._rendergraph import RenderGraph
    import synthetic_image_generator.synthetic_image_generator as implementation

    graph = RenderGraph.compile(
        [113, 129, 128, 0],
        implementation.GPU_LEVEL_FAMILIES,
        implementation._RENDER_GRAPH_STAGES,
    )
    submitted = []

    class RecordingExecutor:
        def map(self, function, arguments, *, chunksize):
            assert chunksize == 4
            values = list(arguments)
            submitted.extend(values)
            return map(function, values)

    specifications = {
        "reaction_diffusion": (lambda value: value, ["reaction"]),
        "transform": (lambda value: value, ["transform"]),
        "primitive_ir": (lambda value: value, ["primitive_ir"]),
    }
    pending = implementation._submit_render_graph_preparation(
        graph, RecordingExecutor(), specifications, chunksize=4
    )

    assert submitted == ["reaction", "transform", "primitive_ir"]
    assert [
        result
        for family in ("reaction_diffusion", "transform", "primitive_ir")
        for result in pending[family]
    ] == submitted


def test_primitive_plans_preserve_legacy_rng_and_cpu_raster():
    import synthetic_image_generator.synthetic_image_generator as implementation
    from synthetic_image_generator._primitive_ir import PrimitiveOp

    for level in (0, 1, 2, 3, 4, 6, 15, 16, 17, 18, 142, 144):
        samples = (0, 7, 41) if level < 4 else (0, 7)
        for sample in samples:
            idx = sig.level_start(level) + sample
            initial, commands, rng = implementation._prepare_primitive_plan(
                idx, level, 7
            )
            rendered = initial.copy()
            for command in commands:
                operation = PrimitiveOp(int(command[0]))
                color = command[9:12]
                if operation == PrimitiveOp.RECT:
                    x0 = max(0, int(np.ceil(command[1])))
                    y0 = max(0, int(np.ceil(command[2])))
                    x1 = min(sig.W - 1, int(np.floor(command[3])))
                    y1 = min(sig.H - 1, int(np.floor(command[4])))
                    rendered[y0 : y1 + 1, x0 : x1 + 1] = color
                elif operation == PrimitiveOp.BRESENHAM:
                    implementation._bresenham(
                        rendered,
                        int(command[1]),
                        int(command[2]),
                        int(command[3]),
                        int(command[4]),
                        int(command[5]),
                        color,
                    )
                elif operation in {
                    PrimitiveOp.SOFT_DISC,
                    PrimitiveOp.ADDITIVE_SOFT_DISC,
                }:
                    alpha = (
                        implementation._soft_disc(
                            command[1], command[2], command[3]
                        )
                        * command[8]
                    ).reshape(sig.H, sig.W, 1)
                    if operation == PrimitiveOp.SOFT_DISC:
                        rendered = np.clip(
                            rendered * (1 - alpha)
                            + color.reshape(1, 1, 3) * alpha,
                            0,
                            1,
                        )
                    else:
                        rendered = np.clip(
                            rendered + color.reshape(1, 1, 3) * alpha,
                            0,
                            1,
                        )
                elif operation == PrimitiveOp.AFFINE_RECT:
                    x0 = max(0, int(command[1]))
                    y0 = max(0, int(command[2]))
                    x1 = min(sig.W - 1, int(command[3]))
                    y1 = min(sig.H - 1, int(command[4]))
                    rendered[y0 : y1 + 1, x0 : x1 + 1] = np.clip(
                        rendered[y0 : y1 + 1, x0 : x1 + 1] * command[8]
                        + color,
                        0,
                        1,
                    )
                elif operation in {
                    PrimitiveOp.ELLIPSE,
                    PrimitiveOp.ELLIPSE_GRADIENT,
                    PrimitiveOp.MAX_ELLIPSE,
                }:
                    yy, xx = np.ogrid[: sig.H, : sig.W]
                    dx, dy = xx - command[1], yy - command[2]
                    angle = (
                        0.0
                        if operation == PrimitiveOp.ELLIPSE_GRADIENT
                        else command[5]
                    )
                    cosine, sine = np.cos(angle), np.sin(angle)
                    rotated_x = dx * cosine + dy * sine
                    rotated_y = -dx * sine + dy * cosine
                    distance = np.sqrt(
                        (rotated_x / command[3]) ** 2
                        + (rotated_y / command[4]) ** 2
                    )
                    mask = distance <= 1
                    if operation == PrimitiveOp.ELLIPSE:
                        rendered[mask] = color
                    elif operation == PrimitiveOp.MAX_ELLIPSE:
                        rendered[mask] = np.maximum(rendered[mask], color)
                    else:
                        gradient = 1 - distance / command[12]
                        for channel in range(sig.C):
                            rendered[:, :, channel][mask] = np.clip(
                                color[channel]
                                + gradient[mask] * command[8],
                                0,
                                1,
                            )
                elif operation in {
                    PrimitiveOp.SPHERE_PHONG,
                    PrimitiveOp.CYLINDER_PHONG,
                    PrimitiveOp.TORUS_PHONG,
                }:
                    yy, xx = np.ogrid[: sig.H, : sig.W]
                    if operation == PrimitiveOp.SPHERE_PHONG:
                        normal_x = (xx - command[1]) / command[3]
                        normal_y = (yy - command[2]) / command[3]
                        distance = normal_x**2 + normal_y**2
                        mask = distance <= 1
                        normal_z = np.sqrt(np.maximum(0, 1 - distance))
                        light = command[4:7]
                        rim_strength = command[7]
                        iridescence = command[8]
                    elif operation == PrimitiveOp.CYLINDER_PHONG:
                        dx, dy = xx - command[1], yy - command[2]
                        radial = np.sqrt(dx**2 + dy**2)
                        mask = radial <= command[3]
                        normal_x = dx / (radial + 1e-8)
                        normal_y = dy / (radial + 1e-8)
                        normal_z = np.zeros_like(normal_x)
                        light = command[4:7]
                        rim_strength = 0
                        iridescence = 0
                    else:
                        radius = command[3]
                        minor_ratio = command[4]
                        dx = (xx - command[1]) / radius
                        dy = (yy - command[2]) / radius
                        radial = np.sqrt(dx**2 + dy**2)
                        radial_delta = radial - 1
                        depth_squared = (
                            minor_ratio**2 - radial_delta**2
                        )
                        mask = (depth_squared >= 0) & (radial > 0.15)
                        normal_z = (
                            np.sqrt(np.maximum(0, depth_squared))
                            / minor_ratio
                        )
                        normal_radius = radial_delta / (radial + 1e-8)
                        normal_x = (
                            dx / (radial + 1e-8) * normal_radius
                        )
                        normal_y = (
                            dy / (radial + 1e-8) * normal_radius
                        )
                        light = command[5:8]
                        rim_strength = command[8]
                        iridescence = command[12]
                    lighting = implementation._phong(
                        normal_x, normal_y, normal_z, light
                    )
                    rim = (
                        (1 - np.abs(normal_z)) ** 3 * rim_strength
                    )
                    for channel in range(sig.C):
                        value = color[channel] * lighting
                        if iridescence:
                            value = value * (
                                0.62
                                + 0.5
                                * np.sin(
                                    normal_z * 5.5
                                    + channel * 2.1
                                    + iridescence
                                )
                            )
                        value = value + rim * color[channel]
                        rendered[:, :, channel][mask] = np.clip(
                            value[mask], 0, 1
                        )
                elif operation == PrimitiveOp.MAX_RECT:
                    x0 = max(0, int(np.ceil(command[1])))
                    y0 = max(0, int(np.ceil(command[2])))
                    x1 = min(sig.W - 1, int(np.floor(command[3])))
                    y1 = min(sig.H - 1, int(np.floor(command[4])))
                    rendered[y0 : y1 + 1, x0 : x1 + 1] = np.maximum(
                        rendered[y0 : y1 + 1, x0 : x1 + 1], color
                    )
                else:
                    raise AssertionError(f"unexpected operation {operation}")
            portable_control = implementation._finish_primitive_item_task(
                (idx, rendered, rng, None, None)
            )
            legacy = sig.make_image(idx)
            if level < 4:
                assert np.array_equal(portable_control, legacy)
            else:
                assert np.allclose(
                    portable_control, legacy, rtol=1e-5, atol=1e-5
                ), (level, sample, float(np.max(np.abs(portable_control - legacy))))


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
    with pytest.raises(ValueError):
        sig.make_image(-1)
    with pytest.raises(TypeError):
        sig.make_image(1.5)
    with pytest.raises(ValueError):
        sig.make_image(42, force_level=sig.N_LEVELS)
    with pytest.raises(TypeError):
        sig.make_image(42, force_level=3.5)
    with pytest.raises(TypeError):
        sig.make_image(42, step=True)
    with pytest.raises(TypeError):
        sig.make_image(42, step=1.5)
    with pytest.raises(ValueError):
        sig.make_image(42, step=0)


def test_threaded_generation_matches_serial_baseline():
    levels = (0, 15, 67, 72, 80, 108, 113, 128, 137, 148)
    indices = [sig.level_start(level) + 3 for level in levels] * 4
    baseline = {idx: sig.make_image(idx) for idx in set(indices)}

    with ThreadPoolExecutor(max_workers=8) as executor:
        threaded = list(executor.map(sig.make_image, indices))

    for idx, image in zip(indices, threaded):
        assert np.array_equal(image, baseline[idx])
    assert (sig.H, sig.W, sig.C) == (32, 32, 3)


def test_threaded_scene_and_supersampling_do_not_cross_contaminate():
    scene = sig.SceneSpec(
        background=sig.Background(kind="transparent"),
        objects=[sig.ObjectSpec(kind="sphere_3d", radius=7)],
    )
    raster = sig.RasterSpec(width=48, height=24, mode="rgba")
    scene_baseline = sig.make_scene(scene, seed=9, raster=raster)
    indexed_idx = sig.level_start(80) + 7
    indexed_baseline = sig.make_image(indexed_idx)

    with ThreadPoolExecutor(max_workers=8) as executor:
        scene_futures = [
            executor.submit(sig.make_scene, scene, 9, raster) for _ in range(12)
        ]
        indexed_futures = [
            executor.submit(sig.make_image, indexed_idx) for _ in range(12)
        ]

    assert all(
        np.array_equal(future.result(), scene_baseline)
        for future in scene_futures
    )
    assert all(
        np.array_equal(future.result(), indexed_baseline)
        for future in indexed_futures
    )


def test_batch_backends_preserve_order_and_values():
    indices = [sig.level_start(level) + 5 for level in (0, 15, 67, 80, 113, 148)]
    expected = np.stack([sig.make_image(idx) for idx in indices])

    serial = sig.make_images(indices, backend="serial")
    parallel = sig.make_images(
        indices,
        backend="process",
        workers=2,
        chunksize=2,
        start_method="spawn",
    )
    empty = sig.make_images([], backend="serial")
    empty_rgba = sig.make_images(
        [],
        backend="serial",
        raster=sig.RasterSpec(width=11, height=7, mode="rgba"),
    )
    empty_binary = sig.make_images(
        [],
        backend="serial",
        raster=sig.RasterSpec(
            width=13,
            height=5,
            mode="binary",
            bits_per_channel=1,
            packed=True,
        ),
    )

    assert np.array_equal(serial, expected)
    assert np.array_equal(parallel, expected)
    assert empty.shape == (0, 32, 32, 3)
    assert empty_rgba.shape == (0, 7, 11, 4)
    assert empty_rgba.dtype == np.uint8
    assert empty_binary.shape == (0, 5, 2)
    assert empty_binary.dtype == np.uint8


def test_integer_alpha_and_stricter_raster_validation():
    rgb8 = np.array([[[0, 0, 0], [128, 128, 128], [255, 255, 255]]], np.uint8)
    alpha = sig.extract_alpha(rgb8, "luminance")

    assert np.allclose(alpha, [[0.0, 128 / 255, 1.0]])
    with pytest.raises(ValueError, match="packed grayscale"):
        sig.convert_raster(
            rgb8,
            sig.RasterSpec(
                width=3,
                height=1,
                mode="grayscale",
                bits_per_channel=8,
                packed=True,
            ),
        )


def test_wave_video_and_audio_render_consistently_with_theta():
    from synthetic_image_generator import wave

    lvl, (n, mode) = next(iter(wave.WAVE_LEVELS.items()))
    th = wave.theta(lvl * 325)

    video = wave.render_video(th, n, mode, n_frames=8)
    audio = wave.render_audio(th, n, mode, n_samples=4000)

    assert video.shape == (32, 32, 3, 8)
    assert video.dtype == np.float32
    assert -1e-6 <= video.min() and video.max() <= 1 + 1e-6
    assert np.abs(np.diff(video, axis=-1)).mean() > 1e-3  # waves actually travel

    assert audio.shape == (4000,)
    assert audio.dtype == np.float32
    assert -1 - 1e-6 <= audio.min() and audio.max() <= 1 + 1e-6
    assert np.abs(audio).max() > 0.3  # not silent

    # same theta -> same output, every time
    assert np.array_equal(video, wave.render_video(th, n, mode, n_frames=8))
    assert np.array_equal(audio, wave.render_audio(th, n, mode, n_samples=4000))


def test_raster_floyd_steinberg_dither_with_bicubic_resize():
    image = sig.make_image(123)
    raster = sig.RasterSpec(
        width=17, height=11, mode="rgb", bits_per_channel=4,
        resize="bicubic", dither="floyd_steinberg",
    )
    result = sig.convert_raster(image, raster)

    assert result.shape == (11, 17, 3)
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 15  # 4 bits/channel

    # dithering must actually diffuse error: "none" would band identical
    # regions to the same flat level, floyd_steinberg breaks that up.
    none_raster = sig.RasterSpec(
        width=17, height=11, mode="rgb", bits_per_channel=4,
        resize="bicubic", dither="none",
    )
    result_none = sig.convert_raster(image, none_raster)
    assert not np.array_equal(result, result_none)

    # deterministic given the same inputs
    assert np.array_equal(result, sig.convert_raster(image, raster))


def test_extract_alpha_background_matte_separates_subject_from_backdrop():
    sprite_level = 90
    image = sig.make_image(0, force_level=sprite_level)

    matte = sig.extract_alpha(image, "background", level=sprite_level)

    assert matte.shape == (32, 32)
    assert matte.dtype == np.float32
    assert 0.0 <= matte.min() and matte.max() <= 1.0
    assert matte.std() > 0.01  # not a flat/trivial mask

    # opaque and luminance modes must disagree with the background matte on
    # the same image -- otherwise "background" isn't doing anything distinct.
    opaque = sig.extract_alpha(image, "opaque")
    assert not np.array_equal(matte, opaque)


def test_scene_validation_rejects_bad_seeds_and_cycles():
    with pytest.raises(TypeError):
        sig.make_scene(sig.SceneSpec(), seed=True)
    with pytest.raises(ValueError):
        sig.make_scene(sig.SceneSpec(), seed=-1)

    parent = sig.ObjectSpec(kind="disc")
    parent.children.append(parent)
    with pytest.raises(ValueError, match="reference cycle"):
        sig.make_scene(sig.SceneSpec(objects=[parent]))
