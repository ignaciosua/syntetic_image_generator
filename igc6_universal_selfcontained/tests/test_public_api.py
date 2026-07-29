from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import synthetic_image_generator as sig


def test_public_indexed_contract():
    first = sig.make_image(42)
    second = sig.make_image(42)

    assert isinstance(sig.__version__, str) and len(sig.__version__) > 0
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
        "BoundingBox",
        "CollisionShape",
        "LightSpec",
        "ObjectSpec",
        "PostSpec",
        "RasterSpec",
        "SceneQuery",
        "SceneSpec",
        "check_collision",
        "convert_raster",
        "cpu_info",
        "accelerator_info",
        "extract_alpha",
        "find_collisions",
        "level_block",
        "level_of",
        "level_start",
        "make_image",
        "make_images",
        "make_image_raster",
        "make_scene",
        "make_scene_raster",
        "object_aabb",
        # scene-graph/mini-engine phases 2-13
        "CameraSpec",
        "world_to_screen",
        "screen_to_world",
        "aabb_in_view",
        "Layer",
        "LayerManager",
        "AtlasRegion",
        "AtlasSpec",
        "FlipbookClip",
        "stamp_sprite",
        "flipbook_region",
        "ParticleEmitterSpec",
        "ParticlePool",
        "RigidBodySpec",
        "PhysicsWorld",
        "Contact",
        "ContactHandler",
        "RayCastHit",
        "InputState",
        "InputProvider",
        "NullInputProvider",
        "ReplayInputProvider",
        "WebInputProvider",
        "PygameInputProvider",
        "GameClock",
        "SceneGraph",
        "WidgetSpec",
        "HUD",
        "Tween",
        "AnimationTrack",
        "Keyframe",
        "KeyframeTrack",
        "TilemapSpec",
        "TilemapRenderer",
        "FrameRecord",
        "SessionRecorder",
        "SessionPlayer",
        "serialize_objects",
        "export_html",
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


def test_bounding_box_math():
    a = sig.BoundingBox(0, 0, 10, 10)
    b = sig.BoundingBox(5, 5, 15, 15)
    c = sig.BoundingBox(20, 20, 30, 30)

    assert a.overlaps(b) and b.overlaps(a)
    assert not a.overlaps(c)
    assert a.contains(5, 5)
    assert not a.contains(11, 5)
    assert a.area() == 100.0


def test_object_metadata_is_inert_for_rendering():
    def make(with_metadata: bool):
        obj_kwargs = dict(kind="sphere_3d", x=16, y=16, radius=8, color=(0.8, 0.3, 0.2))
        if with_metadata:
            obj_kwargs.update(
                tags={"player"},
                layer=3,
                name="hero",
                collision_shape=sig.CollisionShape.CIRCLE,
            )
        scene = sig.SceneSpec(objects=[sig.ObjectSpec(**obj_kwargs)])
        return sig.make_scene(scene, seed=7)

    plain = make(with_metadata=False)
    tagged = make(with_metadata=True)

    assert np.array_equal(plain, tagged)


def test_scene_query_at_point_in_rect_tagged_named():
    hero = sig.ObjectSpec(kind="sphere_3d", x=10, y=10, radius=4, tags={"player"}, name="hero")
    enemy = sig.ObjectSpec(kind="sphere_3d", x=12, y=10, radius=4, tags={"enemy"})
    child = sig.ObjectSpec(kind="disc", x=1, y=1, radius=1, tags={"pickup"}, name="coin")
    wall = sig.ObjectSpec(kind="box_3d", x=100, y=100, size=8, name="wall", children=[child])

    scene = sig.SceneSpec(objects=[hero, enemy, wall])
    query = sig.SceneQuery(scene)

    assert query.named("hero") is hero
    assert query.named("coin") is child  # nested children are flattened
    assert query.named("missing") is None
    assert query.tagged("enemy") == [enemy]
    assert hero in query.in_rect(0, 0, 20, 20)
    assert wall not in query.in_rect(0, 0, 20, 20)
    assert hero in query.at_point(10, 10)


def test_check_collision_and_find_collisions():
    circle_a = sig.ObjectSpec(
        kind="sphere_3d", x=0, y=0, radius=4, collision_shape=sig.CollisionShape.CIRCLE
    )
    circle_b = sig.ObjectSpec(
        kind="sphere_3d", x=6, y=0, radius=4, collision_shape=sig.CollisionShape.CIRCLE
    )
    circle_far = sig.ObjectSpec(
        kind="sphere_3d", x=100, y=100, radius=4, collision_shape=sig.CollisionShape.CIRCLE
    )
    box = sig.ObjectSpec(kind="box_3d", x=0, y=0, size=10, collision_shape=sig.CollisionShape.AABB)
    box_far = sig.ObjectSpec(
        kind="box_3d", x=200, y=200, size=10, collision_shape=sig.CollisionShape.AABB
    )
    passive = sig.ObjectSpec(kind="sphere_3d", x=0, y=0, radius=4)  # collision_shape=NONE

    assert sig.check_collision(circle_a, circle_b)
    assert not sig.check_collision(circle_a, circle_far)
    assert sig.check_collision(box, circle_a)
    assert not sig.check_collision(box, circle_far)
    assert not sig.check_collision(box, box_far)
    assert not sig.check_collision(circle_a, passive)

    pairs = sig.find_collisions([circle_a, circle_b, circle_far, box, box_far, passive])
    assert (circle_a, circle_b) in pairs or (circle_b, circle_a) in pairs
    assert (box, circle_a) in pairs or (circle_a, box) in pairs
    assert all(passive not in pair for pair in pairs)


def test_camera_world_screen_round_trip_and_culling():
    cam = sig.CameraSpec(viewport_width=640, viewport_height=480, world_x=100, world_y=50, zoom=2.0)
    sx, sy = sig.world_to_screen(100, 50, cam)
    assert (sx, sy) == (320.0, 240.0)
    wx, wy = sig.screen_to_world(sx, sy, cam)
    assert (round(wx, 6), round(wy, 6)) == (100.0, 50.0)

    assert sig.aabb_in_view(sig.BoundingBox(90, 40, 110, 60), cam)
    assert not sig.aabb_in_view(sig.BoundingBox(10_000, 10_000, 10_010, 10_010), cam)


def test_layer_manager_sorts_by_layer_then_depth():
    mgr = sig.LayerManager()
    mgr.add_layer("bg", order=0, parallax=0.5)
    mgr.add_layer("world", order=1, parallax=1.0)

    far = sig.ObjectSpec(kind="tree", depth=0.9, layer=1)
    near = sig.ObjectSpec(kind="tree", depth=0.2, layer=1)
    bg = sig.ObjectSpec(kind="tree", depth=0.5, layer=0)

    ordered = mgr.sort_objects([far, near, bg])
    assert ordered[0] is bg
    assert ordered[1] is far and ordered[2] is near  # far-to-near within a layer


def test_sprite_atlas_stamp_and_flipbook():
    atlas_img = np.zeros((16, 16, 4), np.uint8)
    atlas_img[0:8, 0:8] = (255, 0, 0, 255)
    region = sig.AtlasRegion(name="red", x=0, y=0, width=8, height=8)
    atlas = sig.AtlasSpec(image=atlas_img, regions=(region,), default_region="red")

    canvas = np.zeros((32, 32, 4), np.uint8)
    sig.stamp_sprite(canvas, atlas, atlas.get("red"), x=16, y=16)
    assert tuple(canvas[16, 16]) == (255, 0, 0, 255)
    assert tuple(canvas[0, 0]) == (0, 0, 0, 0)

    clip = sig.FlipbookClip(frames=(region, region), frame_duration_ms=100, loop=True)
    assert sig.flipbook_region(clip, 50) is clip.frames[0]
    assert sig.flipbook_region(clip, 250) is clip.frames[0]  # wraps


def test_particle_pool_emit_update_render_lifecycle():
    rng = np.random.RandomState(0)
    emitter = sig.ParticleEmitterSpec(lifetime=(1.0, 1.0), speed=(0, 0), max_particles=8)
    pool = sig.ParticlePool(emitter)

    assert pool.emit(5, (16, 16), rng) == 5
    pool.update(0.5)
    assert pool.alive.sum() == 5
    pool.update(0.6)  # ages exceed lifetime
    assert pool.alive.sum() == 0

    pool.emit(1, (16, 16), rng)
    canvas = np.zeros((32, 32, 4), np.uint8)
    pool.render(canvas)
    assert canvas[16, 16, 3] > 0


def test_physics_world_gravity_collision_and_raycast():
    ground = sig.ObjectSpec(kind="box_3d", x=0, y=0, size=20, collision_shape=sig.CollisionShape.AABB)
    ball = sig.ObjectSpec(kind="sphere_3d", x=0, y=5, radius=2, collision_shape=sig.CollisionShape.CIRCLE)
    world = sig.PhysicsWorld(gravity=(0, 0))
    world.add_body(ground, sig.RigidBodySpec(is_static=True))
    world.add_body(ball, sig.RigidBodySpec(mass=1.0, restitution=0.5))

    hit = world.ray_cast((0, 50), (0, -1), 100)
    assert hit is not None and hit.obj is ground

    contacts = world.step(0.016)
    assert len(contacts) == 1

    falling = sig.ObjectSpec(kind="sphere_3d", x=0, y=100)
    freefall = sig.PhysicsWorld(gravity=(0, -100))
    freefall.add_body(falling, sig.RigidBodySpec(mass=1.0))
    freefall.step(1.0)
    assert falling.y < 100


def test_input_providers_replay_and_web():
    frames = [sig.InputState(keys_down={"left"}), sig.InputState(keys_down={"right"})]
    replay = sig.ReplayInputProvider(frames)
    assert replay.poll().keys_down == {"left"}
    assert replay.poll().keys_down == {"right"}
    assert replay.exhausted

    web = sig.WebInputProvider()
    web.push_event({"type": "keydown", "key": "a"})
    state = web.poll()
    assert state.keys_down == {"a"} and state.keys_pressed == {"a"}
    assert web.poll().keys_pressed == set()  # transient set clears after poll

    assert sig.NullInputProvider().poll() == sig.InputState()


def test_game_clock_fixed_steps_and_scene_graph_tick():
    clock = sig.GameClock(fixed_timestep=1 / 60)
    assert clock.tick(1 / 30) == 2  # two 60Hz steps fit in one 1/30s frame
    assert clock.tick(10.0) == round(0.25 / (1 / 60))  # spiral-of-death guard

    scene = sig.SceneSpec(objects=[sig.ObjectSpec(kind="sphere_3d", x=16, y=16, radius=6)])
    graph = sig.SceneGraph(scene_spec=scene, camera=sig.CameraSpec(viewport_width=32, viewport_height=32))
    frame = graph.tick(1 / 60, sig.InputState())
    assert frame.shape == (32, 32, 4) and frame.dtype == np.uint8


def test_hud_renders_and_dispatches_click_on_press_edge():
    button = sig.WidgetSpec(kind="button", x=10, y=10, width=40, height=20, text="OK", on_click="confirm")
    hud = sig.HUD([button])

    canvas = np.zeros((64, 64, 4), np.uint8)
    hud.render(canvas)
    assert canvas[15, 15, 3] > 0

    click = sig.InputState(mouse_x=20, mouse_y=15, mouse_buttons={0})
    assert hud.handle_input(click) == ["confirm"]
    assert hud.handle_input(click) == []  # held, not a new press


def test_tween_and_keyframe_track():
    obj = sig.ObjectSpec(kind="sphere_3d", name="hero", x=0)
    scene = sig.SceneSpec(objects=[obj])
    track = sig.AnimationTrack([sig.Tween(target="hero", property="x", from_value=0, to_value=100, duration=1.0)])
    track.update(0.5, scene)
    assert obj.x == 50.0
    track.update(0.5, scene)
    assert obj.x == 100.0 and track.finished

    kf = sig.KeyframeTrack("hero", [sig.Keyframe(0, {"y": 0}), sig.Keyframe(2, {"y": 10})])
    assert kf.sample(1)["y"] == 5.0


def test_tilemap_storage_collision_mask_and_render():
    atlas_img = np.zeros((8, 16, 4), np.uint8)
    atlas_img[:, 0:8] = (255, 0, 0, 255)
    atlas_img[:, 8:16] = (0, 255, 0, 255)
    tileset = sig.AtlasSpec(
        image=atlas_img,
        regions=(
            sig.AtlasRegion(name="grass", x=0, y=0, width=8, height=8),
            sig.AtlasRegion(name="water", x=8, y=0, width=8, height=8),
        ),
    )
    tiles = np.zeros((4, 4), np.uint16)
    tiles[1, 1] = 1
    tilemap = sig.TilemapSpec(width=4, height=4, tile_size=8, tiles=tiles, tileset=tileset)

    assert tilemap.tile_at(1, 1) == 1
    tilemap.set_tile(0, 0, 2)
    assert tilemap.collision_mask()[0, 0] and not tilemap.collision_mask()[3, 3]

    cam = sig.CameraSpec(viewport_width=32, viewport_height=32, world_x=16, world_y=16, zoom=1.0)
    frame = sig.TilemapRenderer.render(tilemap, cam)
    assert frame.shape == (32, 32, 4)
    assert frame[..., 3].sum() > 0


def test_session_record_save_load_round_trip(tmp_path):
    obj = sig.ObjectSpec(kind="sphere_3d", x=1.0, y=2.0, name="hero")
    snap = sig.serialize_objects([obj])

    recorder = sig.SessionRecorder()
    recorder.record(sig.FrameRecord(frame=0, time=0.0, input=sig.InputState(keys_down={"left"}), objects_state=snap))

    path = tmp_path / "session.json"
    recorder.save(str(path))
    loaded = sig.SessionRecorder.load(str(path))

    assert len(loaded.frames) == 1
    assert loaded.frames[0].input.keys_down == {"left"}
    assert loaded.frames[0].objects_state == snap


def test_export_html_writes_self_contained_interactive_file(tmp_path):
    class FakeGraph:
        objects = [
            sig.ObjectSpec(kind="sphere_3d", x=10, y=10, radius=4, color=(1, 0, 0), tags={"player"}),
            sig.ObjectSpec(kind="box_3d", x=50, y=50, size=8, color=(0, 1, 0)),
        ]

    path = tmp_path / "scene.html"
    sig.export_html(FakeGraph(), str(path), width=320, height=240, title="Test Scene")

    content = path.read_text()
    assert "<canvas" in content and "Test Scene" in content
    assert '"player"' in content


def test_scene_graph_end_to_end_determinism_and_cx_cy_normalization():
    def build_graph():
        ground = sig.ObjectSpec(
            kind="box_3d", cx=16, cy=4, size=32, collision_shape=sig.CollisionShape.AABB, tags={"ground"}
        )
        ball = sig.ObjectSpec(
            kind="sphere_3d", x=16, y=28, radius=3, collision_shape=sig.CollisionShape.CIRCLE, name="ball"
        )
        scene = sig.SceneSpec(objects=[ground, ball])
        physics = sig.PhysicsWorld(gravity=(0, -50))
        physics.add_body(ground, sig.RigidBodySpec(is_static=True))
        physics.add_body(ball, sig.RigidBodySpec(mass=1.0, restitution=0.2))
        graph = sig.SceneGraph(
            scene_spec=scene,
            camera=sig.CameraSpec(viewport_width=32, viewport_height=32),
            physics=physics,
            seed=3,
        )
        return graph, ground, ball

    graph_a, ground_a, ball_a = build_graph()
    graph_b, ground_b, ball_b = build_graph()

    # ground was built with cx/cy; add_body must normalize it into x/y or
    # physics would silently move a field resolved_x/y never reads.
    assert ground_a.cx is None and ground_a.x == 16.0

    frames_a = [graph_a.tick(1 / 60, sig.InputState()) for _ in range(5)]
    frames_b = [graph_b.tick(1 / 60, sig.InputState()) for _ in range(5)]

    for fa, fb in zip(frames_a, frames_b):
        assert np.array_equal(fa, fb)
    assert ball_a.x == ball_b.x and ball_a.y == ball_b.y
    assert ball_a.y < 28  # gravity actually moved it


def test_scene_graph_composes_tilemap_sprites_and_hud_in_one_frame():
    tile_img = np.zeros((8, 8, 4), np.uint8)
    tile_img[:, :] = (255, 0, 0, 255)
    tile_region = sig.AtlasRegion(name="tile", x=0, y=0, width=8, height=8)
    tile_atlas = sig.AtlasSpec(image=tile_img, regions=(tile_region,), default_region="tile")
    tiles = np.ones((4, 4), np.uint16)
    tilemap = sig.TilemapSpec(width=4, height=4, tile_size=8, tiles=tiles, tileset=tile_atlas)

    sprite_img = np.zeros((8, 8, 4), np.uint8)
    sprite_img[:, :] = (0, 255, 0, 255)
    sprite_region = sig.AtlasRegion(name="hero", x=0, y=0, width=8, height=8)
    sprite_atlas = sig.AtlasSpec(image=sprite_img, regions=(sprite_region,), default_region="hero")

    hero = sig.ObjectSpec(kind="sphere_3d", x=16, y=16, sprite=sprite_region, name="hero")
    scene = sig.SceneSpec(background=sig.Background(kind="none"), objects=[hero])
    hud = sig.HUD([sig.WidgetSpec(kind="rect", x=0, y=0, width=6, height=6, color=(1, 1, 1))])

    graph = sig.SceneGraph(
        scene_spec=scene,
        camera=sig.CameraSpec(viewport_width=32, viewport_height=32, world_x=16, world_y=16),
        atlas=sprite_atlas,
        tilemaps=[tilemap],
        hud=hud,
    )
    frame = graph.tick(1 / 60, sig.InputState())

    assert frame.shape == (32, 32, 4)
    assert tuple(frame[16, 16][:3]) == (0, 255, 0)  # sprite wins over the tile beneath it
    assert tuple(frame[28, 28][:3]) == (255, 0, 0)  # tile still shows where nothing sits on top
    assert tuple(frame[0, 0][:3]) == (255, 255, 255)  # HUD wins over everything


def test_scene_graph_update_drives_animations_without_manual_wiring():
    hero = sig.ObjectSpec(kind="sphere_3d", name="hero", x=0, y=16, radius=4)
    scene = sig.SceneSpec(objects=[hero])
    track = sig.AnimationTrack([sig.Tween(target="hero", property="x", from_value=0, to_value=32, duration=1.0)])
    graph = sig.SceneGraph(
        scene_spec=scene, camera=sig.CameraSpec(viewport_width=32, viewport_height=32), animations=[track]
    )

    graph.update(0.5, sig.InputState())
    assert hero.x == 16.0
    graph.update(0.5, sig.InputState())
    assert hero.x == 32.0 and track.finished


def test_scene_graph_update_auto_emits_particles_with_origin_deterministically():
    emitter = sig.ParticleEmitterSpec(rate=100.0, lifetime=(1.0, 1.0), speed=(0, 0), max_particles=8)
    pool = sig.ParticlePool(emitter, origin=(16.0, 16.0))
    graph = sig.SceneGraph(scene_spec=sig.SceneSpec(), particles=[pool])

    graph.update(0.5, sig.InputState())
    assert pool.alive.sum() > 0  # emitted with no manual pool.emit*() call

    pool_a = sig.ParticlePool(sig.ParticleEmitterSpec(rate=50.0, max_particles=16), origin=(0.0, 0.0))
    pool_b = sig.ParticlePool(sig.ParticleEmitterSpec(rate=50.0, max_particles=16), origin=(0.0, 0.0))
    graph_a = sig.SceneGraph(scene_spec=sig.SceneSpec(), particles=[pool_a], seed=5)
    graph_b = sig.SceneGraph(scene_spec=sig.SceneSpec(), particles=[pool_b], seed=5)
    graph_a.update(0.5, sig.InputState())
    graph_b.update(0.5, sig.InputState())
    assert np.array_equal(pool_a.positions, pool_b.positions)  # same seed -> same draws


def test_scene_graph_particle_origin_can_follow_a_named_object():
    hero = sig.ObjectSpec(kind="sphere_3d", name="hero", x=5, y=7, radius=2)
    emitter = sig.ParticleEmitterSpec(rate=100.0, lifetime=(1.0, 1.0), speed=(0, 0), max_particles=8)
    pool = sig.ParticlePool(emitter, origin="hero")
    graph = sig.SceneGraph(scene_spec=sig.SceneSpec(objects=[hero]), particles=[pool])

    graph.update(0.5, sig.InputState())
    alive = pool.alive
    assert alive.any()
    assert np.allclose(pool.positions[alive][0], (5.0, 7.0))
