import json
import numpy as np
import synthetic_image_generator as sig


def test_scene_spec_json_roundtrip_renders_identically():
    scene = sig.SceneSpec(
        objects=[sig.ObjectSpec(kind="sphere_3d", name="ball", x=12, y=14,
                                radius=4, color=(.2, .5, .8),
                                material=sig.MaterialSpec("wood", seed=4),
                                children=[sig.ObjectSpec(kind="disc", x=12, y=20, radius=2)])],
        layout=sig.LayoutSpec((sig.LayoutRelation("above", "ball", "missing"),)),
    )
    restored = sig.scene_from_dict(json.loads(json.dumps(sig.scene_to_dict(scene))))
    assert np.array_equal(sig.make_scene(scene), sig.make_scene(restored))
    assert restored.objects[0].children[0].kind == "disc"


def test_scene_graph_snapshot_restores_clock_and_body_state():
    obj = sig.ObjectSpec(kind="sphere_3d", name="ball", x=4, y=4,
                         radius=2, collision_shape=sig.CollisionShape.CIRCLE)
    graph = sig.SceneGraph(scene_spec=sig.SceneSpec(objects=[obj]), seed=3)
    graph.clock.tick(1 / 30)
    checkpoint = graph.snapshot()
    obj.x = 25
    graph.clock.elapsed = 99
    graph.restore_snapshot(checkpoint)
    assert graph.objects[0].x == 4
    assert graph.clock.elapsed < 1


def test_scene_graph_snapshot_restores_camera_and_particles():
    emitter = sig.ParticleEmitterSpec(max_particles=4, lifetime=(2.0, 2.0), speed=(0.0, 0.0))
    pool = sig.ParticlePool(emitter, origin=(3.0, 3.0))
    pool.emit(2, (3.0, 3.0), np.random.RandomState(1))
    graph = sig.SceneGraph(scene_spec=sig.SceneSpec(), particles=[pool])
    graph.camera.zoom = 2.5
    checkpoint = graph.snapshot()
    graph.camera.zoom = 0.2
    pool.positions[:] = 99
    graph.restore_snapshot(checkpoint)
    assert graph.camera.zoom == 2.5
    assert np.array_equal(pool.positions, np.asarray(checkpoint["particles"][0]["positions"], np.float32))
