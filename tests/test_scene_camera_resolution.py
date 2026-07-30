import numpy as np
import synthetic_image_generator as sig


def test_scene_graph_supports_high_resolution_moving_viewport():
    obj = sig.ObjectSpec(kind="sphere_3d", name="landmark", x=300, y=220, radius=24, color=(.2, .8, .4))
    graph = sig.SceneGraph(
        scene_spec=sig.SceneSpec(objects=[obj]),
        camera=sig.CameraSpec(viewport_width=512, viewport_height=512, world_x=300, world_y=220),
    )
    frame = graph.render()
    assert frame.shape == (512, 512, 4)
    assert frame.dtype == np.uint8
    center_alpha = frame[256, 256, 3]
    assert center_alpha > 0
    graph.camera.world_x = 700
    moved = graph.render()
    assert not np.array_equal(moved[256, 256, :3], frame[256, 256, :3])
