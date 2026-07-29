"""Export a few interactive `export_html` demos plus a browsable gallery.

Mirrors the media/ output convention of `render_generator_atlas.py`, but for
the scene-graph layer's web export instead of indexed-image atlases. Each
demo is a self-contained, single-file interactive HTML page (arrow keys /
WASD move any object tagged "player"); the gallery embeds them all in one
page via <iframe> for a quick visual check.
"""

from pathlib import Path

from synthetic_image_generator import (
    Background,
    ObjectSpec,
    SceneGraph,
    SceneSpec,
    export_html,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "media" / "scene_graph_demos"


def _graph(objects):
    return SceneGraph(scene_spec=SceneSpec(background=Background(kind="solid"), objects=objects))


def demo_platformer():
    ground = ObjectSpec(kind="box_3d", x=320, y=460, width=640, height=40, color=(0.3, 0.3, 0.35), tags={"ground"})
    platforms = [
        ObjectSpec(kind="box_3d", x=150, y=340, width=120, height=18, color=(0.55, 0.4, 0.25), tags={"platform"}),
        ObjectSpec(kind="box_3d", x=340, y=260, width=120, height=18, color=(0.55, 0.4, 0.25), tags={"platform"}),
        ObjectSpec(kind="box_3d", x=520, y=180, width=120, height=18, color=(0.55, 0.4, 0.25), tags={"platform"}),
    ]
    player = ObjectSpec(kind="sphere_3d", x=60, y=420, radius=14, color=(0.2, 0.9, 0.3), tags={"player"}, name="hero")
    enemy = ObjectSpec(kind="sphere_3d", x=520, y=140, radius=12, color=(0.9, 0.2, 0.2), tags={"enemy"})
    return _graph([ground, *platforms, player, enemy])


def demo_shapes_reference():
    objects = []
    colors = [(0.9, 0.2, 0.2), (0.2, 0.6, 0.9), (0.9, 0.8, 0.2), (0.4, 0.9, 0.4), (0.7, 0.3, 0.9)]
    for i, color in enumerate(colors):
        objects.append(ObjectSpec(kind="sphere_3d", x=60 + i * 90, y=90, radius=18 + i * 4, color=color, tags={"circle"}))
        objects.append(
            ObjectSpec(kind="box_3d", x=60 + i * 90, y=220, width=30 + i * 6, height=30 + i * 6, color=color, tags={"rect"})
        )
    return _graph(objects)


def demo_top_down_arena():
    wall_color = (0.25, 0.25, 0.3)
    walls = [
        ObjectSpec(kind="box_3d", x=240, y=10, width=480, height=20, color=wall_color, tags={"wall"}),
        ObjectSpec(kind="box_3d", x=240, y=310, width=480, height=20, color=wall_color, tags={"wall"}),
        ObjectSpec(kind="box_3d", x=10, y=160, width=20, height=320, color=wall_color, tags={"wall"}),
        ObjectSpec(kind="box_3d", x=470, y=160, width=20, height=320, color=wall_color, tags={"wall"}),
        ObjectSpec(kind="box_3d", x=240, y=160, width=100, height=20, color=wall_color, tags={"wall"}),
    ]
    player = ObjectSpec(kind="sphere_3d", x=100, y=250, radius=12, color=(0.2, 0.9, 0.3), tags={"player"}, name="hero")
    pickups = [
        ObjectSpec(kind="sphere_3d", x=380, y=60, radius=6, color=(1.0, 0.85, 0.2), tags={"pickup"}),
        ObjectSpec(kind="sphere_3d", x=380, y=250, radius=6, color=(1.0, 0.85, 0.2), tags={"pickup"}),
    ]
    return _graph([*walls, player, *pickups])


DEMOS = {
    "platformer": ("Platformer layout", demo_platformer, 640, 480),
    "shapes_reference": ("Shape/color reference (no player)", demo_shapes_reference, 480, 260),
    "top_down_arena": ("Top-down arena", demo_top_down_arena, 480, 320),
}


def render(output_dir: Path = OUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for slug, (title, builder, width, height) in DEMOS.items():
        path = output_dir / f"{slug}.html"
        export_html(builder(), str(path), width=width, height=height, title=title)
        cards.append((slug, title, width, height))

    rows = "\n".join(
        f"""    <section>
      <h2>{title}</h2>
      <iframe src="{slug}.html" width="{width}" height="{height}" style="border:1px solid #444"></iframe>
    </section>"""
        for slug, title, width, height in cards
    )
    index = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Scene-graph export_html demos</title>
<style>
  body {{ background:#111; color:#eee; font-family:sans-serif; padding:24px; }}
  section {{ margin-bottom:32px; }}
  h2 {{ font-size:16px; font-weight:600; }}
  p.hint {{ color:#999; font-size:13px; }}
</style>
</head>
<body>
<h1>Scene-graph export_html demos</h1>
<p class="hint">Click a frame, then use arrow keys / WASD — only the green "player"-tagged object moves. This is the flat-shaded 2D proxy exporter (not the Phong 3D renderer); see README.md's "Scene graph / mini game engine" section.</p>
{rows}
</body>
</html>
"""
    (output_dir / "index.html").write_text(index)
    return output_dir / "index.html"


if __name__ == "__main__":
    index_path = render()
    print(f"OK — wrote {len(DEMOS)} demos + gallery to {index_path}")
