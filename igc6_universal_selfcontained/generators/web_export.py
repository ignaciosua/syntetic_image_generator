"""Single self-contained HTML file exporter.

ponytail: exports objects as flat-shaded 2D proxy shapes (circle/rect) driven
by a small inline JS loop, not a JS port of the Phong-shaded 3D renderer —
reimplementing that renderer in JS is a project of its own with no caller
yet. Objects tagged ``"player"`` are arrow-key/WASD-movable; that's the
extent of client-side interactivity. If you need pixel-identical visuals,
pre-render frames with ``make_scene`` server-side instead of using this
export. No build tools, no npm, one `<script>` tag, drop in a browser.
"""

from __future__ import annotations

import json

from .scene_spec import CollisionShape, ObjectSpec

_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>html,body{{margin:0;background:#111}}canvas{{display:block;margin:0 auto;background:#222}}</style>
</head>
<body>
<canvas id="c" width="{width}" height="{height}"></canvas>
<script>
const OBJECTS = {objects_json};
const AUTO_PLAY = {auto_play_json};
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const keys = new Set();
addEventListener("keydown", e => keys.add(e.key.toLowerCase()));
addEventListener("keyup", e => keys.delete(e.key.toLowerCase()));

const SPEED = 80; // px/s

function update(dt) {{
  let dx = 0, dy = 0;
  if (keys.has("arrowleft") || keys.has("a")) dx -= 1;
  if (keys.has("arrowright") || keys.has("d")) dx += 1;
  if (keys.has("arrowup") || keys.has("w")) dy -= 1;
  if (keys.has("arrowdown") || keys.has("s")) dy += 1;
  for (const o of OBJECTS) {{
    if (o.tags.includes("player")) {{
      o.x += dx * SPEED * dt;
      o.y += dy * SPEED * dt;
    }}
  }}
}}

function render() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const o of OBJECTS) {{
    if (!o.visible) continue;
    ctx.fillStyle = o.color;
    if (o.shape === "circle") {{
      ctx.beginPath();
      ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2);
      ctx.fill();
    }} else {{
      ctx.fillRect(o.x - o.w / 2, o.y - o.h / 2, o.w, o.h);
    }}
  }}
}}

let last = performance.now();
function frame(now) {{
  const dt = Math.min((now - last) / 1000, 0.25);
  last = now;
  update(dt);
  render();
  if (AUTO_PLAY) requestAnimationFrame(frame);
}}
render();
if (AUTO_PLAY) requestAnimationFrame(frame);
</script>
</body>
</html>
"""


def _object_to_json(obj: ObjectSpec) -> dict:
    is_circle = obj.collision_shape is CollisionShape.CIRCLE or obj.radius and not obj.width
    color = obj.color or (0.8, 0.8, 0.8)
    css_color = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
    entry = {
        "x": obj.resolved_x,
        "y": obj.resolved_y,
        "color": css_color,
        "visible": obj.visible,
        "tags": sorted(obj.tags),
        "shape": "circle" if is_circle else "rect",
    }
    if is_circle:
        entry["r"] = obj.radius
    else:
        entry["w"] = obj.width if obj.width is not None else obj.size
        entry["h"] = obj.height if obj.height is not None else obj.size
    return entry


def export_html(
    scene_graph,
    path: str,
    width: int = 640,
    height: int = 480,
    title: str = "Scene",
    auto_play: bool = True,
) -> None:
    """Write a single self-contained interactive HTML file for ``scene_graph``."""

    objects = [_object_to_json(obj) for obj in scene_graph.objects]
    html = _TEMPLATE.format(
        title=title,
        width=width,
        height=height,
        objects_json=json.dumps(objects),
        auto_play_json=json.dumps(bool(auto_play)),
    )
    with open(path, "w") as fh:
        fh.write(html)


if __name__ == "__main__":
    import tempfile

    class _FakeGraph:
        objects = [
            ObjectSpec(kind="sphere_3d", x=10, y=10, radius=4, color=(1, 0, 0), tags={"player"}),
            ObjectSpec(kind="box_3d", x=50, y=50, size=8, color=(0, 1, 0)),
        ]

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        path = tmp.name
    export_html(_FakeGraph(), path, width=320, height=240, title="Test Scene")

    with open(path) as fh:
        content = fh.read()
    assert "<canvas" in content and "Test Scene" in content
    assert '"player"' in content
    assert content.count('"shape"') == 2

    print("OK — export_html writes a self-contained interactive HTML file")
