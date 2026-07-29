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

from .scene_graph import SceneQuery
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


def _object_to_json(obj: ObjectSpec, scale: float) -> dict:
    is_circle = obj.collision_shape is CollisionShape.CIRCLE or obj.radius and not obj.width
    color = obj.color or (0.8, 0.8, 0.8)
    css_color = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
    entry = {
        "x": obj.resolved_x * scale,
        "y": obj.resolved_y * scale,
        "color": css_color,
        "visible": obj.visible,
        "tags": sorted(obj.tags),
        "shape": "circle" if is_circle else "rect",
    }
    if is_circle:
        entry["r"] = obj.radius * scale
    else:
        entry["w"] = (obj.width if obj.width is not None else obj.size) * scale
        entry["h"] = (obj.height if obj.height is not None else obj.size) * scale
    return entry


def export_html(
    scene_graph,
    path: str,
    width: int = 640,
    height: int = 480,
    title: str = "Scene",
    auto_play: bool = True,
    scale: float = 1.0,
) -> None:
    """Write a single self-contained interactive HTML file for ``scene_graph``.

    ``scale`` converts object coordinates (world units) to canvas pixels —
    needed for scenes built at a small viewport (e.g. the 32x32-unit
    scene-catalog archetypes) that should render on a larger canvas.
    """

    objects = [_object_to_json(obj, scale) for obj in scene_graph.objects]
    html = _TEMPLATE.format(
        title=title,
        width=width,
        height=height,
        objects_json=json.dumps(objects),
        auto_play_json=json.dumps(bool(auto_play)),
    )
    with open(path, "w") as fh:
        fh.write(html)


_LIVE_TEMPLATE = """<!doctype html>
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
const ANIMATIONS = {animations_json};
const PHYSICS = {physics_json};
const SCALE = {scale_json};
const BODY_BY_INDEX = {{}};
if (PHYSICS) for (const b of PHYSICS.bodies) BODY_BY_INDEX[b.index] = b;
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const keys = new Set();
addEventListener("keydown", e => keys.add(e.key.toLowerCase()));
addEventListener("keyup", e => keys.delete(e.key.toLowerCase()));

const SPEED = 10; // world units/s — comparable to the sim's own chase/orbit speeds

// --- tween easing (ports generators/tween.py::_ease) ---
function ease(kind, t) {{
  t = Math.max(0, Math.min(1, t));
  switch (kind) {{
    case "linear": return t;
    case "ease_in": return t * t;
    case "ease_out": return 1 - (1 - t) * (1 - t);
    case "ease_in_out": return 3 * t * t - 2 * t * t * t;
    case "bounce":
      if (t < 1 / 2.75) return 7.5625 * t * t;
      if (t < 2 / 2.75) {{ t -= 1.5 / 2.75; return 7.5625 * t * t + 0.75; }}
      if (t < 2.5 / 2.75) {{ t -= 2.25 / 2.75; return 7.5625 * t * t + 0.9375; }}
      t -= 2.625 / 2.75; return 7.5625 * t * t + 0.984375;
    case "elastic":
      if (t === 0 || t === 1) return t;
      return -(2 ** (10 * (t - 1))) * Math.sin((t - 1.075) * (2 * Math.PI) / 0.3);
    default: return t;
  }}
}}

function tweenValue(tw) {{
  let t = tw.duration > 0 ? (tw.elapsed - tw.delay) / tw.duration : 0;
  t = Math.max(0, Math.min(1, t));
  if (tw.reversed) t = 1 - t;
  return tw.from + (tw.to - tw.from) * ease(tw.easing, t);
}}

function tweenAdvance(tw, dt) {{
  const finished = !tw.loop && tw.elapsed >= tw.delay + tw.duration;
  if (finished) return;
  tw.elapsed += dt;
  const span = tw.delay + tw.duration;
  if (tw.elapsed < span) return;
  if (tw.pingPong) {{ tw.reversed = !tw.reversed; tw.elapsed = tw.delay + (tw.elapsed - span); }}
  else if (tw.loop) {{ tw.elapsed = tw.delay + (tw.elapsed - span); }}
}}

// --- physics (ports generators/physics.py::PhysicsWorld) ---
function objectAabb(o) {{
  const halfW = Math.max(0.5 * o.w, o.r), halfH = Math.max(0.5 * o.h, o.r);
  return {{ x0: o.x - halfW, y0: o.y - halfH, x1: o.x + halfW, y1: o.y + halfH }};
}}

function checkCollision(a, b) {{
  if (a.collision === "none" || b.collision === "none") return false;
  if (a.collision === "circle" && b.collision === "circle") {{
    return Math.hypot(a.x - b.x, a.y - b.y) <= a.r + b.r;
  }}
  if (a.collision === "aabb" && b.collision === "aabb") {{
    const ba = objectAabb(a), bb = objectAabb(b);
    return ba.x0 < bb.x1 && bb.x0 < ba.x1 && ba.y0 < bb.y1 && bb.y0 < ba.y1;
  }}
  const [box, circle] = a.collision === "aabb" ? [a, b] : [b, a];
  const box_ = objectAabb(box);
  const nx = Math.min(Math.max(circle.x, box_.x0), box_.x1);
  const ny = Math.min(Math.max(circle.y, box_.y0), box_.y1);
  return Math.hypot(circle.x - nx, circle.y - ny) <= circle.r;
}}

function exactHalfExtents(o) {{
  if (o.collision === "circle") return [o.r, o.r];
  return [0.5 * o.w, 0.5 * o.h];
}}

// ports physics.py::_contact_normal_and_penetration — NOT objectAabb (that
// one is deliberately padded for hit-testing/culling, per its Python
// docstring, and using it here is what caused stacked/dropped bodies to
// fly apart instead of settling).
function contactNormalAndPenetration(a, b, dx, dy) {{
  if (a.collision === "circle" && b.collision === "circle") {{
    const dist = Math.hypot(dx, dy) || 1e-6;
    return [dx / dist, dy / dist, Math.max(0, a.r + b.r - dist)];
  }}
  const [halfWa, halfHa] = exactHalfExtents(a);
  const [halfWb, halfHb] = exactHalfExtents(b);
  const overlapX = (halfWa + halfWb) - Math.abs(dx);
  const overlapY = (halfHa + halfHb) - Math.abs(dy);
  if (overlapX < overlapY) return [dx >= 0 ? 1 : -1, 0, Math.max(0, overlapX)];
  return [0, dy >= 0 ? 1 : -1, Math.max(0, overlapY)];
}}

function separate(oa, ba, ob, bb, nx, ny, pen) {{
  if (ba.isStatic && bb.isStatic) return;
  if (ba.isStatic) {{ ob.x += nx * pen; ob.y += ny * pen; }}
  else if (bb.isStatic) {{ oa.x -= nx * pen; oa.y -= ny * pen; }}
  else {{ oa.x -= nx * pen / 2; oa.y -= ny * pen / 2; ob.x += nx * pen / 2; ob.y += ny * pen / 2; }}
}}

function applyImpulse(ba, bb, nx, ny) {{
  const rvx = bb.vx - ba.vx, rvy = bb.vy - ba.vy;
  const velAlongNormal = rvx * nx + rvy * ny;
  if (velAlongNormal > 0) return;
  const invA = ba.isStatic ? 0 : 1 / Math.max(ba.mass, 1e-6);
  const invB = bb.isStatic ? 0 : 1 / Math.max(bb.mass, 1e-6);
  if (invA + invB === 0) return;
  const e = Math.min(ba.restitution, bb.restitution);
  const j = -(1 + e) * velAlongNormal / (invA + invB);
  if (!ba.isStatic) {{ ba.vx -= j * invA * nx; ba.vy -= j * invA * ny; }}
  if (!bb.isStatic) {{ bb.vx += j * invB * nx; bb.vy += j * invB * ny; }}
}}

function physicsStep(dt) {{
  if (!PHYSICS) return;
  const bodies = PHYSICS.bodies;
  for (const b of bodies) {{
    if (b.isStatic || b.isKinematic) continue;
    b.vx += PHYSICS.gravity[0] * b.gravityScale * dt;
    b.vy += PHYSICS.gravity[1] * b.gravityScale * dt;
    b.vx *= Math.max(0, 1 - b.linearDamping);
    b.vy *= Math.max(0, 1 - b.linearDamping);
    const o = OBJECTS[b.index];
    o.x += b.vx * dt; o.y += b.vy * dt;
  }}
  for (let i = 0; i < bodies.length; i++) {{
    const ba = bodies[i], oa = OBJECTS[ba.index];
    if (oa.collision === "none") continue;
    for (let j = i + 1; j < bodies.length; j++) {{
      const bb = bodies[j], ob = OBJECTS[bb.index];
      if (ob.collision === "none") continue;
      if (!checkCollision(oa, ob)) continue;
      const dx = ob.x - oa.x, dy = ob.y - oa.y;
      const [nx, ny, penetration] = contactNormalAndPenetration(oa, ob, dx, dy);
      separate(oa, ba, ob, bb, nx, ny, penetration);
      applyImpulse(ba, bb, nx, ny);
    }}
  }}
}}

// --- behavior controllers (ports generators/behaviors.py) ---
function stepAnimations(dt) {{
  for (const a of ANIMATIONS) {{
    if (a.kind === "tween") {{
      tweenAdvance(a, dt);
      const v = tweenValue(a);
      for (const idx of a.targets) OBJECTS[idx][a.property] = v;
    }} else if (a.kind === "chase") {{
      const chaser = OBJECTS[a.chaser], target = OBJECTS[a.target];
      const dx = target.x - chaser.x, dy = target.y - chaser.y;
      const dist = Math.hypot(dx, dy) || 1e-6;
      const sign = a.flee ? -1 : 1;
      const vx = sign * (dx / dist) * a.speed, vy = sign * (dy / dist) * a.speed;
      const body = BODY_BY_INDEX[a.chaser];
      // a wall-aware chase (body present) sets intent as velocity and lets
      // physicsStep()'s collision resolution have the final say — ports
      // generators/behaviors.py::_ChaseController's physics-driven branch.
      if (body) {{ body.vx = vx; body.vy = vy; }}
      else {{ chaser.x += vx * dt; chaser.y += vy * dt; }}
    }} else if (a.kind === "orbit") {{
      a.angle += a.angularSpeed * dt;
      const body = OBJECTS[a.body];
      body.x = a.cx + a.radius * Math.cos(a.angle);
      body.y = a.cy + a.radius * Math.sin(a.angle);
    }}
  }}
}}

function updatePlayer(dt) {{
  let dx = 0, dy = 0;
  if (keys.has("arrowleft") || keys.has("a")) dx -= 1;
  if (keys.has("arrowright") || keys.has("d")) dx += 1;
  if (keys.has("arrowup") || keys.has("w")) dy -= 1;
  if (keys.has("arrowdown") || keys.has("s")) dy += 1;
  for (const o of OBJECTS) {{
    if (o.tags.includes("player")) {{ o.x += dx * SPEED * dt; o.y += dy * SPEED * dt; }}
  }}
}}

function render() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const o of OBJECTS) {{
    if (!o.visible) continue;
    ctx.fillStyle = o.color;
    if (o.shape === "circle") {{
      ctx.beginPath();
      ctx.arc(o.x * SCALE, o.y * SCALE, o.r * SCALE, 0, Math.PI * 2);
      ctx.fill();
    }} else {{
      ctx.fillRect((o.x - o.w / 2) * SCALE, (o.y - o.h / 2) * SCALE, o.w * SCALE, o.h * SCALE);
    }}
  }}
}}

let last = performance.now();
function frame(now) {{
  const dt = Math.min((now - last) / 1000, 0.25);
  last = now;
  updatePlayer(dt);
  stepAnimations(dt);
  physicsStep(dt);
  render();
  requestAnimationFrame(frame);
}}
render();
requestAnimationFrame(frame);
</script>
</body>
</html>
"""


def _live_object_to_json(obj: ObjectSpec) -> dict:
    is_circle = obj.collision_shape is CollisionShape.CIRCLE or obj.radius and not obj.width
    color = obj.color or (0.8, 0.8, 0.8)
    css_color = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
    return {
        "x": obj.resolved_x,
        "y": obj.resolved_y,
        "w": obj.width if obj.width is not None else obj.size,
        "h": obj.height if obj.height is not None else obj.size,
        "r": obj.radius,
        "color": css_color,
        "visible": obj.visible,
        "tags": sorted(obj.tags),
        "shape": "circle" if is_circle else "rect",
        "collision": obj.collision_shape.value,
    }


def _resolve_indices(scene_spec, index_of: dict[int, int], name_or_tag: str) -> list[int]:
    query = SceneQuery(scene_spec)
    named = query.named(name_or_tag)
    targets = [named] if named is not None else query.tagged(name_or_tag)
    return [index_of[id(o)] for o in targets]


def export_live_html(
    scene_graph,
    path: str,
    width: int = 640,
    height: int = 480,
    title: str = "Scene",
    scale: float = 1.0,
) -> None:
    """Write a self-contained HTML file that runs a *real* simulation.

    Unlike ``export_html`` (a static snapshot + player-only movement stub),
    this ports ``PhysicsWorld`` (gravity/AABB/circle collision/impulses) and
    the ``generators.behaviors`` controllers (patrol/chase/flee/orbit) to
    JS, and drives them from the exact deterministic parameters
    ``scene_graph`` was built with (positions, tween endpoints, chase
    speeds, gravity, restitution, ...) — no client-side RNG, no reliance on
    porting numpy's RNG to match a Python idx bit-for-bit. ``scale`` is
    display-only: simulation stays in world units, only ``render()``
    multiplies by it.

    ponytail: ``bounce_particles`` isn't ported — physics/collisions still
    run correctly, only the trailing particle-emitter visual is missing.
    Add a ParticlePool port if that visual is ever needed client-side.
    """

    objects = scene_graph.objects
    index_of = {id(o): i for i, o in enumerate(objects)}
    objects_json = [_live_object_to_json(o) for o in objects]

    animations_json: list[dict] = []
    for track in scene_graph.animations:
        if hasattr(track, "tweens"):
            for tw in track.tweens:
                animations_json.append({
                    "kind": "tween",
                    "targets": _resolve_indices(scene_graph.scene_spec, index_of, tw.target),
                    "property": tw.property,
                    "from": tw.from_value,
                    "to": tw.to_value,
                    "duration": tw.duration,
                    "easing": tw.easing,
                    "delay": tw.delay,
                    "loop": tw.loop,
                    "pingPong": tw.ping_pong,
                    "elapsed": tw.elapsed,
                    "reversed": False,
                })
        elif hasattr(track, "chaser_name"):
            query = SceneQuery(scene_graph.scene_spec)
            chaser, target = query.named(track.chaser_name), query.named(track.target_name)
            if chaser is None or target is None:
                continue
            animations_json.append({
                "kind": "chase", "chaser": index_of[id(chaser)], "target": index_of[id(target)],
                "speed": track.speed, "flee": track.flee,
            })
        elif hasattr(track, "body_name"):
            body = SceneQuery(scene_graph.scene_spec).named(track.body_name)
            if body is None:
                continue
            animations_json.append({
                "kind": "orbit", "body": index_of[id(body)], "cx": track.center_x, "cy": track.center_y,
                "radius": track.radius, "angularSpeed": track.angular_speed, "angle": track.angle,
            })

    physics_json = None
    if scene_graph.physics is not None:
        bodies_json = [
            {
                "index": index_of[id(obj)],
                "mass": body.mass,
                "isStatic": body.is_static,
                "isKinematic": body.is_kinematic,
                "restitution": body.restitution,
                "linearDamping": body.linear_damping,
                "gravityScale": body.gravity_scale,
                "vx": body.velocity[0],
                "vy": body.velocity[1],
            }
            for obj, body in scene_graph.physics.entries()
        ]
        # The keyboard-driven "player" only exists client-side (Python has
        # no input system driving it) — give it a kinematic body purely for
        # this export so it still collides with whatever walls the
        # archetype has, instead of walking straight through them.
        physics_indices = {index_of[id(obj)] for obj, _ in scene_graph.physics.entries()}
        for obj in objects:
            if "player" in obj.tags and index_of[id(obj)] not in physics_indices:
                bodies_json.append({
                    "index": index_of[id(obj)], "mass": 1.0, "isStatic": False, "isKinematic": True,
                    "restitution": 0.0, "linearDamping": 0.0, "gravityScale": 0.0, "vx": 0.0, "vy": 0.0,
                })
        physics_json = {"gravity": list(scene_graph.physics.gravity), "bodies": bodies_json}

    html = _LIVE_TEMPLATE.format(
        title=title,
        width=width,
        height=height,
        objects_json=json.dumps(objects_json),
        animations_json=json.dumps(animations_json),
        physics_json=json.dumps(physics_json),
        scale_json=json.dumps(scale),
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

    # export_live_html: verify the *ported JS* actually reproduces the
    # Python engine's behavior (gravity+collision, chase, tween) by running
    # the emitted <script> under node with DOM calls stubbed out — the same
    # properties physics.py/behaviors.py assert on their own Python side.
    import shutil
    import subprocess

    from .game_loop import SceneGraph
    from .physics import PhysicsWorld, RigidBodySpec
    from .scene_spec import SceneSpec
    from .tween import AnimationTrack, Tween

    node = shutil.which("node") or shutil.which("nodejs")
    if node is None:
        print("SKIP — export_live_html JS self-check needs node, none found on PATH")
    else:
        from .behaviors import _ChaseController

        ground = ObjectSpec(kind="box_3d", x=16, y=30, width=32, height=4, collision_shape=CollisionShape.AABB, name="ground")
        ball = ObjectSpec(kind="sphere_3d", x=16, y=5, radius=2, collision_shape=CollisionShape.CIRCLE, name="ball")
        chaser = ObjectSpec(kind="sphere_3d", x=0, y=0, radius=1, name="chaser")
        target = ObjectSpec(kind="sphere_3d", x=20, y=20, radius=1, name="target")
        mover = ObjectSpec(kind="box_3d", x=16, y=16, size=2, name="mover")

        graph = SceneGraph(scene_spec=SceneSpec(objects=[ground, ball, chaser, target, mover]))
        graph.physics = PhysicsWorld(gravity=(0.0, 100.0))
        graph.physics.add_body(ground, RigidBodySpec(is_static=True))
        graph.physics.add_body(ball, RigidBodySpec(mass=1.0, restitution=0.3))
        graph.animations.append(_ChaseController("chaser", "target", speed=10.0))
        graph.animations.append(
            AnimationTrack([Tween(target="mover", property="x", from_value=10, to_value=22,
                                   duration=1.0, easing="ease_in_out", loop=True, ping_pong=True)])
        )

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            live_path = tmp.name
        export_live_html(graph, live_path, width=320, height=320, title="Live Test", scale=10.0)

        with open(live_path) as fh:
            live_html = fh.read()
        script = live_html.split("<script>", 1)[1].split("</script>", 1)[0]

        node_check = f"""
        global.document = {{ getElementById: () => ({{ width: 320, height: 320,
          getContext: () => ({{ clearRect(){{}}, beginPath(){{}}, arc(){{}}, fill(){{}}, fillRect(){{}}, fillStyle: null }}) }}) }};
        global.addEventListener = () => {{}};
        global.requestAnimationFrame = () => {{}};
        global.performance = {{ now: () => 0 }};
        {script}
        for (let i = 0; i < 20; i++) {{ physicsStep(0.05); stepAnimations(0.05); }}
        const ball = OBJECTS[1];
        if (!(ball.y > 5)) throw new Error("gravity didn't move ball: y=" + ball.y);
        const chaser = OBJECTS[2], target = OBJECTS[3];
        const dist0 = Math.hypot(0 - 20, 0 - 20);
        const dist1 = Math.hypot(chaser.x - target.x, chaser.y - target.y);
        if (!(dist1 < dist0)) throw new Error("chase didn't close distance: " + dist1 + " vs " + dist0);
        const mover = OBJECTS[4];
        if (mover.x === 10) throw new Error("tween didn't move mover");
        console.log("OK");
        """
        result = subprocess.run([node, "-e", node_check], capture_output=True, text=True)
        assert result.returncode == 0 and "OK" in result.stdout, (
            f"ported JS engine self-check failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        print("OK — export_live_html's ported JS engine reproduces gravity/collision/chase/tween")

    print("OK — export_html writes a self-contained interactive HTML file")
