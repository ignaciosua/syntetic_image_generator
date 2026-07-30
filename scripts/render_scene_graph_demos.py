"""Export interactive `export_html`/`export_live_html` demos plus a
browsable gallery.

Mirrors the media/ output convention of `render_generator_atlas.py`, but for
the scene-graph layer's web export instead of indexed-image atlases.

The 8 archetype demos are built from the scene-catalog registry
(`generators.scene_levels.SCENE_ARCHETYPES`) with real behaviors attached —
same selection as `scene_catalog.make_scene_trajectory` (1-3 compatible
behaviors drawn from the same RNG stream) — then exported with
`export_live_html`, which ports `PhysicsWorld` + the behavior controllers to
JS so the page runs the actual simulation (gravity, collisions, patrol,
chase, flee, orbit), not a static snapshot or a recording. Archetypes are
composed at a 32x32-unit viewport; `scale` is display-only (canvas pixels),
the simulation itself stays in world units.
"""

from pathlib import Path

import numpy as np

from synthetic_image_generator import Background, ObjectSpec, SceneGraph, SceneSpec, export_html, export_live_html
from synthetic_image_generator.behaviors import compatible_behaviors
from synthetic_image_generator.scene_levels import SCENE_ARCHETYPES, _VIEWPORT

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "media" / "scene_graph_demos"

ARCHETYPE_SCALE = 10
ARCHETYPE_CANVAS = _VIEWPORT * ARCHETYPE_SCALE  # 320x320
_MAX_BEHAVIORS = 3  # mirrors generators.scene_catalog._MAX_BEHAVIORS


def demo_shapes_reference():
    objects = []
    colors = [(0.9, 0.2, 0.2), (0.2, 0.6, 0.9), (0.9, 0.8, 0.2), (0.4, 0.9, 0.4), (0.7, 0.3, 0.9)]
    for i, color in enumerate(colors):
        objects.append(ObjectSpec(kind="sphere_3d", x=60 + i * 90, y=90, radius=18 + i * 4, color=color, tags={"circle"}))
        objects.append(
            ObjectSpec(kind="box_3d", x=60 + i * 90, y=220, width=30 + i * 6, height=30 + i * 6, color=color, tags={"rect"})
        )
    return SceneGraph(scene_spec=SceneSpec(background=Background(kind="solid"), objects=objects))


def _build_showcase(archetype, max_tries: int = 30):
    """Same 1-3-behaviors-from-compatible draw as ``make_scene_trajectory``,
    but tries a few seeds so the demo actually exercises whatever makes the
    archetype interesting (gravity_drop, chase, orbit, ...) instead of
    risking a seed where only the always-compatible ``patrol`` got picked."""

    # patrol is always compatible (not a distinguishing pick) and
    # bounce_particles has no visible effect in this port (no particle
    # rendering) — neither counts toward "this seed shows something".
    all_candidates = compatible_behaviors(archetype.name)
    interesting = {b.name for b in all_candidates} - {"patrol", "bounce_particles"}
    # patrol's tween can land on the same object gravity_drop is dropping
    # and drag it (and, via collision, its neighbors) sideways with its own
    # swing — a real interaction, not a bug, but a bad look for a demo
    # that's supposed to showcase gravity/collision settling.
    has_gravity = any(b.name == "gravity_drop" for b in all_candidates)
    candidates_pool = [b for b in all_candidates if not (has_gravity and b.name == "patrol")]

    graph, chosen_names = None, []
    for seed in range(max_tries):
        rng = np.random.RandomState(seed)
        graph = archetype.build(rng)
        candidates = candidates_pool
        chosen_names = []
        if candidates:
            n_pick = min(1 + int(rng.randint(0, _MAX_BEHAVIORS)), len(candidates))
            picks = sorted(rng.choice(len(candidates), size=n_pick, replace=False).tolist())
            for i in picks:
                candidates[i].attach(graph, rng)
                chosen_names.append(candidates[i].name)
        if not interesting or (set(chosen_names) & interesting):
            break
    return graph, chosen_names


def render(output_dir: Path = OUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    export_html(demo_shapes_reference(), str(output_dir / "shapes_reference.html"), width=480, height=260, title="Shape/color reference")

    cards = []
    for archetype in SCENE_ARCHETYPES:
        graph, chosen_names = _build_showcase(archetype)

        path = output_dir / f"{archetype.name}.html"
        export_live_html(
            graph, str(path), width=ARCHETYPE_CANVAS, height=ARCHETYPE_CANVAS,
            title=f"{archetype.name} — scene-graph demo", scale=ARCHETYPE_SCALE,
        )
        behaviors = " · ".join(chosen_names) if chosen_names else "none this seed"
        cards.append((archetype.name, behaviors))

    grid_rows = "\n".join(
        f"""<div class="card">
  <iframe src="{name}.html" width="{ARCHETYPE_CANVAS}" height="{ARCHETYPE_CANVAS}" style="border:none"></iframe>
  <div class="label">{name} <span class="behaviors">{behaviors}</span></div>
</div>"""
        for name, behaviors in cards
    )

    index = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Scene-graph export_html demos</title>
<style>
  body {{ background:#111; color:#eee; font-family:sans-serif; padding:24px; }}
  h1 {{ font-size:20px; margin-bottom:4px; }}
  section {{ margin-bottom:32px; }}
  h2 {{ font-size:15px; font-weight:600; margin:20px 0 6px; }}
  p.hint {{ color:#999; font-size:13px; margin-top:2px; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:20px; }}
  .card {{ background:#1a1a1a; border:1px solid #333; border-radius:6px; overflow:hidden; }}
  .label {{ padding:4px 8px; font-size:12px; color:#aaa; }}
  .behaviors {{ font-size:11px; color:#666; }}
  iframe {{ display:block; }}
</style>
</head>
<body>
<h1>Scene-graph export_html demos <span style="font-weight:400;font-size:14px;color:#888">— {len(cards)} archetypes</span></h1>
<p class="hint">Archetypes run a real client-side simulation (ported PhysicsWorld + behavior controllers — gravity, collisions, patrol/chase/flee/orbit all actually compute in JS, not a recording). Click a frame, then arrow keys / WASD move the green player-tagged object. Flat-shaded 2D proxy renderer (not the Phong 3D one).</p>

<h2>Shapes reference</h2>
<div class="card" style="display:inline-block"><iframe src="shapes_reference.html" width="480" height="260" style="border:none"></iframe></div>
<p class="hint">One of every shape kind — no player, no behaviors.</p>

<h2>Archetypes</h2>
<div class="grid">
{grid_rows}
</div>

<h2>Behaviors</h2>
<p class="hint">
  <b>patrol</b> — ping-pong tween on x or y axis ·
  <b>chase</b> — steer enemy toward player ·
  <b>flee</b> — steer away from player ·
  <b>orbit</b> — constant-radius circular motion around center ·
  <b>gravity_drop</b> — attach PhysicsWorld + gravity, tagged balls/boxes fall ·
  <b>bounce_particles</b> — trailing particle emitter on moving bodies
</p>
</body></html>
"""
    (output_dir / "index.html").write_text(index)
    return output_dir / "index.html"


if __name__ == "__main__":
    index_path = render()
    print(f"OK — wrote {len(SCENE_ARCHETYPES) + 1} demos + gallery to {index_path}")
