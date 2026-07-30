"""Generate a deterministic HTML gallery for camera post-effects."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from generators.scene_level_catalog import make_scene_level_graph, scene_level_start, SCENE_LEVELS

EFFECTS = ("vignette", "grayscale", "scanlines", "flash", "motion_blur",
           "chromatic_aberration", "pixelate", "film_grain", "depth_of_field",
           "fog", "underwater", "night_vision", "thermal", "damage", "rain", "dust")

def render(output=ROOT / "media" / "camera_effect_demos"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    idx = scene_level_start(next(i for i, x in enumerate(SCENE_LEVELS) if x.archetype == "forest"))
    cards = []
    for effect in EFFECTS:
        graph = make_scene_level_graph(idx)
        graph.camera.viewport_width = graph.camera.viewport_height = 512
        graph.camera.effects = (effect,)
        path = output / f"{effect}.png"
        plt.imsave(path, graph.render())
        cards.append(f'<article><img src="{path.name}"><b>{effect}</b><small>SceneGraph real · 512×512</small></article>')
    html = """<!doctype html><meta charset=utf-8><title>Camera effects</title>
<style>body{background:#111;color:#eee;font:14px sans-serif;padding:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(512px,1fr));gap:18px}article{background:#1d1d1d;padding-bottom:8px}img{display:block;width:512px;height:512px}b,small{display:block;padding:4px 8px}small{color:#999}</style>
<h1>Camera post-effects</h1><p>Variantes deterministas renderizadas por el SceneGraph.</p><div class=grid>""" + "".join(cards) + "</div>"
    (output / "index.html").write_text(html, encoding="utf-8")
    return output / "index.html"

if __name__ == "__main__":
    print(render())
