"""Render a compact gallery of the 154 scene-level catalog entries."""
from pathlib import Path
import numpy as np
import synthetic_image_generator as sig


def render_gallery(output="media/scene_level_gallery.npy"):
    frames = [sig.make_scene_level(sig.scene_level_start(i)) for i in range(sig.SCENE_N_LEVELS)]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.stack(frames))
    return path


def render_html(output="media/scene_level_gallery"):
    """Write per-level PNGs and a navigable HTML index (matplotlib optional)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    cards = []
    for level in range(sig.SCENE_N_LEVELS):
        idx = sig.scene_level_start(level)
        image = sig.make_scene_level(idx)
        filename = f"level_{level:03d}.png"
        plt.imsave(root / filename, image)
        spec = sig.scene_level_spec(idx)
        cards.append(f'<figure><img src="{filename}"><figcaption>{level}: {spec.name}<br>{spec.archetype} · {spec.style} · {spec.simulation}</figcaption></figure>')
    (root / "index.html").write_text("<html><style>body{background:#222;color:#eee;font-family:sans-serif;display:grid;grid-template-columns:repeat(6,1fr)}figure{margin:4px;font-size:11px}img{width:100%}</style>" + "".join(cards) + "</html>", encoding="utf-8")
    return root / "index.html"


if __name__ == "__main__":
    print(render_html())
