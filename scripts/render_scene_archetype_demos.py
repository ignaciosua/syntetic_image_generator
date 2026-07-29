"""Export an interactive SceneGraph gallery for all 48 scene archetypes."""
from pathlib import Path

from synthetic_image_generator import scene_level_start
from synthetic_image_generator.scene_level_catalog import ARCHETYPE_NAMES, SCENE_LEVELS, make_scene_level_graph, scene_level_frame_count, make_scene_camera_trajectory
from synthetic_image_generator import export_live_html

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media" / "scene_archetype_demos"
SCALE = 12


def render(output: Path = OUT, quality: str = "fast"):
    output.mkdir(parents=True, exist_ok=True)
    cards = []
    tour_cards = []
    raster_frames = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for archetype in ARCHETYPE_NAMES:
        level_index = next(i for i, level in enumerate(SCENE_LEVELS) if level.archetype == archetype)
        graph = make_scene_level_graph(scene_level_start(level_index))
        real_frame = graph.render()
        raster_frames.append((archetype, real_frame))
        filename = f"{archetype}.html"
        preview = f"{archetype}.png"
        plt.imsave(output / preview, real_frame)
        sequence_dir = output / "frames" / archetype
        sequence_dir.mkdir(parents=True, exist_ok=True)
        sequence = []
        frame_count = scene_level_frame_count(scene_level_start(level_index), quality)
        from synthetic_image_generator import InputState
        for frame_index in range(frame_count):
            frame_path = sequence_dir / f"{frame_index:03d}.png"
            plt.imsave(frame_path, graph.render())
            sequence.append(f"frames/{archetype}/{frame_index:03d}.png")
            graph.update(1 / 30, InputState())
        export_live_html(graph, str(output / filename), width=384, height=384,
                         title=f"{archetype} — SceneGraph", scale=SCALE)
        spec = SCENE_LEVELS[level_index]
        cards.append(f'<article><img class="real sequence" data-frames="{",".join(sequence)}" data-frame-count="{len(sequence)}" src="{sequence[0]}" alt="{archetype} exact simulation"><b>{archetype}</b><small>{spec.variant} · {spec.composition} · {spec.simulation} · {len(sequence)} frames</small></article>')
        if spec.world_width > 32:
            tour_dir = output / "camera_tours" / archetype
            tour_dir.mkdir(parents=True, exist_ok=True)
            tour = []
            camera_modes = ("top_down", "top_down_follow", "pan", "zoom", "diagonal", "orbit", "dolly",
                            "first_person", "third_person", "follow", "lead",
                            "rail", "shake", "rotate")
            camera_mode = camera_modes[level_index % len(camera_modes)]
            for j, frame in enumerate(make_scene_camera_trajectory(scene_level_start(level_index), n_frames=48, viewport=(512, 512), mode=camera_mode)):
                path = tour_dir / f"{j:03d}.png"
                plt.imsave(path, frame)
                tour.append(f"camera_tours/{archetype}/{j:03d}.png")
            tour_cards.append(f'<article><img class="real sequence" data-frames="{",".join(tour)}" src="{tour[0]}" alt="{archetype} camera tour"><b>{archetype} · camera {camera_mode}</b><small>trayectoria {camera_mode} en mundo 512×512 · {len(tour)} frames</small></article>')
    index = """<!doctype html><meta charset='utf-8'><title>48 SceneGraph archetypes</title>
<style>body{background:#111;color:#eee;font:14px sans-serif;padding:20px}h1{font-size:22px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(512px,1fr));gap:18px}article{background:#1d1d1d;padding-bottom:8px}.real{display:block;width:512px;height:512px;image-rendering:auto}b,small{display:block;padding:3px 8px}small{color:#999}</style>
<h1>48 SceneGraph archetypes</h1><p>Animación exacta: cada frame fue producido por el renderer Python real a partir del mismo SceneGraph, no por un proxy JavaScript.</p><h2>Arquetipos y simulación</h2><div class='grid'>""" + "".join(cards) + """</div><h2>Recorridos de cámara en mundos grandes</h2><p>Estas tarjetas desplazan la cámara por distintas regiones de escenas de 512×512.</p><div class='grid'>""" + "".join(tour_cards) + """</div><button id='toggle'>Pausar</button><label> FPS <input id='fps' type='range' min='1' max='30' value='15'></label><script>
const images=[...document.querySelectorAll('.sequence')]; const frames=images.map(i=>i.dataset.frames.split(',')); let frame=0, playing=true, timer;
function play(){clearInterval(timer); timer=setInterval(()=>{images.forEach((image,index)=>{let list=frames[index]; image.src=list[frame%list.length]}); frame++},1000/Number(document.querySelector('#fps').value))}
document.querySelector('#toggle').onclick=()=>{playing=!playing; document.querySelector('#toggle').textContent=playing?'Pausar':'Reproducir'; if(playing)play(); else clearInterval(timer)};
document.querySelector('#fps').oninput=()=>{if(playing)play()}; play();
</script>"""
    (output / "index.html").write_text(index, encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 8, figsize=(16, 12))
    for axis, (name, frame) in zip(axes.flat, raster_frames):
        axis.imshow(frame)
        axis.set_title(name, fontsize=7)
        axis.axis("off")
    for axis in axes.flat[len(raster_frames):]: axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "contact_sheet.png", dpi=150)
    plt.close(fig)
    return output / "index.html"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("preview", "fast", "training"), default="fast")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    print(render(Path(args.output), args.quality))
