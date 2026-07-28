# synthetic-image-generator

Deterministic indexed image generator with composable scenes, alpha, and raster conversion.

```
pip install synthetic-image-generator
```

```python
from synthetic_image_generator import make_image, make_scene, make_scene_raster

img = make_image(42)  # (32, 32, 3) float32
spec, scene = make_scene(42)
raster = make_scene_raster(spec, scene, 256, 256)
```

See [igc6_universal_selfcontained/README.md](igc6_universal_selfcontained/README.md) for full docs.
