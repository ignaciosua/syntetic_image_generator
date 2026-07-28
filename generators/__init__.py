"""Public API for the deterministic synthetic image generator."""

from .levels import (
    GENERATOR_CONTRACT_VERSION,
    LEGACY_LEVEL_SAMPLE_SHA256,
    LEVEL_CYCLE_SIZE,
    LEVEL_NAMES,
    LEVEL_TABLE,
    N_BLOCKS,
    N_LEVELS,
    SAMPLES_PER_LEVEL,
    level_block,
    level_of,
    level_start,
)
from .scene_spec import (
    Background,
    LightSpec,
    ObjectSpec,
    PostSpec,
    RasterSpec,
    SceneSpec,
)
from .synthetic_image_generator import (
    C,
    H,
    W,
    convert_raster,
    extract_alpha,
    make_image,
    make_image_raster,
    make_scene,
    make_scene_raster,
)
from .wave import WAVE_LEVELS

__version__ = "0.2.2"

__all__ = [
    "Background",
    "C",
    "GENERATOR_CONTRACT_VERSION",
    "H",
    "LEGACY_LEVEL_SAMPLE_SHA256",
    "LEVEL_CYCLE_SIZE",
    "LEVEL_NAMES",
    "LEVEL_TABLE",
    "LightSpec",
    "N_BLOCKS",
    "N_LEVELS",
    "ObjectSpec",
    "PostSpec",
    "RasterSpec",
    "SAMPLES_PER_LEVEL",
    "SceneSpec",
    "WAVE_LEVELS",
    "W",
    "__version__",
    "convert_raster",
    "extract_alpha",
    "level_block",
    "level_of",
    "level_start",
    "make_image",
    "make_image_raster",
    "make_scene",
    "make_scene_raster",
]
