"""Canonical level catalog and unbounded deterministic index schedule.

Every generator used to own its LEVEL_TABLE, and each padded it with a different
list of "give this level extra samples" blocks. The tables therefore had different
lengths, so SAMPLES_PER_LEVEL came out different (240, 242, 221, 248, 250, ...) and
`idx // SAMPLES_PER_LEVEL` landed on a different level in every modality.

Measured level agreement across 52000 indices with the old per-generator tables:

                image  stereo   audio  binaur   video  stereo_video
    image       100.0%   32.7%   67.8%    7.1%    5.8%   11.9%
    stereo_im    32.7%  100.0%   27.9%    7.4%    7.3%   17.1%
    audio        67.8%   27.9%  100.0%    2.8%    8.2%    7.2%
    binaural      7.1%    7.4%    2.8%  100.0%    1.9%   10.4%
    video         5.8%    7.3%    8.2%    1.9%  100.0%   30.6%
    stereo_vid   11.9%   17.1%    7.2%   10.4%   30.6%  100.0%

So "level" was not a shared factor at all — it was six unrelated labels that happened
to share a name. A model reading the level off `audio[idx]` was right about
`video[idx]`'s level 8% of the time. That is the ceiling every cross-modal pair
outside the three hard-linked ones was training against.

One table fixes it: same idx → same level in all six, agreement 100% everywhere.

The catalog is flat — one block per level, no extra-sample padding.  Indices
cycle through the catalog forever:

    level = (idx // 325) % 154

The absolute block number remains part of the renderer seed.  Later cycles
therefore select the same level sequence without repeating the first cycle's
geometry.  Indices 0..50049 retain the original byte-exact contract.

Levels 148-153 are the plane-wave family (wave.py): one 30-number parameter vector,
Sobol-sampled, rendered into all six modalities. They are the only levels where
image[idx] and audio[idx] are the same scene rather than two independent draws.
"""

import operator

N_LEVELS = 154
SAMPLES_PER_LEVEL = 325
LEVEL_CYCLE_SIZE = N_LEVELS * SAMPLES_PER_LEVEL
LEVEL_TABLE = list(range(N_LEVELS))
# Backward-compatible name: this is the number of blocks in one catalog cycle,
# not a maximum supported block count.
N_BLOCKS = N_LEVELS

LEVEL_NAMES = (
    "dots", "pixels", "lines", "polylines", "ellipses", "hollow_2d",
    "radial_gradient", "organic_blob_2d", "edge_detection", "monochrome",
    "analogous", "complementary", "triadic", "warm", "cool", "sphere_3d",
    "box_3d", "cylinder_3d", "torus_3d", "organic_blob_3d", "splat_3d",
    "mixed_3d", "brushed", "wood", "marble", "striated", "depth_fog",
    "perspective_grid", "cluttered_3d", "landscape", "low_sun",
    "occlusion_pile", "composite", "hollow_3d_glass", "glass_objects",
    "blur_3d", "sensor_artifacts", "one_over_f", "perlin", "lit_3d",
    "hollow_outlines", "edge_image", "gradient_fills", "voronoi",
    "checkerboard", "fog_blobs", "textured_surface", "placeholder_47",
    "insect_2d", "quadruped_2d", "bird_2d", "fish_2d", "spider_2d",
    "human_3d", "quadruped_3d", "bird_3d", "fish_3d",
    "abstract_critter_3d", "textured_critter", "round_tree_2d",
    "conifer_tree_2d", "palm_tree_2d", "bush_flower", "building_2d",
    "water", "fire", "gas_cloud", "element_combos", "nature_scene",
    "city_scene", "mixed_environment", "chaos", "style_raw",
    "style_realistic", "style_cartoon", "style_sketch", "style_watercolor",
    "style_neon", "style_vintage", "style_pixel", "supersample",
    "random_crop", "car_2d", "airplane_2d", "ship_2d", "head_closeup",
    "depth_of_field", "motion_blur", "natural_color", "tone_vignette",
    "backlit_silhouette", "grass_field", "fur_feather", "fractal_branches",
    "manmade_repetition", "sky_clouds", "weather", "reflection_symmetry",
    "jpeg_artifacts", "coherent_scene", "round_tree_3d",
    "conifer_palm_3d", "bush_flower_3d", "building_3d", "car_3d",
    "airplane_3d", "ship_3d", "insect_3d", "spider_3d", "wireframe_3d",
    "texture_3d", "particle_fields", "emissive_points",
    "reaction_diffusion", "transmission", "turbulence_spiral",
    "nonlinear_optics", "hot_matter", "nonoptical_imaging",
    "coherent_light", "closed_organic", "text_symbols", "known_symbols",
    "single_symbol", "indoor_scene", "human_figures", "underwater",
    "flower_field", "pore_texture", "mirror_fold", "kaleidoscope",
    "recursive_131", "recursive_132", "diffraction", "recursive_134",
    "refraction_2d", "refraction_3d", "hair_fur_3d", "woven", "glass_3d",
    "caustics", "aerial_view", "food", "medical", "crowds", "macro",
    "sunset", "placeholder_147", "wave_2", "wave_4", "wave_8",
    "wave_mono", "wave_posterize", "wave_polar",
)
assert len(LEVEL_NAMES) == N_LEVELS

GENERATOR_CONTRACT_VERSION = "sig-154-cycle-v1"
LEGACY_LEVEL_SAMPLE_SHA256 = (
    "fdc94129dedac51a89eb11168e3003128611d6c053a57ce8d14654daa2546a2c"
)


def _non_negative_integer(value, name):
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def level_block(idx):
    """Return the absolute, unbounded schedule block for ``idx``."""
    return _non_negative_integer(idx, "idx") // SAMPLES_PER_LEVEL


def level_of(idx):
    """Map any non-negative integer index to one of the 154 source levels."""
    return LEVEL_TABLE[level_block(idx) % N_LEVELS]


def level_start(lvl, rng=None, *, cycle=0):
    """Return the first index for ``lvl`` in a selected catalog ``cycle``.

    ``rng`` remains accepted and ignored for compatibility with historical
    callers that passed it as the second positional argument.
    """
    level = _non_negative_integer(lvl, "lvl")
    cycle_index = _non_negative_integer(cycle, "cycle")
    if level >= N_LEVELS:
        raise ValueError(f"lvl must be between 0 and {N_LEVELS - 1}")
    return (cycle_index * N_LEVELS + level) * SAMPLES_PER_LEVEL


if __name__ == '__main__':
    assert sorted(set(LEVEL_TABLE)) == list(range(N_LEVELS)), 'levels not contiguous'
    assert level_of(0) == 0 and level_of(49999) == N_LEVELS - 1
    assert level_of(SAMPLES_PER_LEVEL - 1) == 0 and level_of(SAMPLES_PER_LEVEL) == 1
    for cycle in (0, 1, 10, 10_000):
        start = cycle * LEVEL_CYCLE_SIZE
        assert level_of(start) == 0
        assert level_of(start + LEVEL_CYCLE_SIZE - 1) == N_LEVELS - 1
        assert level_start(37, cycle=cycle) == start + 37 * SAMPLES_PER_LEVEL

    if __package__:
        from .wave import WAVE_LEVELS
    else:
        from wave import WAVE_LEVELS
    assert set(WAVE_LEVELS) <= set(range(N_LEVELS)), 'wave levels fall outside the table'
    print(
        f"OK — {N_LEVELS} levels × {SAMPLES_PER_LEVEL} indices, "
        "unbounded deterministic cycles"
    )
