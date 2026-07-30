import hashlib

import synthetic_image_generator as sig


def test_one_exact_sample_per_level():
    digest = hashlib.sha256()
    for level in range(sig.N_LEVELS):
        index = sig.level_start(level)
        image = sig.make_image(index, force_level=level)
        digest.update(image.tobytes())

    assert digest.hexdigest() == sig.LEGACY_LEVEL_SAMPLE_SHA256
