"""scene.py — the attributes every modality can express, shared by index.

levels.py gives all six generators the same discrete label. Within a level they
still draw independent content, so any pair that is not hard-linked
(image↔stereo_image, audio↔binaural, video↔stereo_video) has nothing finer than
the level to predict — which is why per-sample cross-modal retrieval sat at chance
while level classification reached 54.9%.

This adds three continuous attributes per index and renders each one in the
modality's own terms:

    energy    peak brightness      ↔  peak loudness
    warmth    red-vs-blue tilt     ↔  low-vs-high spectral tilt
    contrast  pixel spread         ↔  dynamic range

They are read off the index alone and applied AFTER the level renders, so no level
code changes and the attributes are exactly as recoverable from one modality as
from another. All three survive a 32x32 frame and an 80-bin log-mel.

ponytail: these are GLOBAL attributes. They give all 30 cross-modal pairs real
per-sample information, but they do not make image[idx] and audio[idx] depict the
same object — that would mean one shared scene graph rendered six ways, i.e.
rewriting 700 KB of level code. The three hard pairs carry the fine-grained
correspondence; these three dims carry the rest.

The order matters: tilt, then contrast, then normalise, then energy. Energy is
applied last and nothing renormalises after it, so it lands as exactly the peak —
otherwise the three would be entangled and a model could only recover their product.
"""

import hashlib
import numpy as np

PRIMO = 2246822519
RANGES = {'energy': (0.45, 1.00), 'warmth': (-1.0, 1.0), 'contrast': (0.60, 1.40)}
ATTRS = tuple(RANGES)


def scene(idx):
    """Deterministic per-index attributes. Independent of level and of modality."""
    r = np.random.RandomState(int.from_bytes(
        hashlib.sha256(f"{idx}_{PRIMO}_scene".encode()).digest()[:4], "little"))
    return {k: float(r.uniform(lo, hi)) for k, (lo, hi) in RANGES.items()}


def scene_vector(idx):
    """The same attributes as a (3,) float32 in [0, 1] — a regression target.

    Rescaled so one MSE treats all three equally; raw, `warmth` spans 2.0 and
    `energy` 0.55, and the head would spend its capacity on whichever is widest.
    """
    s = scene(idx)
    return np.array([(s[k] - RANGES[k][0]) / (RANGES[k][1] - RANGES[k][0])
                     for k in ATTRS], np.float32)


def apply_visual(img, s, caxis=-1):
    """(..., 3, ...) in [0,1] → same shape, modulated. `caxis` is the RGB axis."""
    x = np.clip((np.asarray(img, np.float32) - 0.5) * s['contrast'] + 0.5, 0.0, 1.0)
    tilt = np.array([1.0 + 0.25 * s['warmth'], 1.0, 1.0 - 0.25 * s['warmth']], np.float32)
    shape = [1] * x.ndim
    shape[caxis] = 3
    x = np.clip(x * tilt.reshape(shape), 0.0, 1.0)
    return (x * s['energy']).astype(np.float32)


def apply_audio(wav, s):
    """(..., N) in [-1,1] → same shape, modulated. Filters along the last axis."""
    x = np.asarray(wav, np.float32)
    peak0 = float(np.abs(x).max())
    if peak0 < 1e-8:
        return x                                  # silence stays silence

    # Spectral tilt: log-linear in frequency, zero-phase, one FFT.
    # warmth=+1 → -12 dB at Nyquist (dark), warmth=-1 → +12 dB (bright).
    F = np.fft.rfft(x, axis=-1)
    f = np.linspace(0.0, 1.0, F.shape[-1], dtype=np.float32)
    x = np.fft.irfft(F * (2.0 ** (-2.0 * s['warmth'] * f)), n=x.shape[-1], axis=-1)

    # Dynamic range: >1 pushes quiet parts down, <1 lifts them.
    x = np.sign(x) * np.abs(x) ** s['contrast']

    peak = float(np.abs(x).max())
    return (x / max(peak, 1e-8) * s['energy']).astype(np.float32)


if __name__ == '__main__':
    a, b = scene(1234), scene(1234)
    assert a == b, 'scene is not deterministic'
    assert scene(10) != scene(11), 'scene does not vary with idx'

    v = np.array([scene_vector(i) for i in range(2000)])
    assert v.shape == (2000, 3) and v.min() >= 0.0 and v.max() <= 1.0
    assert np.abs(v.mean(0) - 0.5).max() < 0.05, f'attributes are not centred: {v.mean(0)}'

    rng = np.random.RandomState(0)

    # Each attribute must move its own statistic, in every modality, monotonically.
    img = rng.uniform(0.1, 0.9, (32, 32, 3)).astype(np.float32)
    wav = (rng.randn(32000).astype(np.float32) * 0.2).clip(-1, 1)
    lo = {'energy': 0.5, 'warmth': -0.9, 'contrast': 0.7}
    hi = {'energy': 1.0, 'warmth': 0.9, 'contrast': 1.3}

    def hf(w):  # high-frequency share of the spectrum
        P = np.abs(np.fft.rfft(w)) ** 2
        return float(P[len(P) // 2:].sum() / P.sum())

    for k in ATTRS:
        s_lo, s_hi = dict(hi, **{k: lo[k]}), dict(hi, **{k: hi[k]})
        v_lo, v_hi = apply_visual(img, s_lo), apply_visual(img, s_hi)
        a_lo, a_hi = apply_audio(wav, s_lo), apply_audio(wav, s_hi)
        if k == 'energy':
            assert v_lo.max() < v_hi.max() and np.abs(a_lo).max() < np.abs(a_hi).max()
        elif k == 'warmth':
            rb = lambda v: v[..., 0].mean() - v[..., 2].mean()
            assert rb(v_lo) < rb(v_hi), 'visual warmth has no effect'
            assert hf(a_hi) < hf(a_lo), 'audio warmth has no effect'
        else:
            assert v_lo.std() < v_hi.std(), 'visual contrast has no effect'
            # higher contrast = more of the signal near the peak, less in the middle
            q = lambda w: float(np.percentile(np.abs(w), 50) / np.abs(w).max())
            assert q(a_hi) < q(a_lo), 'audio contrast has no effect'

    # Non-trailing channel axis (video is H,W,C,T).
    vid = rng.uniform(0.1, 0.9, (32, 32, 3, 8)).astype(np.float32)
    out = apply_visual(vid, hi, caxis=2)
    assert out.shape == vid.shape
    assert np.allclose(out[..., 0], apply_visual(vid[..., 0], hi), atol=1e-6)

    assert apply_visual(img, hi).max() <= 1.0 and np.abs(apply_audio(wav, hi)).max() <= 1.0
    assert np.abs(apply_audio(np.zeros(1000, np.float32), hi)).max() == 0.0
    print('OK')
