"""wave.py — a sum of plane waves, rendered into every modality from ONE parameter vector.

    v(x,y) = Σ_{i<n} sin( 2π (fx_i·x + fy_i·y) + φ_i )
    v̂      = (v - min v) / (max v - min v)
    C_k    = sin( v̂·π·(0.5 + 2·α_k) + β_k·π ) · 0.5 + 0.5        k ∈ {R,G,B}
    I      = ( R·(0.4 + 0.6·v̂),  G·(0.4 + 0.6·(1-v̂)),  B·(0.3 + 0.7·2|v̂-0.5|) )

θ is 30 numbers in [0,1]: six colour parameters, then up to eight (fx, fy, φ) triples.
Sampled with a scrambled Sobol sequence indexed by `idx`, so consecutive indices spread
over the parameter space instead of clustering the way independent uniform draws do.

Why this earns its place in a project about inverting generators
────────────────────────────────────────────────────────────────
It is the exact opposite of what genparams.py measured at R² −0.063. There the target
was the generator's raw rng draws: draw k means "sphere radius" in one level and
"particle count" in the next, levels loop over objects so the vector is an unordered
SET rendered as an ordered list, and many draws leave no trace at 32x32. Here θ is
fixed-length, every position means the same thing in every sample, there is no set,
and the image is a smooth deterministic function of all 30 numbers. If parameter
recovery works anywhere, it works here — see probe_wave.py.

And the SAME θ drives all six modalities, which nothing else in this dataset does:

    image         the formula over (x, y)
    video         the formula over (x, y, t) — φ advances, so the waves travel
    audio         the formula over (t) — additive synthesis, one partial per term,
                  fx sets pitch, fy sets a tremolo rate, φ sets phase; the R colour
                  mapping becomes the waveshaper
    stereo_*      inherited from their mono partners, as everywhere else
    binaural

So on these levels image[idx] and audio[idx] are literally the same 30 numbers in a
different domain. Everywhere else in the dataset the cross-group pairs share only the
level and three global attributes (scene.py), which is why they beat a global-mean
baseline but not a level-mean one.
"""

import numpy as np

D = 30                      # 6 colour + 8 triples
N_SOBOL = 65536             # 2^16 — Sobol wants a power of two
SOBOL_SEED = 12345
_POINTS = None

# level → (number of terms, rendering mode)
WAVE_LEVELS = {
    148: (2, 'rgb'),        # two waves: moiré / plain stripes
    149: (4, 'rgb'),        # the formula as given
    150: (8, 'rgb'),        # dense interference
    151: (4, 'mono'),       # one channel mapping on all three — greyscale texture
    152: (4, 'posterize'),  # v̂ quantised to 5 bands / audio bitcrushed to match
    153: (6, 'polar'),      # domain warped to (radius, angle) — rings and spirals
}
MODES = ('rgb', 'mono', 'posterize', 'polar')
N_BANDS = 5                 # posterize
VIDEO_SPEED = 0.5           # wave cycles per clip, per unit of mean spatial frequency
AUDIO_F0 = 55.0             # Hz at fx = 1
AUDIO_OCT = 0.75            # octaves per unit of fx → fx = 9 lands at 3520 Hz
AUDIO_TREM_HZ = 6.0         # tremolo rate at fy = 9


def theta(idx):
    """(30,) float32 in [0,1] — the parameter vector for this index."""
    global _POINTS
    if _POINTS is None:
        from scipy.stats import qmc      # local: only wave levels pay the import
        _POINTS = qmc.Sobol(d=D, scramble=True, seed=SOBOL_SEED).random(N_SOBOL)
    return _POINTS[int(idx) % N_SOBOL].astype(np.float32)


def unpack(th, n):
    """θ → (fx, fy, φ) each (n,), sorted canonically, and the six colour parameters.

    Colour comes from dims 0-5 and the terms from 6 onward, because Sobol's leading
    dimensions are the best distributed and colour is used by every mode and every
    term count. Within the terms, triple i occupies dims 6+3i .. 8+3i, so a level
    using 2 terms and one using 8 agree on the first two waves.

    The terms are SORTED by (fx, fy) — and this is what makes the parameters
    recoverable at all. A sum is permutation-invariant: swapping term 0 with term 3
    leaves every rendered pixel identical, so "which wave was drawn first" is not in
    the output and regressing it is asking for information that does not exist.
    Measured, before sorting: colour R² +0.614 from an image, terms R² -0.017. That is
    the same failure genparams.py hit ("an unordered SET rendered as an ordered list"),
    and here it costs one lexsort to remove, because sorting changes the LABEL and not
    a single pixel.
    """
    fx = 1.0 + 8.0 * th[6:6 + 3 * n:3]
    fy = 1.0 + 8.0 * th[7:7 + 3 * n:3]
    ph = 2.0 * np.pi * th[8:8 + 3 * n:3]
    o = np.lexsort((fy, fx))                    # fx primary, fy to break ties
    return fx[o], fy[o], ph[o], th[:6]


# Which θ dimensions each modality's output actually depends on. Scoring a probe on
# the rest measures nothing: audio never touches the green or blue colour mapping.
def used_dims(n, modality):
    d = [0, 1] if modality in ('audio', 'binaural') else [0, 1, 2, 3, 4, 5]
    return d + list(range(6, 6 + 3 * n))


def canonical(th, n):
    """θ with its terms in the same sorted order the renderers use — the probe target.

    Identical to θ up to a permutation of the (fx, fy, φ) triples, so it describes the
    exact same image; it just names the waves in an order the output can actually
    reveal (lowest horizontal frequency first).
    """
    fx, fy, ph, col = unpack(th, n)
    out = np.array(th, np.float32).copy()
    out[6:6 + 3 * n:3] = (fx - 1.0) / 8.0
    out[7:7 + 3 * n:3] = (fy - 1.0) / 8.0
    out[8:8 + 3 * n:3] = ph / (2.0 * np.pi)
    return out


def _unit(v):
    mn, mx = v.min(), v.max()
    return ((v - mn) / (mx - mn + 1e-9)).astype(np.float32)


def _domain(h, w, mode):
    """(X, Y) in [0,1]. 'polar' swaps cartesian for (radius, angle)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    X, Y = xx / max(w - 1, 1), yy / max(h - 1, 1)
    if mode != 'polar':
        return X, Y
    dx, dy = X - 0.5, Y - 0.5
    return np.hypot(dx, dy) * 2.0, (np.arctan2(dy, dx) / (2 * np.pi) + 0.5)


def _colour(vh, col, mode):
    """v̂ → (H, W, 3) or (H, W, 3, T)."""
    if mode == 'posterize':
        vh = np.round(vh * (N_BANDS - 1)) / (N_BANDS - 1)
    chan = [np.sin(vh * np.pi * (0.5 + 2 * col[2 * k]) + col[2 * k + 1] * np.pi) * 0.5 + 0.5
            for k in range(3)]
    if mode == 'mono':
        out = np.stack([chan[0]] * 3, axis=2)
    else:
        out = np.stack([chan[0] * (0.4 + 0.6 * vh),
                        chan[1] * (0.4 + 0.6 * (1.0 - vh)),
                        chan[2] * (0.3 + 0.7 * 2 * np.abs(vh - 0.5))], axis=2)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def render_image(th, n, mode, h=32, w=32):
    """(h, w, 3) float32 in [0,1]."""
    fx, fy, ph, col = unpack(th, n)
    X, Y = _domain(h, w, mode)
    v = sum(np.sin(2 * np.pi * (fx[i] * X + fy[i] * Y) + ph[i]) for i in range(n))
    return _colour(_unit(v), col, mode)


def render_video(th, n, mode, h=32, w=32, n_frames=32):
    """(h, w, 3, n_frames) float32 in [0,1] — the same waves, travelling.

    v̂ is normalised over the whole volume rather than per frame, or the clip would
    pulse in brightness for no reason: each frame would be stretched to fill [0,1]
    independently of the others.
    """
    fx, fy, ph, col = unpack(th, n)
    X, Y = (a[..., None] for a in _domain(h, w, mode))
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)[None, None, :]
    v = sum(np.sin(2 * np.pi * (fx[i] * X + fy[i] * Y
                                - (fx[i] + fy[i]) / 2 * VIDEO_SPEED * t) + ph[i])
            for i in range(n))
    return _colour(_unit(v), col, mode)


def render_audio(th, n, mode, sr=16000, n_samples=32000):
    """(n_samples,) float32 in [-1,1] — the same formula over time.

    One partial per term: fx sets its pitch, fy a tremolo rate, φ its phase. The R
    colour mapping is reused as the waveshaper, so the timbre and the image's red
    channel come from the same two numbers.
    """
    fx, fy, ph, col = unpack(th, n)
    t = np.arange(n_samples, dtype=np.float32) / sr
    v = np.zeros(n_samples, np.float32)
    for i in range(n):
        f = AUDIO_F0 * 2.0 ** ((fx[i] - 1.0) * AUDIO_OCT)
        trem = 0.5 + 0.5 * np.sin(2 * np.pi * (fy[i] / 9.0 * AUDIO_TREM_HZ) * t + ph[i])
        v += trem * np.sin(2 * np.pi * f * t + ph[i])
    vh = _unit(v)
    if mode == 'posterize':
        vh = np.round(vh * (N_BANDS - 1)) / (N_BANDS - 1)
    out = np.sin(vh * np.pi * (0.5 + 2 * col[0]) + col[1] * np.pi)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


if __name__ == '__main__':
    for lvl, (n, mode) in WAVE_LEVELS.items():
        th = theta(lvl * 325)
        im, vid = render_image(th, n, mode), render_video(th, n, mode)
        au = render_audio(th, n, mode)
        assert im.shape == (32, 32, 3) and vid.shape == (32, 32, 3, 32)
        assert au.shape == (32000,)
        for a, lo, hi in ((im, 0, 1), (vid, 0, 1), (au, -1, 1)):
            assert np.isfinite(a).all() and lo - 1e-6 <= a.min() and a.max() <= hi + 1e-6
        assert im.std() > 0.02, f'L{lvl} image is flat'
        assert np.abs(au).max() > 0.3, f'L{lvl} audio is silent'
        assert np.abs(np.diff(vid, axis=-1)).mean() > 1e-3, f'L{lvl} video is frozen'
        print(f"  L{lvl}: n={n} {mode:<9s} img_std={im.std():.3f} "
              f"motion={np.abs(np.diff(vid, axis=-1)).mean():.4f} "
              f"audio_rms={np.sqrt((au ** 2).mean()):.3f}")

    assert np.array_equal(theta(777), theta(777)), 'theta is not deterministic'
    assert not np.array_equal(theta(10), theta(11)), 'theta does not vary with idx'
    assert theta(0).shape == (D,)

    # The same θ must drive every modality, or the point of these levels is lost.
    th = theta(42)
    assert np.array_equal(render_image(th, 4, 'rgb'), render_image(theta(42), 4, 'rgb'))

    # Sobol should spread better than independent uniforms: measure the mean distance
    # to each point's nearest neighbour — higher means fewer clumps and fewer gaps.
    from scipy.spatial import cKDTree
    S = np.stack([theta(i) for i in range(2048)])
    U = np.random.RandomState(0).rand(2048, D).astype(np.float32)
    d = [float(cKDTree(A).query(A, k=2)[0][:, 1].mean()) for A in (S, U)]
    print(f"  mean nearest-neighbour distance: sobol {d[0]:.4f}  uniform {d[1]:.4f}")
    assert d[0] > d[1], 'Sobol is not spreading better than uniform — check the sampler'

    # Distinct levels must look distinct, not just be labelled differently.
    ims = {l: render_image(theta(9999), n, m) for l, (n, m) in WAVE_LEVELS.items()}
    pairs = [(a, b) for a in ims for b in ims if a < b]
    assert min(float(np.abs(ims[a] - ims[b]).mean()) for a, b in pairs) > 0.02
    print('OK')
