"""A small CPU particle system: emit, integrate, render onto an RGBA canvas.

Deterministic: callers pass in a ``np.random.RandomState`` (same convention
``make_image``/``make_scene`` use internally), so emission is reproducible
given the same seed and call sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraSpec, world_to_screen

Color = tuple[float, float, float] | tuple[float, float, float, float]


@dataclass
class ParticleEmitterSpec:
    kind: str = "point"  # point, line, circle, rect
    rate: float = 10.0
    lifetime: tuple[float, float] = (0.5, 2.0)
    speed: tuple[float, float] = (10.0, 50.0)
    direction: tuple[float, float] = (0.0, -1.0)
    spread: float = 30.0  # degrees
    start_color: Color = (1.0, 1.0, 1.0)
    end_color: Color = (1.0, 1.0, 1.0, 0.0)
    start_size: float = 4.0
    end_size: float = 0.0
    gravity: tuple[float, float] = (0.0, -98.0)
    max_particles: int = 500


def _rgba(c: Color) -> tuple[float, float, float, float]:
    return (c[0], c[1], c[2], c[3] if len(c) == 4 else 1.0)


class ParticlePool:
    """Fixed-capacity particle buffer for one emitter.

    ``origin`` is optional and only consulted by ``SceneGraph.update`` (this
    module itself never reads it): a fixed ``(x, y)`` world point, or an
    ``ObjectSpec.name`` string to follow that object's position each frame.
    Leave it ``None`` to keep driving ``emit``/``emit_continuous`` manually.
    """

    def __init__(self, emitter: ParticleEmitterSpec, origin: tuple[float, float] | str | None = None):
        self.emitter = emitter
        self.origin = origin
        n = emitter.max_particles
        self.positions = np.zeros((n, 2), np.float32)
        self.velocities = np.zeros((n, 2), np.float32)
        self.ages = np.zeros(n, np.float32)
        self.lifetimes = np.zeros(n, np.float32)
        self.alive = np.zeros(n, bool)
        self._start_sizes = np.zeros(n, np.float32)
        self._end_sizes = np.zeros(n, np.float32)
        self._start_colors = np.zeros((n, 4), np.float32)
        self._end_colors = np.zeros((n, 4), np.float32)
        self._emit_carry = 0.0

    def emit(self, count: int, origin: tuple[float, float], rng: np.random.RandomState) -> int:
        """Spawn up to ``count`` particles at ``origin``. Returns how many actually spawned."""

        free = np.flatnonzero(~self.alive)
        n = min(count, len(free))
        if n == 0:
            return 0
        slots = free[:n]
        e = self.emitter

        base_angle = np.degrees(np.arctan2(e.direction[1], e.direction[0]))
        angles = np.radians(base_angle + rng.uniform(-e.spread / 2, e.spread / 2, n))
        speeds = rng.uniform(e.speed[0], e.speed[1], n)

        self.positions[slots] = origin
        self.velocities[slots, 0] = np.cos(angles) * speeds
        self.velocities[slots, 1] = np.sin(angles) * speeds
        self.lifetimes[slots] = rng.uniform(e.lifetime[0], e.lifetime[1], n)
        self.ages[slots] = 0.0
        self.alive[slots] = True
        self._start_sizes[slots] = e.start_size
        self._end_sizes[slots] = e.end_size
        self._start_colors[slots] = _rgba(e.start_color)
        self._end_colors[slots] = _rgba(e.end_color)
        return n

    def emit_continuous(self, dt: float, origin: tuple[float, float], rng: np.random.RandomState) -> int:
        """Emit at ``emitter.rate`` particles/second, carrying fractional remainders across calls."""

        self._emit_carry += self.emitter.rate * dt
        count = int(self._emit_carry)
        self._emit_carry -= count
        return self.emit(count, origin, rng)

    def update(self, dt: float) -> None:
        if not self.alive.any():
            return
        a = self.alive
        self.velocities[a] += np.asarray(self.emitter.gravity, np.float32) * dt
        self.positions[a] += self.velocities[a] * dt
        self.ages[a] += dt
        self.alive[a] = self.ages[a] < self.lifetimes[a]

    def _current(self) -> tuple[np.ndarray, np.ndarray]:
        """(sizes, colors) for currently alive particles, lerped by age fraction."""

        idx = np.flatnonzero(self.alive)
        t = (self.ages[idx] / np.maximum(self.lifetimes[idx], 1e-6))[:, None]
        sizes = self._start_sizes[idx] * (1 - t[:, 0]) + self._end_sizes[idx] * t[:, 0]
        colors = self._start_colors[idx] * (1 - t) + self._end_colors[idx] * t
        return sizes, colors

    def render(self, img: np.ndarray, camera: CameraSpec | None = None) -> np.ndarray:
        """Alpha-composite alive particles onto RGBA ``img`` as flat-shaded discs."""

        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return img
        sizes, colors = self._current()
        h, w = img.shape[:2]

        for i, size, color in zip(idx, sizes, colors):
            px, py = self.positions[i]
            if camera is not None:
                px, py = world_to_screen(px, py, camera)
            r = max(size, 0.0) / 2.0
            if r <= 0:
                continue
            x0, x1 = max(int(px - r), 0), min(int(px + r) + 1, w)
            y0, y1 = max(int(py - r), 0), min(int(py + r) + 1, h)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            mask = (xx - px) ** 2 + (yy - py) ** 2 <= r * r
            if not mask.any():
                continue
            alpha = float(color[3]) * mask.astype(np.float32)
            patch = img[y0:y1, x0:x1].astype(np.float32)
            rgb = np.asarray(color[:3], np.float32) * 255.0
            blended = rgb * alpha[..., None] + patch[..., :3] * (1 - alpha[..., None])
            out_a = alpha[..., None] * 255.0 + patch[..., 3:4] * (1 - alpha[..., None])
            img[y0:y1, x0:x1, :3] = blended.clip(0, 255).astype(np.uint8)
            img[y0:y1, x0:x1, 3:4] = out_a.clip(0, 255).astype(np.uint8)
        return img


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    emitter = ParticleEmitterSpec(rate=100.0, lifetime=(1.0, 1.0), speed=(0, 0), max_particles=8)
    pool = ParticlePool(emitter)

    spawned = pool.emit(5, (16, 16), rng)
    assert spawned == 5 and pool.alive.sum() == 5

    pool.update(0.5)
    assert np.all(pool.ages[pool.alive] > 0)

    pool.update(0.6)  # ages now > lifetime(1.0) -> dead
    assert pool.alive.sum() == 0

    canvas = np.zeros((32, 32, 4), np.uint8)
    pool.emit(1, (16, 16), rng)
    pool.render(canvas)
    assert canvas[16, 16, 3] > 0  # something got drawn at the particle center

    print("OK — particle emit/update/render lifecycle behaves deterministically")
