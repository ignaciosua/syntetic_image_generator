"""``idx -> SceneSample``: the public API this whole plan was for.

``make_scene_catalog`` mirrors ``make_image(idx)`` exactly — one index, one
deterministic frame, purely *compositional* (no behaviors attached, physics
only runs a fixed settle window so spawned objects aren't mid-overlap).

``make_scene_trajectory`` adds the time axis: 1-3 compatible behaviors are
drawn deterministically from the same seed and attached before ticking
``n_frames`` forward, recording every frame plus its object/event state —
exactly ``FrameRecord``'s shape, so a trajectory doubles as replay input.

Neither function touches ``make_image``'s indexed contract or reaches into
renderer internals — both are built entirely from the existing public
``SceneGraph``/``PhysicsWorld``/behavior API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .behaviors import compatible_behaviors
from .input_system import InputState, NullInputProvider
from .replay import serialize_objects
from .scene_levels import scene_archetype_of, scene_seed_of

_SETTLE_STEPS = 30
_SETTLE_DT = 1 / 60
_TRAJECTORY_FRAMES = 60
_TRAJECTORY_DT = 1 / 60
_MAX_BEHAVIORS = 3

# One exact trajectory sample per archetype (idx = archetype_index *
# SAMPLES_PER_ARCHETYPE, n_frames=_TRAJECTORY_FRAMES), hashed over the
# concatenated frame bytes. Computed by scripts/print_scene_catalog_hash.py;
# any change touching archetype/behavior geometry or RNG draw order must
# update this alongside a version bump — same rule LEGACY_LEVEL_SAMPLE_SHA256
# already follows for the indexed image contract.
SCENE_TRAJECTORY_SAMPLE_SHA256 = "c368086ad6da66b5e134e2a7074343ff7345ef293c08effb48efa3939a0914a6"


@dataclass
class SceneSample:
    idx: int
    archetype: str
    behaviors: list[str]
    seed: int
    frames: list[np.ndarray]
    objects_state: list[dict]
    events: list[dict]
    input_log: list[InputState] | None = None


def _validate_idx(idx: int) -> None:
    if not isinstance(idx, int) or isinstance(idx, bool):
        raise TypeError("idx must be an int")
    if idx < 0:
        raise ValueError("idx must be non-negative")


def make_scene_catalog(idx: int, *, settle_steps: int = _SETTLE_STEPS) -> SceneSample:
    """One deterministic composed scene, mirroring ``make_image(idx)``.

    Builds the archetype from a seeded RNG, steps physics ``settle_steps``
    times at a fixed dt (so spawned objects aren't caught mid-overlap), then
    renders once. No behaviors are attached — this is compositional
    diversity only; use ``make_scene_trajectory`` for dynamics. Events
    aren't tracked during the settle window (there's no "this frame" to
    attach them to yet), so ``events`` is a single empty placeholder.
    """

    _validate_idx(idx)
    archetype = scene_archetype_of(idx)
    seed = scene_seed_of(idx)
    rng = np.random.RandomState(seed)
    graph = archetype.build(rng)

    null_input = NullInputProvider()
    for _ in range(settle_steps):
        graph.update(_SETTLE_DT, null_input.poll())

    frame = graph.render()
    return SceneSample(
        idx=idx,
        archetype=archetype.name,
        behaviors=[],
        seed=seed,
        frames=[frame],
        objects_state=[serialize_objects(graph.objects)],
        events=[{"contacts": [], "hud": []}],
        input_log=None,
    )


def make_scene_trajectory(
    idx: int, n_frames: int = _TRAJECTORY_FRAMES, *, fixed_dt: float = _TRAJECTORY_DT
) -> SceneSample:
    """``n_frames`` of deterministic behavior, mirroring the same
    ``idx -> seed`` schedule ``make_scene_catalog`` uses.

    1-3 behaviors compatible with the archetype are chosen deterministically
    from the same RNG stream used to build the scene (so a given ``idx``
    always draws the same behaviors), then ticked forward with
    ``NullInputProvider`` (no scripted "agent" driving a player object in
    this pass — see SCENE_CATALOG_PLAN.md's open questions).
    """

    _validate_idx(idx)
    archetype = scene_archetype_of(idx)
    seed = scene_seed_of(idx)
    rng = np.random.RandomState(seed)
    graph = archetype.build(rng)

    candidates = compatible_behaviors(archetype.name)
    chosen_names: list[str] = []
    if candidates:
        n_pick = min(1 + int(rng.randint(0, _MAX_BEHAVIORS)), len(candidates))
        picks = sorted(rng.choice(len(candidates), size=n_pick, replace=False).tolist())
        for i in picks:
            candidates[i].attach(graph, rng)
            chosen_names.append(candidates[i].name)

    input_provider = NullInputProvider()
    frames: list[np.ndarray] = []
    objects_states: list[dict] = []
    events: list[dict] = []
    for _ in range(n_frames):
        graph.update(fixed_dt, input_provider.poll())
        frames.append(graph.render())
        objects_states.append(serialize_objects(graph.objects))
        events.append(
            {
                "contacts": [
                    {"a": c.body_a.name, "b": c.body_b.name} for c in graph.last_contacts
                ],
                "hud": list(graph.last_hud_events),
            }
        )

    return SceneSample(
        idx=idx,
        archetype=archetype.name,
        behaviors=chosen_names,
        seed=seed,
        frames=frames,
        objects_state=objects_states,
        events=events,
        input_log=None,
    )


def make_scene_catalog_batch(indices) -> list[SceneSample]:
    """Ordered batch of ``make_scene_catalog`` samples.

    ponytail: plain sequential loop, no process-pool/autotuning like
    ``make_images()`` has — add that only if a real export is measured slow
    enough to need it; guessing at a parallel strategy now risks the wrong
    one.
    """

    return [make_scene_catalog(idx) for idx in indices]


def make_scene_trajectory_batch(indices, n_frames: int = _TRAJECTORY_FRAMES) -> list[SceneSample]:
    """Ordered batch of ``make_scene_trajectory`` samples. See
    ``make_scene_catalog_batch`` for why this stays sequential for now."""

    return [make_scene_trajectory(idx, n_frames=n_frames) for idx in indices]


if __name__ == "__main__":
    sample1 = make_scene_catalog(0)
    sample2 = make_scene_catalog(0)
    assert np.array_equal(sample1.frames[0], sample2.frames[0])  # deterministic
    assert sample1.archetype == "platformer"
    assert sample1.behaviors == []
    assert set(sample1.objects_state[0]["player"]) == {"x", "y", "tags", "bbox"}

    traj1 = make_scene_trajectory(325, n_frames=10)  # top_down_arena
    traj2 = make_scene_trajectory(325, n_frames=10)
    assert len(traj1.frames) == 10
    for f1, f2 in zip(traj1.frames, traj2.frames):
        assert np.array_equal(f1, f2)  # deterministic across independent calls
    assert traj1.behaviors == traj2.behaviors and traj1.behaviors  # something got attached

    try:
        make_scene_catalog(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative idx")

    batch = make_scene_catalog_batch([0, 1, 2])
    assert [s.idx for s in batch] == [0, 1, 2]

    print("OK — make_scene_catalog/make_scene_trajectory are deterministic and idx-indexed")
