"""Record/replay a game session: (recorded input log) + (deterministic
update/render) reproduces every frame byte-identically.

Determinism relies on the caller's own update logic being deterministic
(``SceneGraph.update`` itself has no RNG — ``PhysicsWorld.step`` and
``ParticlePool.update`` are pure math over the current state); this module
only records/replays the input driving it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from .input_system import InputState
from .scene_graph import object_aabb
from .scene_spec import ObjectSpec


@dataclass
class FrameRecord:
    frame: int
    time: float
    input: InputState
    objects_state: dict = field(default_factory=dict)


def serialize_objects(objects: list[ObjectSpec]) -> dict:
    """A per-frame snapshot for replay verification and scene-catalog labels:
    position, tags, and an approximate bounding box (``object_aabb``) per
    object — enough to double as detection-style labels without a second
    pass over the scene."""

    result = {}
    for i, obj in enumerate(objects):
        box = object_aabb(obj)
        result[obj.name or f"#{i}"] = {
            "x": obj.resolved_x,
            "y": obj.resolved_y,
            "tags": sorted(obj.tags),
            "bbox": [box.x0, box.y0, box.x1, box.y1],
        }
    return result


def _input_to_dict(state: InputState) -> dict:
    d = asdict(state)
    d["keys_down"] = sorted(d["keys_down"])
    d["keys_pressed"] = sorted(d["keys_pressed"])
    d["keys_released"] = sorted(d["keys_released"])
    d["mouse_buttons"] = sorted(d["mouse_buttons"])
    return d


def _input_from_dict(d: dict) -> InputState:
    return InputState(
        keys_down=set(d["keys_down"]),
        keys_pressed=set(d["keys_pressed"]),
        keys_released=set(d["keys_released"]),
        mouse_x=d["mouse_x"],
        mouse_y=d["mouse_y"],
        mouse_buttons=set(d["mouse_buttons"]),
        mouse_wheel=d["mouse_wheel"],
    )


class SessionRecorder:
    def __init__(self):
        self.frames: list[FrameRecord] = []

    def record(self, frame: FrameRecord) -> None:
        self.frames.append(frame)

    def save(self, path: str) -> None:
        payload = [
            {
                "frame": f.frame,
                "time": f.time,
                "input": _input_to_dict(f.input),
                "objects_state": f.objects_state,
            }
            for f in self.frames
        ]
        with open(path, "w") as fh:
            json.dump(payload, fh)

    @staticmethod
    def load(path: str) -> "SessionRecorder":
        with open(path) as fh:
            payload = json.load(fh)
        recorder = SessionRecorder()
        for entry in payload:
            recorder.record(
                FrameRecord(
                    frame=entry["frame"],
                    time=entry["time"],
                    input=_input_from_dict(entry["input"]),
                    objects_state=entry["objects_state"],
                )
            )
        return recorder


class SessionPlayer:
    """Deterministic replay engine.

    Given a ``SceneGraph`` (already constructed with the same seed/scene as
    the original session) and a recorded input log, replicates every frame
    byte-identically by driving ``SceneGraph.update``/``render`` with the
    recorded input in order.
    """

    def __init__(self, scene_graph):
        self.scene_graph = scene_graph

    def play(self, recorder: SessionRecorder) -> Iterator:
        for record in recorder.frames:
            self.scene_graph.update(self.scene_graph.clock.fixed_timestep, record.input)
            yield self.scene_graph.render()


if __name__ == "__main__":
    import tempfile

    obj = ObjectSpec(kind="sphere_3d", x=1.0, y=2.0, name="hero", tags={"player"})
    snap = serialize_objects([obj])
    assert snap["hero"]["x"] == 1.0 and snap["hero"]["y"] == 2.0
    assert snap["hero"]["tags"] == ["player"]
    assert len(snap["hero"]["bbox"]) == 4

    recorder = SessionRecorder()
    recorder.record(FrameRecord(frame=0, time=0.0, input=InputState(keys_down={"left"}), objects_state=snap))
    recorder.record(FrameRecord(frame=1, time=1 / 60, input=InputState(), objects_state=snap))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    recorder.save(path)
    loaded = SessionRecorder.load(path)

    assert len(loaded.frames) == 2
    assert loaded.frames[0].input.keys_down == {"left"}
    assert loaded.frames[0].objects_state == snap

    print("OK — session record/save/load round-trips input and object snapshots")
