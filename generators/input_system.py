"""Input abstraction: same ``InputState`` shape whether driven by a replay
log (deterministic), Pygame, externally-pushed browser events, or nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class InputState:
    keys_down: set[str] = field(default_factory=set)
    keys_pressed: set[str] = field(default_factory=set)
    keys_released: set[str] = field(default_factory=set)
    mouse_x: float = 0.0
    mouse_y: float = 0.0
    mouse_buttons: set[int] = field(default_factory=set)
    mouse_wheel: float = 0.0


class InputProvider(ABC):
    @abstractmethod
    def poll(self) -> InputState: ...


class NullInputProvider(InputProvider):
    """Always-empty input, for headless/server use."""

    def poll(self) -> InputState:
        return InputState()


class ReplayInputProvider(InputProvider):
    """Replays a recorded sequence of ``InputState`` frames, one per ``poll()``.

    Deterministic by construction: given the same recorded log, every replay
    produces the same sequence of states. After the log is exhausted, keeps
    returning an empty ``InputState`` (keys/buttons released, nothing new
    pressed).
    """

    def __init__(self, frames: list[InputState]):
        self._frames = frames
        self._index = 0

    def poll(self) -> InputState:
        if self._index >= len(self._frames):
            return InputState()
        state = self._frames[self._index]
        self._index += 1
        return state

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._frames)


class WebInputProvider(InputProvider):
    """Accumulates events pushed from a browser bridge (e.g. a websocket
    handler calling ``push_event``), returning the accumulated state on
    ``poll()`` and resetting the per-frame transient sets afterwards.
    """

    def __init__(self):
        self._state = InputState()

    def push_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "keydown":
            key = event["key"]
            if key not in self._state.keys_down:
                self._state.keys_pressed.add(key)
            self._state.keys_down.add(key)
        elif kind == "keyup":
            key = event["key"]
            self._state.keys_down.discard(key)
            self._state.keys_released.add(key)
        elif kind == "mousemove":
            self._state.mouse_x = event["x"]
            self._state.mouse_y = event["y"]
        elif kind == "mousedown":
            self._state.mouse_buttons.add(event["button"])
        elif kind == "mouseup":
            self._state.mouse_buttons.discard(event["button"])
        elif kind == "wheel":
            self._state.mouse_wheel += event["delta"]

    def poll(self) -> InputState:
        snapshot = InputState(
            keys_down=set(self._state.keys_down),
            keys_pressed=set(self._state.keys_pressed),
            keys_released=set(self._state.keys_released),
            mouse_x=self._state.mouse_x,
            mouse_y=self._state.mouse_y,
            mouse_buttons=set(self._state.mouse_buttons),
            mouse_wheel=self._state.mouse_wheel,
        )
        self._state.keys_pressed.clear()
        self._state.keys_released.clear()
        self._state.mouse_wheel = 0.0
        return snapshot


class PygameInputProvider(InputProvider):
    """Real-time input from a Pygame window.

    ponytail: lazy-imports ``pygame`` inside ``poll()`` so it stays an
    optional dependency — the package never requires pygame just to import
    ``generators.input_system``.
    """

    def __init__(self):
        self._prev_keys_down: set[str] = set()

    def poll(self) -> InputState:
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "PygameInputProvider requires the 'pygame' package (pip install pygame)"
            ) from exc

        pygame.event.pump()
        pressed = pygame.key.get_pressed()
        keys_down = {pygame.key.name(i) for i in range(len(pressed)) if pressed[i]}
        mx, my = pygame.mouse.get_pos()
        buttons = {i for i, down in enumerate(pygame.mouse.get_pressed()) if down}

        state = InputState(
            keys_down=keys_down,
            keys_pressed=keys_down - self._prev_keys_down,
            keys_released=self._prev_keys_down - keys_down,
            mouse_x=float(mx),
            mouse_y=float(my),
            mouse_buttons=buttons,
        )
        self._prev_keys_down = keys_down
        return state


if __name__ == "__main__":
    null = NullInputProvider()
    assert null.poll() == InputState()

    frames = [InputState(keys_down={"left"}), InputState(keys_down={"right"})]
    replay = ReplayInputProvider(frames)
    assert replay.poll().keys_down == {"left"}
    assert replay.poll().keys_down == {"right"}
    assert replay.exhausted
    assert replay.poll() == InputState()  # empty after exhaustion

    web = WebInputProvider()
    web.push_event({"type": "keydown", "key": "a"})
    state = web.poll()
    assert state.keys_down == {"a"} and state.keys_pressed == {"a"}
    state2 = web.poll()
    assert state2.keys_pressed == set()  # transient set cleared after poll
    web.push_event({"type": "keyup", "key": "a"})
    state3 = web.poll()
    assert state3.keys_released == {"a"} and "a" not in state3.keys_down

    print("OK — input providers behave deterministically (replay/web/null)")
