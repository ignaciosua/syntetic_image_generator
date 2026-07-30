"""Composable rect/label/button/panel widgets for a HUD, drawn onto an RGBA
canvas. Reuses the generator's existing 5x7 bitmap font (``_stamp_glyph``)
instead of reinventing text rendering.

No callbacks embedded on widgets — ``on_click`` is just an event name; the
caller's game loop dispatches it to whatever handler it wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .input_system import InputState
from .synthetic_image_generator import _FONT, _stamp_glyph

Color = tuple[float, float, float]

_GLYPH_ADVANCE_RATIO = 0.7


@dataclass
class WidgetSpec:
    kind: str  # rect, label, button, slider, image, panel
    x: float
    y: float
    width: float
    height: float
    text: str = ""
    color: Color = (0.2, 0.2, 0.3)
    text_color: Color = (1.0, 1.0, 1.0)
    font_size: int = 12
    children: list["WidgetSpec"] = field(default_factory=list)
    on_click: str | None = None
    visible: bool = True
    layer: int = 999


def _draw_rect(canvas: np.ndarray, x, y, w, h, color: Color, alpha: float = 1.0) -> None:
    h_img, w_img = canvas.shape[:2]
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x + w), w_img), min(int(y + h), h_img)
    if x0 >= x1 or y0 >= y1:
        return
    rgb = (np.asarray(color, np.float32) * 255.0).clip(0, 255)
    patch = canvas[y0:y1, x0:x1].astype(np.float32)
    blended = rgb * alpha + patch[..., :3] * (1 - alpha)
    canvas[y0:y1, x0:x1, :3] = blended.astype(np.uint8)
    if canvas.shape[2] == 4:
        canvas[y0:y1, x0:x1, 3] = np.maximum(canvas[y0:y1, x0:x1, 3], int(alpha * 255))


def _draw_text(canvas: np.ndarray, x, y, text: str, color: Color, font_size: int) -> None:
    advance = max(1, int(font_size * _GLYPH_ADVANCE_RATIO))
    cx = int(x)
    rgb = (np.asarray(color, np.float32) * 255.0).clip(0, 255)
    for ch in text.upper():
        if ch == " ":
            cx += advance
            continue
        if ch not in _FONT:
            cx += advance
            continue
        mask = _stamp_glyph(ch, font_size)
        h_img, w_img = canvas.shape[:2]
        x0, y0 = max(cx, 0), max(int(y), 0)
        x1, y1 = min(cx + font_size, w_img), min(int(y) + font_size, h_img)
        if x0 < x1 and y0 < y1:
            sub_mask = mask[y0 - int(y) : y1 - int(y), x0 - cx : x1 - cx]
            patch = canvas[y0:y1, x0:x1].astype(np.float32)
            alpha = sub_mask[..., None]
            blended = rgb * alpha + patch[..., :3] * (1 - alpha)
            canvas[y0:y1, x0:x1, :3] = blended.astype(np.uint8)
            if canvas.shape[2] == 4:
                canvas[y0:y1, x0:x1, 3] = np.maximum(canvas[y0:y1, x0:x1, 3], (alpha[..., 0] * 255).astype(np.uint8))
        cx += advance


def _flatten(widgets: list[WidgetSpec]) -> list[WidgetSpec]:
    out: list[WidgetSpec] = []
    for w in widgets:
        out.append(w)
        out.extend(_flatten(w.children))
    return out


def _hit(w: WidgetSpec, x: float, y: float) -> bool:
    return w.x <= x <= w.x + w.width and w.y <= y <= w.y + w.height


class HUD:
    def __init__(self, widgets: list[WidgetSpec] | None = None):
        self.widgets = widgets or []
        self._prev_buttons: set[int] = set()

    def handle_input(self, input_state: InputState) -> list[str]:
        """Returns event names for widgets whose ``on_click`` fired this frame."""

        newly_pressed = input_state.mouse_buttons - self._prev_buttons
        self._prev_buttons = set(input_state.mouse_buttons)
        if not newly_pressed:
            return []

        events: list[str] = []
        for w in reversed(_flatten(self.widgets)):  # topmost (last-drawn) first
            if not w.visible or w.on_click is None:
                continue
            if _hit(w, input_state.mouse_x, input_state.mouse_y):
                events.append(w.on_click)
        return events

    def render(self, viewport: np.ndarray) -> np.ndarray:
        for w in sorted(_flatten(self.widgets), key=lambda w: w.layer):
            if not w.visible:
                continue
            if w.kind in ("rect", "button", "panel", "slider"):
                _draw_rect(viewport, w.x, w.y, w.width, w.height, w.color)
            if w.text:
                pad = 2
                _draw_text(viewport, w.x + pad, w.y + pad, w.text, w.text_color, w.font_size)
        return viewport


if __name__ == "__main__":
    button = WidgetSpec(kind="button", x=10, y=10, width=40, height=20, text="OK", on_click="confirm")
    hud = HUD([button])

    canvas = np.zeros((64, 64, 4), np.uint8)
    hud.render(canvas)
    assert canvas[15, 15, 3] > 0  # button rect drawn

    click = InputState(mouse_x=20, mouse_y=15, mouse_buttons={0})
    events = hud.handle_input(click)
    assert events == ["confirm"]

    held = InputState(mouse_x=20, mouse_y=15, mouse_buttons={0})
    assert hud.handle_input(held) == []  # not a *new* press, no repeat event

    miss = InputState(mouse_x=0, mouse_y=0, mouse_buttons=set())
    hud.handle_input(miss)
    outside_click = InputState(mouse_x=0, mouse_y=0, mouse_buttons={0})
    assert hud.handle_input(outside_click) == []

    print("OK — HUD renders widgets and dispatches click events on press edges only")
