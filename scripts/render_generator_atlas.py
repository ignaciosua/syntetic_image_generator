"""Render a compact GitHub-friendly atlas from the bundled generator."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "generators"))
from levels import N_LEVELS, SAMPLES_PER_LEVEL, level_start
from synthetic_image_generator import make_image
from wave import WAVE_LEVELS


WAVE_NAMES = {
    148: "wave-2 rgb",
    149: "wave-4 rgb",
    150: "wave-8 rgb",
    151: "wave mono",
    152: "wave posterize",
    153: "wave polar",
}


def font(path, size):
    return ImageFont.truetype(ROOT / "assets/fonts" / path, size)


def render(output, cards_per_row=7, examples=5, scale=2):
    tile = 32 * scale
    card_padding = 8
    label_height = 25
    card_width = card_padding * 2 + examples * tile
    card_height = card_padding * 2 + label_height + tile
    outer = 22
    gap = 10
    header = 76
    rows = math.ceil(N_LEVELS / cards_per_row)
    width = outer * 2 + cards_per_row * card_width + (cards_per_row - 1) * gap
    height = outer * 2 + header + rows * card_height + (rows - 1) * gap

    canvas = Image.new("RGB", (width, height), (14, 17, 23))
    draw = ImageDraw.Draw(canvas)
    title_font = font("DejaVuSans-Bold.ttf", 25)
    subtitle_font = font("DejaVuSans.ttf", 14)
    label_font = font("DejaVuSans-Bold.ttf", 12)
    draw.text(
        (outer, outer),
        "Atlas universal del generador sintético",
        fill=(248, 249, 252),
        font=title_font,
    )
    draw.text(
        (outer, outer + 36),
        f"{N_LEVELS} niveles · {examples} ejemplos deterministas por nivel · RGB 32×32",
        fill=(157, 170, 191),
        font=subtitle_font,
    )

    for level in range(N_LEVELS):
        row, column = divmod(level, cards_per_row)
        x = outer + column * (card_width + gap)
        y = outer + header + row * (card_height + gap)
        draw.rounded_rectangle(
            (x, y, x + card_width - 1, y + card_height - 1),
            radius=6,
            fill=(27, 32, 42),
            outline=(52, 61, 78),
        )
        name = WAVE_NAMES.get(level, "visual")
        draw.text(
            (x + card_padding, y + 6),
            f"L{level:03d} · {name}",
            fill=(229, 234, 242),
            font=label_font,
        )
        start = level_start(level)
        stride = max(1, SAMPLES_PER_LEVEL // examples)
        for example in range(examples):
            index = start + example * stride
            image = np.clip(make_image(index, step=7) * 255, 0, 255)
            image = Image.fromarray(image.round().astype(np.uint8)).resize(
                (tile, tile), Image.Resampling.NEAREST
            )
            image_x = x + card_padding + example * tile
            image_y = y + card_padding + label_height
            canvas.paste(image, (image_x, image_y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(f"saved {N_LEVELS * examples} examples to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "media/synthetic_generator_universal_atlas.png",
    )
    parser.add_argument("--cards-per-row", type=int, default=7)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    if args.cards_per_row < 1 or args.examples < 1 or args.scale < 1:
        raise ValueError("layout arguments must be positive")
    render(args.output, args.cards_per_row, args.examples, args.scale)


if __name__ == "__main__":
    main()
