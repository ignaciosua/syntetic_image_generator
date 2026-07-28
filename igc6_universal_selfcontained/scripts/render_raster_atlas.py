"""Render one indexed source per level through the supported raster formats."""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
from synthetic_image_generator import (
    N_LEVELS,
    RasterSpec,
    convert_raster,
    extract_alpha,
    level_start,
    make_image,
)


VARIANT_LABELS = (
    "RGB8", "GRIS8", "GRIS16", "B/N", "RGB565", "RGBA2222", "RGBA8888"
)


def raster_specs(width, height):
    return (
        RasterSpec(width=width, height=height, mode="rgb", bits_per_channel=8),
        RasterSpec(
            width=width, height=height, mode="grayscale", bits_per_channel=8
        ),
        RasterSpec(
            width=width, height=height, mode="grayscale", bits_per_channel=16
        ),
        RasterSpec(
            width=width,
            height=height,
            mode="binary",
            bits_per_channel=1,
            dither="ordered",
        ),
        RasterSpec(
            width=width,
            height=height,
            mode="rgb",
            bits_per_channel=(5, 6, 5),
            packed=True,
        ),
        RasterSpec(width=width, height=height, mode="rgba2222"),
        RasterSpec(width=width, height=height, mode="rgba", bits_per_channel=8),
    )


def display_rgb(array, label):
    """Convert one raster representation to RGB8 only for atlas display."""
    if label == "RGB8":
        return np.asarray(array, np.uint8)
    if label == "RGBA8888":
        rgba = np.asarray(array, np.uint8)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        yy, xx = np.indices(rgba.shape[:2])
        checker = np.where(((xx // 4 + yy // 4) % 2)[:, :, None], 190, 130)
        return np.rint(
            rgba[:, :, :3] * alpha + checker * (1.0 - alpha)
        ).astype(np.uint8)
    if label == "GRIS8":
        gray = np.asarray(array, np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)
    if label == "GRIS16":
        gray = np.rint(np.asarray(array, np.float64) / 257.0).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)
    if label == "B/N":
        gray = np.asarray(array, bool).astype(np.uint8) * 255
        return np.repeat(gray[:, :, None], 3, axis=2)
    packed = np.asarray(array, np.uint32)
    if label == "RGB565":
        red = ((packed >> 11) & 31) * 255 // 31
        green = ((packed >> 5) & 63) * 255 // 63
        blue = (packed & 31) * 255 // 31
        return np.stack([red, green, blue], axis=-1).astype(np.uint8)
    else:
        red = ((packed >> 6) & 3) * 85
        green = ((packed >> 4) & 3) * 85
        blue = ((packed >> 2) & 3) * 85
        alpha = (packed & 3).astype(np.float32) / 3.0
        rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
        yy, xx = np.indices(packed.shape)
        checker = np.where(((xx // 4 + yy // 4) % 2)[:, :, None], 190, 130)
        return np.rint(
            rgb * alpha[:, :, None] + checker * (1.0 - alpha[:, :, None])
        ).astype(np.uint8)


def font(filename, size):
    return ImageFont.truetype(ROOT / "assets/fonts" / filename, size)


def centered_text(draw, box_x, box_width, y, text, selected_font, fill):
    bounds = draw.textbbox((0, 0), text, font=selected_font)
    text_width = bounds[2] - bounds[0]
    draw.text(
        (box_x + (box_width - text_width) / 2, y),
        text,
        fill=fill,
        font=selected_font,
    )


def render(output, width=48, height=32, cards_per_row=6, scale=1):
    specs = raster_specs(width, height)
    tile_width, tile_height = width * scale, height * scale
    padding = 7
    level_height = 18
    variant_label_height = 12
    card_width = padding * 2 + len(specs) * tile_width
    card_height = padding * 2 + level_height + variant_label_height + tile_height
    outer = 20
    gap = 8
    header = 82
    rows = math.ceil(N_LEVELS / cards_per_row)
    canvas_width = (
        outer * 2
        + cards_per_row * card_width
        + (cards_per_row - 1) * gap
    )
    canvas_height = (
        outer * 2 + header + rows * card_height + (rows - 1) * gap
    )

    canvas = Image.new("RGB", (canvas_width, canvas_height), (14, 17, 23))
    draw = ImageDraw.Draw(canvas)
    title_font = font("DejaVuSans-Bold.ttf", 23)
    subtitle_font = font("DejaVuSans.ttf", 13)
    level_font = font("DejaVuSans-Bold.ttf", 11)
    variant_font = font("DejaVuSans-Bold.ttf", 7)
    draw.text(
        (outer, outer),
        "Atlas raster del generador sintético",
        fill=(248, 249, 252),
        font=title_font,
    )
    draw.text(
        (outer, outer + 34),
        (
            f"{N_LEVELS} niveles · 1 fuente por nivel · resolución X={width}, Y={height} · "
            "RGB8 / gris8 / gris16 / binario / RGB565 / RGBA2222 / RGBA8888"
        ),
        fill=(157, 170, 191),
        font=subtitle_font,
    )

    for level in range(N_LEVELS):
        row, column = divmod(level, cards_per_row)
        x = outer + column * (card_width + gap)
        y = outer + header + row * (card_height + gap)
        draw.rounded_rectangle(
            (x, y, x + card_width - 1, y + card_height - 1),
            radius=5,
            fill=(27, 32, 42),
            outline=(52, 61, 78),
        )
        draw.text(
            (x + padding, y + 4),
            f"L{level:03d}",
            fill=(229, 234, 242),
            font=level_font,
        )
        source = make_image(level_start(level))
        for variant, (label, spec) in enumerate(zip(VARIANT_LABELS, specs)):
            tile_x = x + padding + variant * tile_width
            label_y = y + padding + level_height
            centered_text(
                draw,
                tile_x,
                tile_width,
                label_y,
                label,
                variant_font,
                (160, 173, 193),
            )
            raster_source = source
            if label in {"RGBA2222", "RGBA8888"}:
                alpha = extract_alpha(source, spec.alpha_mode, level=level)
                raster_source = np.concatenate(
                    [source, alpha[:, :, None]], axis=2
                ).astype(np.float32)
            raster = convert_raster(raster_source, spec)
            image = Image.fromarray(display_rgb(raster, label), "RGB")
            if scale != 1:
                image = image.resize(
                    (tile_width, tile_height), Image.Resampling.NEAREST
                )
            canvas.paste(
                image,
                (tile_x, y + padding + level_height + variant_label_height),
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(
        f"saved {N_LEVELS} sources × {len(specs)} raster formats "
        f"to {output} ({canvas_width}×{canvas_height})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "media/synthetic_generator_raster_atlas.png",
    )
    parser.add_argument("--width", type=int, default=48, help="X resolution")
    parser.add_argument("--height", type=int, default=32, help="Y resolution")
    parser.add_argument("--cards-per-row", type=int, default=6)
    parser.add_argument("--scale", type=int, default=1)
    args = parser.parse_args()
    if min(args.width, args.height, args.cards_per_row, args.scale) < 1:
        raise ValueError("width, height, cards-per-row, and scale must be positive")
    render(
        args.output,
        width=args.width,
        height=args.height,
        cards_per_row=args.cards_per_row,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
