"""Compact ordered primitive IR shared by CPU planners and WebGPU.

Each command is sixteen float32 values.  The representation deliberately keeps
control flow on CPU and describes only the data-parallel raster work.  Integer
geometry remains exactly representable, while the first four operations retain
the smooth-disc behavior originally introduced for level 128.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


COMMAND_STRIDE = 16


class PrimitiveOp(IntEnum):
    SOFT_DISC = 0
    SOFT_RING = 1
    CLIPPED_SOFT_RING = 2
    CLIPPED_SOFT_DISC = 3
    RECT = 4
    BRESENHAM = 5
    ADDITIVE_SOFT_DISC = 6
    AFFINE_RECT = 7
    ELLIPSE = 8
    ELLIPSE_GRADIENT = 9
    SPHERE_PHONG = 10
    MAX_RECT = 11
    CYLINDER_PHONG = 12
    MAX_ELLIPSE = 13
    TORUS_PHONG = 14


def command(
    operation: PrimitiveOp,
    *,
    x0: float,
    y0: float,
    p0: float,
    p1: float = 0.0,
    p2: float = 0.0,
    clip_x: float = 0.0,
    clip_y: float = 0.0,
    clip_radius: float = 0.0,
    opacity: float = 1.0,
    color,
    extras=(0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build one validated float32 IR command."""
    try:
        operation = PrimitiveOp(operation)
    except ValueError as error:
        raise ValueError(f"unsupported primitive operation: {operation}") from error
    rgb = np.asarray(color, np.float32)
    if rgb.shape != (3,):
        raise ValueError("primitive color must contain exactly three channels")
    extra_values = np.asarray(extras, np.float32)
    if extra_values.shape != (4,):
        raise ValueError("primitive extras must contain exactly four values")
    values = np.asarray(
        [
            int(operation),
            x0,
            y0,
            p0,
            p1,
            p2,
            clip_x,
            clip_y,
            opacity,
            rgb[0],
            rgb[1],
            rgb[2],
            extra_values[0],
            extra_values[1],
            extra_values[2],
            extra_values[3],
        ],
        np.float32,
    )
    # For the clipped operations, fields 5–7 describe the parent disc.
    if operation in {
        PrimitiveOp.CLIPPED_SOFT_RING,
        PrimitiveOp.CLIPPED_SOFT_DISC,
    }:
        values[5:8] = (clip_x, clip_y, clip_radius)
    return values


def validate(commands) -> np.ndarray:
    """Return one contiguous command matrix or reject malformed IR."""
    value = np.asarray(commands, np.float32)
    if value.ndim != 2 or value.shape[1] != COMMAND_STRIDE:
        raise ValueError(
            f"primitive commands must have shape (N, {COMMAND_STRIDE})"
        )
    if not len(value):
        raise ValueError("primitive command lists cannot be empty")
    modes = value[:, 0].astype(np.int64)
    if np.any(modes < int(min(PrimitiveOp))) or np.any(
        modes > int(max(PrimitiveOp))
    ):
        raise ValueError("primitive command contains an unsupported operation")
    if not np.isfinite(value).all():
        raise ValueError("primitive commands must contain finite values")
    return np.ascontiguousarray(value)
