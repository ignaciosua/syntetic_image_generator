"""Optional, vendor-neutral WebGPU acceleration.

The module is intentionally lazy: importing the public package never imports
``wgpu`` or initializes a graphics driver.  The first accelerated call selects
the best hardware adapter exposed by wgpu (Vulkan, Metal, DX12, or browser
WebGPU depending on the platform), builds three reusable compute pipelines, and
keeps them cached for subsequent batches.

The first portable kernel targets the plane-wave family.  It is a particularly
good accelerator workload because all pixels are independent except for the
per-image min/max reduction used by the canonical normalization.
"""

from __future__ import annotations

from functools import lru_cache
import os
import threading
from typing import Any, Sequence

import numpy as np

from ._primitive_ir import COMMAND_STRIDE as PRIMITIVE_COMMAND_STRIDE
from ._primitive_ir import validate as validate_primitive_commands
from .scene import scene_values_batch
from .wave import WAVE_LEVELS, theta, unpack


WIDTH = 32
HEIGHT = 32
CHANNELS = 3
PIXELS = WIDTH * HEIGHT
PARAM_STRIDE = 36
_HARDWARE_TYPES = {"DISCRETE_GPU", "INTEGRATED_GPU"}
_ADAPTER_RANK = {
    "DISCRETE_GPU": 4,
    "INTEGRATED_GPU": 3,
    "VIRTUAL_GPU": 2,
    "CPU": 1,
}
_MODE_NUMBER = {
    "rgb": 0,
    "mono": 1,
    "posterize": 2,
    "polar": 3,
}


_SHADER = """
const WIDTH: u32 = 32u;
const HEIGHT: u32 = 32u;
const PIXELS: u32 = 1024u;
const PARAM_STRIDE: u32 = 36u;
const PI: f32 = 3.14159265358979323846;

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read_write> raw_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> min_max: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> reduce_min: array<f32, 256>;
var<workgroup> reduce_max: array<f32, 256>;

fn parameter_base(image: u32) -> u32 {
    return image * PARAM_STRIDE;
}

@compute @workgroup_size(8, 8, 1)
fn render_raw(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= WIDTH || gid.y >= HEIGHT) {
        return;
    }
    let image = gid.z;
    let base = parameter_base(image);
    let mode = u32(params[base + 31u]);
    var x = f32(gid.x) / f32(WIDTH - 1u);
    var y = f32(gid.y) / f32(HEIGHT - 1u);
    if (mode == 3u) {
        let dx = x - 0.5;
        let dy = y - 0.5;
        x = sqrt(dx * dx + dy * dy) * 2.0;
        y = atan2(dy, dx) / (2.0 * PI) + 0.5;
    }

    let term_count = u32(params[base + 30u]);
    var value = 0.0;
    for (var term = 0u; term < 8u; term = term + 1u) {
        if (term < term_count) {
            let fx = params[base + 6u + term];
            let fy = params[base + 14u + term];
            let phase = params[base + 22u + term];
            value = value + sin(2.0 * PI * (fx * x + fy * y) + phase);
        }
    }
    raw_values[image * PIXELS + gid.y * WIDTH + gid.x] = value;
}

@compute @workgroup_size(256, 1, 1)
fn reduce_image(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let image = workgroup.x;
    let lane = local.x;
    var local_min = 3.402823466e+38;
    var local_max = -3.402823466e+38;
    var pixel = lane;
    loop {
        if (pixel >= PIXELS) {
            break;
        }
        let value = raw_values[image * PIXELS + pixel];
        local_min = min(local_min, value);
        local_max = max(local_max, value);
        pixel = pixel + 256u;
    }
    reduce_min[lane] = local_min;
    reduce_max[lane] = local_max;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            reduce_min[lane] = min(reduce_min[lane], reduce_min[lane + stride]);
            reduce_max[lane] = max(reduce_max[lane], reduce_max[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lane == 0u) {
        min_max[image * 2u] = reduce_min[0];
        min_max[image * 2u + 1u] = reduce_max[0];
    }
}

fn wave_channel(value: f32, scale: f32, phase: f32) -> f32 {
    return sin(value * PI * (0.5 + 2.0 * scale) + phase * PI) * 0.5 + 0.5;
}

@compute @workgroup_size(8, 8, 1)
fn color_and_scene(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= WIDTH || gid.y >= HEIGHT) {
        return;
    }
    let image = gid.z;
    let base = parameter_base(image);
    let pixel = gid.y * WIDTH + gid.x;
    let minimum = min_max[image * 2u];
    let maximum = min_max[image * 2u + 1u];
    var unit = (raw_values[image * PIXELS + pixel] - minimum)
        / (maximum - minimum + 1e-9);
    let mode = u32(params[base + 31u]);
    if (mode == 2u) {
        unit = round(unit * 4.0) / 4.0;
    }

    let red_wave = wave_channel(unit, params[base], params[base + 1u]);
    let green_wave = wave_channel(unit, params[base + 2u], params[base + 3u]);
    let blue_wave = wave_channel(unit, params[base + 4u], params[base + 5u]);
    var color: vec3<f32>;
    if (mode == 1u) {
        color = vec3<f32>(red_wave, red_wave, red_wave);
    } else {
        color = vec3<f32>(
            red_wave * (0.4 + 0.6 * unit),
            green_wave * (0.4 + 0.6 * (1.0 - unit)),
            blue_wave * (0.3 + 0.7 * 2.0 * abs(unit - 0.5)),
        );
    }
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    let energy = params[base + 32u];
    let warmth = params[base + 33u];
    let contrast = params[base + 34u];
    color = clamp(
        (color - vec3<f32>(0.5)) * contrast + vec3<f32>(0.5),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    color = clamp(
        color * vec3<f32>(1.0 + 0.25 * warmth, 1.0, 1.0 - 0.25 * warmth),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    ) * energy;

    let output_base = (image * PIXELS + pixel) * 3u;
    output[output_base] = color.x;
    output[output_base + 1u] = color.y;
    output[output_base + 2u] = color.z;
}
"""


_REACTION_SHADER = """
const PIXELS: u32 = 1024u;
const WIDTH: u32 = 32u;

@group(0) @binding(0) var<storage, read> initial: array<f32>;
@group(0) @binding(1) var<storage, read> parameters: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Four complete 32x32 fields use exactly 16 KiB, the portable WebGPU
// minimum for workgroup storage. One 256-lane group owns one whole image,
// so every simulation step can synchronize without another dispatch.
var<workgroup> u_current: array<f32, 1024>;
var<workgroup> v_current: array<f32, 1024>;
var<workgroup> u_next: array<f32, 1024>;
var<workgroup> v_next: array<f32, 1024>;

@compute @workgroup_size(256, 1, 1)
fn reaction_diffusion(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let image = workgroup.x;
    let lane = local.x;
    let initial_base = image * PIXELS * 2u;
    for (var quarter = 0u; quarter < 4u; quarter = quarter + 1u) {
        let pixel = lane + quarter * 256u;
        u_current[pixel] = initial[initial_base + pixel];
        v_current[pixel] = initial[initial_base + PIXELS + pixel];
    }
    workgroupBarrier();

    let parameter_base = image * 4u;
    let feed = parameters[parameter_base];
    let kill = parameters[parameter_base + 1u];
    let steps = u32(parameters[parameter_base + 2u]);
    for (var step = 0u; step < steps; step = step + 1u) {
        for (var quarter = 0u; quarter < 4u; quarter = quarter + 1u) {
            let pixel = lane + quarter * 256u;
            let x = pixel % WIDTH;
            let y = pixel / WIDTH;
            let up = ((y + 31u) % 32u) * WIDTH + x;
            let down = ((y + 1u) % 32u) * WIDTH + x;
            let left = y * WIDTH + ((x + 31u) % 32u);
            let right = y * WIDTH + ((x + 1u) % 32u);
            let u = u_current[pixel];
            let v = v_current[pixel];
            let laplace_u = u_current[up] + u_current[down]
                + u_current[left] + u_current[right] - 4.0 * u;
            let laplace_v = v_current[up] + v_current[down]
                + v_current[left] + v_current[right] - 4.0 * v;
            let uvv = u * v * v;
            u_next[pixel] = u + (0.16 * laplace_u - uvv + feed * (1.0 - u));
            v_next[pixel] = v + (0.08 * laplace_v + uvv - (feed + kill) * v);
        }
        workgroupBarrier();
        for (var quarter = 0u; quarter < 4u; quarter = quarter + 1u) {
            let pixel = lane + quarter * 256u;
            u_current[pixel] = u_next[pixel];
            v_current[pixel] = v_next[pixel];
        }
        workgroupBarrier();
    }

    var local_min = 3.402823466e+38;
    var local_max = -3.402823466e+38;
    for (var quarter = 0u; quarter < 4u; quarter = quarter + 1u) {
        let value = v_current[lane + quarter * 256u];
        local_min = min(local_min, value);
        local_max = max(local_max, value);
    }
    u_next[lane] = local_min;
    v_next[lane] = local_max;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            u_next[lane] = min(u_next[lane], u_next[lane + stride]);
            v_next[lane] = max(v_next[lane], v_next[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let minimum = u_next[0];
    let maximum = v_next[0];
    for (var quarter = 0u; quarter < 4u; quarter = quarter + 1u) {
        let pixel = lane + quarter * 256u;
        output[image * PIXELS + pixel] =
            (v_current[pixel] - minimum) / (maximum - minimum + 1e-8);
    }
}
"""


TRANSFORM_PARAM_STRIDE = 12

_TRANSFORM_SHADER = """
const WIDTH: u32 = 32u;
const HEIGHT: u32 = 32u;
const PIXELS: u32 = 1024u;
const CHANNELS: u32 = 3u;
const PARAM_STRIDE: u32 = 12u;
const PI: f32 = 3.14159265358979323846;

@group(0) @binding(0) var<storage, read> images: array<f32>;
@group(0) @binding(1) var<storage, read> parameters: array<f32>;
@group(0) @binding(2) var<storage, read> displacement: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn source(image: u32, x: i32, y: i32, channel: u32) -> f32 {
    let sx = u32(clamp(x, 0, i32(WIDTH) - 1));
    let sy = u32(clamp(y, 0, i32(HEIGHT) - 1));
    return images[((image * PIXELS + sy * WIDTH + sx) * CHANNELS) + channel];
}

// scipy.ndimage.map_coordinates(order=1, mode="nearest") for a 32x32 plane.
fn bilinear(image: u32, px: f32, py: f32, channel: u32) -> f32 {
    let x = clamp(px, 0.0, f32(WIDTH - 1u));
    let y = clamp(py, 0.0, f32(HEIGHT - 1u));
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let x1 = min(x0 + 1, i32(WIDTH) - 1);
    let y1 = min(y0 + 1, i32(HEIGHT) - 1);
    let tx = x - f32(x0);
    let ty = y - f32(y0);
    let top = mix(source(image, x0, y0, channel),
                  source(image, x1, y0, channel), tx);
    let bottom = mix(source(image, x0, y1, channel),
                     source(image, x1, y1, channel), tx);
    return mix(top, bottom, ty);
}

fn positive_mod(value: f32, modulus: f32) -> f32 {
    return value - floor(value / modulus) * modulus;
}

fn phong(nx: f32, ny: f32, nz: f32, light: vec3<f32>) -> f32 {
    let diffuse = max(0.0, nx * light.x + ny * light.y + nz * light.z);
    let specular = pow(max(0.0, 2.0 * nz * nz - 1.0), 8.0);
    return clamp(0.12 + 0.65 * diffuse + 0.35 * specular, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= WIDTH || gid.y >= HEIGHT) {
        return;
    }
    let image = gid.z;
    let pixel = gid.y * WIDTH + gid.x;
    let parameter_base = image * PARAM_STRIDE;
    let mode = u32(parameters[parameter_base]);
    let x = f32(gid.x);
    let y = f32(gid.y);
    let cx0 = 15.5;
    let cy0 = 15.5;
    let dx0 = x - cx0;
    let dy0 = y - cy0;
    var sx = x;
    var sy = y;
    var fringe = 1.0;
    var lens = false;
    var lens_nx = 0.0;
    var lens_ny = 0.0;
    var lens_nz = 1.0;

    if (mode == 0u) {
        let theta = parameters[parameter_base + 1u];
        let c = cos(theta);
        let s = sin(theta);
        let u = dx0 * c + dy0 * s;
        let v = abs(-dx0 * s + dy0 * c);
        sx = cx0 + u * c - v * s;
        sy = cy0 + u * s + v * c;
    } else if (mode == 1u) {
        let wedges = parameters[parameter_base + 1u];
        let rotation = parameters[parameter_base + 2u];
        let radius = sqrt(dx0 * dx0 + dy0 * dy0);
        let wedge = 2.0 * PI / wedges;
        var angle = positive_mod(atan2(dy0, dx0) - rotation, wedge);
        angle = min(angle, wedge - angle) + rotation;
        sx = cx0 + radius * cos(angle);
        sy = cy0 + radius * sin(angle);
    } else if (mode == 2u) {
        let angle = parameters[parameter_base + 1u];
        let frequency = parameters[parameter_base + 2u];
        let dispersion = parameters[parameter_base + 3u];
        fringe = 0.6 + 0.4 * sin(
            (x * cos(angle) + y * sin(angle)) * frequency
        );
        // The channel-dependent displacement is applied below.
        sx = x;
        sy = y;
    } else if (mode == 3u) {
        let field_base = (image * PIXELS + pixel) * 2u;
        sx = x + displacement[field_base];
        sy = y + displacement[field_base + 1u];
    } else {
        let lens_cx = parameters[parameter_base + 1u];
        let lens_cy = parameters[parameter_base + 2u];
        let radius = parameters[parameter_base + 3u];
        let magnification = parameters[parameter_base + 4u];
        let lens_dx = x - lens_cx;
        let lens_dy = y - lens_cy;
        let distance = (lens_dx * lens_dx + lens_dy * lens_dy)
            / (radius * radius);
        lens = distance <= 1.0;
        if (lens) {
            sx = lens_cx + lens_dx / magnification;
            sy = lens_cy + lens_dy / magnification;
            lens_nx = lens_dx / radius;
            lens_ny = lens_dy / radius;
            lens_nz = sqrt(max(0.0, 1.0 - distance));
        }
    }

    let output_base = (image * PIXELS + pixel) * CHANNELS;
    for (var channel = 0u; channel < CHANNELS; channel = channel + 1u) {
        var channel_sx = sx;
        var channel_sy = sy;
        if (mode == 2u) {
            let angle = parameters[parameter_base + 1u];
            let shift = (f32(channel) - 1.0)
                * parameters[parameter_base + 3u];
            channel_sx = x + shift * cos(angle);
            channel_sy = y + shift * sin(angle);
        }
        var value = bilinear(image, channel_sx, channel_sy, channel);
        if (mode == 2u) {
            value = clamp(value * fringe, 0.0, 1.0);
        } else if (mode == 4u && lens) {
            let light = vec3<f32>(
                parameters[parameter_base + 5u],
                parameters[parameter_base + 6u],
                parameters[parameter_base + 7u],
            );
            let lighting = phong(lens_nx, lens_ny, lens_nz, light);
            let edge = pow(1.0 - abs(lens_nz), 3.0);
            value = clamp(value * (0.7 + 0.3 * lighting) + edge * 0.25,
                          0.0, 1.0);
        }
        output[output_base + channel] = value;
    }
}
"""


_PRIMITIVE_SHADER = """
const WIDTH: u32 = 32u;
const HEIGHT: u32 = 32u;
const PIXELS: u32 = 1024u;
const CHANNELS: u32 = 3u;
const COMMAND_STRIDE: u32 = 16u;

@group(0) @binding(0) var<storage, read> initial: array<f32>;
@group(0) @binding(1) var<storage, read> spans: array<f32>;
@group(0) @binding(2) var<storage, read> commands: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn soft_disc(x: f32, y: f32, cx: f32, cy: f32, radius: f32) -> f32 {
    let dx = x - cx;
    let dy = y - cy;
    return clamp(radius - sqrt(dx * dx + dy * dy) + 0.5, 0.0, 1.0);
}

fn ellipse_distance(
    x: f32,
    y: f32,
    cx: f32,
    cy: f32,
    radius_x: f32,
    radius_y: f32,
    angle: f32,
) -> f32 {
    let dx = x - cx;
    let dy = y - cy;
    let cosine = cos(angle);
    let sine = sin(angle);
    let rotated_x = dx * cosine + dy * sine;
    let rotated_y = -dx * sine + dy * cosine;
    return sqrt(
        (rotated_x * rotated_x) / (radius_x * radius_x)
        + (rotated_y * rotated_y) / (radius_y * radius_y)
    );
}

fn primitive_phong(nx: f32, ny: f32, nz: f32, light: vec3<f32>) -> f32 {
    let diffuse = max(0.0, nx * light.x + ny * light.y + nz * light.z);
    let specular = pow(max(0.0, 2.0 * nz * nz - 1.0), 8.0);
    return clamp(0.12 + 0.65 * diffuse + 0.35 * specular, 0.0, 1.0);
}

fn bresenham_mask(
    pixel_x: i32,
    pixel_y: i32,
    start_x: i32,
    start_y: i32,
    end_x: i32,
    end_y: i32,
    thickness: i32,
) -> f32 {
    var x = start_x;
    var y = start_y;
    let dx = abs(end_x - start_x);
    let dy = abs(end_y - start_y);
    let step_x = select(-1, 1, start_x < end_x);
    let step_y = select(-1, 1, start_y < end_y);
    var error = dx - dy;
    loop {
        if (abs(pixel_x - x) <= thickness
            && abs(pixel_y - y) <= thickness) {
            return 1.0;
        }
        if (x == end_x && y == end_y) {
            break;
        }
        let twice_error = 2 * error;
        if (twice_error > -dy) {
            error = error - dy;
            x = x + step_x;
        }
        if (twice_error < dx) {
            error = error + dx;
            y = y + step_y;
        }
    }
    return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn render_primitives(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= WIDTH || gid.y >= HEIGHT) {
        return;
    }
    let image = gid.z;
    let pixel = gid.y * WIDTH + gid.x;
    let image_base = (image * PIXELS + pixel) * CHANNELS;
    var color = vec3<f32>(
        initial[image_base],
        initial[image_base + 1u],
        initial[image_base + 2u],
    );
    let command_start = u32(spans[image * 2u]);
    let command_count = u32(spans[image * 2u + 1u]);
    for (var index = 0u; index < command_count; index = index + 1u) {
        let base = (command_start + index) * COMMAND_STRIDE;
        let mode = u32(commands[base]);
        let x = f32(gid.x);
        let y = f32(gid.y);
        let command_color = vec3<f32>(
            commands[base + 9u],
            commands[base + 10u],
            commands[base + 11u],
        );
        var outer = 0.0;
        var alpha = 0.0;
        var direct_color = command_color;
        var direct_write = false;
        var maximum_write = false;
        if (mode <= 3u || mode == 6u) {
            outer = soft_disc(
                x, y, commands[base + 1u], commands[base + 2u],
                commands[base + 3u]
            );
            alpha = outer;
        }
        if (mode == 1u) {
            let inner = soft_disc(
                x, y, commands[base + 1u], commands[base + 2u],
                commands[base + 4u]
            );
            alpha = clamp(outer - inner, 0.0, 1.0);
        } else if (mode == 2u) {
            let inner = soft_disc(
                x, y, commands[base + 1u], commands[base + 2u],
                commands[base + 4u]
            );
            let pore = soft_disc(
                x, y, commands[base + 5u], commands[base + 6u],
                commands[base + 7u]
            );
            alpha = clamp(outer - inner, 0.0, 1.0) * pore;
        } else if (mode == 3u) {
            let pore = soft_disc(
                x, y, commands[base + 5u], commands[base + 6u],
                commands[base + 7u]
            );
            alpha = outer * pore;
        } else if (mode == 4u) {
            alpha = select(
                0.0,
                1.0,
                x >= commands[base + 1u]
                    && x <= commands[base + 3u]
                    && y >= commands[base + 2u]
                    && y <= commands[base + 4u],
            );
        } else if (mode == 5u) {
            alpha = bresenham_mask(
                i32(gid.x),
                i32(gid.y),
                i32(commands[base + 1u]),
                i32(commands[base + 2u]),
                i32(commands[base + 3u]),
                i32(commands[base + 4u]),
                i32(commands[base + 5u]),
            );
        } else if (mode == 7u) {
            alpha = select(
                0.0,
                1.0,
                x >= commands[base + 1u]
                    && x <= commands[base + 3u]
                    && y >= commands[base + 2u]
                    && y <= commands[base + 4u],
            );
        } else if (mode == 8u || mode == 9u || mode == 13u) {
            let ellipse_angle = select(
                commands[base + 5u], 0.0, mode == 9u
            );
            let distance = ellipse_distance(
                x,
                y,
                commands[base + 1u],
                commands[base + 2u],
                commands[base + 3u],
                commands[base + 4u],
                ellipse_angle,
            );
            alpha = select(0.0, 1.0, distance <= 1.0);
            if (mode == 9u) {
                let gradient = 1.0 - distance / commands[base + 12u];
                direct_color = clamp(
                    command_color + vec3<f32>(gradient * commands[base + 8u]),
                    vec3<f32>(0.0),
                    vec3<f32>(1.0),
                );
                direct_write = true;
            } else if (mode == 13u) {
                maximum_write = true;
            }
        } else if (mode == 10u) {
            let nx = (x - commands[base + 1u]) / commands[base + 3u];
            let ny = (y - commands[base + 2u]) / commands[base + 3u];
            let distance = nx * nx + ny * ny;
            alpha = select(0.0, 1.0, distance <= 1.0);
            let nz = sqrt(max(0.0, 1.0 - distance));
            let light = vec3<f32>(
                commands[base + 4u],
                commands[base + 5u],
                commands[base + 6u],
            );
            let lighting = primitive_phong(nx, ny, nz, light);
            direct_color = command_color * lighting;
            let iridescence = commands[base + 8u];
            if (iridescence != 0.0) {
                direct_color = direct_color * vec3<f32>(
                    0.62 + 0.5 * sin(nz * 5.5 + iridescence),
                    0.62 + 0.5 * sin(nz * 5.5 + 2.1 + iridescence),
                    0.62 + 0.5 * sin(nz * 5.5 + 4.2 + iridescence),
                );
            }
            let rim = pow(1.0 - abs(nz), 3.0) * commands[base + 7u];
            direct_color = clamp(
                direct_color + command_color * rim,
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
            direct_write = true;
        } else if (mode == 11u) {
            alpha = select(
                0.0,
                1.0,
                x >= commands[base + 1u]
                    && x <= commands[base + 3u]
                    && y >= commands[base + 2u]
                    && y <= commands[base + 4u],
            );
            maximum_write = true;
        } else if (mode == 12u) {
            let dx = x - commands[base + 1u];
            let dy = y - commands[base + 2u];
            let radial = sqrt(dx * dx + dy * dy);
            alpha = select(
                0.0, 1.0, radial <= commands[base + 3u]
            );
            let light = vec3<f32>(
                commands[base + 4u],
                commands[base + 5u],
                commands[base + 6u],
            );
            let lighting = primitive_phong(
                dx / (radial + 1e-8),
                dy / (radial + 1e-8),
                0.0,
                light,
            );
            direct_color = clamp(
                command_color * lighting,
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
            direct_write = true;
        } else if (mode == 14u) {
            let radius = commands[base + 3u];
            let minor_ratio = commands[base + 4u];
            let dx = (x - commands[base + 1u]) / radius;
            let dy = (y - commands[base + 2u]) / radius;
            let radial = sqrt(dx * dx + dy * dy);
            let radial_delta = radial - 1.0;
            let depth_squared = minor_ratio * minor_ratio
                - radial_delta * radial_delta;
            alpha = select(
                0.0, 1.0, depth_squared >= 0.0 && radial > 0.15
            );
            let nz = sqrt(max(0.0, depth_squared)) / minor_ratio;
            let normal_radius = radial_delta / (radial + 1e-8);
            let nx = dx / (radial + 1e-8) * normal_radius;
            let ny = dy / (radial + 1e-8) * normal_radius;
            let light = vec3<f32>(
                commands[base + 5u],
                commands[base + 6u],
                commands[base + 7u],
            );
            let lighting = primitive_phong(nx, ny, nz, light);
            direct_color = command_color * lighting;
            let iridescence = commands[base + 12u];
            if (iridescence != 0.0) {
                direct_color = direct_color * vec3<f32>(
                    0.62 + 0.5 * sin(nz * 5.5 + iridescence),
                    0.62 + 0.5 * sin(nz * 5.5 + 2.1 + iridescence),
                    0.62 + 0.5 * sin(nz * 5.5 + 4.2 + iridescence),
                );
            }
            let rim = pow(1.0 - abs(nz), 3.0) * commands[base + 8u];
            direct_color = clamp(
                direct_color + command_color * rim,
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
            direct_write = true;
        }
        if (mode <= 6u || mode == 8u) {
            alpha = alpha * commands[base + 8u];
        }
        if (direct_write && alpha > 0.0) {
            color = direct_color;
        } else if (maximum_write && alpha > 0.0) {
            color = max(color, command_color);
        } else if ((mode == 4u || mode == 5u || mode == 8u) && alpha > 0.0) {
            color = command_color;
        } else if (mode == 6u) {
            color = clamp(
                color + command_color * alpha,
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
        } else if (mode == 7u && alpha > 0.0) {
            color = clamp(
                color * commands[base + 8u] + command_color,
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
        } else {
            color = clamp(
                color * (1.0 - alpha) + command_color * alpha,
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
        }
    }
    output[image_base] = color.x;
    output[image_base + 1u] = color.y;
    output[image_base + 2u] = color.z;
}
"""


def _adapter_type(info: dict[str, Any]) -> str:
    raw = (
        str(info.get("adapter_type", "UNKNOWN"))
        .upper()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    return {
        "DISCRETEGPU": "DISCRETE_GPU",
        "INTEGRATEDGPU": "INTEGRATED_GPU",
        "VIRTUALGPU": "VIRTUAL_GPU",
        "CPU": "CPU",
    }.get(raw, raw or "UNKNOWN")


def _adapter_summary(adapter: Any) -> dict[str, Any]:
    info = dict(adapter.info)
    adapter_type = _adapter_type(info)
    return {
        "device": str(info.get("device", "")),
        "adapter_type": adapter_type.lower(),
        "backend": str(info.get("backend_type", "")).lower(),
        "vendor": str(info.get("vendor", "")),
        "hardware": adapter_type in _HARDWARE_TYPES,
    }


def _load_wgpu():
    try:
        import wgpu
    except ImportError as error:
        raise RuntimeError(
            "WebGPU acceleration requires the optional 'gpu' dependency; "
            "install synthetic-image-generator[gpu]"
        ) from error
    return wgpu


def _adapters(wgpu) -> list[Any]:
    try:
        adapters = list(wgpu.gpu.enumerate_adapters_sync())
    except Exception:
        adapters = []
    if not adapters:
        try:
            adapters = [
                wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            ]
        except Exception:
            pass
    return [adapter for adapter in adapters if adapter is not None]


def _eligible_adapters(
    wgpu, allow_software: bool, preference: str = ""
) -> list[Any]:
    candidates = _adapters(wgpu)
    candidates.sort(
        key=lambda adapter: _ADAPTER_RANK.get(
            _adapter_type(dict(adapter.info)), 0
        ),
        reverse=True,
    )
    eligible = [
        adapter
        for adapter in candidates
        if _adapter_type(dict(adapter.info)) in _HARDWARE_TYPES or allow_software
    ]
    if preference:
        normalized = preference.casefold()
        eligible = [
            adapter
            for adapter in eligible
            if normalized
            in " ".join(
                str(value)
                for value in (
                    adapter.info.get("vendor", ""),
                    adapter.info.get("device", ""),
                    adapter.info.get("backend_type", ""),
                    adapter.info.get("adapter_type", ""),
                )
            ).casefold()
        ]
        if not eligible:
            raise RuntimeError(
                f"no WebGPU adapter matches SIG_WEBGPU_ADAPTER={preference!r}"
            )
    if eligible:
        return eligible
    if candidates:
        summary = _adapter_summary(candidates[0])
        raise RuntimeError(
            "WebGPU found only a software/CPU adapter "
            f"({summary['device'] or 'unknown'} via {summary['backend'] or 'unknown'}); "
            "a hardware adapter is required for acceleration"
        )
    raise RuntimeError(
        "no WebGPU adapter is visible; install a Vulkan, Metal, or DirectX 12 "
        "driver and expose the GPU device to this process"
    )


@lru_cache(maxsize=8)
def _accelerator_info(preference: str) -> dict[str, Any]:
    try:
        wgpu = _load_wgpu()
    except RuntimeError as error:
        return {
            "available": False,
            "api": "webgpu",
            "reason": str(error),
        }
    candidates = _adapters(wgpu)
    if not candidates:
        return {
            "available": False,
            "api": "webgpu",
            "reason": "no WebGPU adapter is visible to this process",
        }
    candidates.sort(
        key=lambda adapter: _ADAPTER_RANK.get(
            _adapter_type(dict(adapter.info)), 0
        ),
        reverse=True,
    )
    if preference:
        candidates = [
            adapter
            for adapter in candidates
            if preference
            in " ".join(
                str(value)
                for value in (
                    adapter.info.get("vendor", ""),
                    adapter.info.get("device", ""),
                    adapter.info.get("backend_type", ""),
                    adapter.info.get("adapter_type", ""),
                )
            ).casefold()
        ]
        if not candidates:
            return {
                "available": False,
                "api": "webgpu",
                "reason": (
                    "no WebGPU adapter matches "
                    f"SIG_WEBGPU_ADAPTER={preference!r}"
                ),
            }
    summary = _adapter_summary(candidates[0])
    summary.update(
        {
            "available": bool(summary["hardware"]),
            "api": "webgpu",
            "reason": (
                ""
                if summary["hardware"]
                else "only a software/CPU WebGPU adapter is visible"
            ),
        }
    )
    return summary


def accelerator_info() -> dict[str, Any]:
    """Describe the selected WebGPU adapter without raising when unavailable."""
    preference = os.environ.get("SIG_WEBGPU_ADAPTER", "").strip().casefold()
    return dict(_accelerator_info(preference))


def _prepare_parameters(
    indices: Sequence[int], levels: Sequence[int]
) -> np.ndarray:
    parameters = np.zeros((len(indices), PARAM_STRIDE), np.float32)
    for row, (idx, level) in enumerate(zip(indices, levels)):
        term_count, mode = WAVE_LEVELS[int(level)]
        fx, fy, phase, color = unpack(theta(int(idx)), term_count)
        parameters[row, :6] = color
        parameters[row, 6 : 6 + term_count] = fx
        parameters[row, 14 : 14 + term_count] = fy
        parameters[row, 22 : 22 + term_count] = phase
        parameters[row, 30] = term_count
        parameters[row, 31] = _MODE_NUMBER[mode]
    parameters[:, 32:35] = scene_values_batch(indices)
    return parameters


class _WaveRuntime:
    def __init__(self, allow_software: bool, preference: str = ""):
        self.wgpu = _load_wgpu()
        errors = []
        for adapter in _eligible_adapters(
            self.wgpu, allow_software, preference
        ):
            try:
                device = adapter.request_device_sync(label="sig-webgpu")
            except Exception as error:
                errors.append(f"{adapter.info.get('device', 'unknown')}: {error}")
                continue
            self.adapter = adapter
            self.device = device
            break
        else:
            detail = "; ".join(errors) or "no eligible adapter"
            raise RuntimeError(
                f"WebGPU adapters were found but no device could be created: {detail}"
            )
        self.info = _adapter_summary(self.adapter)
        self.lock = threading.Lock()
        self._build_pipelines()
        self._build_reaction_pipeline()
        self._build_transform_pipeline()
        self._build_primitive_pipeline()
        binding_limit = int(
            self.adapter.limits.get(
                "max-storage-buffer-binding-size", 128 * 1024 * 1024
            )
        )
        self.binding_limit = binding_limit
        dimension_limit = int(
            self.adapter.limits.get(
                "max-compute-workgroups-per-dimension", 65535
            )
        )
        per_image_output = PIXELS * CHANNELS * np.dtype(np.float32).itemsize
        self.max_batch = max(
            1, min(dimension_limit, binding_limit // per_image_output)
        )

    def _build_pipelines(self) -> None:
        wgpu = self.wgpu
        entries = []
        for binding in range(4):
            entries.append(
                {
                    "binding": binding,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": (
                            wgpu.BufferBindingType.read_only_storage
                            if binding == 0
                            else wgpu.BufferBindingType.storage
                        ),
                        "has_dynamic_offset": False,
                    },
                }
            )
        self.bind_group_layout = self.device.create_bind_group_layout(
            label="sig-wave-layout", entries=entries
        )
        pipeline_layout = self.device.create_pipeline_layout(
            label="sig-wave-pipeline-layout",
            bind_group_layouts=[self.bind_group_layout],
        )
        shader = self.device.create_shader_module(
            label="sig-wave-wgsl", code=_SHADER
        )
        self.pipelines = {
            entry: self.device.create_compute_pipeline(
                label=f"sig-{entry}",
                layout=pipeline_layout,
                compute={"module": shader, "entry_point": entry},
            )
            for entry in ("render_raw", "reduce_image", "color_and_scene")
        }

    def _build_reaction_pipeline(self) -> None:
        wgpu = self.wgpu
        self.reaction_bind_group_layout = (
            self.device.create_bind_group_layout(
                label="sig-reaction-layout",
                entries=[
                    {
                        "binding": binding,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {
                            "type": (
                                wgpu.BufferBindingType.read_only_storage
                                if binding < 2
                                else wgpu.BufferBindingType.storage
                            ),
                            "has_dynamic_offset": False,
                        },
                    }
                    for binding in range(3)
                ],
            )
        )
        pipeline_layout = self.device.create_pipeline_layout(
            label="sig-reaction-pipeline-layout",
            bind_group_layouts=[self.reaction_bind_group_layout],
        )
        shader = self.device.create_shader_module(
            label="sig-reaction-wgsl", code=_REACTION_SHADER
        )
        self.reaction_pipeline = self.device.create_compute_pipeline(
            label="sig-reaction-diffusion",
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "reaction_diffusion"},
        )

    def _build_transform_pipeline(self) -> None:
        wgpu = self.wgpu
        self.transform_bind_group_layout = self.device.create_bind_group_layout(
            label="sig-transform-layout",
            entries=[
                {
                    "binding": binding,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": (
                            wgpu.BufferBindingType.read_only_storage
                            if binding < 3
                            else wgpu.BufferBindingType.storage
                        ),
                        "has_dynamic_offset": False,
                    },
                }
                for binding in range(4)
            ],
        )
        pipeline_layout = self.device.create_pipeline_layout(
            label="sig-transform-pipeline-layout",
            bind_group_layouts=[self.transform_bind_group_layout],
        )
        shader = self.device.create_shader_module(
            label="sig-transform-wgsl", code=_TRANSFORM_SHADER
        )
        self.transform_pipeline = self.device.create_compute_pipeline(
            label="sig-transform",
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "transform"},
        )

    def _build_primitive_pipeline(self) -> None:
        wgpu = self.wgpu
        self.primitive_bind_group_layout = self.device.create_bind_group_layout(
            label="sig-primitive-layout",
            entries=[
                {
                    "binding": binding,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": (
                            wgpu.BufferBindingType.read_only_storage
                            if binding < 3
                            else wgpu.BufferBindingType.storage
                        ),
                        "has_dynamic_offset": False,
                    },
                }
                for binding in range(4)
            ],
        )
        pipeline_layout = self.device.create_pipeline_layout(
            label="sig-primitive-pipeline-layout",
            bind_group_layouts=[self.primitive_bind_group_layout],
        )
        shader = self.device.create_shader_module(
            label="sig-primitive-wgsl", code=_PRIMITIVE_SHADER
        )
        self.primitive_pipeline = self.device.create_compute_pipeline(
            label="sig-primitive",
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "render_primitives"},
        )

    def _render_chunk(self, parameters: np.ndarray) -> np.ndarray:
        wgpu = self.wgpu
        count = len(parameters)
        params_buffer = self.device.create_buffer_with_data(
            label="sig-wave-params",
            data=np.ascontiguousarray(parameters, np.float32),
            usage=wgpu.BufferUsage.STORAGE,
        )
        raw_buffer = self.device.create_buffer(
            label="sig-wave-raw",
            size=count * PIXELS * 4,
            usage=wgpu.BufferUsage.STORAGE,
        )
        stats_buffer = self.device.create_buffer(
            label="sig-wave-minmax",
            size=count * 2 * 4,
            usage=wgpu.BufferUsage.STORAGE,
        )
        output_buffer = self.device.create_buffer(
            label="sig-wave-output",
            size=count * PIXELS * CHANNELS * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        buffers = (params_buffer, raw_buffer, stats_buffer, output_buffer)
        bind_group = self.device.create_bind_group(
            label="sig-wave-bind-group",
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": binding,
                    "resource": {
                        "buffer": buffer,
                        "offset": 0,
                        "size": buffer.size,
                    },
                }
                for binding, buffer in enumerate(buffers)
            ],
        )

        encoder = self.device.create_command_encoder(label="sig-wave-commands")
        for entry, dispatch in (
            ("render_raw", (4, 4, count)),
            ("reduce_image", (count, 1, 1)),
            ("color_and_scene", (4, 4, count)),
        ):
            compute_pass = encoder.begin_compute_pass(
                label=f"sig-{entry}-pass"
            )
            compute_pass.set_pipeline(self.pipelines[entry])
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(*dispatch)
            compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        mapped = self.device.queue.read_buffer(output_buffer)
        return np.frombuffer(mapped, dtype=np.float32).reshape(
            count, HEIGHT, WIDTH, CHANNELS
        ).copy()

    def render(self, parameters: np.ndarray) -> np.ndarray:
        if not len(parameters):
            return np.empty((0, HEIGHT, WIDTH, CHANNELS), np.float32)
        chunks = []
        with self.lock:
            for start in range(0, len(parameters), self.max_batch):
                chunks.append(
                    self._render_chunk(parameters[start : start + self.max_batch])
                )
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def _reaction_chunk(
        self, initial: np.ndarray, parameters: np.ndarray
    ) -> np.ndarray:
        wgpu = self.wgpu
        count = len(parameters)
        initial_buffer = self.device.create_buffer_with_data(
            label="sig-reaction-initial",
            data=np.ascontiguousarray(initial, np.float32),
            usage=wgpu.BufferUsage.STORAGE,
        )
        parameter_buffer = self.device.create_buffer_with_data(
            label="sig-reaction-params",
            data=np.ascontiguousarray(parameters, np.float32),
            usage=wgpu.BufferUsage.STORAGE,
        )
        output_buffer = self.device.create_buffer(
            label="sig-reaction-output",
            size=count * PIXELS * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        buffers = (initial_buffer, parameter_buffer, output_buffer)
        bind_group = self.device.create_bind_group(
            label="sig-reaction-bind-group",
            layout=self.reaction_bind_group_layout,
            entries=[
                {
                    "binding": binding,
                    "resource": {
                        "buffer": buffer,
                        "offset": 0,
                        "size": buffer.size,
                    },
                }
                for binding, buffer in enumerate(buffers)
            ],
        )
        encoder = self.device.create_command_encoder(
            label="sig-reaction-commands"
        )
        compute_pass = encoder.begin_compute_pass(label="sig-reaction-pass")
        compute_pass.set_pipeline(self.reaction_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(count, 1, 1)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        mapped = self.device.queue.read_buffer(output_buffer)
        return np.frombuffer(mapped, dtype=np.float32).reshape(
            count, HEIGHT, WIDTH
        ).copy()

    def render_reaction(
        self, initial: np.ndarray, parameters: np.ndarray
    ) -> np.ndarray:
        if not len(parameters):
            return np.empty((0, HEIGHT, WIDTH), np.float32)
        chunks = []
        with self.lock:
            for start in range(0, len(parameters), self.max_batch):
                stop = start + self.max_batch
                chunks.append(
                    self._reaction_chunk(
                        initial[start:stop], parameters[start:stop]
                    )
                )
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def _transform_chunk(
        self,
        images: np.ndarray,
        parameters: np.ndarray,
        displacement: np.ndarray,
    ) -> np.ndarray:
        wgpu = self.wgpu
        count = len(parameters)
        image_buffer = self.device.create_buffer_with_data(
            label="sig-transform-images",
            data=np.ascontiguousarray(images, np.float32),
            usage=wgpu.BufferUsage.STORAGE,
        )
        parameter_buffer = self.device.create_buffer_with_data(
            label="sig-transform-params",
            data=np.ascontiguousarray(parameters, np.float32),
            usage=wgpu.BufferUsage.STORAGE,
        )
        displacement_buffer = self.device.create_buffer_with_data(
            label="sig-transform-displacement",
            data=np.ascontiguousarray(displacement, np.float32),
            usage=wgpu.BufferUsage.STORAGE,
        )
        output_buffer = self.device.create_buffer(
            label="sig-transform-output",
            size=count * PIXELS * CHANNELS * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        buffers = (
            image_buffer,
            parameter_buffer,
            displacement_buffer,
            output_buffer,
        )
        bind_group = self.device.create_bind_group(
            label="sig-transform-bind-group",
            layout=self.transform_bind_group_layout,
            entries=[
                {
                    "binding": binding,
                    "resource": {
                        "buffer": buffer,
                        "offset": 0,
                        "size": buffer.size,
                    },
                }
                for binding, buffer in enumerate(buffers)
            ],
        )
        encoder = self.device.create_command_encoder(
            label="sig-transform-commands"
        )
        compute_pass = encoder.begin_compute_pass(label="sig-transform-pass")
        compute_pass.set_pipeline(self.transform_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(4, 4, count)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        mapped = self.device.queue.read_buffer(output_buffer)
        return np.frombuffer(mapped, dtype=np.float32).reshape(
            count, HEIGHT, WIDTH, CHANNELS
        ).copy()

    def render_transform(
        self,
        images: np.ndarray,
        parameters: np.ndarray,
        displacement: np.ndarray,
    ) -> np.ndarray:
        if not len(parameters):
            return np.empty((0, HEIGHT, WIDTH, CHANNELS), np.float32)
        chunks = []
        with self.lock:
            for start in range(0, len(parameters), self.max_batch):
                stop = start + self.max_batch
                chunks.append(
                    self._transform_chunk(
                        images[start:stop],
                        parameters[start:stop],
                        displacement[start:stop],
                    )
                )
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def _primitive_chunk(
        self,
        initial: np.ndarray,
        spans: np.ndarray,
        commands: np.ndarray,
    ) -> np.ndarray:
        wgpu = self.wgpu
        count = len(initial)
        initial_buffer = self.device.create_buffer_with_data(
            label="sig-primitive-initial",
            data=np.ascontiguousarray(initial, np.float32),
            usage=wgpu.BufferUsage.STORAGE,
        )
        span_buffer = self.device.create_buffer_with_data(
            label="sig-primitive-spans",
            data=np.ascontiguousarray(spans, np.float32),
            usage=wgpu.BufferUsage.STORAGE,
        )
        command_buffer = self.device.create_buffer_with_data(
            label="sig-primitive-commands",
            data=np.ascontiguousarray(commands, np.float32),
            usage=wgpu.BufferUsage.STORAGE,
        )
        output_buffer = self.device.create_buffer(
            label="sig-primitive-output",
            size=count * PIXELS * CHANNELS * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        buffers = (
            initial_buffer,
            span_buffer,
            command_buffer,
            output_buffer,
        )
        bind_group = self.device.create_bind_group(
            label="sig-primitive-bind-group",
            layout=self.primitive_bind_group_layout,
            entries=[
                {
                    "binding": binding,
                    "resource": {
                        "buffer": buffer,
                        "offset": 0,
                        "size": buffer.size,
                    },
                }
                for binding, buffer in enumerate(buffers)
            ],
        )
        encoder = self.device.create_command_encoder(
            label="sig-primitive-commands"
        )
        compute_pass = encoder.begin_compute_pass(label="sig-primitive-pass")
        compute_pass.set_pipeline(self.primitive_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(4, 4, count)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        mapped = self.device.queue.read_buffer(output_buffer)
        return np.frombuffer(mapped, dtype=np.float32).reshape(
            count, HEIGHT, WIDTH, CHANNELS
        ).copy()

    def render_primitives(
        self,
        initial: np.ndarray,
        command_lists: Sequence[np.ndarray],
    ) -> np.ndarray:
        if not len(initial):
            return np.empty((0, HEIGHT, WIDTH, CHANNELS), np.float32)
        chunks = []
        with self.lock:
            start = 0
            while start < len(initial):
                stop = start
                command_bytes = 0
                while stop < len(initial) and stop - start < self.max_batch:
                    next_bytes = (
                        np.asarray(command_lists[stop]).size
                        * np.dtype(np.float32).itemsize
                    )
                    if (
                        stop > start
                        and command_bytes + next_bytes > self.binding_limit
                    ):
                        break
                    if next_bytes > self.binding_limit:
                        raise RuntimeError(
                            "one render plan exceeds the adapter's storage "
                            "buffer binding limit"
                        )
                    command_bytes += next_bytes
                    stop += 1
                selected = command_lists[start:stop]
                counts = np.asarray([len(item) for item in selected], np.int64)
                offsets = np.concatenate(([0], np.cumsum(counts[:-1])))
                spans = np.column_stack((offsets, counts)).astype(np.float32)
                commands = np.concatenate(selected, axis=0)
                chunks.append(
                    self._primitive_chunk(
                        initial[start:stop], spans, commands
                    )
                )
                start = stop
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


_RUNTIME_LOCK = threading.Lock()
_RUNTIMES: dict[tuple[bool, str], _WaveRuntime] = {}
_PARAMETER_LOCK = threading.Lock()


def _software_enabled() -> bool:
    return os.environ.get("SIG_WEBGPU_ALLOW_SOFTWARE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _runtime(allow_software: bool | None = None) -> _WaveRuntime:
    if allow_software is None:
        allow_software = _software_enabled()
    preference = os.environ.get("SIG_WEBGPU_ADAPTER", "").strip()
    key = (bool(allow_software), preference.casefold())
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(key)
        if runtime is None:
            runtime = _WaveRuntime(bool(allow_software), preference)
            _RUNTIMES[key] = runtime
        return runtime


def render_wave_images(
    indices: Sequence[int],
    levels: Sequence[int],
    *,
    allow_software: bool | None = None,
) -> np.ndarray:
    """Render scene-adjusted wave images through the selected WebGPU adapter."""
    if len(indices) != len(levels):
        raise ValueError("indices and levels must have the same length")
    if any(int(level) not in WAVE_LEVELS for level in levels):
        raise ValueError("WebGPU wave rendering received a non-wave level")
    # wave.theta() lazily initializes a shared Sobol table.  Keep that one-time
    # initialization safe without holding the legacy renderer lock during GPU
    # submission or readback.
    with _PARAMETER_LOCK:
        parameters = _prepare_parameters(indices, levels)
    return _runtime(allow_software).render(parameters)


def render_reaction_fields(
    initial_u: np.ndarray,
    initial_v: np.ndarray,
    parameters: np.ndarray,
    *,
    allow_software: bool | None = None,
) -> np.ndarray:
    """Evolve a homogeneous batch of prepared 32x32 Gray-Scott fields."""
    u = np.asarray(initial_u, np.float32)
    v = np.asarray(initial_v, np.float32)
    params = np.asarray(parameters, np.float32)
    if u.shape != v.shape or u.ndim != 3 or u.shape[1:] != (HEIGHT, WIDTH):
        raise ValueError("initial reaction fields must have shape (N, 32, 32)")
    if params.shape != (len(u), 4):
        raise ValueError("reaction parameters must have shape (N, 4)")
    initial = np.concatenate(
        [u.reshape(len(u), PIXELS), v.reshape(len(v), PIXELS)], axis=1
    )
    return _runtime(allow_software).render_reaction(initial, params)


def render_transform_images(
    images: np.ndarray,
    parameters: np.ndarray,
    displacement: np.ndarray | None = None,
    *,
    allow_software: bool | None = None,
) -> np.ndarray:
    """Apply a mixed batch of portable symmetry/optics transforms."""
    source = np.asarray(images, np.float32)
    params = np.asarray(parameters, np.float32)
    if source.ndim != 4 or source.shape[1:] != (
        HEIGHT,
        WIDTH,
        CHANNELS,
    ):
        raise ValueError("transform images must have shape (N, 32, 32, 3)")
    if params.shape != (len(source), TRANSFORM_PARAM_STRIDE):
        raise ValueError(
            f"transform parameters must have shape "
            f"(N, {TRANSFORM_PARAM_STRIDE})"
        )
    if displacement is None:
        fields = np.zeros((len(source), HEIGHT, WIDTH, 2), np.float32)
    else:
        fields = np.asarray(displacement, np.float32)
        if fields.shape != (len(source), HEIGHT, WIDTH, 2):
            raise ValueError(
                "transform displacement must have shape (N, 32, 32, 2)"
            )
    modes = params[:, 0]
    if np.any(modes < 0) or np.any(modes > 4):
        raise ValueError("transform modes must be between 0 and 4")
    return _runtime(allow_software).render_transform(source, params, fields)


def render_primitive_images(
    initial: np.ndarray,
    command_lists: Sequence[np.ndarray],
    *,
    allow_software: bool | None = None,
) -> np.ndarray:
    """Evaluate variable-length ordered primitive IR in one GPU batch."""
    source = np.asarray(initial, np.float32)
    if source.ndim != 4 or source.shape[1:] != (
        HEIGHT,
        WIDTH,
        CHANNELS,
    ):
        raise ValueError("primitive images must have shape (N, 32, 32, 3)")
    if len(command_lists) != len(source):
        raise ValueError("one primitive command list is required per image")
    normalized = []
    for commands in command_lists:
        normalized.append(validate_primitive_commands(commands))
    return _runtime(allow_software).render_primitives(source, normalized)


def render_pore_images(
    initial: np.ndarray,
    command_lists: Sequence[np.ndarray],
    *,
    allow_software: bool | None = None,
) -> np.ndarray:
    """Backward-compatible alias for the original level-128 helper."""
    return render_primitive_images(
        initial, command_lists, allow_software=allow_software
    )
