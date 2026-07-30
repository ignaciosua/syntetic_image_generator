"""Grid tilemaps: storage, collision mask, and camera-culled rendering.

Tile id 0 means empty. Tile id ``n >= 1`` maps to ``tileset.regions[n - 1]``
(index-based convention — ponytail: simplest mapping that needs no extra
lookup table; switch to name-keyed ids if a tileset ever needs sparse ids).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .atlas import AtlasSpec, stamp_sprite
from .camera import CameraSpec, world_to_screen


@dataclass
class TilemapSpec:
    width: int
    height: int
    tile_size: int
    tiles: np.ndarray  # (height, width) uint16 tile ids, 0 = empty
    tileset: AtlasSpec
    layer: int = 0

    def tile_at(self, x: int, y: int) -> int:
        return int(self.tiles[y, x])

    def set_tile(self, x: int, y: int, tile_id: int) -> None:
        self.tiles[y, x] = tile_id

    def collision_mask(self) -> np.ndarray:
        """Bool mask of solid (non-zero) tiles."""

        return self.tiles != 0


class TilemapRenderer:
    @staticmethod
    def render(tilemap: TilemapSpec, camera: CameraSpec) -> np.ndarray:
        """Renders only tiles whose world-space cell overlaps the camera's
        viewport, stamped via numpy slice-and-composite."""

        canvas = np.zeros((camera.viewport_height, camera.viewport_width, 4), np.uint8)
        ts = tilemap.tile_size

        half_w = camera.viewport_width / 2.0 / camera.zoom
        half_h = camera.viewport_height / 2.0 / camera.zoom
        tx0 = max(0, math.floor((camera.world_x - half_w) / ts))
        tx1 = min(tilemap.width, math.ceil((camera.world_x + half_w) / ts) + 1)
        ty0 = max(0, math.floor((camera.world_y - half_h) / ts))
        ty1 = min(tilemap.height, math.ceil((camera.world_y + half_h) / ts) + 1)

        for ty in range(ty0, ty1):
            for tx in range(tx0, tx1):
                tile_id = tilemap.tile_at(tx, ty)
                if tile_id == 0:
                    continue
                region_idx = tile_id - 1
                if not (0 <= region_idx < len(tilemap.tileset.regions)):
                    continue
                region = tilemap.tileset.regions[region_idx]
                wx, wy = (tx + 0.5) * ts, (ty + 0.5) * ts
                sx, sy = world_to_screen(wx, wy, camera)
                scale = (ts * camera.zoom) / region.width
                stamp_sprite(canvas, tilemap.tileset, region, sx, sy, scale=scale)
        return canvas


if __name__ == "__main__":
    atlas_img = np.zeros((8, 16, 4), np.uint8)
    atlas_img[:, 0:8] = (255, 0, 0, 255)
    atlas_img[:, 8:16] = (0, 255, 0, 255)
    from .scene_spec import AtlasRegion

    tileset = AtlasSpec(
        image=atlas_img,
        regions=(
            AtlasRegion(name="grass", x=0, y=0, width=8, height=8),
            AtlasRegion(name="water", x=8, y=0, width=8, height=8),
        ),
    )
    tiles = np.zeros((4, 4), np.uint16)
    tiles[1, 1] = 1  # grass
    tiles[2, 2] = 2  # water
    tilemap = TilemapSpec(width=4, height=4, tile_size=8, tiles=tiles, tileset=tileset)

    assert tilemap.tile_at(1, 1) == 1
    tilemap.set_tile(0, 0, 2)
    assert tilemap.tile_at(0, 0) == 2
    mask = tilemap.collision_mask()
    assert mask[1, 1] and not mask[3, 3]

    cam = CameraSpec(viewport_width=32, viewport_height=32, world_x=16, world_y=16, zoom=1.0)
    frame = TilemapRenderer.render(tilemap, cam)
    assert frame.shape == (32, 32, 4)
    assert frame[..., 3].sum() > 0  # something got stamped

    print("OK — tilemap storage/collision mask/camera-culled rendering work")
