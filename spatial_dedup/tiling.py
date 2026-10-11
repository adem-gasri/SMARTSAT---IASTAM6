"""
spatial_dedup/tiling.py

Splits a large satellite image into fixed-size overlapping tiles, and
converts per-tile detection boxes back into global image coordinates.

Matches the methodology described in the paper (Section III-B):z
640x640 tiles, 20% overlap by default, so that a ship near a tile
boundary is likely to appear intact in at least one neighbouring tile.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Tile:
    x_offset: int
    y_offset: int
    width: int
    height: int


def generate_tiles(image_width: int, image_height: int,
                    tile_size: int = 640, overlap_ratio: float = 0.20) -> List[Tile]:
    """Generate a list of overlapping tile offsets covering the full image."""
    stride = int(tile_size * (1 - overlap_ratio))
    tiles = []

    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            w = min(tile_size, image_width - x)
            h = min(tile_size, image_height - y)
            tiles.append(Tile(x_offset=x, y_offset=y, width=w, height=h))
            if x + tile_size >= image_width:
                break
            x += stride
        if y + tile_size >= image_height:
            break
        y += stride

    return tiles


def tile_box_to_global(bbox_xyxy: List[float], tile: Tile) -> List[float]:
    """Convert a bounding box expressed in tile-local pixel coordinates
    into global image-pixel coordinates."""
    x1, y1, x2, y2 = bbox_xyxy
    return [
        x1 + tile.x_offset,
        y1 + tile.y_offset,
        x2 + tile.x_offset,
        y2 + tile.y_offset,
    ]


if __name__ == "__main__":
    # quick self-test
    tiles = generate_tiles(1800, 1200, tile_size=640, overlap_ratio=0.20)
    print(f"{len(tiles)} tiles generated for a 1800x1200 image")
    for t in tiles[:3]:
        print(t)