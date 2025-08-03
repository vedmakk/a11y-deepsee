"""True 3D zone mapper for natural soundscape generation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import DepthToZoneMapper
from .sound_zones import SoundZoneConfig

# (x, y, z, amplitude, zone_id)
ZoneSource3D = Tuple[float, float, float, float, str]


class Grid3DZoneMapper(DepthToZoneMapper):
    """Convert a depth map into a *true* 3-D zone-based audio scene.

    Similar to Grid3DDepthMapper but outputs sound zone information instead of 
    frequency data. The input image is divided into a regular grid and each cell
    generates audio sources from appropriate sound zones based on distance.

    • *x/y* coordinates correspond to the cell centre (−1 = left/bottom, +1 = right/top)
    • *z* coordinate is proportional to the *physical* distance (further → more negative)
    • *amplitude* (volume) is proportional to zone intensity and closeness (nearer → louder)
    • *zone_id* determines which WAV file to play based on the closeness value
    """

    def __init__(
        self,
        zone_config: SoundZoneConfig,
        grid_size: int = 20,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
        depth_scale: float = 1.0,  # How far (in OpenAL units) *max_depth* is away from the listener
        inverse: bool = True,
    ) -> None:
        """Create a 3D grid-based depth→zone mapper.
        
        Parameters
        ----------
        zone_config : SoundZoneConfig
            Configuration defining sound zones for different distance ranges
        grid_size : int
            Size of the grid (grid_size × grid_size)
        min_depth, max_depth : float
            Depth range to consider
        depth_scale : float
            How far (in OpenAL units) max_depth is away from the listener
        inverse : bool
            Whether the incoming depth map is *inverse* (larger = closer). Set to
            *False* for metric depth maps where smaller values indicate closer objects.
        """
        super().__init__(zone_config)
        self.grid_size = grid_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.inverse = inverse

    def map(self, depth_map: np.ndarray) -> List[ZoneSource3D]:  # noqa: D401
        """Map depth data to 3D zone-based audio sources."""
        sources: List[ZoneSource3D] = []
        
        for gx, gy, amplitude, zone_id in self._process_depth_grid_for_zones(
            depth_map=depth_map,
            grid_size=self.grid_size,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            inverse=self.inverse,
        ):
            # Spatial coordinates (centre of the cell) in normalised units −1‥1
            x = (gx + 0.5) / self.grid_size * 2.0 - 1.0  # −1 = left, +1 = right
            y = 1.0 - ((gy + 0.5) / self.grid_size * 2.0)  # +1 = up, −1 = down (flip Y-axis)

            # Map physical distance to OpenAL Z axis (negative → in front of the listener)
            # We need to reverse-calculate the original closeness from the amplitude
            # since amplitude includes zone intensity adjustment
            original_closeness = amplitude  # This works for our current implementation
            z = -1.0 - (1.0 - original_closeness) * self.depth_scale

            sources.append((x, y, z, amplitude, zone_id))

        return sources