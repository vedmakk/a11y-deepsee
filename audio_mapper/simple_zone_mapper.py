"""Simple grid-based zone mapper for natural soundscape generation."""

from __future__ import annotations

from typing import List

import numpy as np

from .base import DepthToZoneMapper, ZoneSource
from .sound_zones import SoundZoneConfig


class SimpleZoneMapper(DepthToZoneMapper):
    """Grid-based zone mapper that maps depth to natural sound zones.

    Similar to SimpleDepthToAudioMapper but outputs sound zone information
    instead of frequency data. The depth map is divided into a grid, and each
    cell contributes audio sources from appropriate sound zones based on distance.

    * The **amplitude** is proportional to the zone intensity and closeness.
    * The **azimuth** corresponds to the horizontal position of the cell (−1 = full left, +1 = full right).
    * The **zone_id** determines which WAV file to play based on the closeness value.
    """

    def __init__(
        self,
        zone_config: SoundZoneConfig,
        grid_size: int = 10,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
        inverse: bool = True,
    ) -> None:
        """Create a simple grid-based depth→zone mapper.

        Parameters
        ----------
        zone_config : SoundZoneConfig
            Configuration defining sound zones for different distance ranges
        grid_size : int
            Size of the grid (grid_size × grid_size)
        min_depth, max_depth : float
            Depth range to consider
        inverse : bool
            If *True* (default) the incoming depth map is assumed to be **inverse**
            (larger values = closer). Set this to *False* for *metric* depth maps
            where **smaller** numbers mean closer objects.
        """
        super().__init__(zone_config)
        self.grid_size = grid_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.inverse = inverse

    def map(self, depth_map: np.ndarray) -> List[ZoneSource]:  # noqa: D401
        """Map depth data to zone-based audio sources."""
        sources: List[ZoneSource] = []
        
        for gx, gy, amplitude, zone_id in self._process_depth_grid_for_zones(
            depth_map=depth_map,
            grid_size=self.grid_size,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            inverse=self.inverse,
        ):
            # Calculate azimuth from grid position
            azimuth = (gx + 0.5) / self.grid_size * 2.0 - 1.0  # −1..1 (left .. right)
            
            # Calculate actual closeness for this cell (for audio output processing)
            # We need to pass the original closeness value, not the zone-adjusted amplitude
            closeness = amplitude  # This is already the zone-adjusted closeness
            
            sources.append((azimuth, amplitude, closeness, zone_id))

        return sources