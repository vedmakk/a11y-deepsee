from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

import numpy as np

from .sound_zones import SoundZoneConfig

# Legacy frequency-based source (for backwards compatibility)
Source = Tuple[float, float, float]  # azimuth (-1..1), amplitude (0..1), frequency (Hz)

# New zone-based source format
ZoneSource = Tuple[float, float, float, str]  # azimuth (-1..1), amplitude (0..1), closeness (0..1), zone_id

# Data structure for processed grid cells
ProcessedCell = Tuple[int, int, float, float]  # (gx, gy, closeness, frequency)


class DepthToAudioMapper(ABC):
    """Abstract interface that converts a depth map into a set of audio sources."""

    @abstractmethod
    def map(self, depth_map: np.ndarray) -> List[Source]:  # noqa: D401
        """Return a list of `(azimuth, amplitude, frequency)` tuples."""


class DepthToZoneMapper(ABC):
    """Abstract interface that converts a depth map into zone-based audio sources.
    
    This is the new interface for natural soundscape generation using WAV files
    mapped to different distance zones.
    """

    def __init__(self, zone_config: SoundZoneConfig):
        """Initialize with sound zone configuration."""
        self.zone_config = zone_config

    @abstractmethod
    def map(self, depth_map: np.ndarray) -> List[ZoneSource]:  # noqa: D401
        """Return a list of `(azimuth, amplitude, closeness, zone_id)` tuples."""

    def _process_depth_grid_for_zones(
        self,
        depth_map: np.ndarray,
        grid_size: int,
        min_depth: float,
        max_depth: float,
        inverse: bool,
        min_closeness: float = 0.05,
    ) -> Iterator[Tuple[int, int, float, str]]:
        """Process depth map into grid cells and yield zone-based processed cell data.
        
        Similar to the base _process_depth_grid but returns zone information
        instead of frequency data.
        
        Parameters
        ----------
        depth_map : np.ndarray
            2D depth map to process
        grid_size : int
            Size of the grid (grid_size × grid_size)
        min_depth, max_depth : float
            Depth range to consider
        inverse : bool
            Whether larger depth values mean closer objects
        min_closeness : float
            Minimum closeness threshold (skip sources below this)
            
        Yields
        ------
        Tuple[int, int, float, str]
            Tuple of (gx, gy, closeness, zone_id) for each valid cell
        """
        h, w = depth_map.shape
        cell_h = max(1, h // grid_size)
        cell_w = max(1, w // grid_size)

        for gy in range(grid_size):
            for gx in range(grid_size):
                cell = depth_map[
                    gy * cell_h : (gy + 1) * cell_h,
                    gx * cell_w : (gx + 1) * cell_w,
                ]
                if cell.size == 0:
                    continue

                # Determine the pixel that is physically closest inside this grid cell.
                closest = float(cell.max()) if inverse else float(cell.min())

                if closest > max_depth or closest < min_depth:
                    continue  # outside the user-defined range

                # Normalise depth to 0‥1 where 1 = at *min_depth* (closest), 0 = at *max_depth*.
                clipped = np.clip(closest, min_depth, max_depth)
                if inverse:
                    closeness = (clipped - min_depth) / (max_depth - min_depth)
                else:
                    closeness = (max_depth - clipped) / (max_depth - min_depth)
                    
                if closeness < min_closeness:
                    continue  # ignore very faint sources

                # Get the primary zone for this closeness value
                primary_zone = self.zone_config.get_primary_zone(closeness)
                if primary_zone is None:
                    continue  # no zone covers this closeness range
                
                zone, zone_intensity = primary_zone
                yield (gx, gy, closeness * zone_intensity, zone.zone_id)

    def _process_depth_grid(
        self,
        depth_map: np.ndarray,
        grid_size: int,
        min_depth: float,
        max_depth: float,
        base_freq: float,
        freq_span: float,
        inverse: bool,
        min_closeness: float = 0.05,
    ) -> Iterator[ProcessedCell]:
        """Process depth map into grid cells and yield processed cell data.
        
        This method handles all the common logic for:
        - Grid cell extraction
        - Finding closest pixel per cell
        - Range filtering
        - Closeness/amplitude calculation
        - Frequency calculation
        
        Parameters
        ----------
        depth_map : np.ndarray
            2D depth map to process
        grid_size : int
            Size of the grid (grid_size × grid_size)
        min_depth, max_depth : float
            Depth range to consider
        base_freq, freq_span : float
            Frequency mapping parameters
        inverse : bool
            Whether larger depth values mean closer objects
        min_closeness : float
            Minimum closeness threshold (skip sources below this)
            
        Yields
        ------
        ProcessedCell
            Tuple of (gx, gy, closeness, frequency) for each valid cell
        """
        h, w = depth_map.shape
        cell_h = max(1, h // grid_size)
        cell_w = max(1, w // grid_size)

        for gy in range(grid_size):
            for gx in range(grid_size):
                cell = depth_map[
                    gy * cell_h : (gy + 1) * cell_h,
                    gx * cell_w : (gx + 1) * cell_w,
                ]
                if cell.size == 0:
                    continue

                # Determine the pixel that is physically closest inside this grid cell.
                closest = float(cell.max()) if inverse else float(cell.min())

                if closest > max_depth or closest < min_depth:
                    continue  # outside the user-defined range

                # Normalise depth to 0‥1 where 1 = at *min_depth* (closest), 0 = at *max_depth*.
                clipped = np.clip(closest, min_depth, max_depth)
                if inverse:
                    closeness = (clipped - min_depth) / (max_depth - min_depth)
                else:
                    closeness = (max_depth - clipped) / (max_depth - min_depth)
                    
                if closeness < min_closeness:
                    continue  # ignore very faint sources

                # Calculate frequency (far = higher pitch)
                freq = base_freq + (1.0 - closeness) * freq_span

                yield (gx, gy, closeness, freq)
