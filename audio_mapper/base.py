from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

import numpy as np

Source = Tuple[float, float, float]  # azimuth (-1..1), amplitude (0..1), frequency (Hz)

# Data structure for processed grid cells
ProcessedCell = Tuple[int, int, float, float]  # (gx, gy, closeness, frequency)


class DepthToAudioMapper(ABC):
    """Abstract interface that converts a depth map into a set of audio sources."""

    @abstractmethod
    def map(self, depth_map: np.ndarray) -> List[Source]:  # noqa: D401
        """Return a list of `(azimuth, amplitude, frequency)` tuples."""

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
