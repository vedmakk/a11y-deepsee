from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import DepthToAudioMapper, Source


class SimpleDepthToAudioMapper(DepthToAudioMapper):
    """Coarse grid-based sonification.

    The depth map is divided into a *grid_size×grid_size* grid.
    Each cell contributes **at most one** audio source located at the cell centre.

    * The **amplitude** is inversely proportional to the nearest depth value in that cell.
    * The **azimuth** corresponds to the horizontal position of the cell (−1 = full left, +1 = full right).
    * The **frequency** encodes depth as well (further = higher pitch) but can be tuned via `base_freq` / `freq_span`.
    """

    def __init__(
        self,
        grid_size: int = 10,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
        base_freq: float = 440.0,
        freq_span: float = 880.0,
        inverse: bool = True,
    ) -> None:
        """Create a simple grid-based depth→audio mapper.

        Parameters
        ----------
        inverse:
            If *True* (default) the incoming depth map is assumed to be **inverse**
            (larger values = closer).  Set this to *False* for *metric* depth maps
            where **smaller** numbers mean closer objects.
        """
        self.grid_size = grid_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.base_freq = base_freq
        self.freq_span = freq_span
        self.inverse = inverse

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def map(self, depth_map: np.ndarray) -> List[Source]:  # noqa: D401
        sources: List[Source] = []
        
        for gx, gy, closeness, freq in self._process_depth_grid(
            depth_map=depth_map,
            grid_size=self.grid_size,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            base_freq=self.base_freq,
            freq_span=self.freq_span,
            inverse=self.inverse,
        ):
            # Calculate azimuth from grid position
            azimuth = (gx + 0.5) / self.grid_size * 2.0 - 1.0  # −1..1 (left .. right)
            sources.append((azimuth, closeness, freq))

        return sources
