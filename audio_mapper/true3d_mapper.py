from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import DepthToAudioMapper

# (x, y, z, amplitude, frequency)
Source3D = Tuple[float, float, float, float, float]


class Grid3DDepthMapper(DepthToAudioMapper):
    """Convert a depth map into a *true* 3-D audio scene.

    The input image is divided into a regular grid (``grid_size × grid_size``).  For each cell the
    **closest** pixel is taken to generate at most one audio source whose

    • *x/y* coordinates correspond to the cell centre (−1 = left/bottom, +1 = right/top)
    • *z* coordinate is proportional to the *physical* distance (further → more negative)
    • *gain* (volume) is proportional to closeness (nearer → louder)
    • *frequency* is mapped such that distant objects sound higher-pitched (same mapping as the
      existing *SimpleDepthToAudioMapper*)
    """

    def __init__(
        self,
        grid_size: int = 20,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
        base_freq: float = 440.0,
        freq_span: float = 880.0,
        depth_scale: float = 1.0,  # How far (in OpenAL units) *max_depth* is away from the listener
        inverse: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        inverse:
            Whether the incoming depth map is *inverse* (larger = closer). Set to
            *False* for metric depth maps where smaller values indicate closer objects.
        """
        self.grid_size = grid_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.base_freq = base_freq
        self.freq_span = freq_span
        self.depth_scale = depth_scale
        self.inverse = inverse

    # ------------------------------------------------------------------
    # DepthToAudioMapper interface
    # ------------------------------------------------------------------
    def map(self, depth_map: np.ndarray) -> List[Source3D]:  # noqa: D401 – keep interface naming
        sources: List[Source3D] = []
        
        for gx, gy, closeness, freq in self._process_depth_grid(
            depth_map=depth_map,
            grid_size=self.grid_size,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            base_freq=self.base_freq,
            freq_span=self.freq_span,
            inverse=self.inverse,
        ):
            # Spatial coordinates (centre of the cell) in normalised units −1‥1
            x = (gx + 0.5) / self.grid_size * 2.0 - 1.0  # −1 = left, +1 = right
            y = 1.0 - ((gy + 0.5) / self.grid_size * 2.0)  # +1 = up, −1 = down (flip Y-axis)

            # Map physical distance to OpenAL Z axis (negative → in front of the listener)
            z = -1.0 - (1.0 - closeness) * self.depth_scale

            gain = closeness  # 0‥1 – nearer ⇒ louder
            sources.append((x, y, z, gain, freq))

        return sources
