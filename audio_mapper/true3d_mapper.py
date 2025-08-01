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
        h, w = depth_map.shape
        cell_h = max(1, h // self.grid_size)
        cell_w = max(1, w // self.grid_size)

        sources: List[Source3D] = []
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                cell = depth_map[
                    gy * cell_h : (gy + 1) * cell_h,
                    gx * cell_w : (gx + 1) * cell_w,
                ]
                if cell.size == 0:
                    continue

                # Determine the pixel that is physically closest inside this grid cell.
                closest = float(cell.max()) if self.inverse else float(cell.min())

                if closest > self.max_depth or closest < self.min_depth:
                    continue  # outside the user-defined range

                # Normalise depth to 0‥1 where 1 = at *min_depth* (closest), 0 = at *max_depth*.
                clipped = np.clip(closest, self.min_depth, self.max_depth)
                if self.inverse:
                    closeness = (clipped - self.min_depth) / (self.max_depth - self.min_depth)
                else:
                    closeness = (self.max_depth - clipped) / (self.max_depth - self.min_depth)
                if closeness < 0.05:
                    continue  # ignore very faint sources

                # Spatial coordinates (centre of the cell) in normalised units −1‥1
                x = (gx + 0.5) / self.grid_size * 2.0 - 1.0  # −1 = left, +1 = right
                y = 1.0 - ((gy + 0.5) / self.grid_size * 2.0)  # +1 = up, −1 = down (flip Y-axis)

                # Map physical distance to OpenAL Z axis (negative → in front of the listener)
                z = -1.0 - (1.0 - closeness) * self.depth_scale

                gain = closeness  # 0‥1 – nearer ⇒ louder
                freq = self.base_freq + (1.0 - closeness) * self.freq_span

                sources.append((x, y, z, gain, freq))

        return sources
