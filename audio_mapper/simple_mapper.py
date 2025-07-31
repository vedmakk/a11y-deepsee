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
        max_depth: float = 5.0,
        base_freq: float = 440.0,
        freq_span: float = 880.0,
    ) -> None:
        self.grid_size = grid_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.base_freq = base_freq
        self.freq_span = freq_span

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def map(self, depth_map: np.ndarray) -> List[Source]:  # noqa: D401
        h, w = depth_map.shape
        cell_h = max(1, h // self.grid_size)
        cell_w = max(1, w // self.grid_size)

        sources: List[Source] = []
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                cell = depth_map[
                    gy * cell_h : (gy + 1) * cell_h,
                    gx * cell_w : (gx + 1) * cell_w,
                ]
                if cell.size == 0:
                    continue

                closest = float(cell.min())
                if closest > self.max_depth:
                    continue  # ignore far-away things

                # Amplitude 0..1 (1 = touch close, 0 = just within `max_depth`)
                clipped = np.clip(closest, self.min_depth, self.max_depth)
                amp = 1.0 - (clipped - self.min_depth) / (self.max_depth - self.min_depth)
                if amp < 0.05:  # simple threshold to reduce noise
                    continue

                azimuth = (gx + 0.5) / self.grid_size * 2.0 - 1.0  # −1..1
                freq = self.base_freq + (1.0 - amp) * self.freq_span
                sources.append((azimuth, amp, freq))

        return sources
