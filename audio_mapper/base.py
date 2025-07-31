from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

Source = Tuple[float, float, float]  # azimuth (-1..1), amplitude (0..1), frequency (Hz)


class DepthToAudioMapper(ABC):
    """Abstract interface that converts a depth map into a set of audio sources."""

    @abstractmethod
    def map(self, depth_map: np.ndarray) -> List[Source]:  # noqa: D401
        """Return a list of `(azimuth, amplitude, frequency)` tuples."""
