from abc import ABC, abstractmethod
import numpy as np


class DepthProvider(ABC):
    """Abstract interface that converts an RGB image to a single-channel depth map."""

    @abstractmethod
    def get_depth(self, frame: np.ndarray) -> np.ndarray:  # noqa: D401
        """Return a depth map (`H×W`, float32) where **smaller** values mean *closer* objects."""

    @property
    @abstractmethod
    def name(self) -> str:  # noqa: D401
        """Human-readable name of the provider."""
