from abc import ABC, abstractmethod
from typing import List, Tuple

Source = Tuple[float, float, float]  # azimuth, amplitude, frequency


class AudioOutput(ABC):
    """Abstract audio sink that renders one or more sources."""

    @abstractmethod
    def start(self) -> None:  # noqa: D401
        """Begin streaming audio."""

    @abstractmethod
    def stop(self) -> None:  # noqa: D401
        """Terminate the stream and free resources."""

    @abstractmethod
    def update_sources(self, sources: List[Source]) -> None:  # noqa: D401
        """Update the currently audible sources."""
