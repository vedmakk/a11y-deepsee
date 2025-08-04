from abc import ABC, abstractmethod
from typing import List, Tuple

# Legacy frequency-based source (for backwards compatibility)
Source = Tuple[float, float, float]  # azimuth, amplitude, frequency

# New zone-based source format
ZoneSource = Tuple[float, float, float, str]  # azimuth, amplitude, closeness, zone_id
ZoneSource3D = Tuple[float, float, float, float, str]  # x, y, z, amplitude, zone_id


class AudioOutput(ABC):
    """Abstract audio sink that renders one or more frequency-based sources."""

    @abstractmethod
    def start(self) -> None:  # noqa: D401
        """Begin streaming audio."""

    @abstractmethod
    def stop(self) -> None:  # noqa: D401
        """Terminate the stream and free resources."""

    @abstractmethod
    def update_sources(self, sources: List[Source]) -> None:  # noqa: D401
        """Update the currently audible sources."""


class ZoneAudioOutput(ABC):
    """Abstract audio sink that renders zone-based sources using WAV files."""

    @abstractmethod
    def start(self) -> None:  # noqa: D401
        """Begin streaming audio."""

    @abstractmethod
    def stop(self) -> None:  # noqa: D401
        """Terminate the stream and free resources."""

    @abstractmethod
    def update_sources(self, sources: List[ZoneSource]) -> None:  # noqa: D401
        """Update the currently audible zone-based sources."""


class ZoneAudioOutput3D(ABC):
    """Abstract audio sink that renders 3D zone-based sources using WAV files."""

    @abstractmethod
    def start(self) -> None:  # noqa: D401
        """Begin streaming audio."""

    @abstractmethod
    def stop(self) -> None:  # noqa: D401
        """Terminate the stream and free resources."""

    @abstractmethod
    def update_sources(self, sources: List[ZoneSource3D]) -> None:  # noqa: D401
        """Update the currently audible 3D zone-based sources."""
