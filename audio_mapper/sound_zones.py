"""Sound zone configuration system for natural soundscape mapping.

This module defines how distance/closeness values are mapped to different
audio categories (sound zones), each associated with specific WAV files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class SoundZone:
    """Defines a sound zone that maps distance ranges to audio files.
    
    Parameters
    ----------
    zone_id : str
        Unique identifier for this zone (e.g., "ocean", "wind", "footsteps")
    min_closeness : float
        Minimum closeness value (0.0 = farthest, 1.0 = closest)
    max_closeness : float
        Maximum closeness value (0.0 = farthest, 1.0 = closest)
    audio_file : Path
        Path to the WAV file for this zone
    base_volume : float
        Base volume multiplier for this zone (0.0 to 1.0)
    loop : bool
        Whether the audio should loop seamlessly
    fade_distance : float
        Distance over which to fade between zones (0.0 to 1.0)
    """
    zone_id: str
    min_closeness: float
    max_closeness: float
    audio_file: Path
    base_volume: float = 1.0
    loop: bool = True
    fade_distance: float = 0.1

    def __post_init__(self):
        """Validate zone parameters."""
        if not 0.0 <= self.min_closeness <= 1.0:
            raise ValueError(f"min_closeness must be 0.0-1.0, got {self.min_closeness}")
        if not 0.0 <= self.max_closeness <= 1.0:
            raise ValueError(f"max_closeness must be 0.0-1.0, got {self.max_closeness}")
        if self.min_closeness >= self.max_closeness:
            raise ValueError(f"min_closeness ({self.min_closeness}) must be < max_closeness ({self.max_closeness})")
        if not 0.0 <= self.base_volume <= 1.0:
            raise ValueError(f"base_volume must be 0.0-1.0, got {self.base_volume}")
        if not 0.0 <= self.fade_distance <= 1.0:
            raise ValueError(f"fade_distance must be 0.0-1.0, got {self.fade_distance}")

    def contains(self, closeness: float) -> bool:
        """Check if the given closeness value falls within this zone."""
        return self.min_closeness <= closeness <= self.max_closeness

    def get_zone_intensity(self, closeness: float) -> float:
        """Calculate the intensity within this zone (0.0 to 1.0).
        
        Returns 0.0 if closeness is outside the zone, 1.0 at the center,
        and fades towards the edges based on fade_distance.
        """
        if not self.contains(closeness):
            return 0.0
        
        zone_width = self.max_closeness - self.min_closeness
        zone_center = (self.min_closeness + self.max_closeness) / 2
        distance_from_center = abs(closeness - zone_center)
        max_distance_from_center = zone_width / 2
        
        # If fade_distance is 0, return full intensity
        if self.fade_distance <= 0:
            return 1.0
        
        # Calculate fade based on distance from edges
        fade_range = min(self.fade_distance * zone_width / 2, max_distance_from_center)
        
        if distance_from_center <= max_distance_from_center - fade_range:
            return 1.0  # Full intensity in center
        else:
            # Fade towards edges
            fade_factor = (max_distance_from_center - distance_from_center) / fade_range
            return max(0.0, min(1.0, fade_factor))


class SoundZoneConfig:
    """Configuration manager for sound zones in the soundscape.
    
    This class manages multiple sound zones and provides methods to
    determine which zones are active for a given closeness value.
    """
    
    def __init__(self, zones: Optional[List[SoundZone]] = None):
        """Initialize with optional list of sound zones."""
        self.zones = zones or []
        self._validate_zones()
    
    def add_zone(self, zone: SoundZone) -> None:
        """Add a new sound zone to the configuration."""
        self.zones.append(zone)
        self._validate_zones()
    
    def get_active_zones(self, closeness: float) -> List[tuple[SoundZone, float]]:
        """Get all zones active for the given closeness value.
        
        Returns a list of (zone, intensity) tuples where intensity
        is the calculated intensity for that zone (0.0 to 1.0).
        """
        active_zones = []
        for zone in self.zones:
            intensity = zone.get_zone_intensity(closeness)
            if intensity > 0.0:
                active_zones.append((zone, intensity))
        return active_zones
    
    def get_primary_zone(self, closeness: float) -> Optional[tuple[SoundZone, float]]:
        """Get the primary (highest intensity) zone for the given closeness."""
        active_zones = self.get_active_zones(closeness)
        if not active_zones:
            return None
        return max(active_zones, key=lambda x: x[1])
    
    def _validate_zones(self) -> None:
        """Validate that zones don't have conflicting IDs."""
        zone_ids = [zone.zone_id for zone in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("Duplicate zone IDs found")

    @classmethod
    def create_default_config(cls, audio_dir: Path) -> SoundZoneConfig:
        """Create a default sound zone configuration.
        
        This creates a typical setup with:
        - Ocean sounds for far distances (0.0 - 0.3)
        - Wind sounds for medium distances (0.2 - 0.7) 
        - Footstep sounds for close distances (0.6 - 1.0)
        
        Note the overlapping ranges allow for smooth transitions.
        """
        return cls([
            SoundZone(
                zone_id="ocean",
                min_closeness=0.0,
                max_closeness=0.3,
                audio_file=audio_dir / "ocean.wav",
                base_volume=0.8,
                fade_distance=0.2
            ),
            SoundZone(
                zone_id="wind",
                min_closeness=0.2,
                max_closeness=0.7,
                audio_file=audio_dir / "wind.wav",
                base_volume=0.6,
                fade_distance=0.3
            ),
            SoundZone(
                zone_id="footsteps",
                min_closeness=0.6,
                max_closeness=1.0,
                audio_file=audio_dir / "footsteps.wav",
                base_volume=1.0,
                fade_distance=0.2
            )
        ])