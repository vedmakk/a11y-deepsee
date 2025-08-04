"""Stereo zone-based audio output using WAV files."""

from __future__ import annotations

import threading
from typing import Dict, List

import numpy as np
import sounddevice as sd

from .base import ZoneAudioOutput, ZoneSource
from .sample_manager import SampleManager

# Import sound zones with try/except for flexible importing
try:
    from ..audio_mapper.sound_zones import SoundZoneConfig
except ImportError:
    try:
        from audio_mapper.sound_zones import SoundZoneConfig
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from audio_mapper.sound_zones import SoundZoneConfig


class ActiveZoneSource:
    """Represents an active zone-based audio source with playback state."""
    
    def __init__(self, source_id: str, azimuth: float, amplitude: float, closeness: float, zone_id: str):
        self.source_id = source_id  # Unique identifier for tracking across frames
        self.azimuth = azimuth
        self.amplitude = amplitude
        self.closeness = closeness
        self.zone_id = zone_id
        self.playback_position = 0  # Current frame position in the sample
        
    def update(self, azimuth: float, amplitude: float, closeness: float):
        """Update source properties without restarting playback."""
        self.azimuth = azimuth
        self.amplitude = amplitude
        self.closeness = closeness


class StereoZoneOutput(ZoneAudioOutput):
    """Stereo renderer for zone-based natural soundscapes.

    This class loads WAV files for different sound zones and mixes them in real-time
    based on the zone sources provided by the depth mapper. Each zone plays its
    associated audio file with appropriate volume and stereo positioning.
    """

    def __init__(
        self,
        zone_config: SoundZoneConfig,
        sample_rate: int = 44100,
        buffer_size: int = 1024,
        max_sources: int = 32
    ):
        """Initialize stereo zone output.
        
        Parameters
        ----------
        zone_config : SoundZoneConfig
            Configuration defining sound zones and their associated audio files
        sample_rate : int
            Audio sample rate
        buffer_size : int
            Audio buffer size for low-latency playback
        max_sources : int
            Maximum number of simultaneous sources to prevent audio overload
        """
        self.zone_config = zone_config
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_sources = max_sources
        
        # Initialize sample manager and load all zone samples
        self.sample_manager = SampleManager(target_sample_rate=sample_rate)
        self._load_zone_samples()
        
        # Track active sources by unique source ID
        self.active_sources: Dict[str, ActiveZoneSource] = {}
        self._lock = threading.Lock()
        
        # Audio stream
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            channels=2,  # Stereo
            dtype="float32",
            callback=self._audio_callback,
        )

    def _load_zone_samples(self) -> None:
        """Load all WAV files for the configured zones."""
        for zone in self.zone_config.zones:
            try:
                self.sample_manager.load_sample(
                    zone_id=zone.zone_id,
                    file_path=zone.audio_file,
                    loop=zone.loop
                )
                print(f"Loaded sample for zone '{zone.zone_id}': {zone.audio_file}")
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Failed to load sample for zone '{zone.zone_id}': {e}")

    def start(self) -> None:  # noqa: D401
        """Begin streaming audio."""
        self._stream.start()

    def stop(self) -> None:  # noqa: D401
        """Terminate the stream and free resources."""
        self._stream.stop()
        self._stream.close()

    def update_sources(self, sources: List[ZoneSource]) -> None:  # noqa: D401
        """Update the currently audible zone-based sources."""
        with self._lock:
            # Limit number of sources to prevent audio overload
            if len(sources) > self.max_sources:
                # Keep the loudest sources
                sources = sorted(sources, key=lambda s: s[1], reverse=True)[:self.max_sources]
            
            # Create set of current source IDs for this frame
            current_source_ids = set()
            
            # Update or create sources
            for azimuth, amplitude, closeness, zone_id in sources:
                # Create a stable source ID based on zone and approximate position
                # This ensures sources in similar positions maintain continuity
                position_key = f"{zone_id}_{int(azimuth * 10):+03d}"  # Discretize azimuth to nearest 0.1
                source_id = position_key
                current_source_ids.add(source_id)
                
                if source_id in self.active_sources:
                    # Update existing source properties
                    self.active_sources[source_id].update(azimuth, amplitude, closeness)
                else:
                    # Create new source
                    active_source = ActiveZoneSource(source_id, azimuth, amplitude, closeness, zone_id)
                    self.active_sources[source_id] = active_source
            
            # Remove sources that are no longer active
            sources_to_remove = set(self.active_sources.keys()) - current_source_ids
            for source_id in sources_to_remove:
                del self.active_sources[source_id]

    def _audio_callback(self, outdata, frames, time, status):  # noqa: D401
        """Audio callback that mixes zone-based samples."""
        if status:
            print(f"Audio callback status: {status}")

        # Initialize output buffer
        buffer = np.zeros((frames, 2), dtype=np.float32)

        with self._lock:
            # Mix audio from each active source
            for source in self.active_sources.values():
                # Get the sample for this source's zone
                sample = self.sample_manager.get_sample(source.zone_id)
                if sample is None:
                    continue
                
                # Get audio data from the sample
                audio_data = sample.get_samples(source.playback_position, frames)
                
                # Convert to stereo if needed
                if len(audio_data.shape) == 1:  # Mono
                    audio_data = np.column_stack([audio_data, audio_data])
                
                # Apply volume based on amplitude
                volume = source.amplitude
                audio_data *= volume
                
                # Apply stereo panning based on azimuth
                left_gain = (1.0 - source.azimuth) * 0.5
                right_gain = (1.0 + source.azimuth) * 0.5
                
                audio_data[:, 0] *= left_gain   # Left channel
                audio_data[:, 1] *= right_gain  # Right channel
                
                # Add to output buffer
                buffer += audio_data
                
                # Update playback position for continuous playback
                source.playback_position += frames

        # Prevent clipping
        buffer = np.clip(buffer, -1.0, 1.0)
        outdata[:] = buffer