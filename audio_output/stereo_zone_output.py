"""Stereo zone-based audio output using WAV files."""

from __future__ import annotations

import threading
from typing import Dict, List

import numpy as np
import sounddevice as sd

from .base import ZoneAudioOutput, ZoneSource
from .sample_manager import SampleManager
from ..audio_mapper.sound_zones import SoundZoneConfig


class ActiveZoneSource:
    """Represents an active zone-based audio source with playback state."""
    
    def __init__(self, azimuth: float, amplitude: float, closeness: float, zone_id: str):
        self.azimuth = azimuth
        self.amplitude = amplitude
        self.closeness = closeness
        self.zone_id = zone_id
        self.playback_position = 0  # Current frame position in the sample


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
        
        # Track active sources by zone
        self.active_sources: Dict[str, List[ActiveZoneSource]] = {}
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
            # Clear existing sources
            self.active_sources.clear()
            
            # Limit number of sources to prevent audio overload
            if len(sources) > self.max_sources:
                # Keep the loudest sources
                sources = sorted(sources, key=lambda s: s[1], reverse=True)[:self.max_sources]
            
            # Group sources by zone
            for azimuth, amplitude, closeness, zone_id in sources:
                if zone_id not in self.active_sources:
                    self.active_sources[zone_id] = []
                
                # Create active source
                active_source = ActiveZoneSource(azimuth, amplitude, closeness, zone_id)
                self.active_sources[zone_id].append(active_source)

    def _audio_callback(self, outdata, frames, time, status):  # noqa: D401
        """Audio callback that mixes zone-based samples."""
        if status:
            print(f"Audio callback status: {status}")

        # Initialize output buffer
        buffer = np.zeros((frames, 2), dtype=np.float32)

        with self._lock:
            # Mix audio from each active zone
            for zone_id, zone_sources in self.active_sources.items():
                if not zone_sources:
                    continue
                
                # Get the sample for this zone
                sample = self.sample_manager.get_sample(zone_id)
                if sample is None:
                    continue
                
                # Mix all sources in this zone
                for source in zone_sources:
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
                    
                    # Update playback position
                    source.playback_position += frames

        # Prevent clipping
        buffer = np.clip(buffer, -1.0, 1.0)
        outdata[:] = buffer