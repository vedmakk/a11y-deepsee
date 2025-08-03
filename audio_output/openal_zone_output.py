"""True 3D zone-based audio output using OpenAL and WAV files."""

from __future__ import annotations

import tempfile
import threading
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    # The canonical PyPI package is simply called "openal" (PyOpenAL bindings)
    from openal import oalInit, oalQuit, oalOpen, Listener  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover – optional dependency
    raise ImportError(
        "PyOpenAL is required for OpenALZoneOutput. Install with `pip install openal`."
    ) from exc

from .base import ZoneAudioOutput3D, ZoneSource3D
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


class ActiveZoneSource3D:
    """Represents an active 3D zone-based audio source with OpenAL source."""
    
    def __init__(
        self,
        source_id: str,
        x: float,
        y: float,
        z: float,
        amplitude: float,
        zone_id: str,
        openal_source,
        sample_path: Path
    ):
        self.source_id = source_id
        self.x = x
        self.y = y
        self.z = z
        self.amplitude = amplitude
        self.zone_id = zone_id
        self.openal_source = openal_source
        self.sample_path = sample_path
        
    def update(self, x: float, y: float, z: float, amplitude: float):
        """Update source properties and OpenAL state."""
        self.x = x
        self.y = y
        self.z = z
        self.amplitude = amplitude
        
        # Update OpenAL source properties
        self.openal_source.set_position([x, y, z])
        self.openal_source.set_gain(max(0.0, min(1.0, amplitude)))


class OpenALZoneOutput(ZoneAudioOutput3D):
    """True 3D audio output using OpenAL and zone-based WAV files.

    This class creates OpenAL sources for each active zone and positions them
    in 3D space according to the depth mapping. Each zone plays its associated
    WAV file with appropriate 3D positioning and volume.
    """

    MAX_SOURCES = 64  # Practical limit to avoid OpenAL running out of voices

    def __init__(
        self,
        zone_config: SoundZoneConfig,
        sample_rate: int = 44100,
        max_sources_per_zone: int = 8
    ):
        """Initialize OpenAL zone output.
        
        Parameters
        ----------
        zone_config : SoundZoneConfig
            Configuration defining sound zones and their associated audio files
        sample_rate : int
            Audio sample rate  
        max_sources_per_zone : int
            Maximum number of simultaneous sources per zone
        """
        self.zone_config = zone_config
        self.sample_rate = sample_rate
        self.max_sources_per_zone = max_sources_per_zone
        
        # Initialize sample manager
        self.sample_manager = SampleManager(target_sample_rate=sample_rate)
        
        # OpenAL components
        self._listener: Optional[Listener] = None
        self._lock = threading.Lock()
        self._active_sources: Dict[str, ActiveZoneSource3D] = {}
        
        # Cache of temporary WAV files for OpenAL (zone_id -> temp file path)
        self._temp_files: Dict[str, Path] = {}
        
        # Load and prepare all zone samples
        self._prepare_zone_samples()

    def _prepare_zone_samples(self) -> None:
        """Load zone samples and create temporary files for OpenAL."""
        for zone in self.zone_config.zones:
            try:
                # Load the sample using our sample manager
                sample = self.sample_manager.load_sample(
                    zone_id=zone.zone_id,
                    file_path=zone.audio_file,
                    loop=zone.loop
                )
                
                # Create a temporary WAV file that OpenAL can use
                temp_file = self._create_temp_wav_file(sample, zone.zone_id)
                self._temp_files[zone.zone_id] = temp_file
                
                print(f"Prepared zone '{zone.zone_id}' with temp file: {temp_file}")
                
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Failed to prepare zone '{zone.zone_id}': {e}")

    def _create_temp_wav_file(self, sample, zone_id: str) -> Path:
        """Create a temporary WAV file from an AudioSample for OpenAL."""
        temp = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f"_{zone_id}.wav",
            prefix="zone_"
        )
        temp_path = Path(temp.name)
        
        # Convert float32 audio data to 16-bit PCM for OpenAL
        if len(sample.data.shape) == 1:  # Mono
            audio_16bit = (sample.data * 32767).astype(np.int16)
            channels = 1
        else:  # Stereo or multi-channel - convert to mono for 3D positioning
            audio_mono = np.mean(sample.data, axis=1)
            audio_16bit = (audio_mono * 32767).astype(np.int16)
            channels = 1
        
        # Write WAV file
        with wave.open(str(temp_path), 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample.sample_rate)
            wf.writeframes(audio_16bit.tobytes())
        
        return temp_path

    def start(self) -> None:  # noqa: D401
        """Initialize OpenAL and start playback."""
        oalInit()
        
        # Set up listener at origin looking along -Z axis
        self._listener = Listener()
        self._listener.set_position([0, 0, 0])
        
        # Handle different PyOpenAL fork APIs
        try:
            self._listener.set_orientation([0, 0, -1, 0, 1, 0])
        except TypeError:
            self._listener.set_orientation(at=[0, 0, -1], up=[0, 1, 0])

    def stop(self) -> None:  # noqa: D401
        """Stop playback and release all OpenAL resources."""
        with self._lock:
            # Stop and destroy all active sources
            for active_source in self._active_sources.values():
                try:
                    active_source.openal_source.stop()
                    if hasattr(active_source.openal_source, "destroy"):
                        active_source.openal_source.destroy()
                    elif hasattr(active_source.openal_source, "delete"):
                        active_source.openal_source.delete()
                except Exception as e:
                    print(f"Warning: Error cleaning up OpenAL source: {e}")
            
            self._active_sources.clear()
        
        # Shut down OpenAL
        oalQuit()

    def update_sources(self, sources: List[ZoneSource3D]) -> None:  # noqa: D401
        """Update the currently audible 3D zone-based sources."""
        with self._lock:
            # Limit total number of sources
            if len(sources) > self.MAX_SOURCES:
                # Keep the loudest sources
                sources = sorted(sources, key=lambda s: s[3], reverse=True)[:self.MAX_SOURCES]
            
            # Create set of current source IDs for this frame
            current_source_ids = set()
            
            # Update or create sources
            for x, y, z, amplitude, zone_id in sources:
                # Create a stable source ID based on zone and approximate position
                # This ensures sources in similar positions maintain continuity
                position_key = f"{zone_id}_{int(x * 5):+03d}_{int(y * 5):+03d}_{int(z * 2):+03d}"
                source_id = position_key
                current_source_ids.add(source_id)
                
                if source_id in self._active_sources:
                    # Update existing source properties
                    self._active_sources[source_id].update(x, y, z, amplitude)
                else:
                    # Create new source
                    temp_file = self._temp_files.get(zone_id)
                    if temp_file is None:
                        print(f"Warning: No temp file available for zone '{zone_id}'")
                        continue
                    
                    try:
                        # Create OpenAL source
                        openal_source = oalOpen(str(temp_file))
                        openal_source.set_looping(True)
                        openal_source.set_position([x, y, z])
                        openal_source.set_gain(max(0.0, min(1.0, amplitude)))
                        openal_source.play()
                        
                        # Track the active source
                        active_source = ActiveZoneSource3D(
                            source_id, x, y, z, amplitude, zone_id, openal_source, temp_file
                        )
                        self._active_sources[source_id] = active_source
                        
                    except Exception as e:
                        print(f"Warning: Failed to create OpenAL source for zone '{zone_id}': {e}")
            
            # Remove sources that are no longer active
            sources_to_remove = set(self._active_sources.keys()) - current_source_ids
            for source_id in sources_to_remove:
                active_source = self._active_sources[source_id]
                try:
                    active_source.openal_source.stop()
                    if hasattr(active_source.openal_source, "destroy"):
                        active_source.openal_source.destroy()
                    elif hasattr(active_source.openal_source, "delete"):
                        active_source.openal_source.delete()
                except Exception as e:
                    print(f"Warning: Error cleaning up OpenAL source {source_id}: {e}")
                del self._active_sources[source_id]