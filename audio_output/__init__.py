from .base import AudioOutput, ZoneAudioOutput, ZoneAudioOutput3D
from .stereo_output import StereoAudioOutput
from .openal_output import OpenALAudioOutput
from .stereo_zone_output import StereoZoneOutput
from .openal_zone_output import OpenALZoneOutput
from .sample_manager import SampleManager, AudioSample

__all__ = [
    # Legacy frequency-based audio outputs
    "AudioOutput",
    "StereoAudioOutput",
    "OpenALAudioOutput",
    # New zone-based audio outputs
    "ZoneAudioOutput",
    "ZoneAudioOutput3D", 
    "StereoZoneOutput",
    "OpenALZoneOutput",
    # Sample management
    "SampleManager",
    "AudioSample",
]
