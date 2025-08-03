from .base import AudioOutput, ZoneAudioOutput, ZoneAudioOutput3D
from .stereo_output import StereoAudioOutput
from .stereo_zone_output import StereoZoneOutput
from .sample_manager import SampleManager, AudioSample

# OpenAL-based outputs are optional (only available if OpenAL is installed)
_openal_available = True
try:
    from .openal_output import OpenALAudioOutput
    from .openal_zone_output import OpenALZoneOutput
except ImportError:
    _openal_available = False
    
    # Create dummy classes that raise helpful errors
    class OpenALAudioOutput:
        def __init__(self, *args, **kwargs):
            raise ImportError("OpenAL not available. Install with: pip install openal")
    
    class OpenALZoneOutput:
        def __init__(self, *args, **kwargs):
            raise ImportError("OpenAL not available. Install with: pip install openal")

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
