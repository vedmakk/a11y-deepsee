from .base import DepthToAudioMapper, DepthToZoneMapper
from .simple_mapper import SimpleDepthToAudioMapper
from .true3d_mapper import Grid3DDepthMapper
from .simple_zone_mapper import SimpleZoneMapper
from .true3d_zone_mapper import Grid3DZoneMapper
from .sound_zones import SoundZone, SoundZoneConfig

__all__ = [
    # Legacy frequency-based mappers
    "DepthToAudioMapper",
    "SimpleDepthToAudioMapper", 
    "Grid3DDepthMapper",
    # New zone-based mappers
    "DepthToZoneMapper",
    "SimpleZoneMapper",
    "Grid3DZoneMapper",
    # Sound zone configuration
    "SoundZone",
    "SoundZoneConfig",
]
