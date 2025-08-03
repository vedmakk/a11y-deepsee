# Natural Soundscape System Migration Guide

This document explains the new natural soundscape system that replaces synthetic frequency-based audio with real WAV file samples for a more immersive accessibility experience.

## Overview

### Before: Frequency-Based System ❌

- Generated synthetic sine waves at different frequencies
- Higher frequencies = farther objects, lower frequencies = closer objects
- Resulted in unnatural, disturbing sounds
- Only encoded distance through pitch

### After: Zone-Based Natural Soundscape ✅

- Uses real WAV files for natural sounds
- Different distance ranges map to different sound categories
- Ocean sounds for far distances, wind for medium, footsteps for close
- Maintains loudness relationship with closeness
- Much more pleasant and intuitive audio experience

## Architecture Changes

### New Components

1. **Sound Zone System** (`audio_mapper/sound_zones.py`)

   - `SoundZone`: Defines distance ranges and associated WAV files
   - `SoundZoneConfig`: Manages multiple zones with overlap and fading

2. **Zone-Based Mappers**

   - `SimpleZoneMapper`: Grid-based mapping for stereo output
   - `Grid3DZoneMapper`: 3D spatial mapping for OpenAL output
   - Based on new `DepthToZoneMapper` abstract class

3. **WAV-Based Audio Outputs**

   - `StereoZoneOutput`: Stereo mixing of WAV samples with panning
   - `OpenALZoneOutput`: 3D spatial audio using OpenAL
   - Based on new `ZoneAudioOutput` abstract classes

4. **Sample Management**
   - `SampleManager`: Efficient WAV loading, caching, and resampling
   - `AudioSample`: Represents loaded audio with metadata

### Preserved Components

- Original frequency-based classes remain for backward compatibility
- `DepthToAudioMapper`, `SimpleDepthToAudioMapper`, `Grid3DDepthMapper`
- `AudioOutput`, `StereoAudioOutput`, `OpenALAudioOutput`

## Migration Steps

### 1. Install Dependencies (if using 3D audio)

Follow the dependencies installation instructions in the README.md file.

### 2. Prepare Audio Files

Create an `audio_samples` directory with WAV files:

```
audio_samples/
├── ocean.wav      # Far distance sounds (soft, ambient)
├── wind.wav       # Medium distance sounds (transitional)
└── footsteps.wav  # Close distance sounds (immediate)
```

### 3. Update Your Code

**Old frequency-based approach:**

```python
from audio_mapper import SimpleDepthToAudioMapper
from audio_output import StereoAudioOutput

# Old system
mapper = SimpleDepthToAudioMapper(
    grid_size=10,
    base_freq=440.0,
    freq_span=880.0
)
audio_output = StereoAudioOutput()
```

**New zone-based approach:**

```python
from audio_mapper import SimpleZoneMapper, SoundZoneConfig
from audio_output import StereoZoneOutput
from pathlib import Path

# Create sound zone configuration
zone_config = SoundZoneConfig.create_default_config(Path("audio_samples"))

# New system
mapper = SimpleZoneMapper(
    zone_config=zone_config,
    grid_size=10
)
audio_output = StereoZoneOutput(zone_config=zone_config)
```

### 4. Process Audio

The processing workflow remains similar:

```python
# Start audio
audio_output.start()

# Process depth maps
for depth_map in depth_maps:
    sources = mapper.map(depth_map)
    audio_output.update_sources(sources)

# Stop audio
audio_output.stop()
```

## Sound Zone Configuration

### Default Configuration

```python
# Far distances: Ocean sounds (peaceful, ambient)
SoundZone(
    zone_id="ocean",
    min_closeness=0.0,      # Farthest
    max_closeness=0.3,      # To 30% close
    audio_file="ocean.wav",
    base_volume=0.8,
    fade_distance=0.2       # Smooth transitions
)

# Medium distances: Wind sounds (transitional)
SoundZone(
    zone_id="wind",
    min_closeness=0.2,      # Overlaps with ocean
    max_closeness=0.7,      # To 70% close
    audio_file="wind.wav",
    base_volume=0.6,
    fade_distance=0.3
)

# Close distances: Footsteps (immediate, attention-grabbing)
SoundZone(
    zone_id="footsteps",
    min_closeness=0.6,      # Overlaps with wind
    max_closeness=1.0,      # Closest
    audio_file="footsteps.wav",
    base_volume=1.0,
    fade_distance=0.2
)
```

### Custom Configuration

```python
# Create your own zones for specific environments
zones = [
    SoundZone("forest", 0.0, 0.4, "forest_ambient.wav"),
    SoundZone("water", 0.3, 0.8, "water_flowing.wav"),
    SoundZone("gravel", 0.7, 1.0, "gravel_steps.wav")
]
custom_config = SoundZoneConfig(zones)
```

## Audio Output Options

### Stereo Output (Recommended)

- Works with any audio system
- Uses panning for left/right positioning
- Good for headphones and speakers
- No additional dependencies

```python
from audio_output import StereoZoneOutput

audio_output = StereoZoneOutput(
    zone_config=zone_config,
    sample_rate=44100,
    max_sources=16  # Limit for performance
)
```

### 3D Spatial Output (Advanced)

- Requires OpenAL installation
- True 3D positioning of sounds
- Best experience with headphones
- More immersive for complex scenes

```python
from audio_output import OpenALZoneOutput

audio_output = OpenALZoneOutput(
    zone_config=zone_config,
    max_sources_per_zone=4  # Limit per zone
)
```

## Performance Considerations

### Sample Management

- WAV files are loaded once and cached
- Automatic resampling to target sample rate (44.1kHz)
- Memory usage scales with total audio file size

### Source Limiting

- `max_sources`: Total simultaneous sources (stereo)
- `max_sources_per_zone`: Sources per zone (3D)
- Keeps loudest sources when limit exceeded

### Audio Quality vs Performance

- Higher quality WAV files = better sound but more memory
- Shorter looping samples = less memory but potential repetition
- Balance based on your target hardware

## Backward Compatibility

The old frequency-based system remains fully functional:

```python
# Old system still works
from audio_mapper import SimpleDepthToAudioMapper
from audio_output import StereoAudioOutput

mapper = SimpleDepthToAudioMapper()
audio_output = StereoAudioOutput()
# ... rest of code unchanged
```

This allows gradual migration and A/B testing between systems.

## Best Practices

### Audio File Selection

- **Ocean**: Choose looping ocean waves, water sounds
- **Wind**: Gentle wind through trees, ambient air movement
- **Footsteps**: Clear footstep sounds on various surfaces
- **Loop Quality**: Ensure seamless looping without clicks/pops

### Zone Design

- **Overlap zones** for smooth transitions
- **Adjust fade_distance** based on desired transition smoothness
- **Balance base_volume** across zones for pleasant mixing
- **Test with real depth data** for optimal ranges

### Integration

- Start with default configuration and adjust based on testing
- Monitor performance with your target hardware
- Consider user preferences for volume levels and zone ranges
- Provide fallback to frequency-based system if WAV files missing

## Troubleshooting

### Common Issues

**"No temp file available for zone"**

- Check that WAV files exist in the specified paths
- Verify file permissions are readable

**"Failed to load sample"**

- Ensure WAV files are valid format (16/24-bit PCM)
- Check that audio_samples directory exists

**Audio cutting out or distorting**

- Reduce `max_sources` or `max_sources_per_zone`
- Check WAV file quality and sample rates
- Monitor CPU usage during playback

**OpenAL errors**

- Install OpenAL: See README.md (Install the dependencies)
- Check system audio drivers
- Try stereo output as alternative

### Performance Optimization

- Use shorter audio samples for looping
- Reduce grid size if too many sources generated
- Lower audio quality if memory constrained
- Consider audio compression for storage
