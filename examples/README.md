# Natural Soundscape Examples

This directory contains examples demonstrating the new natural soundscape system that replaces synthetic frequency-based audio with real WAV file samples.

## Quick Start

1. **Create an audio samples directory:**

   ```bash
   mkdir audio_samples
   ```

2. **Add your WAV files:**

   - `ocean.wav` - Soft ocean/water sounds for far distances
   - `wind.wav` - Wind or ambient sounds for medium distances
   - `footsteps.wav` - Footsteps or other sounds for close distances

3. **Run the example:**
   ```bash
   python natural_soundscape_example.py
   ```

## Sound Zone Configuration

The new system maps distance ranges to different audio files:

- **Far distances (0.0-0.35 closeness)**: Ocean sounds - soft, ambient background
- **Medium distances (0.25-0.75 closeness)**: Wind sounds - transitional layer
- **Close distances (0.65-1.0 closeness)**: Footsteps - immediate, crisp foreground

Note the overlapping ranges - this creates smooth transitions between zones.

## System Architecture

### Zone-Based Mappers

- `SimpleZoneMapper`: Grid-based mapping with stereo positioning
- `Grid3DZoneMapper`: Full 3D spatial mapping with depth positioning

### Audio Outputs

- `StereoZoneOutput`: Stereo mixing with panning (works with any audio system)
- `OpenALZoneOutput`: True 3D spatial audio (requires `pip install openal`)

### Sample Management

- `SampleManager`: Efficient WAV file loading and caching
- `AudioSample`: Represents loaded audio with looping and resampling

## Customization

You can easily customize the sound zones:

```python
from audio_mapper import SoundZone, SoundZoneConfig

# Create custom zones
custom_config = SoundZoneConfig([
    SoundZone(
        zone_id="nature",
        min_closeness=0.0,
        max_closeness=0.4,
        audio_file=Path("sounds/forest.wav"),
        base_volume=0.8,
        fade_distance=0.1
    ),
    SoundZone(
        zone_id="urban",
        min_closeness=0.3,
        max_closeness=1.0,
        audio_file=Path("sounds/city.wav"),
        base_volume=1.0,
        fade_distance=0.2
    )
])
```

## Integration with Main Application

To integrate with your main depth sensing application:

```python
# Replace the old frequency-based system:
# mapper = SimpleDepthToAudioMapper(...)
# audio_output = StereoAudioOutput(...)

# With the new zone-based system:
zone_config = SoundZoneConfig.create_default_config(Path("audio_samples"))
mapper = SimpleZoneMapper(zone_config=zone_config)
audio_output = StereoZoneOutput(zone_config=zone_config)
```

## WAV File Requirements

- **Format**: WAV files (16-bit or 24-bit PCM)
- **Sample Rate**: Any (will be resampled to 44.1kHz)
- **Channels**: Mono or stereo (stereo will be converted to mono for 3D audio)
- **Length**: Any length (looping is handled automatically)
- **Quality**: Higher quality files will provide better soundscapes

## Performance Tips

- Use compressed audio files if loading time is important
- Limit the number of simultaneous sources (configured via `max_sources`)
- For 3D audio, limit sources per zone (configured via `max_sources_per_zone`)
- Consider using shorter looping samples for better memory efficiency
