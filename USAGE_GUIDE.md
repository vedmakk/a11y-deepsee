# Usage Guide: Natural Soundscape Integration

The `main.py` script has been updated to support both the new natural soundscape system and the legacy frequency-based system.

## Command Line Options

### Audio System Selection

- `--audio-system natural` – Use the new WAV-based natural soundscape system (default)
- `--audio-system frequency` – Use the legacy synthetic frequency-based system

### Audio Output Backend

- `--output stereo` – Use stereo panning output (compatible with any audio system)
- `--output 3d` – Use OpenAL 3D spatial audio (default, requires OpenAL installation)

### Audio Samples Configuration

- `--audio-samples DIR` – Specify directory containing WAV files (default: `audio_samples`)

### Other Options

- `--camera INDEX` – Camera index to use (default: 0)
- `--device DEVICE` – PyTorch device override (cpu, mps, cuda)

## Usage Examples

### Natural Soundscape (Recommended)

```bash
# Default: Natural soundscape with 3D spatial audio
python main.py

# Natural soundscape with stereo output
python main.py --audio-system natural --output stereo

# Custom audio samples directory
python main.py --audio-system natural --audio-samples /path/to/your/audio/files

# Natural soundscape with external camera
python main.py --audio-system natural --camera 1
```

### Legacy Frequency-Based System

```bash
# Frequency-based with 3D spatial audio
python main.py --audio-system frequency --output 3d

# Frequency-based with stereo output
python main.py --audio-system frequency --output stereo
```

## Setting Up Natural Soundscape

1. **Create audio samples directory:**

   ```bash
   mkdir audio_samples
   ```

2. **Add required WAV files:**

   - `ocean.wav` – Soft ocean/water sounds for far distances
   - `wind.wav` – Wind or ambient sounds for medium distances
   - `footsteps.wav` – Footsteps or other sounds for close distances

3. **Generate test audio files (optional):**
   ```bash
   python examples/natural_soundscape_example.py
   # This will create sample audio files if they don't exist
   ```

## Visual Feedback

The application provides visual feedback in the OpenCV window:

### Natural Soundscape Mode

- **Red dots**: Ocean sounds (far distances)
- **Yellow dots**: Wind sounds (medium distances)
- **Blue dots**: Footstep sounds (close distances)

### Frequency-Based Mode

- **Green dots**: Synthetic audio sources

## Automatic Fallback

The system automatically falls back to the frequency-based system if:

- Audio samples directory doesn't exist
- Required WAV files are missing
- There's an error loading audio samples
- OpenAL is not available (for 3D output)

## Performance Tips

- **Natural soundscape**: Uses more memory due to WAV file loading, but provides much better audio experience
- **Frequency-based**: Lower memory usage, but synthetic sounds can be disturbing
- **Stereo output**: Lower CPU usage than 3D spatial audio
- **3D output**: More immersive but requires OpenAL and more processing power

## Integration Notes

The integration maintains full backward compatibility:

- Existing scripts using frequency-based audio continue to work unchanged
- New zone-based mappers and outputs are available alongside legacy classes
- Visual debugging works for both audio systems with appropriate color coding
- Error handling gracefully falls back to frequency-based system when needed
