import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from depth_providers import DepthAnythingV2Provider
from audio_mapper import (
    SimpleDepthToAudioMapper,
    Grid3DDepthMapper,
    SimpleZoneMapper,
    Grid3DZoneMapper,
    SoundZoneConfig,
)
from audio_output import (
    StereoAudioOutput,
    OpenALAudioOutput,
    StereoZoneOutput,
    OpenALZoneOutput,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def colorize(depth: np.ndarray, inverse: bool = True) -> np.ndarray:
    """Convert a single-channel depth map to an RGB image for on-screen debugging.

    Parameters
    ----------
    inverse:
        If *True* the depth map is interpreted as **inverse** depth (larger =
        closer) and the standard *Spectral* colormap is used.  Otherwise the
        reversed variant *Spectral_r* is chosen so that warmer colours still
        represent nearer objects.
    """
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    cmap_name = "Spectral" if inverse else "Spectral_r"
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    colored = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


# -----------------------------------------------------------------------------
# Audio system setup
# -----------------------------------------------------------------------------

def setup_audio_system(output_backend: str, inverse_depth: bool, use_natural_soundscape: bool, audio_samples_dir: Path):
    """Set up the audio mapping and output system."""
    
    if use_natural_soundscape:
        # Try to use the new natural soundscape system
        try:
            # Check if audio samples directory exists
            if not audio_samples_dir.exists():
                print(f"Warning: Audio samples directory '{audio_samples_dir}' not found.")
                print("Falling back to frequency-based audio system.")
                raise FileNotFoundError("Audio samples directory missing")
            
            # Create sound zone configuration
            zone_config = SoundZoneConfig.create_default_config(audio_samples_dir)
            
            # Check if required audio files exist
            missing_files = []
            for zone in zone_config.zones:
                if not zone.audio_file.exists():
                    missing_files.append(str(zone.audio_file))
            
            if missing_files:
                print("Warning: Missing audio files for natural soundscape:")
                for file in missing_files:
                    print(f"  - {file}")
                print("Falling back to frequency-based audio system.")
                raise FileNotFoundError("Required audio files missing")
            
            # Set up zone-based system
            if output_backend == "stereo":
                mapper = SimpleZoneMapper(
                    zone_config=zone_config,
                    grid_size=8,
                    inverse=inverse_depth
                )
                audio_out = StereoZoneOutput(
                    zone_config=zone_config,
                    max_sources=16
                )
                debug_stereo = True
                print("✅ Using natural soundscape with stereo output")
            else:  # "3d" (default)
                mapper = Grid3DZoneMapper(
                    zone_config=zone_config,
                    grid_size=8,
                    inverse=inverse_depth
                )
                audio_out = OpenALZoneOutput(
                    zone_config=zone_config,
                    max_sources_per_zone=4
                )
                debug_stereo = False
                print("✅ Using natural soundscape with 3D spatial output")
                
            return mapper, audio_out, debug_stereo
            
        except (FileNotFoundError, ImportError, Exception) as e:
            print(f"Could not initialize natural soundscape system: {e}")
            print("Falling back to frequency-based audio system.")
    
    # Fallback to original frequency-based system
    if output_backend == "stereo":
        mapper = SimpleDepthToAudioMapper(inverse=inverse_depth)
        audio_out = StereoAudioOutput()
        debug_stereo = True
        print("🔊 Using frequency-based audio with stereo output")
    else:  # "3d" (default)
        mapper = Grid3DDepthMapper(inverse=inverse_depth, grid_size=8)
        audio_out = OpenALAudioOutput()
        debug_stereo = False
        print("🔊 Using frequency-based audio with 3D spatial output")
    
    return mapper, audio_out, debug_stereo


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def run(camera_index: int, device: str | None, output_backend: str, use_natural_soundscape: bool, audio_samples_dir: Path) -> None:  # noqa: D401
    """Capture frames, estimate depth and render spatial audio in real-time."""
    provider = DepthAnythingV2Provider(device=device or None)

    # Heuristic: Depth-Anything *Metric* checkpoints output metric depth (smaller
    # values = closer).  All other variants output inverse depth.  We use this
    # information to automatically configure the depth→audio mapper and colour
    # visualisation.
    inverse_depth = "metric" not in provider.model_id.lower()

    # Set up audio system (new natural soundscape or legacy frequency-based)
    mapper, audio_out, debug_stereo = setup_audio_system(
        output_backend, inverse_depth, use_natural_soundscape, audio_samples_dir
    )

    audio_out.start()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            depth = provider.get_depth(frame)
            sources = mapper.map(depth)
            audio_out.update_sources(sources)

            # --------------------------------------------------------------
            # Visualisation – RGB | Depth | Audio debug overlay
            # --------------------------------------------------------------
            depth_vis = colorize(depth, inverse=inverse_depth)
            combined = np.hstack([frame, depth_vis])

            # Width of a single frame (left = RGB, right = Depth)
            frame_width = frame.shape[1]

            # Debug color mapping
            zone_colors = zone_colors = {"ocean": (255, 0, 0), "wind": (255, 255, 0), "footsteps": (0, 0, 255)}
            default_color = (0, 255, 0)

            if debug_stereo:
                # Handle both frequency-based and zone-based stereo sources
                for source in sources:
                    if len(source) == 3:  # Frequency-based: (azimuth, amp, freq)
                        azimuth, amp, _ = source
                        color = default_color  # Green for frequency-based
                    elif len(source) == 4:  # Zone-based: (azimuth, amp, closeness, zone_id)
                        azimuth, amp, _, zone_id = source
                        # Different colors for different zones (BGR format for OpenCV)
                        color = zone_colors.get(zone_id, default_color)
                    else:
                        continue
                    
                    # Map azimuth −1..1 → depth image X coordinate only (right half)
                    x = frame_width + int((azimuth + 1) / 2 * frame_width)
                    y = combined.shape[0] // 2
                    cv2.circle(combined, (x, y), radius=10, color=color, thickness=-1)
            else:
                # Handle both frequency-based and zone-based 3D sources
                for source in sources:
                    if len(source) == 5:
                        x3d, y3d, z3d, gain, last_element = source
                        
                        # Check if last element is string (zone_id) or float (freq)
                        if isinstance(last_element, str):  # Zone-based
                            zone_id = last_element
                            # Different colors for different zones (BGR format for OpenCV)
                            color = zone_colors.get(zone_id, default_color)
                        else:  # Frequency-based: last_element is freq
                            color = default_color  # Green for frequency-based
                    else:
                        continue
                    
                    # Map X/Y from −1..1 → depth image screen space (right half)
                    x = frame_width + int((x3d + 1) / 2 * frame_width)
                    y = int((1 - y3d) / 2 * combined.shape[0])
                    cv2.circle(combined, (x, y), radius=8, color=color, thickness=-1)

            cv2.imshow("RGB | Depth | Audio Debug", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        audio_out.stop()
        cap.release()
        cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# Entry-point – CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth → Spatial Audio")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu, mps)")
    parser.add_argument(
        "--output",
        choices=["3d", "stereo"],
        default="3d",
        help="Audio backend: '3d' = OpenAL (true spatial) | 'stereo' = simple stereo panning",
    )
    parser.add_argument(
        "--audio-system",
        choices=["natural", "frequency"],
        default="natural",
        help="Audio system: 'natural' = WAV-based natural soundscape | 'frequency' = synthetic frequency-based"
    )
    parser.add_argument(
        "--audio-samples",
        type=Path,
        default=Path("audio_samples"),
        help="Directory containing audio sample files for natural soundscape (default: audio_samples)"
    )
    args = parser.parse_args()

    use_natural_soundscape = args.audio_system == "natural"
    run(args.camera, args.device, args.output, use_natural_soundscape, args.audio_samples)
