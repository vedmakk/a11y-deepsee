#!/usr/bin/env python3
"""Example demonstrating the new natural soundscape system using WAV files.

This example shows how to:
1. Configure sound zones with different WAV files for different distances
2. Use the new zone-based mappers 
3. Output natural soundscapes through stereo or 3D audio

You'll need to provide your own WAV files in an 'audio_samples' directory.
"""

import time
from pathlib import Path

import numpy as np

# Import the new zone-based classes
from audio_mapper import (
    SoundZone, 
    SoundZoneConfig, 
    SimpleZoneMapper, 
    Grid3DZoneMapper
)
from audio_output import (
    StereoZoneOutput,
    # OpenALZoneOutput  # Uncomment if you have OpenAL installed
)


def create_example_sound_config() -> SoundZoneConfig:
    """Create an example sound zone configuration.
    
    This sets up three overlapping zones:
    - Ocean sounds for far distances (soft, ambient)
    - Wind sounds for medium distances (transitional)
    - Footsteps for close distances (crisp, immediate)
    """
    audio_dir = Path("audio_samples")
    
    return SoundZoneConfig([
        SoundZone(
            zone_id="ocean",
            min_closeness=0.0,
            max_closeness=0.35,
            audio_file=audio_dir / "ocean.wav",
            base_volume=0.7,
            loop=True,
            fade_distance=0.15  # Smooth fade at edges
        ),
        SoundZone(
            zone_id="wind", 
            min_closeness=0.25,
            max_closeness=0.75,
            audio_file=audio_dir / "wind.wav", 
            base_volume=0.8,
            loop=True,
            fade_distance=0.2
        ),
        SoundZone(
            zone_id="footsteps",
            min_closeness=0.65,
            max_closeness=1.0,
            audio_file=audio_dir / "footsteps.wav",
            base_volume=1.0,
            loop=True,
            fade_distance=0.1
        )
    ])


def simulate_depth_map(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a simulated depth map for testing.
    
    Creates a depth map with:
    - Objects at various distances
    - Some movement over time for dynamic testing
    """
    # Create a depth map with some interesting features
    y, x = np.ogrid[:height, :width]
    
    # Create some "objects" at different distances
    center_x, center_y = width // 2, height // 2
    
    # Background (far away)
    depth_map = np.full((height, width), 0.1, dtype=np.float32)
    
    # Add some circular objects at different distances
    objects = [
        (center_x - 100, center_y, 50, 0.8),  # Close object (left)
        (center_x + 80, center_y - 60, 40, 0.5),  # Medium distance (top-right)
        (center_x, center_y + 80, 60, 0.3),  # Far object (bottom)
    ]
    
    for obj_x, obj_y, radius, distance in objects:
        mask = (x - obj_x)**2 + (y - obj_y)**2 < radius**2
        depth_map[mask] = distance
    
    return depth_map


def run_stereo_example():
    """Run the stereo zone-based soundscape example."""
    print("Setting up stereo natural soundscape...")
    
    # Create sound zone configuration
    zone_config = create_example_sound_config()
    
    # Check if audio files exist
    audio_dir = Path("audio_samples")
    if not audio_dir.exists():
        print(f"Warning: Audio samples directory '{audio_dir}' not found.")
        print("Please create this directory and add ocean.wav, wind.wav, and footsteps.wav files.")
        return
    
    missing_files = []
    for zone in zone_config.zones:
        if not zone.audio_file.exists():
            missing_files.append(str(zone.audio_file))
    
    if missing_files:
        print("Warning: Missing audio files:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please add these WAV files to continue.")
        return
    
    # Create zone mapper and audio output
    mapper = SimpleZoneMapper(
        zone_config=zone_config,
        grid_size=8,  # 8x8 grid for smoother soundscape
        min_depth=0.0,
        max_depth=1.0,
        inverse=True  # Higher values = closer objects
    )
    
    # Create stereo audio output
    audio_output = StereoZoneOutput(
        zone_config=zone_config,
        sample_rate=44100,
        buffer_size=1024,
        max_sources=16
    )
    
    try:
        # Start audio
        audio_output.start()
        print("Audio started. Playing natural soundscape...")
        print("You should hear ocean sounds (far), wind (medium), and footsteps (close).")
        
        # Generate and process depth maps
        for i in range(100):  # Run for about 10 seconds
            # Generate simulated depth map
            depth_map = simulate_depth_map()
            
            # Add some movement/animation
            time_offset = i * 0.1
            depth_map += 0.05 * np.sin(time_offset) * np.random.random(depth_map.shape)
            
            # Map depth to sound zones
            zone_sources = mapper.map(depth_map)
            
            # Update audio output
            audio_output.update_sources(zone_sources)
            
            # Print status
            if i % 10 == 0:
                print(f"Frame {i}: {len(zone_sources)} active sound sources")
                zone_counts = {}
                for _, _, _, zone_id in zone_sources:
                    zone_counts[zone_id] = zone_counts.get(zone_id, 0) + 1
                print(f"  Zone distribution: {zone_counts}")
            
            time.sleep(0.1)  # ~10 FPS
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        audio_output.stop()
        print("Audio stopped.")


def run_3d_example():
    """Run the 3D zone-based soundscape example (requires OpenAL)."""
    print("Setting up 3D natural soundscape...")
    
    try:
        from audio_output import OpenALZoneOutput
    except ImportError:
        print("OpenAL not available. Install with: pip install openal")
        return
    
    # Create sound zone configuration
    zone_config = create_example_sound_config()
    
    # Create 3D zone mapper
    mapper = Grid3DZoneMapper(
        zone_config=zone_config,
        grid_size=12,  # Higher resolution for 3D
        min_depth=0.0,
        max_depth=1.0,
        depth_scale=5.0,  # 5 OpenAL units = max distance
        inverse=True
    )
    
    # Create 3D audio output
    audio_output = OpenALZoneOutput(
        zone_config=zone_config,
        sample_rate=44100,
        max_sources_per_zone=4
    )
    
    try:
        # Start audio
        audio_output.start()
        print("3D Audio started. Move your head to experience spatial audio!")
        print("Different zones should appear at different 3D positions.")
        
        # Generate and process depth maps
        for i in range(150):  # Run for about 15 seconds
            # Generate simulated depth map with movement
            depth_map = simulate_depth_map()
            
            # Add spatial movement
            time_offset = i * 0.05
            spatial_offset = np.sin(time_offset) * 0.1
            depth_map += spatial_offset * np.random.random(depth_map.shape)
            
            # Map depth to 3D sound zones
            zone_sources_3d = mapper.map(depth_map)
            
            # Update 3D audio output
            audio_output.update_sources(zone_sources_3d)
            
            # Print status
            if i % 15 == 0:
                print(f"Frame {i}: {len(zone_sources_3d)} active 3D sound sources")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        audio_output.stop()
        print("3D Audio stopped.")


if __name__ == "__main__":
    print("Natural Soundscape Example")
    print("=" * 40)
    print("This example demonstrates the new zone-based audio system.")
    print("Choose an option:")
    print("1. Stereo soundscape (works without additional dependencies)")
    print("2. 3D soundscape (requires OpenAL: pip install openal)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            run_stereo_example()
            break
        elif choice == "2":
            run_3d_example()
            break
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")