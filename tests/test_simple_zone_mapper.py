import pytest
import numpy as np
from pathlib import Path

from audio_mapper.simple_zone_mapper import SimpleZoneMapper
from audio_mapper.sound_zones import SoundZone, SoundZoneConfig


@pytest.fixture
def test_zone_config():
    """Create a test sound zone configuration for testing."""
    zones = [
        SoundZone(
            zone_id="far",
            min_closeness=0.0,
            max_closeness=0.33,
            audio_file=Path("test_far.wav"),
            base_volume=0.8,
            fade_distance=0.0  # No fade for cleaner testing
        ),
        SoundZone(
            zone_id="medium", 
            min_closeness=0.34,
            max_closeness=0.66,
            audio_file=Path("test_medium.wav"),
            base_volume=0.9,
            fade_distance=0.0  # No fade for cleaner testing
        ),
        SoundZone(
            zone_id="close",
            min_closeness=0.67,
            max_closeness=1.0,
            audio_file=Path("test_close.wav"),
            base_volume=1.0,
            fade_distance=0.0  # No fade for cleaner testing
        )
    ]
    return SoundZoneConfig(zones)


def test_close_depth_map_produces_close_zone_sources(test_zone_config):
    """An all-zero (close) depth map should create 4 audio sources in the 'close' zone."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.zeros((2, 2), dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    assert all(zone_id == "close" for _, _, _, zone_id in sources), "All sources should be in 'close' zone"
    assert all(amp == 1.0 for _, amp, _, _ in sources), "All sources should have max amplitude"


def test_close_depth_map_produces_close_zone_sources_inverse(test_zone_config):
    """An all-max (close) depth map should create 4 audio sources in the 'close' zone (inverse=True)."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=True, min_depth=0.0, max_depth=1.0)
    depth = np.ones((2, 2), dtype=np.float32)  # all-ones = close when inverse=True
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    assert all(zone_id == "close" for _, _, _, zone_id in sources), "All sources should be in 'close' zone"
    assert all(amp == 1.0 for _, amp, _, _ in sources), "All sources should have max amplitude"


def test_far_depth_map_produces_far_zone_sources(test_zone_config):
    """An all-0.9 (far) depth map should create 4 audio sources in the 'far' zone."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.full((2, 2), 0.9, dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    assert all(zone_id == "far" for _, _, _, zone_id in sources), "All sources should be in 'far' zone"
    for _, amp, _, _ in sources:
        assert np.isclose(amp, 0.1), f"Expected amplitude near 0.1, got {amp}"


def test_far_depth_map_produces_far_zone_sources_inverse(test_zone_config):
    """An all-min (far) depth map should create 4 audio sources in the 'far' zone (inverse=True)."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=True, min_depth=0.0, max_depth=1.0)
    depth = np.full((2, 2), 0.1, dtype=np.float32)  # low values = far when inverse=True
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    assert all(zone_id == "far" for _, _, _, zone_id in sources), "All sources should be in 'far' zone"
    for _, amp, _, _ in sources:
        assert np.isclose(amp, 0.1), f"Expected amplitude near 0.1, got {amp}"


def test_correctly_positioned_and_zoned_sources(test_zone_config):
    """A depth map with various depth values should yield correctly positioned sources in appropriate zones."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.array([[0.0, 0.35], [0.5, 0.9]], dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"

    azimuth1, amp1, closeness1, zone_id1 = sources[0]
    azimuth2, amp2, closeness2, zone_id2 = sources[1] 
    azimuth3, amp3, closeness3, zone_id3 = sources[2]
    azimuth4, amp4, closeness4, zone_id4 = sources[3]

    # Top-left cell (0,0): depth=0.0 -> closeness=1.0 -> 'close' zone, azimuth=-0.5
    assert azimuth1 == -0.5 and amp1 == 1.0 and zone_id1 == "close"
    # Top-right cell (0,1): depth=0.35 -> closeness=0.65 -> 'medium' zone, azimuth=0.5
    assert azimuth2 == 0.5 and np.isclose(amp2, 0.65) and zone_id2 == "medium"
    # Bottom-left cell (1,0): depth=0.5 -> closeness=0.5 -> 'medium' zone, azimuth=-0.5
    assert azimuth3 == -0.5 and np.isclose(amp3, 0.5) and zone_id3 == "medium"
    # Bottom-right cell (1,1): depth=0.9 -> closeness=0.1 -> 'far' zone, azimuth=0.5
    assert azimuth4 == 0.5 and np.isclose(amp4, 0.1) and zone_id4 == "far"


def test_correctly_positioned_and_zoned_sources_inverse(test_zone_config):
    """A depth map with various depth values should yield correctly positioned sources in appropriate zones (inverse=True)."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=True, min_depth=0.0, max_depth=1.0)
    depth = np.array([[1.0, 0.65], [0.5, 0.1]], dtype=np.float32)  # inverted values
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"

    azimuth1, amp1, closeness1, zone_id1 = sources[0]
    azimuth2, amp2, closeness2, zone_id2 = sources[1]
    azimuth3, amp3, closeness3, zone_id3 = sources[2]
    azimuth4, amp4, closeness4, zone_id4 = sources[3]

    # Top-left cell (0,0): depth=1.0 -> closeness=1.0 -> 'close' zone, azimuth=-0.5
    assert azimuth1 == -0.5 and amp1 == 1.0 and zone_id1 == "close"
    # Top-right cell (0,1): depth=0.65 -> closeness=0.65 -> 'medium' zone, azimuth=0.5
    assert azimuth2 == 0.5 and np.isclose(amp2, 0.65) and zone_id2 == "medium"
    # Bottom-left cell (1,0): depth=0.5 -> closeness=0.5 -> 'medium' zone, azimuth=-0.5
    assert azimuth3 == -0.5 and np.isclose(amp3, 0.5) and zone_id3 == "medium"
    # Bottom-right cell (1,1): depth=0.1 -> closeness=0.1 -> 'far' zone, azimuth=0.5
    assert azimuth4 == 0.5 and np.isclose(amp4, 0.1) and zone_id4 == "far"


def test_correctly_filtered_sources(test_zone_config):
    """A depth map with various depth values where some are out of bounds should yield correctly positioned sources for valid cells."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=False, min_depth=0.1, max_depth=0.8)
    depth = np.array([[0.0, 0.3], [0.6, 0.9]], dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 2, "Expected 2 sources for a 2x2 grid (2 filtered out)"

    azimuth1, amp1, closeness1, zone_id1 = sources[0]
    azimuth2, amp2, closeness2, zone_id2 = sources[1]

    # Top-right cell (0,1): depth=0.3, azimuth=0.5, closeness ≈ 0.714, close zone
    assert azimuth1 == 0.5 and np.isclose(amp1, 0.714, rtol=1e-2) and zone_id1 == "close"
    # Bottom-left cell (1,0): depth=0.6, azimuth=-0.5, closeness ≈ 0.286, far zone
    assert azimuth2 == -0.5 and np.isclose(amp2, 0.286, rtol=1e-2) and zone_id2 == "far"


def test_correctly_filtered_sources_inverse(test_zone_config):
    """A depth map with various depth values where some are out of bounds should yield correctly positioned sources for valid cells (inverse=True)."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=True, min_depth=0.1, max_depth=0.8)
    depth = np.array([[0.9, 0.3], [0.6, 0.0]], dtype=np.float32)  # inverted values from original test
    sources = mapper.map(depth)
    assert len(sources) == 2, "Expected 2 sources for a 2x2 grid (2 filtered out)"

    azimuth1, amp1, closeness1, zone_id1 = sources[0]
    azimuth2, amp2, closeness2, zone_id2 = sources[1]

    # Top-right cell (0,1): depth=0.3, azimuth=0.5, closeness ≈ 0.286, far zone
    assert azimuth1 == 0.5 and np.isclose(amp1, 0.286, rtol=1e-2) and zone_id1 == "far"
    # Bottom-left cell (1,0): depth=0.6, azimuth=-0.5, closeness ≈ 0.714, close zone
    assert azimuth2 == -0.5 and np.isclose(amp2, 0.714, rtol=1e-2) and zone_id2 == "close"


def test_correctly_interpolated_grid_cell(test_zone_config):
    """A depth map with more depth pixels than grid cells should correctly interpolate pixels and yield correctly positioned sources for each cell."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.array([
        [0.0, 0.7, 0.9, 1.0],
        [0.1, 0.2, 0.3, 0.7],
        [0.7, 0.9, 1.0, 1.2],
        [1.0, 0.6, 0.9, 1.8],
    ], dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"

    azimuth1, amp1, closeness1, zone_id1 = sources[0]
    azimuth2, amp2, closeness2, zone_id2 = sources[1]
    azimuth3, amp3, closeness3, zone_id3 = sources[2]
    azimuth4, amp4, closeness4, zone_id4 = sources[3]

    # Top-left cell: closest=0.0, azimuth=-0.5, closeness=1.0, close zone
    assert azimuth1 == -0.5 and amp1 == 1.0 and zone_id1 == "close"
    # Top-right cell: closest=0.3, azimuth=0.5, closeness=0.7, close zone
    assert azimuth2 == 0.5 and np.isclose(amp2, 0.7) and zone_id2 == "close"
    # Bottom-left cell: closest=0.6, azimuth=-0.5, closeness=0.4, medium zone
    assert azimuth3 == -0.5 and np.isclose(amp3, 0.4) and zone_id3 == "medium"
    # Bottom-right cell: closest=0.9, azimuth=0.5, closeness=0.1, far zone
    assert azimuth4 == 0.5 and np.isclose(amp4, 0.1) and zone_id4 == "far"


def test_correctly_interpolated_grid_cell_inverse(test_zone_config):
    """A depth map with more depth pixels than grid cells should correctly interpolate pixels and yield correctly positioned sources for each cell (inverse=True)."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=True, min_depth=0.0, max_depth=1.0)
    depth = np.array([
        [1.0, 0.3, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.3],
        [0.3, 0.1, 0.0, -0.2],
        [0.0, 0.4, 0.1, -0.8],
    ], dtype=np.float32)  # inverted values from original test
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"

    azimuth1, amp1, closeness1, zone_id1 = sources[0]
    azimuth2, amp2, closeness2, zone_id2 = sources[1]
    azimuth3, amp3, closeness3, zone_id3 = sources[2]
    azimuth4, amp4, closeness4, zone_id4 = sources[3]

    # Top-left cell: closest=1.0, azimuth=-0.5, closeness=1.0, close zone
    assert azimuth1 == -0.5 and amp1 == 1.0 and zone_id1 == "close"
    # Top-right cell: closest=0.7, azimuth=0.5, closeness=0.7, close zone
    assert azimuth2 == 0.5 and np.isclose(amp2, 0.7) and zone_id2 == "close"
    # Bottom-left cell: closest=0.4, azimuth=-0.5, closeness=0.4, medium zone
    assert azimuth3 == -0.5 and np.isclose(amp3, 0.4) and zone_id3 == "medium"
    # Bottom-right cell: closest=0.1, azimuth=0.5, closeness=0.1, far zone
    assert azimuth4 == 0.5 and np.isclose(amp4, 0.1) and zone_id4 == "far"


def test_medium_depth_produces_medium_zone(test_zone_config):
    """A depth map with medium values should produce sources in the medium zone."""
    mapper = SimpleZoneMapper(zone_config=test_zone_config, grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.full((2, 2), 0.5, dtype=np.float32)  # closeness = 0.5 -> medium zone
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    assert all(zone_id == "medium" for _, _, _, zone_id in sources), "All sources should be in 'medium' zone"
    assert all(np.isclose(amp, 0.5) for _, amp, _, _ in sources), "All sources should have 0.5 amplitude"