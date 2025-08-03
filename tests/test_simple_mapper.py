import numpy as np

from audio_mapper.simple_mapper import SimpleDepthToAudioMapper

def test_close_depth_map_produces_equal_sources():
    """An all-zero (close) depth map should create 4 audio sources with equal max amplitude and base frequency."""
    mapper = SimpleDepthToAudioMapper(grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.zeros((2, 2), dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    assert all(amp == 1.0 and freq == 440.0 for _, amp, freq in sources), "All sources should have equal amplitude and frequency"

def test_close_depth_map_produces_equal_sources_inverse():
    """An all-max (close) depth map should create 4 audio sources with equal max amplitude and base frequency (inverse=True)."""
    mapper = SimpleDepthToAudioMapper(grid_size=2, inverse=True, min_depth=0.0, max_depth=1.0)
    depth = np.ones((2, 2), dtype=np.float32)  # all-ones = close when inverse=True
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    assert all(amp == 1.0 and freq == 440.0 for _, amp, freq in sources), "All sources should have equal amplitude and frequency"

def test_far_depth_map_produces_equal_sources():
    """An all-0.9 (far) depth map should create 4 audio sources with equal min amplitude and high frequency."""
    mapper = SimpleDepthToAudioMapper(grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.full((2, 2), 0.9, dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    for _, amp, freq in sources:
        assert np.isclose(amp, 0.1), f"Expected amplitude near 0.1, got {amp}"
        assert np.isclose(freq, 1232.0), f"Expected freq near 1232, got {freq}"

def test_far_depth_map_produces_equal_sources_inverse():
    """An all-min (far) depth map should create 4 audio sources with equal min amplitude and high frequency (inverse=True)."""
    mapper = SimpleDepthToAudioMapper(grid_size=2, inverse=True, min_depth=0.0, max_depth=1.0)
    depth = np.full((2, 2), 0.1, dtype=np.float32)  # low values = far when inverse=True
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    for _, amp, freq in sources:
        assert np.isclose(amp, 0.1), f"Expected amplitude near 0.1, got {amp}"
        assert np.isclose(freq, 1232.0), f"Expected freq near 1232, got {freq}"

def test_correctly_positioned_sources():
    """A depth map with various depth values should yield correctly positioned sources for each cell."""
    mapper = SimpleDepthToAudioMapper(grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.array([[0.0, 0.3], [0.6, 0.9]], dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"

    azimuth1, amp1, freq1 = sources[0]
    azimuth2, amp2, freq2 = sources[1]
    azimuth3, amp3, freq3 = sources[2]
    azimuth4, amp4, freq4 = sources[3]

    # Top-left cell (0,0): depth=0.0, azimuth=-0.5
    assert azimuth1 == -0.5 and amp1 == 1.0 and freq1 == 440.0
    # Top-right cell (0,1): depth=0.3, azimuth=0.5
    assert azimuth2 == 0.5 and np.isclose(amp2, 0.7) and np.isclose(freq2, 704.0)
    # Bottom-left cell (1,0): depth=0.6, azimuth=-0.5
    assert azimuth3 == -0.5 and np.isclose(amp3, 0.4) and np.isclose(freq3, 968.0)
    # Bottom-right cell (1,1): depth=0.9, azimuth=0.5
    assert azimuth4 == 0.5 and np.isclose(amp4, 0.1) and np.isclose(freq4, 1232.0)

def test_correctly_positioned_sources_inverse():
    """A depth map with various depth values should yield correctly positioned sources for each cell (inverse=True)."""
    mapper = SimpleDepthToAudioMapper(grid_size=2, inverse=True, min_depth=0.0, max_depth=1.0)
    depth = np.array([[1.0, 0.7], [0.4, 0.1]], dtype=np.float32)  # inverted values from original test
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"

    azimuth1, amp1, freq1 = sources[0]
    azimuth2, amp2, freq2 = sources[1]
    azimuth3, amp3, freq3 = sources[2]
    azimuth4, amp4, freq4 = sources[3]

    # Top-left cell (0,0): depth=1.0, azimuth=-0.5
    assert azimuth1 == -0.5 and amp1 == 1.0 and freq1 == 440.0
    # Top-right cell (0,1): depth=0.7, azimuth=0.5
    assert azimuth2 == 0.5 and np.isclose(amp2, 0.7) and np.isclose(freq2, 704.0)
    # Bottom-left cell (1,0): depth=0.4, azimuth=-0.5
    assert azimuth3 == -0.5 and np.isclose(amp3, 0.4) and np.isclose(freq3, 968.0)
    # Bottom-right cell (1,1): depth=0.1, azimuth=0.5
    assert azimuth4 == 0.5 and np.isclose(amp4, 0.1) and np.isclose(freq4, 1232.0)

def test_correctly_filtered_sources():
    """A depth map with various depth values where some are out of bounds should yield correctly positioned sources for valid cells."""
    mapper = SimpleDepthToAudioMapper(grid_size=2, inverse=False, min_depth=0.1, max_depth=0.8)
    depth = np.array([[0.0, 0.3], [0.6, 0.9]], dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 2, "Expected 2 sources for a 2x2 grid (2 filtered out)"

    azimuth1, amp1, freq1 = sources[0]
    azimuth2, amp2, freq2 = sources[1]

    # Top-right cell (0,1): depth=0.3, azimuth=0.5
    assert azimuth1 == 0.5 and np.isclose(amp1, 0.714, rtol=1e-2) and np.isclose(freq1, 691.4, rtol=1e-1)
    # Bottom-left cell (1,0): depth=0.6, azimuth=-0.5  
    assert azimuth2 == -0.5 and np.isclose(amp2, 0.286, rtol=1e-2) and np.isclose(freq2, 1068.6, rtol=1e-1)

def test_correctly_filtered_sources_inverse():
    """A depth map with various depth values where some are out of bounds should yield correctly positioned sources for valid cells (inverse=True)."""
    mapper = SimpleDepthToAudioMapper(grid_size=2, inverse=True, min_depth=0.1, max_depth=0.8)
    depth = np.array([[0.9, 0.3], [0.6, 0.0]], dtype=np.float32)  # inverted values from original test
    sources = mapper.map(depth)
    assert len(sources) == 2, "Expected 2 sources for a 2x2 grid (2 filtered out)"

    azimuth1, amp1, freq1 = sources[0]
    azimuth2, amp2, freq2 = sources[1]

    # Top-right cell (0,1): depth=0.3, azimuth=0.5
    assert azimuth1 == 0.5 and np.isclose(amp1, 0.286, rtol=1e-2) and np.isclose(freq1, 1068.6, rtol=1e-2)
    # Bottom-left cell (1,0): depth=0.6, azimuth=-0.5
    assert azimuth2 == -0.5 and np.isclose(amp2, 0.714, rtol=1e-2) and np.isclose(freq2, 691.4, rtol=1e-2)

def test_correctly_interpolated_grid_cell():
    """A depth map with more depth pixels than grid cells should correctly interpolate pixels and yield correctly positioned sources for each cell."""
    mapper = SimpleDepthToAudioMapper(grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.array([
        [0.0, 0.7, 0.9, 1.0],
        [0.1, 0.2, 0.3, 0.7],
        [0.7, 0.9, 1.0, 1.2],
        [1.0, 0.6, 0.9, 1.8],
    ], dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"

    azimuth1, amp1, freq1 = sources[0]
    azimuth2, amp2, freq2 = sources[1]
    azimuth3, amp3, freq3 = sources[2]
    azimuth4, amp4, freq4 = sources[3]

    # Top-left cell: closest=0.0, azimuth=-0.5
    assert azimuth1 == -0.5 and amp1 == 1.0 and freq1 == 440.0
    # Top-right cell: closest=0.3, azimuth=0.5
    assert azimuth2 == 0.5 and np.isclose(amp2, 0.7) and np.isclose(freq2, 704.0)
    # Bottom-left cell: closest=0.6, azimuth=-0.5
    assert azimuth3 == -0.5 and np.isclose(amp3, 0.4) and np.isclose(freq3, 968.0)
    # Bottom-right cell: closest=0.9, azimuth=0.5
    assert azimuth4 == 0.5 and np.isclose(amp4, 0.1) and np.isclose(freq4, 1232.0)

def test_correctly_interpolated_grid_cell_inverse():
    """A depth map with more depth pixels than grid cells should correctly interpolate pixels and yield correctly positioned sources for each cell (inverse=True)."""
    mapper = SimpleDepthToAudioMapper(grid_size=2, inverse=True, min_depth=0.0, max_depth=1.0)
    depth = np.array([
        [1.0, 0.3, 0.1, 0.0],
        [0.9, 0.8, 0.7, 0.3],
        [0.3, 0.1, 0.0, -0.2],
        [0.0, 0.4, 0.1, -0.8],
    ], dtype=np.float32)  # inverted values from original test
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"

    azimuth1, amp1, freq1 = sources[0]
    azimuth2, amp2, freq2 = sources[1]
    azimuth3, amp3, freq3 = sources[2]
    azimuth4, amp4, freq4 = sources[3]

    # Top-left cell: closest=1.0, azimuth=-0.5
    assert azimuth1 == -0.5 and amp1 == 1.0 and freq1 == 440.0
    # Top-right cell: closest=0.7, azimuth=0.5
    assert azimuth2 == 0.5 and np.isclose(amp2, 0.7) and np.isclose(freq2, 704.0)
    # Bottom-left cell: closest=0.4, azimuth=-0.5
    assert azimuth3 == -0.5 and np.isclose(amp3, 0.4) and np.isclose(freq3, 968.0)
    # Bottom-right cell: closest=0.1, azimuth=0.5
    assert azimuth4 == 0.5 and np.isclose(amp4, 0.1) and np.isclose(freq4, 1232.0)