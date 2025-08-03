import numpy as np

from audio_mapper.true3d_mapper import Grid3DDepthMapper

def test_close_depth_map_produces_equal_sources():
    """An all-zero (close) depth map should create 4 audio sources with equal max gain and base frequency."""
    mapper = Grid3DDepthMapper(grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.zeros((2, 2), dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    assert all(gain == 1.0 and freq == 440.0 for _, _, _, gain, freq in sources), "All sources should have equal gain and frequency"

def test_far_depth_map_produces_equal_sources():
    """An all-one (far) depth map should create 4 audio sources with equal min gain and base frequency."""
    mapper = Grid3DDepthMapper(grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.full((2, 2), 0.9, dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"
    for _, _, _, gain, freq in sources:
        assert np.isclose(gain, 0.1), f"Expected gain near 0.1, got {gain}"
        assert np.isclose(freq, 1232.0), f"Expected freq near 1232, got {freq}"

def test_correctly_positioned_sources():
    """A depth map with various depth pixel should yield correctly positioned sources for each pixel."""
    mapper = Grid3DDepthMapper(grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.array([[0.0, 0.3], [0.6, 0.9]], dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"

    x1, y1, z1, gain1, freq1 = sources[0]
    x2, y2, z2, gain2, freq2 = sources[1]
    x3, y3, z3, gain3, freq3 = sources[2]
    x4, y4, z4, gain4, freq4 = sources[3]

    assert x1 == -0.5 and y1 == 0.5 and z1 == -1.0 and gain1 == 1.0 and freq1 == 440.0
    assert x2 == 0.5 and y2 == 0.5 and np.isclose(z2, -1.3) and np.isclose(gain2, 0.7) and np.isclose(freq2, 704.0)
    assert x3 == -0.5 and y3 == -0.5 and np.isclose(z3, -1.6) and np.isclose(gain3, 0.4) and np.isclose(freq3, 968.0)
    assert x4 == 0.5 and y4 == -0.5 and np.isclose(z4, -1.9) and np.isclose(gain4, 0.1) and np.isclose(freq4, 1232.0)

def test_correctly_filtered_sources():
    """A depth map with various depth pixel where some are out of bounds should yield correctly positioned sources for each pixel."""
    mapper = Grid3DDepthMapper(grid_size=2, inverse=False, min_depth=0.1, max_depth=0.8)
    depth = np.array([[0.0, 0.3], [0.6, 0.9]], dtype=np.float32)
    sources = mapper.map(depth)
    print(sources)
    assert len(sources) == 2, "Expected 2 sources for a 2x2 grid"

    x2, y2, z2, gain2, freq2 = sources[0]
    x3, y3, z3, gain3, freq3 = sources[1]

    assert x2 == 0.5 and y2 == 0.5 and np.isclose(z2, -1.28, rtol=1e-2) and np.isclose(gain2, 0.7, rtol=1e-1) and np.isclose(freq2, 691.0, rtol=1e-1)
    assert x3 == -0.5 and y3 == -0.5 and np.isclose(z3, -1.71, rtol=1e-2) and np.isclose(gain3, 0.28, rtol=1e-1) and np.isclose(freq3, 1068.0, rtol=1e-1)

def test_correctly_interpolated_grid_cell():
    """A depth map with more depth pixel than grid cells should correctly interpolate pixels and yield correctly positioned sources for each cell."""
    mapper = Grid3DDepthMapper(grid_size=2, inverse=False, min_depth=0.0, max_depth=1.0)
    depth = np.array([
        [0.0, 0.7, 0.9, 1.0],
        [0.1, 0.2, 0.3, 0.7],
        [0.7, 0.9, 1.0, 1.2],
        [1.0, 0.6, 0.9, 1.8],
    ], dtype=np.float32)
    sources = mapper.map(depth)
    assert len(sources) == 4, "Expected 4 sources for a 2x2 grid"

    x1, y1, z1, gain1, freq1 = sources[0]
    x2, y2, z2, gain2, freq2 = sources[1]
    x3, y3, z3, gain3, freq3 = sources[2]
    x4, y4, z4, gain4, freq4 = sources[3]

    assert x1 == -0.5 and y1 == 0.5 and z1 == -1.0 and gain1 == 1.0 and freq1 == 440.0
    assert x2 == 0.5 and y2 == 0.5 and np.isclose(z2, -1.3) and np.isclose(gain2, 0.7) and np.isclose(freq2, 704.0)
    assert x3 == -0.5 and y3 == -0.5 and np.isclose(z3, -1.6) and np.isclose(gain3, 0.4) and np.isclose(freq3, 968.0)
    assert x4 == 0.5 and y4 == -0.5 and np.isclose(z4, -1.9) and np.isclose(gain4, 0.1) and np.isclose(freq4, 1232.0)
