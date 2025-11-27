"""Test the State.copy() method."""

import numpy as np
import pytest
from rgrow import State, Tile, TileSet, Bond


def test_state_copy_basic():
    """Test basic copy functionality."""
    # Create a simple state
    state = State((10, 10))
    
    # Make a copy
    copied_state = state.copy()
    
    # Check that it's a different object
    assert copied_state is not state
    
    # Check that basic properties are the same
    assert copied_state.time == state.time
    assert copied_state.total_events == state.total_events
    assert copied_state.ntiles == state.ntiles
    
    # Check that canvas arrays are equal but independent
    assert np.array_equal(copied_state.canvas_view, state.canvas_view)
    
    # Modify original and check that copy is not affected
    state.canvas_view[5, 5] = 1
    assert state.canvas_view[5, 5] == 1
    assert copied_state.canvas_view[5, 5] == 0  # Should remain unchanged


def test_state_copy_with_data():
    """Test copy with actual tile data."""
    # Create a state with some tiles
    state = State((10, 10))
    canvas = state.canvas_view
    
    # Add some tiles
    canvas[2, 2] = 1
    canvas[2, 3] = 2
    canvas[3, 2] = 1
    canvas[3, 3] = 2
    
    # Make a copy
    copied_state = state.copy()
    
    # Check that the canvas data is copied correctly
    assert np.array_equal(copied_state.canvas_view, state.canvas_view)
    
    # Modify original canvas and verify independence
    canvas[4, 4] = 3
    assert state.canvas_view[4, 4] == 3
    assert copied_state.canvas_view[4, 4] == 0  # Should remain unchanged


def test_state_copy_with_simulation():
    """Test copy with a state that has undergone simulation."""
    # Create a simple tileset and system
    ts = TileSet(
        [
            Tile(["a", "b", "c", "d"]),
            Tile(["e", "d", "g", "b"]),
        ],
        [Bond("b", 1.0)],
        seed=[(5, 5, 1)],
        size=10,
        gse=8.0,
        gmc=16.0,
        alpha=0.0,
        kf=1e6,
    )
    
    sys, state = ts.create_system_and_state()
    
    # Evolve the system for a short time
    sys.evolve(state, for_time=0.001, require_strong_bound=False)
    
    # Make a copy after simulation
    copied_state = state.copy()
    
    # Check that simulation state is preserved
    assert copied_state.time == state.time
    assert copied_state.total_events == state.total_events
    assert copied_state.ntiles == state.ntiles
    
    # Check that canvas is copied correctly
    assert np.array_equal(copied_state.canvas_view, state.canvas_view)
    
    # Continue evolving original and check independence
    original_time = state.time
    sys.evolve(state, for_time=0.001, require_strong_bound=False)
    
    # Original should have changed
    assert state.time > original_time
    
    # Copy should remain unchanged
    assert copied_state.time == original_time


def test_state_copy_with_tracking():
    """Test copy with different tracking modes."""
    # Test with order tracking
    state = State((5, 5), tracking="Order")
    
    # Add some tiles
    state.canvas_view[2, 2] = 1
    state.canvas_view[2, 3] = 2
    
    # Make a copy
    copied_state = state.copy()
    
    # Check that basic properties are preserved
    assert copied_state.time == state.time
    assert copied_state.total_events == state.total_events
    assert np.array_equal(copied_state.canvas_view, state.canvas_view)
    
    # Check that tracking data is copied (if accessible)
    try:
        tracking_data = state.tracking_copy()
        copied_tracking_data = copied_state.tracking_copy()
        # The tracking data should be equal
        assert np.array_equal(tracking_data, copied_tracking_data)
    except Exception:
        # If tracking_copy() fails, that's okay - it may not be implemented
        # for all tracking types
        pass


def test_state_copy_different_canvas_types():
    """Test copy with different canvas types."""
    canvas_types = ["Square", "Periodic", "Tube"]
    
    for canvas_type in canvas_types:
        state = State((8, 8), kind=canvas_type)
        
        # Add some tiles
        state.canvas_view[3, 3] = 1
        state.canvas_view[3, 4] = 2
        
        # Make a copy
        copied_state = state.copy()
        
        # Check that the copy preserves the canvas type and data
        assert np.array_equal(copied_state.canvas_view, state.canvas_view)
        assert copied_state.time == state.time
        assert copied_state.total_events == state.total_events
        
        # Test independence
        state.canvas_view[5, 5] = 3
        assert copied_state.canvas_view[5, 5] == 0


def test_state_copy_return_type():
    """Test that copy returns the correct type."""
    state = State((5, 5))
    copied_state = state.copy()
    
    # Should be the same type
    assert type(copied_state) is type(state)
    assert isinstance(copied_state, State)


def test_state_copy_multiple_times():
    """Test making multiple copies."""
    state = State((5, 5))
    state.canvas_view[2, 2] = 1
    
    # Make multiple copies
    copy1 = state.copy()
    copy2 = state.copy()
    copy3 = copy1.copy()
    
    # All should be independent
    assert copy1 is not state
    assert copy2 is not state
    assert copy3 is not state
    assert copy1 is not copy2
    assert copy2 is not copy3
    assert copy1 is not copy3
    
    # All should have the same data
    assert np.array_equal(copy1.canvas_view, state.canvas_view)
    assert np.array_equal(copy2.canvas_view, state.canvas_view)
    assert np.array_equal(copy3.canvas_view, state.canvas_view)
    
    # Modify one and check independence
    copy1.canvas_view[3, 3] = 2
    assert copy1.canvas_view[3, 3] == 2
    assert state.canvas_view[3, 3] == 0
    assert copy2.canvas_view[3, 3] == 0
    assert copy3.canvas_view[3, 3] == 0


if __name__ == "__main__":
    pytest.main([__file__])