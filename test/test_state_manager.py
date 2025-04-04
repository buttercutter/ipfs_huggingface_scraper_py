"""Test the state manager component."""

import os
import json
import tempfile
import shutil
import pytest
from ipfs_huggingface_scraper_py.state_manager import StateManager

@pytest.fixture
def state_manager():
    """Create a temporary state manager for testing."""
    # Create a temporary directory for state files
    temp_dir = tempfile.mkdtemp()
    manager = StateManager(temp_dir)
    
    # Return manager and cleanup function
    yield manager
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_initialization(state_manager):
    """Test that state manager initializes correctly."""
    # Check initial state values
    assert state_manager.state["models_processed"] == 0
    assert state_manager.state["models_completed"] == 0
    assert state_manager.state["models_errored"] == 0
    assert len(state_manager.state["processed_model_ids"]) == 0
    assert len(state_manager.state["completed_model_ids"]) == 0
    assert len(state_manager.state["errored_model_ids"]) == 0
    
    # Check that state file was created
    state_file_path = os.path.join(state_manager.state_dir, "scraper_state.json")
    assert os.path.exists(state_file_path)
    
    # Check that state file contains valid JSON
    with open(state_file_path, 'r') as f:
        state_data = json.load(f)
        assert "models_processed" in state_data
        assert "started_at" in state_data

def test_mark_model_processed(state_manager):
    """Test marking a model as processed."""
    # Mark a model as processed
    state_manager.mark_model_processed("test/model1")
    
    # Check that model is marked as processed
    assert state_manager.is_model_processed("test/model1")
    assert state_manager.state["models_processed"] == 1
    
    # Check that marking again doesn't increment counter
    state_manager.mark_model_processed("test/model1")
    assert state_manager.state["models_processed"] == 1

def test_mark_model_completed(state_manager):
    """Test marking a model as completed."""
    # Mark a model as completed
    state_manager.mark_model_completed("test/model2")
    
    # Check that model is marked as processed and completed
    assert state_manager.is_model_processed("test/model2")
    assert state_manager.is_model_completed("test/model2")
    assert state_manager.state["models_processed"] == 1
    assert state_manager.state["models_completed"] == 1

def test_mark_model_errored(state_manager):
    """Test marking a model as errored."""
    # Mark a model as errored
    state_manager.mark_model_errored("test/model3", "Test error")
    
    # Check that model is marked as processed and has error
    assert state_manager.is_model_processed("test/model3")
    assert state_manager.get_model_error("test/model3") == "Test error"
    assert state_manager.state["models_processed"] == 1
    assert state_manager.state["models_errored"] == 1
    
    # Check that marking as completed removes error
    state_manager.mark_model_completed("test/model3")
    assert state_manager.get_model_error("test/model3") is None

def test_save_load_state(state_manager):
    """Test saving and loading state."""
    # Add some state
    state_manager.mark_model_processed("test/model4")
    state_manager.mark_model_completed("test/model5")
    state_manager.mark_model_errored("test/model6", "Test error")
    
    # Create a new state manager that should load the state
    new_state_manager = StateManager(state_manager.state_dir)
    
    # Check that state was loaded correctly
    assert new_state_manager.is_model_processed("test/model4")
    assert new_state_manager.is_model_completed("test/model5")
    assert new_state_manager.get_model_error("test/model6") == "Test error"
    assert new_state_manager.state["models_processed"] == 3
    assert new_state_manager.state["models_completed"] == 1
    assert new_state_manager.state["models_errored"] == 1

def test_pause_resume(state_manager):
    """Test pausing and resuming."""
    # Test initial state
    assert not state_manager.is_paused()
    
    # Test pause
    state_manager.pause()
    assert state_manager.is_paused()
    
    # Test resume
    state_manager.resume()
    assert not state_manager.is_paused()

def test_reset(state_manager):
    """Test resetting state."""
    # Add some state
    state_manager.mark_model_processed("test/model7")
    state_manager.mark_model_completed("test/model8")
    state_manager.mark_model_errored("test/model9", "Test error")
    
    # Update config
    state_manager.update_config(max_models=500)
    
    # Reset state
    state_manager.reset()
    
    # Check that state was reset
    assert state_manager.state["models_processed"] == 0
    assert state_manager.state["models_completed"] == 0
    assert state_manager.state["models_errored"] == 0
    assert len(state_manager.state["processed_model_ids"]) == 0
    assert len(state_manager.state["completed_model_ids"]) == 0
    assert len(state_manager.state["errored_model_ids"]) == 0
    
    # Check that config was preserved
    assert state_manager.state["config"]["max_models"] == 500

def test_get_progress(state_manager):
    """Test getting progress."""
    # Set total discovered
    state_manager.set_total_discovered(100)
    
    # Add some progress
    for i in range(30):
        state_manager.mark_model_processed(f"test/model{i}")
        if i < 20:
            state_manager.mark_model_completed(f"test/model{i}")
        else:
            state_manager.mark_model_errored(f"test/model{i}", f"Error {i}")
    
    # Get progress
    progress = state_manager.get_progress()
    
    # Check progress values
    assert progress["total_models_discovered"] == 100
    assert progress["models_processed"] == 30
    assert progress["models_completed"] == 20
    assert progress["models_errored"] == 10
    assert progress["completion_percentage"] == 30.0

def test_checkpoint(state_manager):
    """Test checkpointing."""
    # Add some state
    state_manager.mark_model_processed("test/model10")
    state_manager.mark_model_completed("test/model11")
    
    # Create checkpoint
    state_manager.create_checkpoint()
    
    # Check that checkpoint file exists
    checkpoint_file = os.path.join(state_manager.state_dir, "checkpoint_scraper_state.json")
    assert os.path.exists(checkpoint_file)
    
    # Add more state and corrupt main state file
    state_manager.mark_model_processed("test/model12")
    os.remove(os.path.join(state_manager.state_dir, "scraper_state.json"))
    
    # Create a new state manager that should load from checkpoint
    new_state_manager = StateManager(state_manager.state_dir)
    
    # Check that state was loaded from checkpoint
    assert new_state_manager.is_model_processed("test/model10")
    assert new_state_manager.is_model_completed("test/model11")
    assert new_state_manager.state["models_processed"] == 2
    assert new_state_manager.state["models_completed"] == 1

def test_get_unprocessed_models(state_manager):
    """Test getting unprocessed models."""
    # Set current batch
    current_batch = ["model1", "model2", "model3", "model4", "model5"]
    state_manager.set_current_batch(current_batch)
    
    # Mark some models as processed
    state_manager.mark_model_processed("model1")
    state_manager.mark_model_processed("model3")
    state_manager.mark_model_processed("model5")
    
    # Get unprocessed models
    unprocessed = state_manager.get_unprocessed_models()
    
    # Check that only unprocessed models are returned
    assert len(unprocessed) == 2
    assert "model2" in unprocessed
    assert "model4" in unprocessed

def test_update_position(state_manager):
    """Test updating position."""
    # Update position
    state_manager.update_position(42)
    
    # Check that position was updated
    assert state_manager.state["current_position"] == 42
    
    # Check that position is saved in state file
    state_file_path = os.path.join(state_manager.state_dir, "scraper_state.json")
    with open(state_file_path, 'r') as f:
        state_data = json.load(f)
        assert state_data["current_position"] == 42

def test_completion(state_manager):
    """Test marking as completed."""
    # Mark as completed
    state_manager.complete()
    
    # Check that completed flag is set
    assert state_manager.is_completed()
    
    # Check that it's saved in state file
    state_file_path = os.path.join(state_manager.state_dir, "scraper_state.json")
    with open(state_file_path, 'r') as f:
        state_data = json.load(f)
        assert state_data["is_completed"] is True