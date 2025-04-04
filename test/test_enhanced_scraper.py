"""Test the enhanced scraper component."""

import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch, ANY
from ipfs_huggingface_scraper_py.enhanced_scraper import EnhancedScraper
from ipfs_huggingface_scraper_py.config import Config

@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    # Create a temporary file
    fd, config_path = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    
    with open(config_path, 'w') as f:
        f.write("""
[scraper]
output_dir = "test_output"
max_models = 10
save_metadata = true
filename_to_download = "config.json"
batch_size = 5
skip_existing = true
log_level = "INFO"

[api]
base_url = "https://huggingface.co"
api_token = ""
authenticated = false
anonymous_rate_limit = 100.0
daily_anonymous_quota = 1000
max_retries = 2
timeout = 10

[storage]
use_ipfs = false

[state]
state_dir = ".test_state"
auto_resume = true
        """)
    
    yield config_path
    
    # Cleanup
    if os.path.exists(config_path):
        os.remove(config_path)

@pytest.fixture
def enhanced_scraper(temp_config_file):
    """Create an enhanced scraper with test configuration."""
    # Create test output directory
    os.makedirs("test_output", exist_ok=True)
    os.makedirs(".test_state", exist_ok=True)
    
    # Create scraper
    scraper = EnhancedScraper(temp_config_file)
    
    # Replace components with mocks
    scraper.hf_client = MagicMock()
    
    yield scraper
    
    # Cleanup
    import shutil
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")
    if os.path.exists(".test_state"):
        shutil.rmtree(".test_state")

def test_initialization(temp_config_file):
    """Test enhanced scraper initialization."""
    # Create scraper
    scraper = EnhancedScraper(temp_config_file)
    
    # Check components
    assert scraper.config is not None
    assert scraper.state_manager is not None
    assert scraper.rate_limiter is not None
    assert scraper.ipfs_storage is not None
    assert scraper.hf_client is not None
    
    # Check configuration
    assert scraper.config.config["scraper"]["output_dir"] == "test_output"
    assert scraper.config.config["scraper"]["max_models"] == 10
    assert scraper.config.config["api"]["authenticated"] is False
    assert scraper.config.config["storage"]["use_ipfs"] is False

@patch("ipfs_huggingface_scraper_py.enhanced_scraper.list_models")
def test_discover_models(mock_list_models, enhanced_scraper):
    """Test model discovery."""
    # Mock model list
    mock_models = []
    for i in range(5):
        model = MagicMock()
        model.id = f"model{i}"
        mock_models.append(model)
    
    # Configure mock
    mock_list_models.return_value = mock_models
    
    # Discover models
    model_ids = enhanced_scraper._discover_models(max_models=5)
    
    # Check result
    assert len(model_ids) == 5
    assert model_ids == ["model0", "model1", "model2", "model3", "model4"]
    
    # Check that list_models was called
    mock_list_models.assert_called_once()
    
    # Check that state was updated
    assert enhanced_scraper.state_manager.state["total_models_discovered"] == 5

@patch("ipfs_huggingface_scraper_py.enhanced_scraper.list_models")
def test_discover_models_error(mock_list_models, enhanced_scraper):
    """Test model discovery with error."""
    # Configure mock to raise exception
    mock_list_models.side_effect = Exception("API error")
    
    # Discover models
    model_ids = enhanced_scraper._discover_models(max_models=5)
    
    # Check result
    assert model_ids == []

def test_process_model(enhanced_scraper):
    """Test processing a single model."""
    # Mock functions
    enhanced_scraper._save_model_metadata = MagicMock(return_value=True)
    enhanced_scraper._download_model_file = MagicMock(return_value=True)
    enhanced_scraper.ipfs_storage.store_model_files = MagicMock(return_value="QmTestCid")
    
    # Process model
    result = enhanced_scraper._process_model("test/model")
    
    # Check result
    assert result is True
    
    # Check that functions were called
    enhanced_scraper.state_manager.mark_model_processed.assert_called_with("test/model")
    enhanced_scraper._save_model_metadata.assert_called_with("test/model", ANY)
    enhanced_scraper._download_model_file.assert_called_with("test/model", "config.json", ANY)
    
    # Should not call IPFS storage since use_ipfs is False
    enhanced_scraper.ipfs_storage.store_model_files.assert_not_called()
    
    # Test with IPFS enabled
    enhanced_scraper.config.config["storage"]["use_ipfs"] = True
    enhanced_scraper.ipfs_storage.is_ipfs_available = MagicMock(return_value=True)
    
    # Process model
    result = enhanced_scraper._process_model("test/model2")
    
    # Check result
    assert result is True
    
    # Should call IPFS storage
    enhanced_scraper.ipfs_storage.store_model_files.assert_called_once()
    
    # Test metadata failure
    enhanced_scraper._save_model_metadata.return_value = False
    
    # Process model
    result = enhanced_scraper._process_model("test/model3")
    
    # Check result
    assert result is False
    
    # Check that download was not called
    enhanced_scraper._download_model_file.assert_called_once()
    
    # Test download failure
    enhanced_scraper._save_model_metadata.return_value = True
    enhanced_scraper._download_model_file.return_value = False
    
    # Process model
    result = enhanced_scraper._process_model("test/model4")
    
    # Check result
    assert result is False
    
    # Test exception
    enhanced_scraper._save_model_metadata.side_effect = Exception("Test error")
    
    # Process model
    result = enhanced_scraper._process_model("test/model5")
    
    # Check result
    assert result is False
    
    # Check that error was recorded
    enhanced_scraper.state_manager.mark_model_errored.assert_called_with("test/model5", ANY)

def test_save_model_metadata(enhanced_scraper):
    """Test saving model metadata."""
    # Mock model info
    model_info = MagicMock()
    model_info.__dict__ = {
        "_some_private": "value",
        "id": "test/model",
        "modelId": "test/model",
        "tags": ["test", "model"],
        "pipeline_tag": "text-generation",
        "last_modified": None
    }
    enhanced_scraper.hf_client.model_info.return_value = model_info
    
    # Create test directory
    model_dir = os.path.join("test_output", "test__model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save metadata
    result = enhanced_scraper._save_model_metadata("test/model", model_dir)
    
    # Check result
    assert result is True
    
    # Check that file was created
    metadata_path = os.path.join(model_dir, "metadata.json")
    assert os.path.exists(metadata_path)
    
    # Check file contents
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        assert metadata["id"] == "test/model"
        assert metadata["modelId"] == "test/model"
        assert metadata["tags"] == ["test", "model"]
        assert metadata["pipeline_tag"] == "text-generation"
    
    # Test with API error
    enhanced_scraper.hf_client.model_info.side_effect = Exception("API error")
    
    # Save metadata
    result = enhanced_scraper._save_model_metadata("test/model2", model_dir)
    
    # Check result
    assert result is False

def test_download_model_file(enhanced_scraper):
    """Test downloading a model file."""
    # Mock hf_hub_download
    with patch("ipfs_huggingface_scraper_py.enhanced_scraper.hf_hub_download") as mock_download:
        # Configure mock to return a path
        model_dir = os.path.join("test_output", "test__model")
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, "config.json")
        mock_download.return_value = file_path
        
        # Create the file
        with open(file_path, 'w') as f:
            f.write("{}")
        
        # Download file
        result = enhanced_scraper._download_model_file("test/model", "config.json", model_dir)
        
        # Check result
        assert result is True
        
        # Check that download was called
        mock_download.assert_called_once_with(
            repo_id="test/model",
            filename="config.json",
            repo_type="model",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        # Test repository not found
        from huggingface_hub.utils import RepositoryNotFoundError
        mock_download.side_effect = RepositoryNotFoundError("Not found")
        
        # Download file
        result = enhanced_scraper._download_model_file("test/model", "config.json", model_dir)
        
        # Check result
        assert result is False
        
        # Test file not found
        from huggingface_hub.utils import EntryNotFoundError
        mock_download.side_effect = EntryNotFoundError("File not found")
        
        # Download file
        result = enhanced_scraper._download_model_file("test/model", "config.json", model_dir)
        
        # Check result
        assert result is False
        
        # Test validation error
        from huggingface_hub.utils import HFValidationError
        mock_download.side_effect = HFValidationError("Validation error")
        
        # Download file
        result = enhanced_scraper._download_model_file("test/model", "config.json", model_dir)
        
        # Check result
        assert result is False
        
        # Test generic exception
        mock_download.side_effect = Exception("Download error")
        
        # Download file
        result = enhanced_scraper._download_model_file("test/model", "config.json", model_dir)
        
        # Check result
        assert result is False

@patch("ipfs_huggingface_scraper_py.enhanced_scraper.list_models")
def test_scrape_models(mock_list_models, enhanced_scraper):
    """Test the scrape_models method."""
    # Mock model list
    mock_models = []
    for i in range(5):
        model = MagicMock()
        model.id = f"model{i}"
        mock_models.append(model)
    
    # Configure mock
    mock_list_models.return_value = mock_models
    
    # Mock process_models_in_batches
    enhanced_scraper._process_models_in_batches = MagicMock()
    
    # Scrape models
    enhanced_scraper.scrape_models(max_models=5)
    
    # Check that discover_models was called
    mock_list_models.assert_called_once()
    
    # Check that process_models_in_batches was called
    enhanced_scraper._process_models_in_batches.assert_called_once_with(
        ["model0", "model1", "model2", "model3", "model4"], 
        5
    )
    
    # Check that scraping was marked as completed
    enhanced_scraper.state_manager.complete.assert_called_once()
    
    # Test with keyboard interrupt
    enhanced_scraper._process_models_in_batches.side_effect = KeyboardInterrupt()
    
    # Scrape models
    enhanced_scraper.scrape_models(max_models=5)
    
    # Check that state was paused
    enhanced_scraper.state_manager.pause.assert_called_once()
    
    # Test with generic exception
    enhanced_scraper._process_models_in_batches.side_effect = Exception("Scraper error")
    
    # Scrape models
    enhanced_scraper.scrape_models(max_models=5)
    
    # Check that state was paused
    assert enhanced_scraper.state_manager.pause.call_count == 2

@patch("ipfs_huggingface_scraper_py.enhanced_scraper.list_models")
def test_process_models_in_batches(mock_list_models, enhanced_scraper):
    """Test processing models in batches."""
    # Mock model list
    model_ids = [f"model{i}" for i in range(10)]
    
    # Mock state
    enhanced_scraper.state_manager.is_paused.return_value = False
    enhanced_scraper.state_manager.is_model_processed.return_value = False
    
    # Mock _process_batch
    enhanced_scraper._process_batch = MagicMock()
    
    # Process models in batches
    enhanced_scraper._process_models_in_batches(model_ids, batch_size=3)
    
    # Check that _process_batch was called for each batch
    assert enhanced_scraper._process_batch.call_count == 4  # 10 models in batches of 3 = 4 batches
    
    # Check batch contents
    enhanced_scraper._process_batch.assert_any_call(["model0", "model1", "model2"])
    enhanced_scraper._process_batch.assert_any_call(["model3", "model4", "model5"])
    enhanced_scraper._process_batch.assert_any_call(["model6", "model7", "model8"])
    enhanced_scraper._process_batch.assert_any_call(["model9"])
    
    # Check that state was updated
    assert enhanced_scraper.state_manager.set_current_batch.call_count == 4
    assert enhanced_scraper.state_manager.update_position.call_count == 4
    assert enhanced_scraper.state_manager.create_checkpoint.call_count == 4
    
    # Test with pause
    enhanced_scraper.state_manager.is_paused.return_value = True
    enhanced_scraper._process_batch.reset_mock()
    
    # Process models in batches
    enhanced_scraper._process_models_in_batches(model_ids, batch_size=3)
    
    # Check that no batches were processed
    enhanced_scraper._process_batch.assert_not_called()
    
    # Test with skip_existing=True
    enhanced_scraper.state_manager.is_paused.return_value = False
    enhanced_scraper.config.config["scraper"]["skip_existing"] = True
    enhanced_scraper._process_batch.reset_mock()
    
    # Make some models appear already processed
    def is_processed(model_id):
        return model_id in ["model0", "model3", "model6"]
    
    enhanced_scraper.state_manager.is_model_processed.side_effect = is_processed
    
    # Process models in batches
    enhanced_scraper._process_models_in_batches(model_ids, batch_size=3)
    
    # Check that _process_batch was called for each batch, with filtered contents
    assert enhanced_scraper._process_batch.call_count == 4
    enhanced_scraper._process_batch.assert_any_call(["model1", "model2"])
    enhanced_scraper._process_batch.assert_any_call(["model4", "model5"])
    enhanced_scraper._process_batch.assert_any_call(["model7", "model8"])
    enhanced_scraper._process_batch.assert_any_call(["model9"])

@patch("ipfs_huggingface_scraper_py.enhanced_scraper.ThreadPoolExecutor")
def test_process_batch(mock_executor, enhanced_scraper):
    """Test processing a batch of models."""
    # Mock threadpool and future
    mock_pool = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = True
    mock_pool.__enter__.return_value = mock_pool
    mock_pool.submit.return_value = mock_future
    mock_executor.return_value = mock_pool
    
    # Process batch
    enhanced_scraper._process_batch(["model1", "model2", "model3"])
    
    # Check that threadpool was created
    mock_executor.assert_called_once_with(max_workers=5)
    
    # Check that submit was called for each model
    assert mock_pool.submit.call_count == 3
    mock_pool.submit.assert_any_call(enhanced_scraper._process_model, "model1")
    mock_pool.submit.assert_any_call(enhanced_scraper._process_model, "model2")
    mock_pool.submit.assert_any_call(enhanced_scraper._process_model, "model3")
    
    # Check that mark_model_completed was called for each successful model
    assert enhanced_scraper.state_manager.mark_model_completed.call_count == 3
    
    # Test with future exception
    mock_future.result.side_effect = Exception("Future error")
    enhanced_scraper.state_manager.mark_model_completed.reset_mock()
    
    # Process batch
    enhanced_scraper._process_batch(["model4"])
    
    # Check that mark_model_completed was not called
    enhanced_scraper.state_manager.mark_model_completed.assert_not_called()
    
    # Check that mark_model_errored was called
    enhanced_scraper.state_manager.mark_model_errored.assert_called_once_with("model4", ANY)

@patch("ipfs_huggingface_scraper_py.enhanced_scraper.list_models")
def test_resume(mock_list_models, enhanced_scraper):
    """Test resuming a paused scraping operation."""
    # Mock model list
    mock_models = []
    for i in range(5):
        model = MagicMock()
        model.id = f"model{i}"
        mock_models.append(model)
    
    # Configure mock
    mock_list_models.return_value = mock_models
    
    # Mock is_paused to return True then False (for the check in resume and then in _process_models_in_batches)
    enhanced_scraper.state_manager.is_paused.side_effect = [True, False]
    
    # Mock process_models_in_batches
    enhanced_scraper._process_models_in_batches = MagicMock()
    
    # Resume
    enhanced_scraper.resume()
    
    # Check that resume was called
    enhanced_scraper.state_manager.resume.assert_called_once()
    
    # Check that discover_models was called
    mock_list_models.assert_called_once()
    
    # Check that process_models_in_batches was called
    enhanced_scraper._process_models_in_batches.assert_called_once_with(
        ["model0", "model1", "model2", "model3", "model4"], 
        5
    )
    
    # Test not paused
    enhanced_scraper.state_manager.is_paused.side_effect = [False]
    enhanced_scraper.state_manager.resume.reset_mock()
    mock_list_models.reset_mock()
    enhanced_scraper._process_models_in_batches.reset_mock()
    
    # Resume
    enhanced_scraper.resume()
    
    # Check that no operations were performed
    enhanced_scraper.state_manager.resume.assert_not_called()
    mock_list_models.assert_not_called()
    enhanced_scraper._process_models_in_batches.assert_not_called()
    
    # Test with cached model count
    enhanced_scraper.state_manager.is_paused.side_effect = [True, False]
    enhanced_scraper.state_manager.state["total_models_discovered"] = 5
    
    # Resume
    enhanced_scraper.resume()
    
    # Check that discover_models was still called (we don't cache the actual model IDs)
    mock_list_models.assert_called_once()
    
    # Test completing after resume
    enhanced_scraper.state_manager.is_paused.side_effect = [True, False]
    enhanced_scraper.state_manager.is_completed.return_value = False
    
    # Resume
    enhanced_scraper.resume()
    
    # Check that complete was called
    enhanced_scraper.state_manager.complete.assert_called_once()