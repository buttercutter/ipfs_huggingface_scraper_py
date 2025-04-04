"""Test the IPFS integration component."""

import os
import json
import tempfile
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from ipfs_huggingface_scraper_py.ipfs_integration import IpfsStorage

# Skip tests if dependencies not available
ipfs_available = True
try:
    from ipfs_kit_py.ipfs_kit import IpfsKit
    from ipfs_datasets_py.ipfs_datasets import IpfsDataset
except ImportError:
    ipfs_available = False

# Mock IPFS client for testing
@pytest.fixture
def mock_ipfs_kit():
    """Create a mock IPFS kit."""
    mock_kit = MagicMock()
    
    # Mock add method
    mock_kit.add.return_value = {"Hash": "QmTestHash1234"}
    
    # Mock pin.add method
    mock_kit.pin = MagicMock()
    mock_kit.pin.add.return_value = {"Pins": ["QmTestHash1234"]}
    
    # Mock get method
    mock_kit.get.return_value = True
    
    # Mock calculate_cid method
    mock_kit.calculate_cid = MagicMock(return_value="QmCalculatedCid1234")
    
    return mock_kit

@pytest.fixture
def mock_ipfs_dataset():
    """Create a mock IPFS dataset."""
    mock_dataset = MagicMock()
    
    # Mock create_car method
    mock_dataset.create_car.return_value = True
    
    return mock_dataset

@pytest.fixture
def ipfs_storage(mock_ipfs_kit, mock_ipfs_dataset):
    """Create an IPFS storage with mocked dependencies."""
    storage = IpfsStorage({"use_ipfs": True})
    
    # Replace the real IPFS kit with the mock
    storage.ipfs_kit = mock_ipfs_kit
    storage.ipfs_dataset = mock_ipfs_dataset
    
    # Temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    
    yield storage
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

@pytest.mark.skipif(not ipfs_available, reason="IPFS dependencies not available")
def test_initialization():
    """Test initialization with real dependencies."""
    # With IPFS enabled
    storage = IpfsStorage({"use_ipfs": True})
    assert hasattr(storage, "ipfs_kit")
    assert hasattr(storage, "ipfs_dataset")
    
    # With IPFS disabled
    storage = IpfsStorage({"use_ipfs": False})
    assert storage.ipfs_kit is None
    assert storage.ipfs_dataset is None

def test_is_ipfs_available(ipfs_storage):
    """Test checking IPFS availability."""
    # Should be available with mocks
    assert ipfs_storage.is_ipfs_available() is True
    
    # Test when not available
    ipfs_storage.ipfs_kit = None
    assert ipfs_storage.is_ipfs_available() is False
    
    # Test when only kit is available
    ipfs_storage.ipfs_kit = MagicMock()
    ipfs_storage.ipfs_dataset = None
    assert ipfs_storage.is_ipfs_available() is False

def test_add_file_to_ipfs(ipfs_storage):
    """Test adding a file to IPFS."""
    # Create a test file
    fd, file_path = tempfile.mkstemp()
    os.write(fd, b"test content")
    os.close(fd)
    
    try:
        # Add file to IPFS
        cid = ipfs_storage.add_file_to_ipfs(file_path)
        
        # Check result
        assert cid == "QmTestHash1234"
        
        # Check that add was called with correct params
        ipfs_storage.ipfs_kit.add.assert_called_once_with(file_path, **{})
    finally:
        # Cleanup
        os.remove(file_path)
    
    # Test with options
    ipfs_storage.config["ipfs_add_options"] = {
        "pin": True,
        "wrap_with_directory": True
    }
    
    # Create another test file
    fd, file_path = tempfile.mkstemp()
    os.write(fd, b"test content")
    os.close(fd)
    
    try:
        # Add file to IPFS
        cid = ipfs_storage.add_file_to_ipfs(file_path)
        
        # Check result
        assert cid == "QmTestHash1234"
        
        # Check that add was called with correct params
        ipfs_storage.ipfs_kit.add.assert_called_with(
            file_path, 
            pin=True, 
            wrap_with_directory=True
        )
    finally:
        # Cleanup
        os.remove(file_path)
    
    # Test error handling
    ipfs_storage.ipfs_kit.add.side_effect = Exception("IPFS error")
    assert ipfs_storage.add_file_to_ipfs("nonexistent.txt") is None

def test_add_directory_to_ipfs(ipfs_storage):
    """Test adding a directory to IPFS."""
    # Create a test directory
    temp_dir = tempfile.mkdtemp()
    
    # Create some files in the directory
    for i in range(3):
        with open(os.path.join(temp_dir, f"file{i}.txt"), "w") as f:
            f.write(f"test content {i}")
    
    # Mock return value for directory with multiple entries
    ipfs_storage.ipfs_kit.add.return_value = [
        {"Name": "file0.txt", "Hash": "QmHash0"},
        {"Name": "file1.txt", "Hash": "QmHash1"},
        {"Name": "file2.txt", "Hash": "QmHash2"},
        {"Name": temp_dir, "Hash": "QmDirHash"}
    ]
    
    try:
        # Add directory to IPFS
        cid = ipfs_storage.add_directory_to_ipfs(temp_dir)
        
        # Check result
        assert cid == "QmDirHash"
        
        # Check that add was called with recursive=True
        ipfs_storage.ipfs_kit.add.assert_called_with(
            temp_dir, 
            recursive=True
        )
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    # Test with unusual return value
    ipfs_storage.ipfs_kit.add.return_value = []
    assert ipfs_storage.add_directory_to_ipfs("/tmp") is None
    
    # Test with error
    ipfs_storage.ipfs_kit.add.side_effect = Exception("IPFS error")
    assert ipfs_storage.add_directory_to_ipfs("/tmp") is None

def test_pin_cid(ipfs_storage):
    """Test pinning a CID."""
    # Pin a CID
    result = ipfs_storage.pin_cid("QmTestCid")
    
    # Check result
    assert result is True
    
    # Check that pin.add was called
    ipfs_storage.ipfs_kit.pin.add.assert_called_once_with("QmTestCid")
    
    # Test with error response
    ipfs_storage.ipfs_kit.pin.add.return_value = {"Other": "value"}
    assert ipfs_storage.pin_cid("QmTestCid") is False
    
    # Test with exception
    ipfs_storage.ipfs_kit.pin.add.side_effect = Exception("IPFS error")
    assert ipfs_storage.pin_cid("QmTestCid") is False
    
    # Test with IPFS not available
    ipfs_storage.ipfs_kit = None
    assert ipfs_storage.pin_cid("QmTestCid") is False

def test_store_metadata_as_json(ipfs_storage):
    """Test storing metadata as JSON."""
    # Test metadata
    metadata = {
        "name": "Test Model",
        "description": "A test model",
        "parameters": 1000000,
        "tags": ["test", "example"]
    }
    
    # Store metadata
    file_path, cid = ipfs_storage.store_metadata_as_json(metadata)
    
    # Check result
    assert file_path is not None
    assert cid == "QmTestHash1234"
    
    # Check that file exists and contains correct data
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        stored_data = json.load(f)
        assert stored_data == metadata
    
    # Cleanup
    os.remove(file_path)
    
    # Test with output path
    output_path = tempfile.mktemp(suffix=".json")
    file_path, cid = ipfs_storage.store_metadata_as_json(metadata, output_path)
    
    # Check result
    assert file_path == output_path
    assert os.path.exists(output_path)
    
    # Cleanup
    os.remove(output_path)
    
    # Test error handling
    with patch("json.dump", side_effect=Exception("JSON error")):
        file_path, cid = ipfs_storage.store_metadata_as_json(metadata)
        assert file_path is None
        assert cid is None

def test_store_metadata_as_parquet(ipfs_storage):
    """Test storing metadata as Parquet."""
    # Test metadata list
    metadata_list = [
        {"name": "Model 1", "parameters": 1000000, "tags": ["test"]},
        {"name": "Model 2", "parameters": 2000000, "tags": ["example"]},
        {"name": "Model 3", "parameters": 3000000, "tags": ["test", "example"]}
    ]
    
    # Store metadata
    file_path, cid = ipfs_storage.store_metadata_as_parquet(metadata_list)
    
    # Check result
    assert file_path is not None
    assert cid == "QmTestHash1234"
    
    # Check that file exists and contains correct data
    assert os.path.exists(file_path)
    df = pd.read_parquet(file_path)
    assert len(df) == 3
    assert list(df["name"]) == ["Model 1", "Model 2", "Model 3"]
    
    # Cleanup
    os.remove(file_path)
    
    # Test with output path
    output_path = tempfile.mktemp(suffix=".parquet")
    file_path, cid = ipfs_storage.store_metadata_as_parquet(metadata_list, output_path)
    
    # Check result
    assert file_path == output_path
    assert os.path.exists(output_path)
    
    # Cleanup
    os.remove(output_path)
    
    # Test error handling
    with patch("pandas.DataFrame.to_parquet", side_effect=Exception("Parquet error")):
        file_path, cid = ipfs_storage.store_metadata_as_parquet(metadata_list)
        assert file_path is None
        assert cid is None

def test_store_model_files(ipfs_storage):
    """Test storing model files."""
    # Create a test directory
    temp_dir = tempfile.mkdtemp()
    
    # Create some files in the directory
    for i in range(3):
        with open(os.path.join(temp_dir, f"file{i}.txt"), "w") as f:
            f.write(f"test content {i}")
    
    try:
        # Store model files
        cid = ipfs_storage.store_model_files(temp_dir)
        
        # Check result
        assert cid == "QmDirHash"
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    # Test with non-existent directory
    assert ipfs_storage.store_model_files("/nonexistent") is None
    
    # Test with error
    ipfs_storage.add_directory_to_ipfs = MagicMock(side_effect=Exception("IPFS error"))
    assert ipfs_storage.store_model_files("/tmp") is None

def test_convert_jsonl_to_parquet(ipfs_storage):
    """Test converting JSONL to Parquet."""
    # Create a test JSONL file
    fd, jsonl_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    
    # Write some JSON lines
    with open(jsonl_path, "w") as f:
        for i in range(3):
            f.write(json.dumps({"id": i, "name": f"Model {i}"}) + "\n")
    
    try:
        # Convert to Parquet
        parquet_path = ipfs_storage.convert_jsonl_to_parquet(jsonl_path)
        
        # Check result
        assert parquet_path is not None
        assert parquet_path.endswith(".parquet")
        assert os.path.exists(parquet_path)
        
        # Check content
        df = pd.read_parquet(parquet_path)
        assert len(df) == 3
        assert list(df["id"]) == [0, 1, 2]
        assert list(df["name"]) == ["Model 0", "Model 1", "Model 2"]
        
        # Cleanup
        os.remove(parquet_path)
    finally:
        # Cleanup
        os.remove(jsonl_path)
    
    # Test with output path
    jsonl_path = tempfile.mktemp(suffix=".jsonl")
    with open(jsonl_path, "w") as f:
        f.write(json.dumps({"id": 0, "name": "Model 0"}) + "\n")
    
    parquet_path = tempfile.mktemp(suffix=".parquet")
    
    try:
        result = ipfs_storage.convert_jsonl_to_parquet(jsonl_path, parquet_path)
        
        # Check result
        assert result == parquet_path
        assert os.path.exists(parquet_path)
        
        # Cleanup
        os.remove(parquet_path)
    finally:
        # Cleanup
        os.remove(jsonl_path)
    
    # Test error handling
    with patch("pandas.read_json", side_effect=Exception("JSON error")):
        assert ipfs_storage.convert_jsonl_to_parquet("nonexistent.jsonl") is None

def test_create_car_file(ipfs_storage):
    """Test creating a CAR file."""
    # Create a test file
    fd, file_path = tempfile.mkstemp()
    os.write(fd, b"test content")
    os.close(fd)
    
    try:
        # Create CAR file
        car_path = ipfs_storage.create_car_file(file_path)
        
        # Check result
        assert car_path == file_path + ".car"
        
        # Check that create_car was called
        ipfs_storage.ipfs_dataset.create_car.assert_called_once_with(file_path, file_path + ".car")
    finally:
        # Cleanup
        os.remove(file_path)
    
    # Test with output path
    fd, file_path = tempfile.mkstemp()
    os.close(fd)
    car_path = tempfile.mktemp(suffix=".car")
    
    try:
        result = ipfs_storage.create_car_file(file_path, car_path)
        
        # Check result
        assert result == car_path
        
        # Check that create_car was called
        ipfs_storage.ipfs_dataset.create_car.assert_called_with(file_path, car_path)
    finally:
        # Cleanup
        os.remove(file_path)
    
    # Test error handling
    ipfs_storage.ipfs_dataset.create_car.return_value = False
    assert ipfs_storage.create_car_file(file_path, car_path) is None
    
    # Test with exception
    ipfs_storage.ipfs_dataset.create_car.side_effect = Exception("CAR error")
    assert ipfs_storage.create_car_file(file_path, car_path) is None
    
    # Test with IPFS not available
    ipfs_storage.ipfs_dataset = None
    assert ipfs_storage.create_car_file(file_path, car_path) is None

def test_get_file_from_ipfs(ipfs_storage):
    """Test retrieving a file from IPFS."""
    # Get file
    output_path = tempfile.mktemp()
    result = ipfs_storage.get_file_from_ipfs("QmTestCid", output_path)
    
    # Check result
    assert result is True
    
    # Check that get was called
    ipfs_storage.ipfs_kit.get.assert_called_once_with("QmTestCid", output_path)
    
    # Test with failed retrieval
    ipfs_storage.ipfs_kit.get.return_value = False
    assert ipfs_storage.get_file_from_ipfs("QmTestCid", output_path) is False
    
    # Test with exception
    ipfs_storage.ipfs_kit.get.side_effect = Exception("IPFS error")
    assert ipfs_storage.get_file_from_ipfs("QmTestCid", output_path) is False
    
    # Test with IPFS not available
    ipfs_storage.ipfs_kit = None
    assert ipfs_storage.get_file_from_ipfs("QmTestCid", output_path) is False

def test_calculate_cid(ipfs_storage):
    """Test calculating a CID without adding to IPFS."""
    # Create a test file
    fd, file_path = tempfile.mkstemp()
    os.write(fd, b"test content")
    os.close(fd)
    
    try:
        # Calculate CID
        cid = ipfs_storage.calculate_cid(file_path)
        
        # Check result
        assert cid == "QmCalculatedCid1234"
        
        # Check that calculate_cid was called
        ipfs_storage.ipfs_kit.calculate_cid.assert_called_once_with(file_path)
    finally:
        # Cleanup
        os.remove(file_path)
    
    # Test with exception
    ipfs_storage.ipfs_kit.calculate_cid.side_effect = Exception("IPFS error")
    assert ipfs_storage.calculate_cid("nonexistent.txt") is None
    
    # Test with method not available
    del ipfs_storage.ipfs_kit.calculate_cid
    assert ipfs_storage.calculate_cid(file_path) is None
    
    # Test with IPFS not available
    ipfs_storage.ipfs_kit = None
    assert ipfs_storage.calculate_cid(file_path) is None