"""Simple tests for the IPFS HuggingFace Scraper components."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from ipfs_huggingface_scraper_py.state_manager import StateManager
from ipfs_huggingface_scraper_py.rate_limiter import RateLimiter
from ipfs_huggingface_scraper_py.config import Config
from ipfs_huggingface_scraper_py.ipfs_integration import IpfsStorage

class TestStateManager(unittest.TestCase):
    """Test the state manager basics."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for state files
        self.temp_dir = tempfile.mkdtemp()
        self.state_manager = StateManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up the test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_basic_functionality(self):
        """Test basic state manager functionality."""
        # Mark a model as processed
        self.state_manager.mark_model_processed("test/model1")
        
        # Check that model is marked as processed
        self.assertTrue(self.state_manager.is_model_processed("test/model1"))
        
        # Mark a model as completed
        self.state_manager.mark_model_completed("test/model2")
        
        # Check that model is marked as processed and completed
        self.assertTrue(self.state_manager.is_model_processed("test/model2"))
        self.assertTrue(self.state_manager.is_model_completed("test/model2"))
        
        # Test pausing and resuming
        self.assertFalse(self.state_manager.is_paused())
        self.state_manager.pause()
        self.assertTrue(self.state_manager.is_paused())
        self.state_manager.resume()
        self.assertFalse(self.state_manager.is_paused())

class TestRateLimiter(unittest.TestCase):
    """Test the rate limiter basics."""
    
    def test_basic_functionality(self):
        """Test basic rate limiter functionality."""
        # Create rate limiter
        limiter = RateLimiter(default_rate=5.0, daily_quota=1000, max_retries=2)
        
        # Test initialization
        self.assertEqual(limiter.default_rate, 5.0)
        self.assertEqual(limiter.daily_quota, 1000)
        self.assertEqual(limiter.max_retries, 2)
        self.assertEqual(limiter.quota_used, 0)
        self.assertFalse(limiter.is_authenticated)
        
        # Test setting rate limits
        limiter.set_rate_limit("test_endpoint", 10.0)
        self.assertEqual(limiter.get_rate_limit("test_endpoint"), 10.0)
        self.assertEqual(limiter.get_rate_limit("unknown_endpoint"), 5.0)
        
        # Test setting authenticated
        limiter.set_authenticated(True)
        self.assertTrue(limiter.is_authenticated)

class TestConfig(unittest.TestCase):
    """Test the configuration system basics."""
    
    def test_basic_functionality(self):
        """Test basic configuration functionality."""
        # Create config
        config = Config()
        
        # Test default values
        self.assertEqual(config.config["scraper"]["output_dir"], "hf_model_data")
        self.assertIsNone(config.config["scraper"]["max_models"])
        self.assertTrue(config.config["scraper"]["save_metadata"])
        
        # Test getting and setting values
        self.assertEqual(config.get("scraper", "output_dir"), "hf_model_data")
        config.set("scraper", "max_models", 100)
        self.assertEqual(config.get("scraper", "max_models"), 100)

class TestIpfsStorage(unittest.TestCase):
    """Test the IPFS storage integration."""
    
    def test_initialization(self):
        """Test initialization."""
        # Test with IPFS disabled
        storage = IpfsStorage({"use_ipfs": False})
        self.assertIsNone(storage.ipfs_kit)
        self.assertIsNone(storage.ipfs_dataset)
        
        # Test is_ipfs_available
        self.assertFalse(storage.is_ipfs_available())
    
    def test_storage_with_mocks(self):
        """Test storage functions with mocked objects."""
        # Create a storage object with mocked IPFS
        storage = IpfsStorage({"use_ipfs": True})
        
        # Manually set mocked objects
        storage.ipfs_kit = MagicMock()
        storage.ipfs_dataset = MagicMock()
        
        # Configure mocks
        storage.ipfs_kit.add.return_value = {"Hash": "QmTestHash1234"}
        storage.ipfs_kit.pin = MagicMock()
        storage.ipfs_kit.pin.add.return_value = {"Pins": ["QmTestHash1234"]}
        
        # Test is_ipfs_available
        self.assertTrue(storage.is_ipfs_available())
        
        # Test add_file_to_ipfs
        fd, file_path = tempfile.mkstemp()
        os.write(fd, b"test content")
        os.close(fd)
        
        try:
            # Add file to IPFS
            cid = storage.add_file_to_ipfs(file_path)
            
            # Check result
            self.assertEqual(cid, "QmTestHash1234")
            
            # Check that add was called
            storage.ipfs_kit.add.assert_called_once()
        finally:
            # Cleanup
            os.remove(file_path)
        
        # Reset mock for pin test
        storage.ipfs_kit.pin.add.return_value = {"Pins": ["QmTestCid"]}
        
        # Test pin_cid
        result = storage.pin_cid("QmTestCid")
        self.assertTrue(result)
        storage.ipfs_kit.pin.add.assert_called_once_with("QmTestCid")

class TestCLI(unittest.TestCase):
    """Test the command-line interface."""
    
    def test_cli_import(self):
        """Test importing the CLI module."""
        try:
            # Test importing CLI module
            from ipfs_huggingface_scraper_py import cli
            self.assertTrue(hasattr(cli, 'main'))
            self.assertTrue(hasattr(cli, 'parse_args'))
            self.assertTrue(hasattr(cli, 'setup_logging'))
            self.assertTrue(hasattr(cli, 'show_status'))
            self.assertTrue(hasattr(cli, 'init_config'))
        except ImportError as e:
            self.fail(f"Failed to import CLI module: {e}")

def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(TestStateManager),
        loader.loadTestsFromTestCase(TestRateLimiter),
        loader.loadTestsFromTestCase(TestConfig),
        loader.loadTestsFromTestCase(TestIpfsStorage),
        loader.loadTestsFromTestCase(TestCLI)
    ])
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run_tests()