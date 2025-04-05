"""Tests for the HuggingFace Spaces Scraper."""

import os
import json
import unittest
import tempfile
from unittest.mock import patch, MagicMock

from ipfs_huggingface_scraper_py.spaces.spaces_scraper import SpacesScraper

class TestSpacesScraper(unittest.TestCase):
    """Test the SpacesScraper class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create temporary config file
        self.config_path = os.path.join(self.temp_dir, "config.toml")
        with open(self.config_path, "w") as f:
            f.write(f"""
[scraper]
output_dir = "{self.output_dir}"
max_spaces = 5
entity_types = ["spaces"]
save_metadata = true
batch_size = 2
skip_existing = true

[api]
base_url = "https://huggingface.co"
api_token = ""
authenticated = false
anonymous_rate_limit = 5.0
authenticated_rate_limit = 10.0
daily_anonymous_quota = 1000
daily_authenticated_quota = 2000
max_retries = 2
timeout = 10

[storage]
use_ipfs = false

[state]
state_dir = "{self.temp_dir}"
checkpoint_interval = 2
auto_resume = true
""")
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch("ipfs_huggingface_scraper_py.spaces.spaces_scraper.list_spaces")
    def test_initialization(self, mock_list_spaces):
        """Test scraper initialization."""
        scraper = SpacesScraper(self.config_path)
        
        # Check that scraper was initialized correctly
        self.assertEqual(scraper.config.config["scraper"]["output_dir"], self.output_dir)
        self.assertEqual(scraper.config.config["scraper"]["max_spaces"], 5)
        self.assertEqual(scraper.config.config["scraper"]["entity_types"], ["spaces"])
    
    @patch("ipfs_huggingface_scraper_py.spaces.spaces_scraper.list_spaces")
    def test_discover_spaces(self, mock_list_spaces):
        """Test discovering spaces."""
        # Set up mock list_spaces
        mock_space1 = MagicMock()
        mock_space1.id = "user1/space1"
        mock_space2 = MagicMock()
        mock_space2.id = "user2/space2"
        mock_list_spaces.return_value = [mock_space1, mock_space2]
        
        # Create scraper and discover spaces
        scraper = SpacesScraper(self.config_path)
        space_ids = scraper._discover_spaces(max_spaces=2)
        
        # Check that spaces were discovered correctly
        self.assertEqual(len(space_ids), 2)
        self.assertEqual(space_ids[0], "user1/space1")
        self.assertEqual(space_ids[1], "user2/space2")
        mock_list_spaces.assert_called_once()
    
    @patch("ipfs_huggingface_scraper_py.spaces.spaces_scraper.list_spaces")
    @patch("ipfs_huggingface_scraper_py.spaces.spaces_scraper.HfApi")
    def test_process_space(self, mock_hf_api, mock_list_spaces):
        """Test processing a single space."""
        # Create scraper
        scraper = SpacesScraper(self.config_path)
        
        # Patch the _save_space_metadata method to avoid needing HfApi
        with patch.object(scraper, '_save_space_metadata', return_value=True):
            # Process space
            result = scraper._process_space("user/space")
            
            # Check that space was processed correctly
            self.assertTrue(result)
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, "spaces", "user__space")))
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, "spaces", "user__space", "thumbnail_info.json")))
            
            # Check thumbnail info content
            with open(os.path.join(self.output_dir, "spaces", "user__space", "thumbnail_info.json"), "r") as f:
                thumbnail_info = json.load(f)
                self.assertEqual(thumbnail_info["space_id"], "user/space")
    
    @patch("ipfs_huggingface_scraper_py.spaces.spaces_scraper.list_spaces")
    def test_scrape_spaces(self, mock_list_spaces):
        """Test the full scrape_spaces method."""
        # Set up mock list_spaces
        mock_space1 = MagicMock()
        mock_space1.id = "user1/space1"
        mock_space2 = MagicMock()
        mock_space2.id = "user2/space2"
        mock_list_spaces.return_value = [mock_space1, mock_space2]
        
        # Patch _process_space to return True
        with patch.object(SpacesScraper, "_process_space", return_value=True) as mock_process:
            # Create scraper and run scrape_spaces
            scraper = SpacesScraper(self.config_path)
            scraper.scrape_spaces(max_spaces=2)
            
            # Check that process_space was called for each space
            self.assertEqual(mock_process.call_count, 2)
            mock_process.assert_any_call("user1/space1")
            mock_process.assert_any_call("user2/space2")

if __name__ == "__main__":
    unittest.main()
