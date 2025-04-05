"""Tests for the HuggingFace Datasets Scraper."""

import os
import json
import unittest
import tempfile
from unittest.mock import patch, MagicMock

from ipfs_huggingface_scraper_py.datasets.datasets_scraper import DatasetsScraper

class TestDatasetsScraper(unittest.TestCase):
    """Test the DatasetsScraper class."""
    
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
max_datasets = 5
entity_types = ["datasets"]
save_metadata = true
dataset_preview_max_rows = 10
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
    
    @patch("ipfs_huggingface_scraper_py.datasets.datasets_scraper.list_datasets")
    def test_initialization(self, mock_list_datasets):
        """Test scraper initialization."""
        scraper = DatasetsScraper(self.config_path)
        
        # Check that scraper was initialized correctly
        self.assertEqual(scraper.config.config["scraper"]["output_dir"], self.output_dir)
        self.assertEqual(scraper.config.config["scraper"]["max_datasets"], 5)
        self.assertEqual(scraper.config.config["scraper"]["entity_types"], ["datasets"])
        self.assertEqual(scraper.config.config["scraper"]["dataset_preview_max_rows"], 10)
    
    @patch("ipfs_huggingface_scraper_py.datasets.datasets_scraper.list_datasets")
    def test_discover_datasets(self, mock_list_datasets):
        """Test discovering datasets."""
        # Set up mock list_datasets
        mock_dataset1 = MagicMock()
        mock_dataset1.id = "user1/dataset1"
        mock_dataset2 = MagicMock()
        mock_dataset2.id = "user2/dataset2"
        mock_list_datasets.return_value = [mock_dataset1, mock_dataset2]
        
        # Create scraper and discover datasets
        scraper = DatasetsScraper(self.config_path)
        dataset_ids = scraper._discover_datasets(max_datasets=2)
        
        # Check that datasets were discovered correctly
        self.assertEqual(len(dataset_ids), 2)
        self.assertEqual(dataset_ids[0], "user1/dataset1")
        self.assertEqual(dataset_ids[1], "user2/dataset2")
        mock_list_datasets.assert_called_once()
    
    @patch("ipfs_huggingface_scraper_py.datasets.datasets_scraper.list_datasets")
    @patch("ipfs_huggingface_scraper_py.datasets.datasets_scraper.hf_hub_download")
    @patch("ipfs_huggingface_scraper_py.datasets.datasets_scraper.HfApi")
    def test_process_dataset(self, mock_hf_api, mock_hub_download, mock_list_datasets):
        """Test processing a single dataset."""
        # Create scraper
        scraper = DatasetsScraper(self.config_path)
        
        # Patch the _save_dataset_metadata method to avoid needing HfApi
        with patch.object(scraper, '_save_dataset_metadata', return_value=True):
            # Process dataset
            result = scraper._process_dataset("user/dataset")
            
            # Check that dataset was processed correctly
            self.assertTrue(result)
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, "datasets", "user__dataset")))
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, "datasets", "user__dataset", "preview_info.json")))
            
            # Check preview content
            with open(os.path.join(self.output_dir, "datasets", "user__dataset", "preview_info.json"), "r") as f:
                preview_info = json.load(f)
                self.assertEqual(preview_info["dataset_id"], "user/dataset")
    
    @patch("ipfs_huggingface_scraper_py.datasets.datasets_scraper.list_datasets")
    def test_scrape_datasets(self, mock_list_datasets):
        """Test the full scrape_datasets method."""
        # Set up mock list_datasets
        mock_dataset1 = MagicMock()
        mock_dataset1.id = "user1/dataset1"
        mock_dataset2 = MagicMock()
        mock_dataset2.id = "user2/dataset2"
        mock_list_datasets.return_value = [mock_dataset1, mock_dataset2]
        
        # Patch _process_dataset to return True
        with patch.object(DatasetsScraper, "_process_dataset", return_value=True) as mock_process:
            # Create scraper and run scrape_datasets
            scraper = DatasetsScraper(self.config_path)
            scraper.scrape_datasets(max_datasets=2)
            
            # Check that process_dataset was called for each dataset
            self.assertEqual(mock_process.call_count, 2)
            mock_process.assert_any_call("user1/dataset1")
            mock_process.assert_any_call("user2/dataset2")

if __name__ == "__main__":
    unittest.main()
