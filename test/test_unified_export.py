import os
import sys
import json
import unittest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path to import module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ipfs_huggingface_scraper_py.unified_export import UnifiedExport

class TestUnifiedExport(unittest.TestCase):
    """Test the UnifiedExport class for storing entity data in Parquet format."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Mock config with temporary directory
        self.test_config = {
            "data_dir": self.temp_path,
            "use_ipfs": False  # Disable IPFS for testing
        }
        
        # Initialize the exporter with test config
        self.exporter = UnifiedExport(self.test_config)
        
        # Sample test data
        self.model_data = [
            {"id": "model1", "name": "Test Model 1", "description": "Description 1", "downloads": 100},
            {"id": "model2", "name": "Test Model 2", "description": "Description 2", "downloads": 200}
        ]
        
        self.dataset_data = [
            {"id": "dataset1", "name": "Test Dataset 1", "description": "Description 1", "downloads": 300},
            {"id": "dataset2", "name": "Test Dataset 2", "description": "Description 2", "downloads": 400}
        ]
        
        self.space_data = [
            {"id": "space1", "name": "Test Space 1", "description": "Description 1", "visits": 500},
            {"id": "space2", "name": "Test Space 2", "description": "Description 2", "visits": 600}
        ]
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_normalize_schema(self):
        """Test normalizing schema for different entity types."""
        # Test model normalization
        normalized_models = self.exporter.normalize_schema(self.model_data, 'model')
        self.assertEqual(len(normalized_models), 2)
        self.assertEqual(normalized_models[0]['entity_type'], 'model')
        self.assertEqual(normalized_models[0]['id'], 'model1')
        self.assertIn('processed_at', normalized_models[0])
        
        # Test dataset normalization
        normalized_datasets = self.exporter.normalize_schema(self.dataset_data, 'dataset')
        self.assertEqual(len(normalized_datasets), 2)
        self.assertEqual(normalized_datasets[0]['entity_type'], 'dataset')
        
        # Test space normalization
        normalized_spaces = self.exporter.normalize_schema(self.space_data, 'space')
        self.assertEqual(len(normalized_spaces), 2)
        self.assertEqual(normalized_spaces[0]['entity_type'], 'space')
    
    def test_save_dataframe_to_parquet_safely(self):
        """Test saving DataFrame to Parquet with safe handling."""
        # Create test DataFrame
        df = pd.DataFrame(self.model_data)
        
        # Test successful save
        output_path = self.temp_path / "test_output.parquet"
        result = self.exporter.save_dataframe_to_parquet_safely(df, output_path)
        
        self.assertTrue(result)
        self.assertTrue(output_path.exists())
        
        # Verify we can read it back
        df_read = pd.read_parquet(output_path)
        self.assertEqual(len(df_read), 2)
    
    def test_store_entity_data(self):
        """Test storing entity data for a specific entity type."""
        # Store model data
        file_path, cid = self.exporter.store_entity_data(
            self.model_data, 'model'
        )
        
        self.assertIsNotNone(file_path)
        self.assertTrue(os.path.exists(file_path))
        self.assertIsNone(cid)  # No IPFS in test config
        
        # Read and verify the stored data
        df = pd.read_parquet(file_path)
        self.assertEqual(len(df), 2)
        self.assertEqual(df['entity_type'].unique()[0], 'model')
    
    def test_store_unified_data(self):
        """Test storing unified data with multiple entity types."""
        # Store all entity types
        file_path, cid = self.exporter.store_unified_data(
            models_list=self.model_data,
            datasets_list=self.dataset_data,
            spaces_list=self.space_data
        )
        
        self.assertIsNotNone(file_path)
        self.assertTrue(os.path.exists(file_path))
        self.assertIsNone(cid)  # No IPFS in test config
        
        # Read and verify the stored data
        df = pd.read_parquet(file_path)
        self.assertEqual(len(df), 6)  # 2 models + 2 datasets + 2 spaces
        
        # Check entity type counts
        entity_counts = df['entity_type'].value_counts().to_dict()
        self.assertEqual(entity_counts['model'], 2)
        self.assertEqual(entity_counts['dataset'], 2)
        self.assertEqual(entity_counts['space'], 2)
    
    def test_merge_with_existing_data(self):
        """Test merging new data with existing data."""
        # First store some data
        file_path, _ = self.exporter.store_unified_data(
            models_list=self.model_data[:1]  # Just the first model
        )
        
        # Now store more data and merge
        file_path, _ = self.exporter.store_unified_data(
            models_list=self.model_data[1:],  # Second model
            datasets_list=self.dataset_data   # Both datasets
        )
        
        # Read and verify the merged data
        df = pd.read_parquet(file_path)
        self.assertEqual(len(df), 3)  # 1 model + 1 model + 2 datasets
        
        # Check entity type counts
        entity_counts = df['entity_type'].value_counts().to_dict()
        self.assertEqual(entity_counts['model'], 2)
        self.assertEqual(entity_counts['dataset'], 2)
    
    @patch('ipfs_huggingface_scraper_py.unified_export.logging')
    def test_handle_duplicate_ids(self, mock_logging):
        """Test handling of duplicate IDs when merging data."""
        # First store original data
        file_path, _ = self.exporter.store_unified_data(
            models_list=self.model_data
        )
        
        # Create updated version of first model
        updated_model = self.model_data[0].copy()
        updated_model['downloads'] = 999  # Changed value
        
        # Store updated model
        file_path, _ = self.exporter.store_unified_data(
            models_list=[updated_model]
        )
        
        # Read and verify the data - should have same number of records
        df = pd.read_parquet(file_path)
        self.assertEqual(len(df), 2)  # Still just 2 models
        
        # Verify the download count was updated for model1
        model1_data = df[df['id'] == 'model1']
        self.assertEqual(model1_data['downloads'].values[0], 999)
    
    def test_get_entity_statistics(self):
        """Test getting statistics about stored entities."""
        # Store unified data
        file_path, _ = self.exporter.store_unified_data(
            models_list=self.model_data,
            datasets_list=self.dataset_data,
            spaces_list=self.space_data
        )
        
        # Get statistics
        stats = self.exporter.get_entity_statistics(file_path)
        
        self.assertTrue(stats['file_exists'])
        self.assertEqual(stats['total_entities'], 6)
        self.assertEqual(stats['entity_types']['model'], 2)
        self.assertEqual(stats['entity_types']['dataset'], 2)
        self.assertEqual(stats['entity_types']['space'], 2)

if __name__ == '__main__':
    unittest.main()