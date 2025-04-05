"""Tests for the ProvenanceTracker component."""

import os
import json
import unittest
import tempfile
from unittest.mock import patch, MagicMock

from ipfs_huggingface_scraper_py.provenance import ProvenanceTracker

class TestProvenanceTracker(unittest.TestCase):
    """Test the ProvenanceTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ProvenanceTracker(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test tracker initialization."""
        # Check that storage directory was created
        self.assertTrue(os.path.exists(self.temp_dir))
        
        # Check initial structure after adding a relationship
        # (this initializes the relationships file)
        self.tracker.add_model_base_relationship("test-model", "base-model")
        
        # Now check for relationships file
        relationships_file = os.path.join(self.temp_dir, "entity_relationships.json")
        self.assertTrue(os.path.exists(relationships_file))
        
        # Check structure
        with open(relationships_file, "r") as f:
            data = json.load(f)
            self.assertIn("models", data)
            self.assertIn("datasets", data)
            self.assertIn("spaces", data)
            self.assertIn("last_updated", data)
    
    def test_model_base_relationships(self):
        """Test tracking model base relationships."""
        # Add a model base relationship
        self.tracker.add_model_base_relationship("derived-model", "base-model")
        
        # Check that the relationship was added
        base_models = self.tracker.get_model_base_models("derived-model")
        self.assertEqual(len(base_models), 1)
        self.assertEqual(base_models[0], "base-model")
        
        # Check inverse relationship
        derived_models = self.tracker.get_model_derived_models("base-model")
        self.assertEqual(len(derived_models), 1)
        self.assertEqual(derived_models[0], "derived-model")
        
        # Check relationships file content
        relationships_file = os.path.join(self.temp_dir, "entity_relationships.json")
        with open(relationships_file, "r") as f:
            data = json.load(f)
            self.assertIn("derived-model", data["models"])
            self.assertIn("base-model", data["models"])
            self.assertIn("base-model", data["models"]["derived-model"]["base_models"])
            self.assertIn("derived-model", data["models"]["base-model"]["derived_models"])
    
    def test_model_dataset_relationships(self):
        """Test tracking model-dataset relationships."""
        # Add a model-dataset relationship
        self.tracker.add_model_dataset_relationship("model-id", "dataset-id", "trained_on")
        
        # Check that the relationship was added
        datasets = self.tracker.get_model_datasets("model-id", "trained_on")
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0], "dataset-id")
        
        # Check without relationship type filter
        all_datasets = self.tracker.get_model_datasets("model-id")
        self.assertEqual(len(all_datasets), 1)
        self.assertEqual(all_datasets[0], "dataset-id")
        
        # Check inverse relationship
        from_dataset = self.tracker.get_dataset_models("dataset-id", "used_by")
        self.assertEqual(len(from_dataset), 1)
        self.assertEqual(from_dataset[0], "model-id")
    
    def test_space_entity_relationships(self):
        """Test tracking space-entity relationships."""
        # Add space-model relationship
        self.tracker.add_space_entity_relationship("space-id", "model-id", "model", "uses")
        
        # Check the relationship
        models = self.tracker.get_space_entities("space-id", "model")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0], "model-id")
        
        # Check inverse relationship
        spaces = self.tracker.get_entity_spaces("model-id", "model")
        self.assertEqual(len(spaces), 1)
        self.assertEqual(spaces[0], "space-id")
        
        # Add space-dataset relationship
        self.tracker.add_space_entity_relationship("space-id", "dataset-id", "dataset", "uses")
        
        # Check that both relationships exist
        models = self.tracker.get_space_entities("space-id", "model")
        datasets = self.tracker.get_space_entities("space-id", "dataset")
        self.assertEqual(len(models), 1)
        self.assertEqual(len(datasets), 1)
    
    def test_extract_relationships_from_metadata(self):
        """Test extracting relationships from metadata."""
        # Create test metadata for a model
        model_metadata = {
            "modelId": "base-model-id",
            "tags": ["dataset:test-dataset"],
            "config": {
                "dataset": "training-dataset"
            }
        }
        
        # Extract relationships
        self.tracker.extract_relationships_from_metadata(model_metadata, "fine-tuned-model", "model")
        
        # Check that relationships were extracted
        base_models = self.tracker.get_model_base_models("fine-tuned-model")
        self.assertEqual(len(base_models), 1)
        self.assertEqual(base_models[0], "base-model-id")
        
        # Check for dataset relationship from tags
        datasets = self.tracker.get_model_datasets("fine-tuned-model")
        self.assertIn("test-dataset", datasets)
        
        # Check for dataset relationship from config
        self.assertIn("training-dataset", datasets)
    
    def test_generate_provenance_graph(self):
        """Test generating provenance graph."""
        # Add some relationships
        self.tracker.add_model_base_relationship("model1", "base-model")
        self.tracker.add_model_dataset_relationship("model1", "dataset1", "trained_on")
        self.tracker.add_space_entity_relationship("space1", "model1", "model", "uses")
        
        # Generate graph
        graph_path = os.path.join(self.temp_dir, "graph.json")
        result = self.tracker.generate_provenance_graph(graph_path)
        
        # Check result
        self.assertTrue(result)
        self.assertTrue(os.path.exists(graph_path))
        
        # Check graph structure
        with open(graph_path, "r") as f:
            graph = json.load(f)
            self.assertIn("nodes", graph)
            self.assertIn("edges", graph)
            
            # Check that all entities are in nodes
            node_ids = [node["id"] for node in graph["nodes"]]
            self.assertIn("model1", node_ids)
            self.assertIn("base-model", node_ids)
            self.assertIn("dataset1", node_ids)
            self.assertIn("space1", node_ids)
            
            # Check edges
            self.assertEqual(len(graph["edges"]), 3)  # Three relationships
            
            # Check edge types
            edge_types = [edge["type"] for edge in graph["edges"]]
            self.assertIn("derived_from", edge_types)
            self.assertIn("trained_on", edge_types)
            self.assertIn("uses", edge_types)

if __name__ == "__main__":
    unittest.main()
