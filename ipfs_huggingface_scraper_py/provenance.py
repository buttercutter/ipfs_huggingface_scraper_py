import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union

class ProvenanceTracker:
    """
    Tracks and manages provenance relationships between HuggingFace entities 
    (models, datasets, spaces).
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize the provenance tracker.
        
        Args:
            storage_dir: Directory for storing provenance information
        """
        self.storage_dir = storage_dir
        self.relationships_file = os.path.join(storage_dir, "entity_relationships.json")
        self.relationships = self._load_relationships()
        
        # Create directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
    
    def _load_relationships(self) -> Dict[str, Any]:
        """
        Load existing relationships from storage.
        
        Returns:
            Dictionary of entity relationships
        """
        if os.path.exists(self.relationships_file):
            try:
                with open(self.relationships_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading relationships: {e}", exc_info=True)
                return self._create_empty_relationships()
        else:
            return self._create_empty_relationships()
    
    def _create_empty_relationships(self) -> Dict[str, Any]:
        """
        Create an empty relationships structure.
        
        Returns:
            Empty relationships dictionary
        """
        return {
            "models": {},
            "datasets": {},
            "spaces": {},
            "last_updated": time.time()
        }
    
    def _save_relationships(self) -> None:
        """Save relationships to storage."""
        try:
            self.relationships["last_updated"] = time.time()
            with open(self.relationships_file, 'w', encoding='utf-8') as f:
                json.dump(self.relationships, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving relationships: {e}", exc_info=True)
    
    def add_model_base_relationship(self, model_id: str, base_model_id: str) -> None:
        """
        Add a relationship indicating a model is derived from a base model.
        
        Args:
            model_id: ID of the derived model
            base_model_id: ID of the base model
        """
        # Ensure models dictionary exists
        if "models" not in self.relationships:
            self.relationships["models"] = {}
        
        # Add/update model entry
        if model_id not in self.relationships["models"]:
            self.relationships["models"][model_id] = {"base_models": [], "derived_models": []}
        
        if base_model_id not in self.relationships["models"]:
            self.relationships["models"][base_model_id] = {"base_models": [], "derived_models": []}
        
        # Add relationship if it doesn't exist
        if base_model_id not in self.relationships["models"][model_id]["base_models"]:
            self.relationships["models"][model_id]["base_models"].append(base_model_id)
        
        # Add inverse relationship
        if model_id not in self.relationships["models"][base_model_id]["derived_models"]:
            self.relationships["models"][base_model_id]["derived_models"].append(model_id)
        
        # Save changes
        self._save_relationships()
    
    def add_model_dataset_relationship(self, model_id: str, dataset_id: str, 
                                       relationship_type: str = "trained_on") -> None:
        """
        Add a relationship between a model and a dataset.
        
        Args:
            model_id: ID of the model
            dataset_id: ID of the dataset
            relationship_type: Type of relationship (e.g., "trained_on", "evaluated_on")
        """
        # Ensure dictionaries exist
        if "models" not in self.relationships:
            self.relationships["models"] = {}
        if "datasets" not in self.relationships:
            self.relationships["datasets"] = {}
        
        # Add/update model entry
        if model_id not in self.relationships["models"]:
            self.relationships["models"][model_id] = {"datasets": {}}
        if "datasets" not in self.relationships["models"][model_id]:
            self.relationships["models"][model_id]["datasets"] = {}
        
        # Add/update dataset entry
        if dataset_id not in self.relationships["datasets"]:
            self.relationships["datasets"][dataset_id] = {"models": {}}
        if "models" not in self.relationships["datasets"][dataset_id]:
            self.relationships["datasets"][dataset_id]["models"] = {}
        
        # Add relationship
        if relationship_type not in self.relationships["models"][model_id]["datasets"]:
            self.relationships["models"][model_id]["datasets"][relationship_type] = []
        
        if dataset_id not in self.relationships["models"][model_id]["datasets"][relationship_type]:
            self.relationships["models"][model_id]["datasets"][relationship_type].append(dataset_id)
        
        # Add inverse relationship
        inverse_type = f"used_by" if relationship_type == "trained_on" else f"evaluated_by"
        if inverse_type not in self.relationships["datasets"][dataset_id]["models"]:
            self.relationships["datasets"][dataset_id]["models"][inverse_type] = []
        
        if model_id not in self.relationships["datasets"][dataset_id]["models"][inverse_type]:
            self.relationships["datasets"][dataset_id]["models"][inverse_type].append(model_id)
        
        # Save changes
        self._save_relationships()
    
    def add_space_entity_relationship(self, space_id: str, entity_id: str, 
                                     entity_type: str, relationship_type: str = "uses") -> None:
        """
        Add a relationship between a space and another entity.
        
        Args:
            space_id: ID of the space
            entity_id: ID of the related entity
            entity_type: Type of entity ("model" or "dataset")
            relationship_type: Type of relationship
        """
        # Ensure dictionaries exist
        if "spaces" not in self.relationships:
            self.relationships["spaces"] = {}
        
        entity_type_plural = f"{entity_type}s"  # Convert to plural (model -> models)
        if entity_type_plural not in self.relationships:
            self.relationships[entity_type_plural] = {}
        
        # Add/update space entry
        if space_id not in self.relationships["spaces"]:
            self.relationships["spaces"][space_id] = {entity_type_plural: {}}
        if entity_type_plural not in self.relationships["spaces"][space_id]:
            self.relationships["spaces"][space_id][entity_type_plural] = {}
        
        # Add/update entity entry
        if entity_id not in self.relationships[entity_type_plural]:
            self.relationships[entity_type_plural][entity_id] = {"spaces": {}}
        if "spaces" not in self.relationships[entity_type_plural][entity_id]:
            self.relationships[entity_type_plural][entity_id]["spaces"] = {}
        
        # Add relationship
        if relationship_type not in self.relationships["spaces"][space_id][entity_type_plural]:
            self.relationships["spaces"][space_id][entity_type_plural][relationship_type] = []
        
        if entity_id not in self.relationships["spaces"][space_id][entity_type_plural][relationship_type]:
            self.relationships["spaces"][space_id][entity_type_plural][relationship_type].append(entity_id)
        
        # Add inverse relationship
        inverse_type = "used_by"
        if inverse_type not in self.relationships[entity_type_plural][entity_id]["spaces"]:
            self.relationships[entity_type_plural][entity_id]["spaces"][inverse_type] = []
        
        if space_id not in self.relationships[entity_type_plural][entity_id]["spaces"][inverse_type]:
            self.relationships[entity_type_plural][entity_id]["spaces"][inverse_type].append(space_id)
        
        # Save changes
        self._save_relationships()
    
    def get_model_base_models(self, model_id: str) -> List[str]:
        """
        Get base models for a given model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            List of base model IDs
        """
        try:
            if (model_id in self.relationships.get("models", {}) and 
                "base_models" in self.relationships["models"][model_id]):
                return self.relationships["models"][model_id]["base_models"]
            return []
        except Exception as e:
            logging.error(f"Error getting base models for {model_id}: {e}")
            return []
    
    def get_model_derived_models(self, model_id: str) -> List[str]:
        """
        Get models derived from a given model.
        
        Args:
            model_id: ID of the base model
            
        Returns:
            List of derived model IDs
        """
        try:
            if (model_id in self.relationships.get("models", {}) and 
                "derived_models" in self.relationships["models"][model_id]):
                return self.relationships["models"][model_id]["derived_models"]
            return []
        except Exception as e:
            logging.error(f"Error getting derived models for {model_id}: {e}")
            return []
    
    def get_model_datasets(self, model_id: str, relationship_type: Optional[str] = None) -> List[str]:
        """
        Get datasets related to a model.
        
        Args:
            model_id: ID of the model
            relationship_type: Type of relationship to filter by (optional)
            
        Returns:
            List of dataset IDs or dict of datasets by relationship type
        """
        try:
            if model_id not in self.relationships.get("models", {}):
                return []
            
            if "datasets" not in self.relationships["models"][model_id]:
                return []
            
            if relationship_type:
                return self.relationships["models"][model_id]["datasets"].get(relationship_type, [])
            else:
                # Return all datasets
                all_datasets = []
                for datasets_list in self.relationships["models"][model_id]["datasets"].values():
                    all_datasets.extend(datasets_list)
                return list(set(all_datasets))  # Remove duplicates
        except Exception as e:
            logging.error(f"Error getting datasets for {model_id}: {e}")
            return []
    
    def get_dataset_models(self, dataset_id: str, relationship_type: Optional[str] = None) -> List[str]:
        """
        Get models related to a dataset.
        
        Args:
            dataset_id: ID of the dataset
            relationship_type: Type of relationship to filter by (optional)
            
        Returns:
            List of model IDs
        """
        try:
            if dataset_id not in self.relationships.get("datasets", {}):
                return []
            
            if "models" not in self.relationships["datasets"][dataset_id]:
                return []
            
            if relationship_type:
                return self.relationships["datasets"][dataset_id]["models"].get(relationship_type, [])
            else:
                # Return all models
                all_models = []
                for models_list in self.relationships["datasets"][dataset_id]["models"].values():
                    all_models.extend(models_list)
                return list(set(all_models))  # Remove duplicates
        except Exception as e:
            logging.error(f"Error getting models for {dataset_id}: {e}")
            return []
    
    def get_space_entities(self, space_id: str, entity_type: str,
                          relationship_type: Optional[str] = None) -> List[str]:
        """
        Get entities related to a space.
        
        Args:
            space_id: ID of the space
            entity_type: Type of entity ("model" or "dataset")
            relationship_type: Type of relationship to filter by (optional)
            
        Returns:
            List of entity IDs
        """
        try:
            entity_type_plural = f"{entity_type}s"  # Convert to plural
            
            if space_id not in self.relationships.get("spaces", {}):
                return []
            
            if entity_type_plural not in self.relationships["spaces"][space_id]:
                return []
            
            if relationship_type:
                return self.relationships["spaces"][space_id][entity_type_plural].get(relationship_type, [])
            else:
                # Return all entities
                all_entities = []
                for entities_list in self.relationships["spaces"][space_id][entity_type_plural].values():
                    all_entities.extend(entities_list)
                return list(set(all_entities))  # Remove duplicates
        except Exception as e:
            logging.error(f"Error getting {entity_type}s for space {space_id}: {e}")
            return []
    
    def get_entity_spaces(self, entity_id: str, entity_type: str,
                         relationship_type: Optional[str] = None) -> List[str]:
        """
        Get spaces related to an entity.
        
        Args:
            entity_id: ID of the entity
            entity_type: Type of entity ("model" or "dataset")
            relationship_type: Type of relationship to filter by (optional)
            
        Returns:
            List of space IDs
        """
        try:
            entity_type_plural = f"{entity_type}s"  # Convert to plural
            
            if entity_id not in self.relationships.get(entity_type_plural, {}):
                return []
            
            if "spaces" not in self.relationships[entity_type_plural][entity_id]:
                return []
            
            if relationship_type:
                return self.relationships[entity_type_plural][entity_id]["spaces"].get(relationship_type, [])
            else:
                # Return all spaces
                all_spaces = []
                for spaces_list in self.relationships[entity_type_plural][entity_id]["spaces"].values():
                    all_spaces.extend(spaces_list)
                return list(set(all_spaces))  # Remove duplicates
        except Exception as e:
            logging.error(f"Error getting spaces for {entity_type} {entity_id}: {e}")
            return []
    
    def extract_relationships_from_metadata(self, metadata: Dict[str, Any], 
                                           entity_id: str, entity_type: str) -> None:
        """
        Extract and store relationships from entity metadata.
        
        Args:
            metadata: Entity metadata dictionary
            entity_id: ID of the entity
            entity_type: Type of entity ("model", "dataset", or "space")
        """
        try:
            if entity_type == "model":
                # Extract base model information
                if "modelId" in metadata and metadata["modelId"] != entity_id:
                    self.add_model_base_relationship(entity_id, metadata["modelId"])
                
                # Look for dataset mentions in tags or description
                if "tags" in metadata and isinstance(metadata["tags"], list):
                    for tag in metadata["tags"]:
                        if tag.startswith("dataset:"):
                            dataset_id = tag.replace("dataset:", "")
                            self.add_model_dataset_relationship(entity_id, dataset_id, "trained_on")
                
                # Look for config information about training dataset
                if "config" in metadata and isinstance(metadata["config"], dict):
                    if "dataset" in metadata["config"]:
                        dataset_info = metadata["config"]["dataset"]
                        if isinstance(dataset_info, str):
                            self.add_model_dataset_relationship(entity_id, dataset_info, "trained_on")
                        elif isinstance(dataset_info, dict) and "name" in dataset_info:
                            self.add_model_dataset_relationship(entity_id, dataset_info["name"], "trained_on")
            
            elif entity_type == "dataset":
                # Nothing specific to extract for datasets at this time
                pass
            
            elif entity_type == "space":
                # Look for model references
                if "sdk" in metadata and metadata["sdk"] == "gradio":
                    # Gradio spaces often use models
                    if "metadata" in metadata and isinstance(metadata["metadata"], dict):
                        if "tags" in metadata["metadata"] and isinstance(metadata["metadata"]["tags"], list):
                            for tag in metadata["metadata"]["tags"]:
                                if tag.startswith("model:"):
                                    model_id = tag.replace("model:", "")
                                    self.add_space_entity_relationship(entity_id, model_id, "model", "uses")
        
        except Exception as e:
            logging.error(f"Error extracting relationships from metadata for {entity_type} {entity_id}: {e}")
    
    def generate_provenance_graph(self, output_file: str) -> bool:
        """
        Generate a graph representation of provenance relationships.
        
        Args:
            output_file: Path to save the graph data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a graph representation in a more universal format
            graph = {
                "nodes": [],
                "edges": []
            }
            
            # Add nodes for models
            for model_id in self.relationships.get("models", {}):
                graph["nodes"].append({
                    "id": model_id,
                    "type": "model",
                    "label": model_id.split("/")[-1] if "/" in model_id else model_id
                })
            
            # Add nodes for datasets
            for dataset_id in self.relationships.get("datasets", {}):
                graph["nodes"].append({
                    "id": dataset_id,
                    "type": "dataset",
                    "label": dataset_id.split("/")[-1] if "/" in dataset_id else dataset_id
                })
            
            # Add nodes for spaces
            for space_id in self.relationships.get("spaces", {}):
                graph["nodes"].append({
                    "id": space_id,
                    "type": "space",
                    "label": space_id.split("/")[-1] if "/" in space_id else space_id
                })
            
            # Add edges for model-model relationships
            for model_id, model_data in self.relationships.get("models", {}).items():
                for base_model_id in model_data.get("base_models", []):
                    graph["edges"].append({
                        "source": base_model_id,
                        "target": model_id,
                        "type": "derived_from"
                    })
            
            # Add edges for model-dataset relationships
            for model_id, model_data in self.relationships.get("models", {}).items():
                for rel_type, dataset_ids in model_data.get("datasets", {}).items():
                    for dataset_id in dataset_ids:
                        graph["edges"].append({
                            "source": model_id,
                            "target": dataset_id,
                            "type": rel_type
                        })
            
            # Add edges for space relationships
            for space_id, space_data in self.relationships.get("spaces", {}).items():
                # Space-model relationships
                for rel_type, model_ids in space_data.get("models", {}).items():
                    for model_id in model_ids:
                        graph["edges"].append({
                            "source": space_id,
                            "target": model_id,
                            "type": rel_type
                        })
                
                # Space-dataset relationships
                for rel_type, dataset_ids in space_data.get("datasets", {}).items():
                    for dataset_id in dataset_ids:
                        graph["edges"].append({
                            "source": space_id,
                            "target": dataset_id,
                            "type": rel_type
                        })
            
            # Save graph to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Generated provenance graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
            return True
            
        except Exception as e:
            logging.error(f"Error generating provenance graph: {e}", exc_info=True)
            return False