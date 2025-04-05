import os
import logging
import tempfile
import pandas as pd
import pyarrow as pa
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UnifiedExport:
    """
    Manages export of scraped entity data (models, datasets, spaces) to a unified Parquet file.
    
    This class handles:
    - Aggregation of metadata from different entity types
    - Schema normalization and validation
    - Safe Parquet conversion with fallback mechanisms
    - Storage in the project data folder
    - IPFS integration (if available)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the unified export manager.
        
        Args:
            config: Configuration dictionary with export settings
        """
        self.config = config or {}
        self.ipfs_storage = None
        
        # Initialize IPFS if available
        if self.config.get("use_ipfs", True):
            try:
                from ipfs_huggingface_scraper_py.ipfs_integration import IpfsStorage
                self.ipfs_storage = IpfsStorage(self.config)
                logging.info("IPFS storage initialized for unified export")
            except ImportError:
                logging.warning("IPFS integration not available. CIDs will not be generated.")
                
        # Set up data directory paths
        self.data_dir = Path(self.config.get("data_dir", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")))
        self.metadata_dir = self.data_dir / "huggingface_hub_metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Default output filenames
        self.default_output_filename = "all_entities_metadata.parquet"
        self.models_output_filename = "all_models_metadata.parquet"
        self.datasets_output_filename = "all_datasets_metadata.parquet"
        self.spaces_output_filename = "all_spaces_metadata.parquet"
    
    def normalize_schema(self, entity_list: List[Dict[str, Any]], entity_type: str) -> List[Dict[str, Any]]:
        """
        Normalize schema across different entity types to ensure consistency.
        
        Args:
            entity_list: List of entity metadata dictionaries
            entity_type: Type of entity ('model', 'dataset', or 'space')
            
        Returns:
            List of normalized entity metadata dictionaries
        """
        normalized_entities = []
        
        for entity in entity_list:
            # Create copy to avoid modifying the original
            normalized_entity = entity.copy()
            
            # Ensure entity_type field exists
            normalized_entity['entity_type'] = entity_type
            
            # Ensure id field is consistent
            if 'id' not in normalized_entity and 'model_id' in normalized_entity:
                normalized_entity['id'] = normalized_entity['model_id']
            elif 'id' not in normalized_entity and 'dataset_id' in normalized_entity:
                normalized_entity['id'] = normalized_entity['dataset_id']
            elif 'id' not in normalized_entity and 'space_id' in normalized_entity:
                normalized_entity['id'] = normalized_entity['space_id']
                
            # Add timestamp for when this record was processed
            if 'processed_at' not in normalized_entity:
                normalized_entity['processed_at'] = datetime.utcnow().isoformat()
                
            # Remove any embedding data that might be present (should be in separate files)
            normalized_entity.pop('config_embedding', None)
            normalized_entity.pop('readme_embedding', None)
            normalized_entity.pop('description_embedding', None)
            
            normalized_entities.append(normalized_entity)
            
        return normalized_entities
    
    def save_dataframe_to_parquet_safely(self, df: pd.DataFrame, filepath: Union[str, Path]) -> bool:
        """
        Saves DataFrame to Parquet with explicit schema handling for mixed types.
        Implements fallback mechanisms if the initial save fails.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the Parquet file
            
        Returns:
            True if saved successfully as Parquet, False if failed or saved as CSV fallback
        """
        filepath = Path(filepath)
        try:
            # First attempt: Convert known problematic columns to string
            df_safe = df.copy()
            
            # Handle the 'gated' column specifically which often causes issues
            if 'gated' in df_safe.columns:
                df_safe['gated'] = df_safe['gated'].astype(str)
            
            # Convert all object columns except id to string to be safe
            for col in df_safe.select_dtypes(include=['object']).columns:
                if col != 'id':  # Keep id as is since it's used as an identifier
                    df_safe[col] = df_safe[col].astype(str)
            
            # Try saving with pandas
            df_safe.to_parquet(filepath, index=False)
            logging.info(f"Successfully saved to {filepath} using pandas")
            return True
            
        except Exception as e:
            logging.warning(f"First attempt to save Parquet failed: {e}")
            
            try:
                # Second attempt: Use PyArrow with explicit schema
                schema = pa.Schema.from_pandas(df)
                fields = list(schema)
                
                # Convert all string/binary fields to string type except id
                for i, field in enumerate(fields):
                    if (pa.types.is_string(field.type) or pa.types.is_binary(field.type)) and field.name != 'id':
                        fields[i] = pa.field(field.name, pa.string())
                
                new_schema = pa.schema(fields)
                
                # Force conversion of problematic columns
                df_safe = df.copy()
                for col in df_safe.select_dtypes(include=['object']).columns:
                    if col != 'id':
                        df_safe[col] = df_safe[col].astype(str)
                
                # Convert to table with schema and write
                table = pa.Table.from_pandas(df_safe, schema=new_schema)
                pa.parquet.write_table(table, filepath)
                logging.info(f"Successfully saved to {filepath} using PyArrow with schema conversion")
                return True
                
            except Exception as e2:
                logging.error(f"Both Parquet saving attempts failed for {filepath}: {e2}")
                
                # Last resort - save to CSV instead
                try:
                    csv_filepath = filepath.with_suffix('.csv')
                    logging.warning(f"Falling back to CSV format: {csv_filepath}")
                    df.to_csv(csv_filepath, index=False)
                    logging.info(f"Saved as CSV instead: {csv_filepath}")
                    return False
                except Exception as e3:
                    logging.error(f"Even CSV fallback failed: {e3}")
                    return False
    
    def load_existing_data(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Load existing data from a Parquet file with fallback to CSV.
        
        Args:
            filepath: Path to the Parquet file
            
        Returns:
            DataFrame with the loaded data or None if file doesn't exist or load fails
        """
        if not filepath.exists():
            return None
            
        try:
            # Try loading Parquet first
            df = pd.read_parquet(filepath)
            logging.info(f"Loaded {len(df)} records from existing Parquet file: {filepath}")
            return df
        except Exception as e:
            logging.warning(f"Failed to load Parquet file {filepath}: {e}")
            
            # Try CSV fallback
            csv_filepath = filepath.with_suffix('.csv')
            if csv_filepath.exists():
                try:
                    df = pd.read_csv(csv_filepath)
                    logging.info(f"Loaded {len(df)} records from existing CSV file: {csv_filepath}")
                    return df
                except Exception as e2:
                    logging.error(f"Failed to load CSV fallback {csv_filepath}: {e2}")
            
            return None
    
    def store_entity_data(self, 
                          entity_list: List[Dict[str, Any]],
                          entity_type: str,
                          output_path: Optional[str] = None,
                          merge_with_existing: bool = True) -> Tuple[Optional[str], Optional[str]]:
        """
        Store entity data as Parquet and add to IPFS if available.
        
        Args:
            entity_list: List of entity metadata dictionaries
            entity_type: Type of entity ('model', 'dataset', or 'space')
            output_path: Path to save the Parquet file (optional)
            merge_with_existing: Whether to merge with existing data (if any)
            
        Returns:
            Tuple of (file_path, cid) or (None, None) if failed
        """
        if not entity_list:
            logging.warning(f"No {entity_type} entities provided to store. Skipping.")
            return None, None
            
        try:
            # Normalize schema across different entity types
            normalized_entities = self.normalize_schema(entity_list, entity_type)
            logging.info(f"Normalized {len(normalized_entities)} {entity_type} entities for storage")
            
            # Create DataFrame from normalized entities
            df_new = pd.DataFrame(normalized_entities)
            
            # Determine output path if not provided
            if output_path is None:
                if entity_type == 'model':
                    output_path = str(self.metadata_dir / self.models_output_filename)
                elif entity_type == 'dataset':
                    output_path = str(self.metadata_dir / self.datasets_output_filename)
                elif entity_type == 'space':
                    output_path = str(self.metadata_dir / self.spaces_output_filename)
                else:
                    output_path = str(self.metadata_dir / self.default_output_filename)
            
            output_path = Path(output_path)
            
            # Merge with existing data if requested and file exists
            if merge_with_existing:
                df_existing = self.load_existing_data(output_path)
                
                if df_existing is not None:
                    # Check for duplicate IDs
                    if 'id' in df_existing.columns and 'id' in df_new.columns:
                        existing_ids = set(df_existing['id'])
                        new_ids = set(df_new['id'])
                        duplicate_ids = existing_ids.intersection(new_ids)
                        
                        if duplicate_ids:
                            logging.info(f"Found {len(duplicate_ids)} duplicate IDs between existing and new data")
                            
                            # Remove duplicates from existing data (keep the new versions)
                            df_existing = df_existing[~df_existing['id'].isin(duplicate_ids)]
                            
                        # Merge datasets
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        logging.info(f"Combined {len(df_existing)} existing and {len(df_new)} new records")
                        
                        # Use combined DataFrame for saving
                        df_to_save = df_combined
                    else:
                        # If no 'id' column, just append
                        df_to_save = pd.concat([df_existing, df_new], ignore_index=True)
                        logging.info(f"Appended {len(df_new)} new records to {len(df_existing)} existing records")
                else:
                    # No existing data, just use new data
                    df_to_save = df_new
            else:
                # Not merging, just use new data
                df_to_save = df_new
            
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame to Parquet safely
            save_success = self.save_dataframe_to_parquet_safely(df_to_save, output_path)
            
            if not save_success:
                logging.warning(f"Could not save as Parquet. Check for CSV fallback at {output_path.with_suffix('.csv')}")
            
            # Add to IPFS if available
            cid = None
            if save_success and self.ipfs_storage and self.ipfs_storage.is_ipfs_available():
                try:
                    cid = self.ipfs_storage.add_file_to_ipfs(str(output_path))
                    if cid:
                        logging.info(f"Added {entity_type} Parquet file to IPFS with CID: {cid}")
                    else:
                        logging.warning(f"Failed to add {entity_type} Parquet file to IPFS")
                except Exception as e:
                    logging.error(f"Error adding file to IPFS: {e}")
            
            return str(output_path), cid
            
        except Exception as e:
            logging.error(f"Error storing {entity_type} data as Parquet: {e}")
            return None, None
    
    def store_unified_data(self, 
                           models_list: Optional[List[Dict[str, Any]]] = None,
                           datasets_list: Optional[List[Dict[str, Any]]] = None,
                           spaces_list: Optional[List[Dict[str, Any]]] = None,
                           output_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Store all entity types in a unified Parquet file.
        
        Args:
            models_list: List of model metadata dictionaries (optional)
            datasets_list: List of dataset metadata dictionaries (optional)
            spaces_list: List of space metadata dictionaries (optional)
            output_path: Path to save the unified Parquet file (optional)
            
        Returns:
            Tuple of (file_path, cid) or (None, None) if failed
        """
        # Create empty lists for any missing parameters
        models_list = models_list or []
        datasets_list = datasets_list or []
        spaces_list = spaces_list or []
        
        # Check if we have any data to store
        if not models_list and not datasets_list and not spaces_list:
            logging.warning("No entity data provided to store_unified_data. Nothing to do.")
            return None, None
        
        try:
            # Normalize schema for each entity type
            normalized_models = self.normalize_schema(models_list, 'model') if models_list else []
            normalized_datasets = self.normalize_schema(datasets_list, 'dataset') if datasets_list else []
            normalized_spaces = self.normalize_schema(spaces_list, 'space') if spaces_list else []
            
            # Combine all entities into a single list
            all_entities = normalized_models + normalized_datasets + normalized_spaces
            
            if not all_entities:
                logging.warning("No entities to store after normalization. Skipping.")
                return None, None
            
            # Create DataFrame
            df_unified = pd.DataFrame(all_entities)
            
            # Determine output path if not provided
            if output_path is None:
                output_path = str(self.metadata_dir / self.default_output_filename)
            
            output_path = Path(output_path)
            
            # Merge with existing data if file exists
            df_existing = self.load_existing_data(output_path)
            
            if df_existing is not None:
                # Check for duplicate IDs
                if 'id' in df_existing.columns and 'id' in df_unified.columns:
                    existing_ids = set(df_existing['id'])
                    new_ids = set(df_unified['id'])
                    duplicate_ids = existing_ids.intersection(new_ids)
                    
                    if duplicate_ids:
                        logging.info(f"Found {len(duplicate_ids)} duplicate IDs between existing and new data")
                        
                        # Remove duplicates from existing data (keep the new versions)
                        df_existing = df_existing[~df_existing['id'].isin(duplicate_ids)]
                        
                    # Merge datasets
                    df_combined = pd.concat([df_existing, df_unified], ignore_index=True)
                    logging.info(f"Combined {len(df_existing)} existing and {len(df_unified)} new records")
                    
                    # Use combined DataFrame for saving
                    df_to_save = df_combined
                else:
                    # If no 'id' column, just append
                    df_to_save = pd.concat([df_existing, df_unified], ignore_index=True)
                    logging.info(f"Appended {len(df_unified)} new records to {len(df_existing)} existing records")
            else:
                # No existing data, just use new data
                df_to_save = df_unified
            
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame to Parquet safely
            save_success = self.save_dataframe_to_parquet_safely(df_to_save, output_path)
            
            if not save_success:
                logging.warning(f"Could not save as Parquet. Check for CSV fallback at {output_path.with_suffix('.csv')}")
            
            # Add to IPFS if available
            cid = None
            if save_success and self.ipfs_storage and self.ipfs_storage.is_ipfs_available():
                try:
                    cid = self.ipfs_storage.add_file_to_ipfs(str(output_path))
                    if cid:
                        logging.info(f"Added unified Parquet file to IPFS with CID: {cid}")
                    else:
                        logging.warning("Failed to add unified Parquet file to IPFS")
                except Exception as e:
                    logging.error(f"Error adding file to IPFS: {e}")
            
            # Log summary of stored entities
            entity_counts = {}
            if 'entity_type' in df_to_save.columns:
                entity_counts = df_to_save['entity_type'].value_counts().to_dict()
                
            logging.info(f"Stored unified data with {len(df_to_save)} total records: {entity_counts}")
            
            return str(output_path), cid
            
        except Exception as e:
            logging.error(f"Error storing unified data as Parquet: {e}")
            return None, None

    def get_entity_statistics(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about the stored entity data.
        
        Args:
            filepath: Path to the Parquet file (optional, uses default if not provided)
            
        Returns:
            Dictionary with statistics
        """
        if filepath is None:
            filepath = str(self.metadata_dir / self.default_output_filename)
            
        filepath = Path(filepath)
        
        stats = {
            "total_entities": 0,
            "entity_types": {},
            "last_updated": None,
            "file_size_bytes": 0,
            "file_exists": False
        }
        
        # Check if file exists
        if not filepath.exists():
            # Check for CSV fallback
            csv_filepath = filepath.with_suffix('.csv')
            if not csv_filepath.exists():
                return stats
            else:
                filepath = csv_filepath
        
        stats["file_exists"] = True
        stats["file_size_bytes"] = filepath.stat().st_size
        stats["last_updated"] = datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
        
        try:
            # Load the data
            if filepath.suffix.lower() == '.parquet':
                df = pd.read_parquet(filepath)
            else:  # CSV
                df = pd.read_csv(filepath)
                
            stats["total_entities"] = len(df)
            
            # Count by entity type if column exists
            if 'entity_type' in df.columns:
                stats["entity_types"] = df['entity_type'].value_counts().to_dict()
                
            return stats
            
        except Exception as e:
            logging.error(f"Error getting entity statistics: {e}")
            return stats