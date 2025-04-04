import os
import logging
import json
import tempfile
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

# Import dependencies from other modules
try:
    from ipfs_kit_py.ipfs_kit import IpfsKit
    from ipfs_datasets_py.ipfs_datasets import IpfsDataset
except ImportError:
    logging.warning("Could not import IPFS modules. Some functionality will be limited.")

class IpfsStorage:
    """
    Manages IPFS integration for storing model metadata and files.
    
    This class handles:
    - Content-addressed storage of model metadata
    - Storage of model files in IPFS
    - Efficient conversion between formats (JSON, Parquet, CAR)
    - Pin management for persistence in IPFS
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize IPFS storage integration.
        
        Args:
            config: Storage configuration dictionary
        """
        self.config = config
        self.ipfs_kit = None
        self.ipfs_dataset = None
        
        # Initialize IPFS if enabled
        if self.config.get("use_ipfs", True):
            self._initialize_ipfs()
    
    def _initialize_ipfs(self) -> None:
        """Initialize IPFS connections and clients."""
        try:
            # Initialize ipfs_kit for low-level IPFS operations
            self.ipfs_kit = IpfsKit()
            logging.info("IPFS Kit initialized successfully")
            
            # Initialize ipfs_dataset for dataset management
            self.ipfs_dataset = IpfsDataset()
            logging.info("IPFS Dataset initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize IPFS: {e}")
            self.ipfs_kit = None
            self.ipfs_dataset = None
    
    def is_ipfs_available(self) -> bool:
        """
        Check if IPFS is available.
        
        Returns:
            True if IPFS is available, False otherwise
        """
        return self.ipfs_kit is not None and self.ipfs_dataset is not None
    
    def add_file_to_ipfs(self, file_path: str) -> Optional[str]:
        """
        Add a file to IPFS.
        
        Args:
            file_path: Path to the file to add
            
        Returns:
            CID of the added file, or None if adding failed
        """
        if not self.is_ipfs_available():
            logging.warning("IPFS not available. File not added.")
            return None
        
        try:
            # Add file to IPFS
            options = self.config.get("ipfs_add_options", {})
            result = self.ipfs_kit.add(file_path, **options)
            
            # Extract and return the CID
            if isinstance(result, dict) and "Hash" in result:
                cid = result["Hash"]
                logging.info(f"File {file_path} added to IPFS with CID {cid}")
                return cid
            else:
                logging.error(f"Unexpected result from IPFS add: {result}")
                return None
        except Exception as e:
            logging.error(f"Error adding file to IPFS: {e}")
            return None
    
    def add_directory_to_ipfs(self, dir_path: str) -> Optional[str]:
        """
        Add a directory to IPFS recursively.
        
        Args:
            dir_path: Path to the directory to add
            
        Returns:
            CID of the added directory, or None if adding failed
        """
        if not self.is_ipfs_available():
            logging.warning("IPFS not available. Directory not added.")
            return None
        
        try:
            # Add directory to IPFS recursively
            options = self.config.get("ipfs_add_options", {})
            options["recursive"] = True
            result = self.ipfs_kit.add(dir_path, **options)
            
            # Extract and return the CID of the directory
            if isinstance(result, list) and result:
                # The last item should be the directory itself
                dir_entry = result[-1]
                if isinstance(dir_entry, dict) and "Hash" in dir_entry:
                    cid = dir_entry["Hash"]
                    logging.info(f"Directory {dir_path} added to IPFS with CID {cid}")
                    return cid
                else:
                    logging.error(f"Unexpected directory entry format: {dir_entry}")
                    return None
            else:
                logging.error(f"Unexpected result from IPFS add: {result}")
                return None
        except Exception as e:
            logging.error(f"Error adding directory to IPFS: {e}")
            return None
    
    def pin_cid(self, cid: str) -> bool:
        """
        Pin a CID to ensure it's not garbage collected.
        
        Args:
            cid: Content identifier to pin
            
        Returns:
            True if pinned successfully, False otherwise
        """
        if not self.is_ipfs_available():
            logging.warning("IPFS not available. CID not pinned.")
            return False
        
        try:
            result = self.ipfs_kit.pin.add(cid)
            if result and "Pins" in result and cid in result["Pins"]:
                logging.info(f"CID {cid} pinned successfully")
                return True
            else:
                logging.error(f"Failed to pin CID {cid}: {result}")
                return False
        except Exception as e:
            logging.error(f"Error pinning CID {cid}: {e}")
            return False
    
    def store_metadata_as_json(self, metadata: Dict[str, Any], 
                               output_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Store metadata as JSON and add to IPFS.
        
        Args:
            metadata: Metadata dictionary to store
            output_path: Path to save the JSON file (optional)
            
        Returns:
            Tuple of (file_path, cid) or (None, None) if failed
        """
        try:
            # Create temporary file if no output path provided
            if output_path is None:
                fd, output_path = tempfile.mkstemp(suffix=".json")
                os.close(fd)
            
            # Write metadata to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Add to IPFS if available
            cid = None
            if self.is_ipfs_available():
                cid = self.add_file_to_ipfs(output_path)
                
            return output_path, cid
            
        except Exception as e:
            logging.error(f"Error storing metadata as JSON: {e}")
            return None, None
    
    def store_metadata_as_parquet(self, metadata_list: List[Dict[str, Any]],
                                  output_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Store metadata as Parquet and add to IPFS.
        
        Args:
            metadata_list: List of metadata dictionaries to store
            output_path: Path to save the Parquet file (optional)
            
        Returns:
            Tuple of (file_path, cid) or (None, None) if failed
        """
        try:
            # Convert metadata list to DataFrame
            df = pd.DataFrame(metadata_list)
            
            # Create temporary file if no output path provided
            if output_path is None:
                fd, output_path = tempfile.mkstemp(suffix=".parquet")
                os.close(fd)
            
            # Write DataFrame to Parquet file
            df.to_parquet(output_path, index=False)
            
            # Add to IPFS if available
            cid = None
            if self.is_ipfs_available():
                cid = self.add_file_to_ipfs(output_path)
                
            return output_path, cid
            
        except Exception as e:
            logging.error(f"Error storing metadata as Parquet: {e}")
            return None, None
    
    def store_model_files(self, model_dir: str) -> Optional[str]:
        """
        Store all model files in IPFS.
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            CID of the model directory, or None if failed
        """
        if not os.path.isdir(model_dir):
            logging.error(f"Model directory {model_dir} does not exist")
            return None
        
        try:
            # Add the entire directory to IPFS
            return self.add_directory_to_ipfs(model_dir)
        except Exception as e:
            logging.error(f"Error storing model files: {e}")
            return None
    
    def convert_jsonl_to_parquet(self, jsonl_path: str, parquet_path: Optional[str] = None) -> Optional[str]:
        """
        Convert JSONL file to Parquet format.
        
        Args:
            jsonl_path: Path to the JSONL file
            parquet_path: Path to save the Parquet file (optional)
            
        Returns:
            Path to the Parquet file, or None if conversion failed
        """
        try:
            # Create output path if not provided
            if parquet_path is None:
                parquet_path = os.path.splitext(jsonl_path)[0] + ".parquet"
            
            # Read JSONL file as DataFrame
            df = pd.read_json(jsonl_path, lines=True)
            
            # Write DataFrame to Parquet
            df.to_parquet(parquet_path, index=False)
            
            logging.info(f"Converted {jsonl_path} to {parquet_path}")
            return parquet_path
            
        except Exception as e:
            logging.error(f"Error converting JSONL to Parquet: {e}")
            return None
    
    def create_car_file(self, content_path: str, car_path: Optional[str] = None) -> Optional[str]:
        """
        Create a CAR (Content Addressable aRchive) file from content.
        
        Args:
            content_path: Path to the content (file or directory)
            car_path: Path to save the CAR file (optional)
            
        Returns:
            Path to the CAR file, or None if creation failed
        """
        if not self.is_ipfs_available() or not hasattr(self.ipfs_dataset, 'create_car'):
            logging.warning("IPFS Dataset not available or missing create_car method. CAR file not created.")
            return None
        
        try:
            # Create default car path if not provided
            if car_path is None:
                car_path = content_path + ".car"
            
            # Create CAR file
            result = self.ipfs_dataset.create_car(content_path, car_path)
            
            if result and os.path.exists(car_path):
                logging.info(f"Created CAR file {car_path} from {content_path}")
                return car_path
            else:
                logging.error(f"Failed to create CAR file from {content_path}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating CAR file: {e}")
            return None
    
    def get_file_from_ipfs(self, cid: str, output_path: str) -> bool:
        """
        Retrieve a file from IPFS.
        
        Args:
            cid: Content identifier of the file
            output_path: Path to save the retrieved file
            
        Returns:
            True if retrieved successfully, False otherwise
        """
        if not self.is_ipfs_available():
            logging.warning("IPFS not available. File not retrieved.")
            return False
        
        try:
            result = self.ipfs_kit.get(cid, output_path)
            
            if result and os.path.exists(output_path):
                logging.info(f"Retrieved file with CID {cid} to {output_path}")
                return True
            else:
                logging.error(f"Failed to retrieve file with CID {cid}")
                return False
                
        except Exception as e:
            logging.error(f"Error retrieving file from IPFS: {e}")
            return False
    
    def calculate_cid(self, file_path: str) -> Optional[str]:
        """
        Calculate the CID for a file without adding it to IPFS.
        
        Args:
            file_path: Path to the file
            
        Returns:
            CID of the file, or None if calculation failed
        """
        if not self.is_ipfs_available() or not hasattr(self.ipfs_kit, 'calculate_cid'):
            logging.warning("IPFS Kit not available or missing calculate_cid method. CID not calculated.")
            return None
        
        try:
            cid = self.ipfs_kit.calculate_cid(file_path)
            logging.info(f"Calculated CID for {file_path}: {cid}")
            return cid
        except Exception as e:
            logging.error(f"Error calculating CID: {e}")
            return None