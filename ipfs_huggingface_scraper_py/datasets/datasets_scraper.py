import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from huggingface_hub import list_datasets, hf_hub_download, HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HFValidationError

from ..config import get_config
from ..state_manager import StateManager
from ..rate_limiter import RateLimiter
from ..ipfs_integration import IpfsStorage

class DatasetsScraper:
    """
    HuggingFace datasets scraper with robust state management,
    rate limiting, and IPFS integration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the datasets scraper.
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        # Load configuration
        self.config = get_config(config_path)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all scraper components."""
        # Get configuration values
        scraper_config = self.config.config["scraper"]
        api_config = self.config.config["api"]
        storage_config = self.config.config["storage"]
        state_config = self.config.config["state"]
        
        # Initialize state manager
        self.state_manager = StateManager(
            state_dir=state_config["state_dir"],
            state_file="datasets_scraper_state.json"
        )
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            default_rate=api_config["anonymous_rate_limit"] if not api_config["authenticated"] else api_config["authenticated_rate_limit"],
            daily_quota=api_config["daily_anonymous_quota"],
            authenticated_quota=api_config["daily_authenticated_quota"],
            max_retries=api_config["max_retries"]
        )
        self.rate_limiter.set_authenticated(api_config["authenticated"])
        
        # Initialize IPFS storage
        self.ipfs_storage = IpfsStorage(storage_config)
        
        # Initialize HuggingFace client
        self.hf_client = HfApi(token=api_config["api_token"])
        # Timeout will be handled by the rate limiter
        
        # Configure for resumable operation
        if state_config["auto_resume"] and not self.state_manager.is_completed():
            logging.info("Auto-resuming from previous state")
    
    def scrape_datasets(self, max_datasets: Optional[int] = None) -> None:
        """
        Scrape datasets from HuggingFace Hub.
        
        Args:
            max_datasets: Maximum number of datasets to scrape (overrides config)
        """
        # Get configuration values
        scraper_config = self.config.config["scraper"]
        
        # Use parameter max_datasets if provided, otherwise use config
        max_datasets = max_datasets if max_datasets is not None else scraper_config["max_datasets"]
        
        # Log start
        logging.info(f"Starting HuggingFace datasets scraper. Max datasets: {max_datasets or 'unlimited'}")
        
        try:
            # Discover datasets
            dataset_ids = self._discover_datasets(max_datasets)
            
            # Process datasets in batches
            self._process_datasets_in_batches(dataset_ids, scraper_config["batch_size"])
            
            # Mark scraping as completed
            self.state_manager.complete()
            logging.info("Datasets scraping completed successfully")
            
            # Log summary
            self._log_summary()
            
        except KeyboardInterrupt:
            logging.info("Scraping interrupted by user. State saved for resuming.")
            self.state_manager.pause()
        except Exception as e:
            logging.error(f"Error in scraping process: {e}", exc_info=True)
            self.state_manager.pause()
    
    def _discover_datasets(self, max_datasets: Optional[int] = None) -> List[str]:
        """
        Discover datasets from HuggingFace Hub.
        
        Args:
            max_datasets: Maximum number of datasets to discover
            
        Returns:
            List of dataset IDs
        """
        logging.info(f"Discovering datasets from HuggingFace Hub (limit: {max_datasets or 'unlimited'})...")
        
        dataset_ids = []
        
        try:
            # Define a rate-limited dataset discovery function
            def get_datasets_page():
                return list_datasets(limit=max_datasets)
            
            # Use rate limiter for the API call
            datasets_iterator = self.rate_limiter.execute_with_rate_limit(
                "list_datasets", get_datasets_page
            )
            
            # Extract dataset IDs
            for dataset_info in datasets_iterator:
                dataset_ids.append(dataset_info.id)
                
                # Check if we've reached the limit
                if max_datasets and len(dataset_ids) >= max_datasets:
                    break
            
            # Update state with total discovered
            self.state_manager.set_total_discovered(len(dataset_ids))
            logging.info(f"Discovered {len(dataset_ids)} datasets")
            
            return dataset_ids
            
        except Exception as e:
            logging.error(f"Error discovering datasets: {e}", exc_info=True)
            return []
    
    def _process_datasets_in_batches(self, dataset_ids: List[str], batch_size: int = 100) -> None:
        """
        Process datasets in batches.
        
        Args:
            dataset_ids: List of dataset IDs to process
            batch_size: Size of each batch
        """
        if not dataset_ids:
            logging.warning("No datasets to process")
            return
        
        # Start from the current position in state
        current_position = self.state_manager.state["current_position"]
        
        # Calculate number of batches
        total_datasets = len(dataset_ids)
        num_batches = (total_datasets + batch_size - 1) // batch_size
        
        logging.info(f"Processing {total_datasets} datasets in {num_batches} batches of {batch_size}")
        
        for batch_idx in range(current_position // batch_size, num_batches):
            # Check if scraping is paused
            if self.state_manager.is_paused():
                logging.info("Scraping is paused. Stopping batch processing.")
                break
            
            # Calculate batch range
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_datasets)
            batch_dataset_ids = dataset_ids[start_idx:end_idx]
            
            # Update state with current batch
            self.state_manager.set_current_batch(batch_dataset_ids)
            self.state_manager.update_position(start_idx)
            
            logging.info(f"Processing batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx-1})")
            
            # Process batch
            self._process_batch(batch_dataset_ids)
            
            # Create checkpoint after each batch
            self.state_manager.create_checkpoint()
    
    def _process_batch(self, dataset_ids: List[str]) -> None:
        """
        Process a batch of datasets.
        
        Args:
            dataset_ids: List of dataset IDs to process
        """
        # Get configuration values
        scraper_config = self.config.config["scraper"]
        skip_existing = scraper_config["skip_existing"]
        
        # Filter out already processed datasets if skipping is enabled
        if skip_existing:
            unprocessed_ids = [
                dataset_id for dataset_id in dataset_ids
                if not self.state_manager.is_model_processed(dataset_id)
            ]
            if len(unprocessed_ids) < len(dataset_ids):
                logging.info(f"Skipping {len(dataset_ids) - len(unprocessed_ids)} already processed datasets")
            dataset_ids = unprocessed_ids
        
        # Create progress bar
        with tqdm(total=len(dataset_ids), desc="Processing datasets") as pbar:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all tasks
                future_to_dataset = {
                    executor.submit(self._process_dataset, dataset_id): dataset_id
                    for dataset_id in dataset_ids
                }
                
                # Process results as they complete
                for future in as_completed(future_to_dataset):
                    dataset_id = future_to_dataset[future]
                    try:
                        # Get result (success/failure)
                        success = future.result()
                        if success:
                            self.state_manager.mark_model_completed(dataset_id)
                    except Exception as e:
                        logging.error(f"Unhandled exception processing dataset {dataset_id}: {e}", exc_info=True)
                        self.state_manager.mark_model_errored(dataset_id, str(e))
                    
                    # Update progress bar
                    pbar.update(1)
    
    def _process_dataset(self, dataset_id: str) -> bool:
        """
        Process a single dataset.
        
        Args:
            dataset_id: Dataset ID to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        # Get configuration values
        scraper_config = self.config.config["scraper"]
        output_dir = scraper_config["output_dir"]
        save_metadata = scraper_config["save_metadata"]
        
        logging.info(f"Processing dataset: {dataset_id}")
        
        try:
            # Mark as processed in state
            self.state_manager.mark_model_processed(dataset_id)
            
            # Create dataset directory
            safe_dataset_dirname = dataset_id.replace("/", "__")
            dataset_save_path = os.path.join(output_dir, "datasets", safe_dataset_dirname)
            os.makedirs(dataset_save_path, exist_ok=True)
            
            # 1. Get and save metadata
            if save_metadata:
                success = self._save_dataset_metadata(dataset_id, dataset_save_path)
                if not success:
                    self.state_manager.mark_model_errored(dataset_id, "Failed to save metadata")
                    return False
            
            # 2. Download dataset preview
            success = self._download_dataset_preview(dataset_id, dataset_save_path, 
                                                   scraper_config["dataset_preview_max_rows"])
            if not success:
                logging.warning(f"Failed to download preview for {dataset_id}")
                # Continue anyway - preview is optional
            
            # 3. Store in IPFS if enabled
            if self.config.config["storage"]["use_ipfs"] and self.ipfs_storage.is_ipfs_available():
                cid = self.ipfs_storage.store_model_files(dataset_save_path)
                if cid:
                    logging.info(f"Dataset {dataset_id} stored in IPFS with CID: {cid}")
                    
                    # Save CID to dataset directory for reference
                    with open(os.path.join(dataset_save_path, "ipfs_cid.txt"), "w") as f:
                        f.write(cid)
                else:
                    logging.warning(f"Failed to store dataset {dataset_id} in IPFS")
            
            # 4. Extract provenance information if enabled
            if self.config.config["provenance"]["extract_dataset_relationships"]:
                self._extract_dataset_provenance(dataset_id, dataset_save_path)
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing dataset {dataset_id}: {e}", exc_info=True)
            self.state_manager.mark_model_errored(dataset_id, str(e))
            return False
    
    def _save_dataset_metadata(self, dataset_id: str, dataset_save_path: str) -> bool:
        """
        Save dataset metadata.
        
        Args:
            dataset_id: Dataset ID
            dataset_save_path: Path to save dataset files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define a rate-limited metadata fetch function
            def fetch_dataset_info():
                return self.hf_client.dataset_info(dataset_id, full=True, with_card_data=True)
            
            # Use rate limiter for the API call
            dataset_info = self.rate_limiter.execute_with_rate_limit(
                "dataset_info", fetch_dataset_info
            )
            
            # Convert dataset info to dictionary
            metadata_dict = {}
            for attr, value in dataset_info.__dict__.items():
                if attr.startswith("_"):
                    continue
                
                try:
                    # Handle special cases like datetime objects
                    if hasattr(value, "isoformat"):
                        metadata_dict[attr] = value.isoformat()
                    elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        metadata_dict[attr] = value
                    else:
                        # Try to convert to string as fallback
                        metadata_dict[attr] = str(value)
                except Exception as e:
                    logging.warning(f"Could not serialize attribute '{attr}' for {dataset_id}: {e}")
            
            # Save metadata to file
            metadata_filepath = os.path.join(dataset_save_path, "metadata.json")
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved metadata for {dataset_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving metadata for {dataset_id}: {e}", exc_info=True)
            return False
    
    def _download_dataset_preview(self, dataset_id: str, dataset_save_path: str, max_rows: int = 100) -> bool:
        """
        Download a dataset preview.
        
        Args:
            dataset_id: Dataset ID
            dataset_save_path: Path to save dataset files
            max_rows: Maximum number of rows to include in preview
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # We'll just save a reference to the dataset_id - actual preview requires
            # the datasets library and would be more complex
            preview_info = {
                "dataset_id": dataset_id,
                "max_preview_rows": max_rows,
                "preview_note": "To preview this dataset, use the datasets library: "
                                "from datasets import load_dataset; "
                                f"dataset = load_dataset('{dataset_id}')"
            }
            
            preview_filepath = os.path.join(dataset_save_path, "preview_info.json")
            with open(preview_filepath, 'w', encoding='utf-8') as f:
                json.dump(preview_info, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved preview info for {dataset_id}")
            return True
                
        except Exception as e:
            logging.error(f"Error saving preview info for {dataset_id}: {e}", exc_info=True)
            return False
    
    def _extract_dataset_provenance(self, dataset_id: str, dataset_save_path: str) -> bool:
        """
        Extract dataset provenance information.
        
        Args:
            dataset_id: Dataset ID
            dataset_save_path: Path to save dataset files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Attempt to extract provenance information from metadata and dataset card
            # This is a simplified version - actual implementation would be more complex
            provenance_info = {
                "dataset_id": dataset_id,
                "derived_from": [],  # Would contain other datasets this was derived from
                "used_by_models": [],  # Would contain models using this dataset
                "versions": [],  # Would contain version history
                "collection": None,  # Would contain collection info if part of a collection
                "timestamp": time.time()
            }
            
            # Save provenance info
            provenance_filepath = os.path.join(dataset_save_path, "provenance.json")
            with open(provenance_filepath, 'w', encoding='utf-8') as f:
                json.dump(provenance_info, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved provenance info for {dataset_id}")
            return True
                
        except Exception as e:
            logging.error(f"Error extracting provenance for {dataset_id}: {e}", exc_info=True)
            return False
    
    def _log_summary(self) -> None:
        """Log a summary of the scraping operation."""
        progress = self.state_manager.get_progress()
        
        logging.info("=== Datasets Scraping Summary ===")
        logging.info(f"Total datasets discovered: {progress['total_models_discovered']}")
        logging.info(f"Datasets processed: {progress['models_processed']}")
        logging.info(f"Datasets completed successfully: {progress['models_completed']}")
        logging.info(f"Datasets with errors: {progress['models_errored']}")
        logging.info(f"Completion percentage: {progress['completion_percentage']:.2f}%")
        logging.info(f"Elapsed time: {progress['elapsed_time'] / 60:.2f} minutes")
        logging.info("=======================")
    
    def resume(self) -> None:
        """Resume a paused scraping operation."""
        if not self.state_manager.is_paused():
            logging.warning("No paused scraping operation to resume")
            return
        
        logging.info("Resuming datasets scraping operation")
        self.state_manager.resume()
        
        # Re-discover datasets or use cached list
        max_datasets = self.config.config["scraper"]["max_datasets"]
        cached_total = self.state_manager.state["total_models_discovered"]
        
        if cached_total > 0:
            # Rediscover only if we need more datasets than we have already discovered
            if max_datasets is None or cached_total < max_datasets:
                dataset_ids = self._discover_datasets(max_datasets)
            else:
                logging.info(f"Using {cached_total} already discovered datasets")
                # We don't actually have the dataset IDs stored, so we need to rediscover
                dataset_ids = self._discover_datasets(max_datasets)
        else:
            dataset_ids = self._discover_datasets(max_datasets)
        
        # Resume processing from the current position
        self._process_datasets_in_batches(dataset_ids, self.config.config["scraper"]["batch_size"])
        
        # Mark as completed if we finished all batches
        if not self.state_manager.is_paused():
            self.state_manager.complete()
            logging.info("Resumed datasets scraping completed successfully")
            
        # Log summary
        self._log_summary()

# Main entry point
def main(config_path: Optional[str] = None, max_datasets: Optional[int] = None) -> None:
    """
    Main entry point for the datasets scraper.
    
    Args:
        config_path: Path to the configuration file (optional)
        max_datasets: Maximum number of datasets to scrape (optional)
    """
    scraper = DatasetsScraper(config_path)
    scraper.scrape_datasets(max_datasets)

if __name__ == "__main__":
    main()