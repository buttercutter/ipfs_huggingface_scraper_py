import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from huggingface_hub import list_models, hf_hub_download, HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HFValidationError

from .config import get_config
from .state_manager import StateManager
from .rate_limiter import RateLimiter
from .ipfs_integration import IpfsStorage

class EnhancedScraper:
    """
    Enhanced HuggingFace model scraper with robust state management,
    rate limiting, and IPFS integration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the enhanced scraper.
        
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
            state_file="scraper_state.json"
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
    
    def scrape_models(self, max_models: Optional[int] = None) -> None:
        """
        Scrape models from HuggingFace Hub.
        
        Args:
            max_models: Maximum number of models to scrape (overrides config)
        """
        # Get configuration values
        scraper_config = self.config.config["scraper"]
        
        # Use parameter max_models if provided, otherwise use config
        max_models = max_models if max_models is not None else scraper_config["max_models"]
        
        # Log start
        logging.info(f"Starting enhanced HuggingFace scraper. Max models: {max_models or 'unlimited'}")
        
        try:
            # Discover models
            model_ids = self._discover_models(max_models)
            
            # Process models in batches
            self._process_models_in_batches(model_ids, scraper_config["batch_size"])
            
            # Mark scraping as completed
            self.state_manager.complete()
            logging.info("Scraping completed successfully")
            
            # Log summary
            self._log_summary()
            
        except KeyboardInterrupt:
            logging.info("Scraping interrupted by user. State saved for resuming.")
            self.state_manager.pause()
        except Exception as e:
            logging.error(f"Error in scraping process: {e}", exc_info=True)
            self.state_manager.pause()
    
    def _discover_models(self, max_models: Optional[int] = None) -> List[str]:
        """
        Discover models from HuggingFace Hub.
        
        Args:
            max_models: Maximum number of models to discover
            
        Returns:
            List of model IDs
        """
        logging.info(f"Discovering models from HuggingFace Hub (limit: {max_models or 'unlimited'})...")
        
        model_ids = []
        
        try:
            # Define a rate-limited model discovery function
            def get_models_page():
                return list_models(limit=max_models)
            
            # Use rate limiter for the API call
            models_iterator = self.rate_limiter.execute_with_rate_limit(
                "list_models", get_models_page
            )
            
            # Extract model IDs
            for model_info in models_iterator:
                model_ids.append(model_info.id)
                
                # Check if we've reached the limit
                if max_models and len(model_ids) >= max_models:
                    break
            
            # Update state with total discovered
            self.state_manager.set_total_discovered(len(model_ids))
            logging.info(f"Discovered {len(model_ids)} models")
            
            return model_ids
            
        except Exception as e:
            logging.error(f"Error discovering models: {e}", exc_info=True)
            return []
    
    def _process_models_in_batches(self, model_ids: List[str], batch_size: int = 100) -> None:
        """
        Process models in batches.
        
        Args:
            model_ids: List of model IDs to process
            batch_size: Size of each batch
        """
        if not model_ids:
            logging.warning("No models to process")
            return
        
        # Start from the current position in state
        current_position = self.state_manager.state["current_position"]
        
        # Calculate number of batches
        total_models = len(model_ids)
        num_batches = (total_models + batch_size - 1) // batch_size
        
        logging.info(f"Processing {total_models} models in {num_batches} batches of {batch_size}")
        
        for batch_idx in range(current_position // batch_size, num_batches):
            # Check if scraping is paused
            if self.state_manager.is_paused():
                logging.info("Scraping is paused. Stopping batch processing.")
                break
            
            # Calculate batch range
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_models)
            batch_model_ids = model_ids[start_idx:end_idx]
            
            # Update state with current batch
            self.state_manager.set_current_batch(batch_model_ids)
            self.state_manager.update_position(start_idx)
            
            logging.info(f"Processing batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx-1})")
            
            # Process batch
            self._process_batch(batch_model_ids)
            
            # Create checkpoint after each batch
            self.state_manager.create_checkpoint()
    
    def _process_batch(self, model_ids: List[str]) -> None:
        """
        Process a batch of models.
        
        Args:
            model_ids: List of model IDs to process
        """
        # Get configuration values
        scraper_config = self.config.config["scraper"]
        skip_existing = scraper_config["skip_existing"]
        
        # Filter out already processed models if skipping is enabled
        if skip_existing:
            unprocessed_ids = [
                model_id for model_id in model_ids
                if not self.state_manager.is_model_processed(model_id)
            ]
            if len(unprocessed_ids) < len(model_ids):
                logging.info(f"Skipping {len(model_ids) - len(unprocessed_ids)} already processed models")
            model_ids = unprocessed_ids
        
        # Create progress bar
        with tqdm(total=len(model_ids), desc="Processing models") as pbar:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all tasks
                future_to_model = {
                    executor.submit(self._process_model, model_id): model_id
                    for model_id in model_ids
                }
                
                # Process results as they complete
                for future in as_completed(future_to_model):
                    model_id = future_to_model[future]
                    try:
                        # Get result (success/failure)
                        success = future.result()
                        if success:
                            self.state_manager.mark_model_completed(model_id)
                        # Note: If not successful, the model is marked as errored inside _process_model
                    except Exception as e:
                        logging.error(f"Unhandled exception processing model {model_id}: {e}", exc_info=True)
                        self.state_manager.mark_model_errored(model_id, str(e))
                    
                    # Update progress bar
                    pbar.update(1)
    
    def _process_model(self, model_id: str) -> bool:
        """
        Process a single model.
        
        Args:
            model_id: Model ID to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        # Get configuration values
        scraper_config = self.config.config["scraper"]
        output_dir = scraper_config["output_dir"]
        save_metadata = scraper_config["save_metadata"]
        download_filename = scraper_config["filename_to_download"]
        
        logging.info(f"Processing model: {model_id}")
        
        try:
            # Mark as processed in state
            self.state_manager.mark_model_processed(model_id)
            
            # Create model directory
            safe_model_dirname = model_id.replace("/", "__")
            model_save_path = os.path.join(output_dir, safe_model_dirname)
            os.makedirs(model_save_path, exist_ok=True)
            
            # 1. Get and save metadata
            if save_metadata:
                success = self._save_model_metadata(model_id, model_save_path)
                if not success:
                    self.state_manager.mark_model_errored(model_id, "Failed to save metadata")
                    return False
            
            # 2. Download model file
            success = self._download_model_file(model_id, download_filename, model_save_path)
            if not success:
                self.state_manager.mark_model_errored(model_id, f"Failed to download {download_filename}")
                return False
            
            # 3. Store in IPFS if enabled
            if self.config.config["storage"]["use_ipfs"] and self.ipfs_storage.is_ipfs_available():
                cid = self.ipfs_storage.store_model_files(model_save_path)
                if cid:
                    logging.info(f"Model {model_id} stored in IPFS with CID: {cid}")
                    
                    # Save CID to model directory for reference
                    with open(os.path.join(model_save_path, "ipfs_cid.txt"), "w") as f:
                        f.write(cid)
                else:
                    logging.warning(f"Failed to store model {model_id} in IPFS")
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing model {model_id}: {e}", exc_info=True)
            self.state_manager.mark_model_errored(model_id, str(e))
            return False
    
    def _save_model_metadata(self, model_id: str, model_save_path: str) -> bool:
        """
        Save model metadata.
        
        Args:
            model_id: Model ID
            model_save_path: Path to save model files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define a rate-limited metadata fetch function
            def fetch_model_info():
                return self.hf_client.model_info(model_id, full=True, with_card_data=True)
            
            # Use rate limiter for the API call
            model_info = self.rate_limiter.execute_with_rate_limit(
                "model_info", fetch_model_info
            )
            
            # Convert model info to dictionary
            metadata_dict = {}
            for attr, value in model_info.__dict__.items():
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
                    logging.warning(f"Could not serialize attribute '{attr}' for {model_id}: {e}")
            
            # Save metadata to file
            metadata_filepath = os.path.join(model_save_path, "metadata.json")
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved metadata for {model_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving metadata for {model_id}: {e}", exc_info=True)
            return False
    
    def _download_model_file(self, model_id: str, filename: str, model_save_path: str) -> bool:
        """
        Download a model file.
        
        Args:
            model_id: Model ID
            filename: Filename to download
            model_save_path: Path to save model files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_filepath = os.path.join(model_save_path, filename)
            
            # Define a rate-limited download function
            def download_file():
                return hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    repo_type="model",
                    local_dir=model_save_path,
                    local_dir_use_symlinks=False
                )
            
            # Use rate limiter for the API call
            downloaded_path = self.rate_limiter.execute_with_rate_limit(
                "hub_download", download_file
            )
            
            if downloaded_path and os.path.exists(downloaded_path):
                logging.info(f"Downloaded {filename} for {model_id}")
                return True
            else:
                logging.error(f"Download returned success but file not found for {model_id}/{filename}")
                return False
                
        except RepositoryNotFoundError:
            logging.warning(f"Repository '{model_id}' not found (maybe private or deleted)")
            return False
        except EntryNotFoundError:
            logging.warning(f"File '{filename}' not found in repository '{model_id}'")
            return False
        except HFValidationError as e:
            logging.error(f"Validation Error for {model_id}/{filename}: {e}")
            return False
        except Exception as e:
            logging.error(f"Error downloading {filename} for {model_id}: {e}", exc_info=True)
            return False
    
    def _log_summary(self) -> None:
        """Log a summary of the scraping operation."""
        progress = self.state_manager.get_progress()
        
        logging.info("=== Scraping Summary ===")
        logging.info(f"Total models discovered: {progress['total_models_discovered']}")
        logging.info(f"Models processed: {progress['models_processed']}")
        logging.info(f"Models completed successfully: {progress['models_completed']}")
        logging.info(f"Models with errors: {progress['models_errored']}")
        logging.info(f"Completion percentage: {progress['completion_percentage']:.2f}%")
        logging.info(f"Elapsed time: {progress['elapsed_time'] / 60:.2f} minutes")
        logging.info("=======================")
    
    def resume(self) -> None:
        """Resume a paused scraping operation."""
        if not self.state_manager.is_paused():
            logging.warning("No paused scraping operation to resume")
            return
        
        logging.info("Resuming scraping operation")
        self.state_manager.resume()
        
        # Re-discover models or use cached list
        max_models = self.config.config["scraper"]["max_models"]
        cached_total = self.state_manager.state["total_models_discovered"]
        
        if cached_total > 0:
            # Rediscover only if we need more models than we have already discovered
            if max_models is None or cached_total < max_models:
                model_ids = self._discover_models(max_models)
            else:
                logging.info(f"Using {cached_total} already discovered models")
                # We don't actually have the model IDs stored, so we need to rediscover
                model_ids = self._discover_models(max_models)
        else:
            model_ids = self._discover_models(max_models)
        
        # Resume processing from the current position
        self._process_models_in_batches(model_ids, self.config.config["scraper"]["batch_size"])
        
        # Mark as completed if we finished all batches
        if not self.state_manager.is_paused():
            self.state_manager.complete()
            logging.info("Resumed scraping completed successfully")
            
        # Log summary
        self._log_summary()

# Main entry point
def main(config_path: Optional[str] = None, max_models: Optional[int] = None) -> None:
    """
    Main entry point for the enhanced scraper.
    
    Args:
        config_path: Path to the configuration file (optional)
        max_models: Maximum number of models to scrape (optional)
    """
    scraper = EnhancedScraper(config_path)
    scraper.scrape_models(max_models)

if __name__ == "__main__":
    main()