import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from huggingface_hub import list_spaces, hf_hub_download, HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HFValidationError

from ..config import get_config
from ..state_manager import StateManager
from ..rate_limiter import RateLimiter
from ..ipfs_integration import IpfsStorage

class SpacesScraper:
    """
    HuggingFace spaces scraper with robust state management,
    rate limiting, and IPFS integration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the spaces scraper.
        
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
            state_file="spaces_scraper_state.json"
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
    
    def scrape_spaces(self, max_spaces: Optional[int] = None) -> None:
        """
        Scrape spaces from HuggingFace Hub.
        
        Args:
            max_spaces: Maximum number of spaces to scrape (overrides config)
        """
        # Get configuration values
        scraper_config = self.config.config["scraper"]
        
        # Use parameter max_spaces if provided, otherwise use config
        max_spaces = max_spaces if max_spaces is not None else scraper_config["max_spaces"]
        
        # Log start
        logging.info(f"Starting HuggingFace spaces scraper. Max spaces: {max_spaces or 'unlimited'}")
        
        try:
            # Discover spaces
            space_ids = self._discover_spaces(max_spaces)
            
            # Process spaces in batches
            self._process_spaces_in_batches(space_ids, scraper_config["batch_size"])
            
            # Mark scraping as completed
            self.state_manager.complete()
            logging.info("Spaces scraping completed successfully")
            
            # Log summary
            self._log_summary()
            
        except KeyboardInterrupt:
            logging.info("Scraping interrupted by user. State saved for resuming.")
            self.state_manager.pause()
        except Exception as e:
            logging.error(f"Error in scraping process: {e}", exc_info=True)
            self.state_manager.pause()
    
    def _discover_spaces(self, max_spaces: Optional[int] = None) -> List[str]:
        """
        Discover spaces from HuggingFace Hub.
        
        Args:
            max_spaces: Maximum number of spaces to discover
            
        Returns:
            List of space IDs
        """
        logging.info(f"Discovering spaces from HuggingFace Hub (limit: {max_spaces or 'unlimited'})...")
        
        space_ids = []
        
        try:
            # Define a rate-limited space discovery function
            def get_spaces_page():
                return list_spaces(limit=max_spaces)
            
            # Use rate limiter for the API call
            spaces_iterator = self.rate_limiter.execute_with_rate_limit(
                "list_spaces", get_spaces_page
            )
            
            # Extract space IDs
            for space_info in spaces_iterator:
                space_ids.append(space_info.id)
                
                # Check if we've reached the limit
                if max_spaces and len(space_ids) >= max_spaces:
                    break
            
            # Update state with total discovered
            self.state_manager.set_total_discovered(len(space_ids))
            logging.info(f"Discovered {len(space_ids)} spaces")
            
            return space_ids
            
        except Exception as e:
            logging.error(f"Error discovering spaces: {e}", exc_info=True)
            return []
    
    def _process_spaces_in_batches(self, space_ids: List[str], batch_size: int = 100) -> None:
        """
        Process spaces in batches.
        
        Args:
            space_ids: List of space IDs to process
            batch_size: Size of each batch
        """
        if not space_ids:
            logging.warning("No spaces to process")
            return
        
        # Start from the current position in state
        current_position = self.state_manager.state["current_position"]
        
        # Calculate number of batches
        total_spaces = len(space_ids)
        num_batches = (total_spaces + batch_size - 1) // batch_size
        
        logging.info(f"Processing {total_spaces} spaces in {num_batches} batches of {batch_size}")
        
        for batch_idx in range(current_position // batch_size, num_batches):
            # Check if scraping is paused
            if self.state_manager.is_paused():
                logging.info("Scraping is paused. Stopping batch processing.")
                break
            
            # Calculate batch range
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_spaces)
            batch_space_ids = space_ids[start_idx:end_idx]
            
            # Update state with current batch
            self.state_manager.set_current_batch(batch_space_ids)
            self.state_manager.update_position(start_idx)
            
            logging.info(f"Processing batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx-1})")
            
            # Process batch
            self._process_batch(batch_space_ids)
            
            # Create checkpoint after each batch
            self.state_manager.create_checkpoint()
    
    def _process_batch(self, space_ids: List[str]) -> None:
        """
        Process a batch of spaces.
        
        Args:
            space_ids: List of space IDs to process
        """
        # Get configuration values
        scraper_config = self.config.config["scraper"]
        skip_existing = scraper_config["skip_existing"]
        
        # Filter out already processed spaces if skipping is enabled
        if skip_existing:
            unprocessed_ids = [
                space_id for space_id in space_ids
                if not self.state_manager.is_model_processed(space_id)
            ]
            if len(unprocessed_ids) < len(space_ids):
                logging.info(f"Skipping {len(space_ids) - len(unprocessed_ids)} already processed spaces")
            space_ids = unprocessed_ids
        
        # Create progress bar
        with tqdm(total=len(space_ids), desc="Processing spaces") as pbar:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all tasks
                future_to_space = {
                    executor.submit(self._process_space, space_id): space_id
                    for space_id in space_ids
                }
                
                # Process results as they complete
                for future in as_completed(future_to_space):
                    space_id = future_to_space[future]
                    try:
                        # Get result (success/failure)
                        success = future.result()
                        if success:
                            self.state_manager.mark_model_completed(space_id)
                    except Exception as e:
                        logging.error(f"Unhandled exception processing space {space_id}: {e}", exc_info=True)
                        self.state_manager.mark_model_errored(space_id, str(e))
                    
                    # Update progress bar
                    pbar.update(1)
    
    def _process_space(self, space_id: str) -> bool:
        """
        Process a single space.
        
        Args:
            space_id: Space ID to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        # Get configuration values
        scraper_config = self.config.config["scraper"]
        output_dir = scraper_config["output_dir"]
        save_metadata = scraper_config["save_metadata"]
        
        logging.info(f"Processing space: {space_id}")
        
        try:
            # Mark as processed in state
            self.state_manager.mark_model_processed(space_id)
            
            # Create space directory
            safe_space_dirname = space_id.replace("/", "__")
            space_save_path = os.path.join(output_dir, "spaces", safe_space_dirname)
            os.makedirs(space_save_path, exist_ok=True)
            
            # 1. Get and save metadata
            if save_metadata:
                success = self._save_space_metadata(space_id, space_save_path)
                if not success:
                    self.state_manager.mark_model_errored(space_id, "Failed to save metadata")
                    return False
            
            # 2. Download space thumbnail
            success = self._download_space_thumbnail(space_id, space_save_path)
            if not success:
                logging.warning(f"Failed to download thumbnail for {space_id}")
                # Continue anyway - thumbnail is optional
            
            # 3. Store in IPFS if enabled
            if self.config.config["storage"]["use_ipfs"] and self.ipfs_storage.is_ipfs_available():
                cid = self.ipfs_storage.store_model_files(space_save_path)
                if cid:
                    logging.info(f"Space {space_id} stored in IPFS with CID: {cid}")
                    
                    # Save CID to space directory for reference
                    with open(os.path.join(space_save_path, "ipfs_cid.txt"), "w") as f:
                        f.write(cid)
                else:
                    logging.warning(f"Failed to store space {space_id} in IPFS")
            
            # 4. Extract relationship information if enabled
            if self.config.config["provenance"]["extract_dataset_relationships"]:
                self._extract_space_relationships(space_id, space_save_path)
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing space {space_id}: {e}", exc_info=True)
            self.state_manager.mark_model_errored(space_id, str(e))
            return False
    
    def _save_space_metadata(self, space_id: str, space_save_path: str) -> bool:
        """
        Save space metadata.
        
        Args:
            space_id: Space ID
            space_save_path: Path to save space files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define a rate-limited metadata fetch function
            def fetch_space_info():
                return self.hf_client.space_info(space_id)
            
            # Use rate limiter for the API call
            space_info = self.rate_limiter.execute_with_rate_limit(
                "space_info", fetch_space_info
            )
            
            # Convert space info to dictionary
            metadata_dict = {}
            for attr, value in space_info.__dict__.items():
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
                    logging.warning(f"Could not serialize attribute '{attr}' for {space_id}: {e}")
            
            # Save metadata to file
            metadata_filepath = os.path.join(space_save_path, "metadata.json")
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved metadata for {space_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving metadata for {space_id}: {e}", exc_info=True)
            return False
    
    def _download_space_thumbnail(self, space_id: str, space_save_path: str) -> bool:
        """
        Download space thumbnail.
        
        Args:
            space_id: Space ID
            space_save_path: Path to save space files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # This would require fetching the thumbnail URL from space info
            # and downloading it. For now, just create a placeholder.
            thumbnail_info = {
                "space_id": space_id,
                "thumbnail_url": f"https://huggingface.co/spaces/{space_id}/thumbnail",
                "note": "Thumbnail URL - actual download not implemented in this version"
            }
            
            thumbnail_info_path = os.path.join(space_save_path, "thumbnail_info.json")
            with open(thumbnail_info_path, 'w', encoding='utf-8') as f:
                json.dump(thumbnail_info, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved thumbnail info for {space_id}")
            return True
                
        except Exception as e:
            logging.error(f"Error saving thumbnail info for {space_id}: {e}", exc_info=True)
            return False
    
    def _extract_space_relationships(self, space_id: str, space_save_path: str) -> bool:
        """
        Extract space relationships to models and datasets.
        
        Args:
            space_id: Space ID
            space_save_path: Path to save space files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Attempt to extract relationship information
            # This is a simplified version - actual implementation would parse
            # space configuration files to identify models and datasets used
            relationships_info = {
                "space_id": space_id,
                "models_used": [],  # Would contain models used by this space
                "datasets_used": [],  # Would contain datasets used by this space
                "timestamp": time.time()
            }
            
            # Save relationships info
            relationships_filepath = os.path.join(space_save_path, "relationships.json")
            with open(relationships_filepath, 'w', encoding='utf-8') as f:
                json.dump(relationships_info, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved relationships info for {space_id}")
            return True
                
        except Exception as e:
            logging.error(f"Error extracting relationships for {space_id}: {e}", exc_info=True)
            return False
    
    def _log_summary(self) -> None:
        """Log a summary of the scraping operation."""
        progress = self.state_manager.get_progress()
        
        logging.info("=== Spaces Scraping Summary ===")
        logging.info(f"Total spaces discovered: {progress['total_models_discovered']}")
        logging.info(f"Spaces processed: {progress['models_processed']}")
        logging.info(f"Spaces completed successfully: {progress['models_completed']}")
        logging.info(f"Spaces with errors: {progress['models_errored']}")
        logging.info(f"Completion percentage: {progress['completion_percentage']:.2f}%")
        logging.info(f"Elapsed time: {progress['elapsed_time'] / 60:.2f} minutes")
        logging.info("=======================")
    
    def resume(self) -> None:
        """Resume a paused scraping operation."""
        if not self.state_manager.is_paused():
            logging.warning("No paused scraping operation to resume")
            return
        
        logging.info("Resuming spaces scraping operation")
        self.state_manager.resume()
        
        # Re-discover spaces or use cached list
        max_spaces = self.config.config["scraper"]["max_spaces"]
        cached_total = self.state_manager.state["total_models_discovered"]
        
        if cached_total > 0:
            # Rediscover only if we need more spaces than we have already discovered
            if max_spaces is None or cached_total < max_spaces:
                space_ids = self._discover_spaces(max_spaces)
            else:
                logging.info(f"Using {cached_total} already discovered spaces")
                # We don't actually have the space IDs stored, so we need to rediscover
                space_ids = self._discover_spaces(max_spaces)
        else:
            space_ids = self._discover_spaces(max_spaces)
        
        # Resume processing from the current position
        self._process_spaces_in_batches(space_ids, self.config.config["scraper"]["batch_size"])
        
        # Mark as completed if we finished all batches
        if not self.state_manager.is_paused():
            self.state_manager.complete()
            logging.info("Resumed spaces scraping completed successfully")
            
        # Log summary
        self._log_summary()

# Main entry point
def main(config_path: Optional[str] = None, max_spaces: Optional[int] = None) -> None:
    """
    Main entry point for the spaces scraper.
    
    Args:
        config_path: Path to the configuration file (optional)
        max_spaces: Maximum number of spaces to scrape (optional)
    """
    scraper = SpacesScraper(config_path)
    scraper.scrape_spaces(max_spaces)

if __name__ == "__main__":
    main()