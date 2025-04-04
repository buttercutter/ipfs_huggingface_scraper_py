import os
import json
import time
import logging
import threading
from typing import Dict, List, Set, Any, Optional

class StateManager:
    """
    Manages the state of scraping operations, enabling resumable scraping.
    
    This class provides:
    - Checkpointing for long-running operations
    - State persistence through JSON files
    - Tracking of processed models and their statuses
    - Resumption of interrupted operations
    - Concurrent state updates with thread safety
    """
    
    def __init__(self, state_dir: str, state_file: str = "scraper_state.json"):
        """
        Initialize the state manager.
        
        Args:
            state_dir: Directory to store state files
            state_file: Filename for the main state file
        """
        self.state_dir = state_dir
        self.state_file_path = os.path.join(state_dir, state_file)
        self.checkpoint_file_path = os.path.join(state_dir, f"checkpoint_{state_file}")
        self.lock = threading.RLock()
        
        # Ensure state directory exists
        os.makedirs(state_dir, exist_ok=True)
        
        # Initialize state
        self.state = {
            "started_at": time.time(),
            "last_updated": time.time(),
            "total_models_discovered": 0,
            "models_processed": 0,
            "models_completed": 0,
            "models_errored": 0,
            "processed_model_ids": set(),
            "completed_model_ids": set(),
            "errored_model_ids": {},  # model_id -> error message
            "current_batch": [],
            "current_position": 0,  # Position in the overall model list
            "is_paused": False,
            "is_completed": False,
            "config": {
                "max_models": None,
                "save_metadata": True,
                "filename": "config.json",
                "batch_size": 100
            }
        }
        
        # Load existing state if available
        self._load_state()
        
    def _load_state(self) -> None:
        """Load state from disk if available."""
        try:
            if os.path.exists(self.state_file_path):
                with open(self.state_file_path, 'r') as f:
                    saved_state = json.load(f)
                
                # Convert lists back to sets for faster lookups
                if "processed_model_ids" in saved_state:
                    saved_state["processed_model_ids"] = set(saved_state["processed_model_ids"])
                if "completed_model_ids" in saved_state:
                    saved_state["completed_model_ids"] = set(saved_state["completed_model_ids"])
                
                # Update state with saved values
                self.state.update(saved_state)
                logging.info(f"Loaded existing state with {self.state['models_completed']} completed models")
            else:
                logging.info("No existing state found. Starting fresh.")
        except Exception as e:
            logging.error(f"Error loading state: {e}. Starting with fresh state.")
            
            # Try to load from checkpoint if available
            try:
                if os.path.exists(self.checkpoint_file_path):
                    with open(self.checkpoint_file_path, 'r') as f:
                        saved_state = json.load(f)
                    
                    # Convert lists back to sets
                    if "processed_model_ids" in saved_state:
                        saved_state["processed_model_ids"] = set(saved_state["processed_model_ids"])
                    if "completed_model_ids" in saved_state:
                        saved_state["completed_model_ids"] = set(saved_state["completed_model_ids"])
                    
                    self.state.update(saved_state)
                    logging.info(f"Recovered state from checkpoint with {self.state['models_completed']} completed models")
            except Exception as e2:
                logging.error(f"Error loading checkpoint: {e2}. Using fresh state.")
    
    def _save_state(self, is_checkpoint: bool = False) -> None:
        """
        Save current state to disk.
        
        Args:
            is_checkpoint: If True, save to the checkpoint file instead of main state file
        """
        try:
            # Update timestamp
            self.state["last_updated"] = time.time()
            
            # Convert sets to lists for JSON serialization
            serializable_state = self.state.copy()
            serializable_state["processed_model_ids"] = list(self.state["processed_model_ids"])
            serializable_state["completed_model_ids"] = list(self.state["completed_model_ids"])
            
            # Determine which file to write to
            file_path = self.checkpoint_file_path if is_checkpoint else self.state_file_path
            
            # Write to a temporary file first, then rename to avoid corruption
            temp_file_path = f"{file_path}.tmp"
            with open(temp_file_path, 'w') as f:
                json.dump(serializable_state, f, indent=2)
            
            # Atomic rename to minimize corruption risk
            os.replace(temp_file_path, file_path)
            
            if not is_checkpoint:
                logging.debug(f"State saved with {self.state['models_completed']} completed models")
        except Exception as e:
            logging.error(f"Error saving state: {e}")
    
    def update_config(self, **kwargs) -> None:
        """
        Update scraper configuration parameters.
        
        Args:
            **kwargs: Configuration key-value pairs to update
        """
        with self.lock:
            self.state["config"].update(kwargs)
            self._save_state()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current scraper configuration."""
        with self.lock:
            return self.state["config"].copy()
    
    def mark_model_processed(self, model_id: str) -> None:
        """
        Mark a model as processed (attempted).
        
        Args:
            model_id: The Hugging Face model ID
        """
        with self.lock:
            if model_id not in self.state["processed_model_ids"]:
                self.state["processed_model_ids"].add(model_id)
                self.state["models_processed"] += 1
                
                # Save state every 10 models
                if self.state["models_processed"] % 10 == 0:
                    self._save_state()
                # Create checkpoint every 50 models
                if self.state["models_processed"] % 50 == 0:
                    self._save_state(is_checkpoint=True)
    
    def mark_model_completed(self, model_id: str) -> None:
        """
        Mark a model as successfully completed.
        
        Args:
            model_id: The Hugging Face model ID
        """
        with self.lock:
            # First ensure it's marked as processed
            if model_id not in self.state["processed_model_ids"]:
                self.mark_model_processed(model_id)
            
            # Then mark as completed
            if model_id not in self.state["completed_model_ids"]:
                self.state["completed_model_ids"].add(model_id)
                self.state["models_completed"] += 1
                
                # Remove from errors if it was there
                if model_id in self.state["errored_model_ids"]:
                    del self.state["errored_model_ids"][model_id]
                
                # Save state every 10 completed models
                if self.state["models_completed"] % 10 == 0:
                    self._save_state()
    
    def mark_model_errored(self, model_id: str, error_message: str) -> None:
        """
        Mark a model as failed with an error.
        
        Args:
            model_id: The Hugging Face model ID
            error_message: Description of the error
        """
        with self.lock:
            # First ensure it's marked as processed
            if model_id not in self.state["processed_model_ids"]:
                self.mark_model_processed(model_id)
            
            # Then record the error
            self.state["errored_model_ids"][model_id] = error_message
            
            # Only increment error count if this is first time we're recording this error
            if model_id not in self.state.get("errored_model_ids", {}):
                self.state["models_errored"] += 1
                
            # Save state on errors
            self._save_state()
    
    def is_model_processed(self, model_id: str) -> bool:
        """
        Check if a model has been processed (attempted).
        
        Args:
            model_id: The Hugging Face model ID
            
        Returns:
            True if the model has been processed, False otherwise
        """
        with self.lock:
            return model_id in self.state["processed_model_ids"]
    
    def is_model_completed(self, model_id: str) -> bool:
        """
        Check if a model has been successfully completed.
        
        Args:
            model_id: The Hugging Face model ID
            
        Returns:
            True if the model has been completed, False otherwise
        """
        with self.lock:
            return model_id in self.state["completed_model_ids"]
    
    def get_model_error(self, model_id: str) -> Optional[str]:
        """
        Get the error message for a model if it failed.
        
        Args:
            model_id: The Hugging Face model ID
            
        Returns:
            Error message string or None if no error
        """
        with self.lock:
            return self.state["errored_model_ids"].get(model_id)
    
    def set_current_batch(self, batch: List[str]) -> None:
        """
        Set the current batch of models being processed.
        
        Args:
            batch: List of model IDs in the current batch
        """
        with self.lock:
            self.state["current_batch"] = batch
            self._save_state()
    
    def update_position(self, position: int) -> None:
        """
        Update the current position in the overall model list.
        
        Args:
            position: Current position
        """
        with self.lock:
            self.state["current_position"] = position
            self._save_state()
    
    def set_total_discovered(self, count: int) -> None:
        """
        Set the total number of models discovered.
        
        Args:
            count: Total count of models
        """
        with self.lock:
            self.state["total_models_discovered"] = count
            self._save_state()
    
    def pause(self) -> None:
        """Pause the scraping operation."""
        with self.lock:
            self.state["is_paused"] = True
            self._save_state()
    
    def resume(self) -> None:
        """Resume the scraping operation."""
        with self.lock:
            self.state["is_paused"] = False
            self._save_state()
    
    def complete(self) -> None:
        """Mark the scraping operation as completed."""
        with self.lock:
            self.state["is_completed"] = True
            self._save_state()
    
    def is_paused(self) -> bool:
        """Check if scraping is paused."""
        with self.lock:
            return self.state["is_paused"]
    
    def is_completed(self) -> bool:
        """Check if scraping is completed."""
        with self.lock:
            return self.state["is_completed"]
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get a summary of current progress.
        
        Returns:
            Dictionary with progress statistics
        """
        with self.lock:
            return {
                "started_at": self.state["started_at"],
                "elapsed_time": time.time() - self.state["started_at"],
                "total_models_discovered": self.state["total_models_discovered"],
                "models_processed": self.state["models_processed"],
                "models_completed": self.state["models_completed"],
                "models_errored": self.state["models_errored"],
                "current_position": self.state["current_position"],
                "is_paused": self.state["is_paused"],
                "is_completed": self.state["is_completed"],
                "completion_percentage": self._calculate_completion_percentage()
            }
    
    def _calculate_completion_percentage(self) -> float:
        """
        Calculate the percentage of completion.
        
        Returns:
            Percentage as a float between 0 and 100
        """
        total = self.state["total_models_discovered"]
        if total == 0:
            return 0.0
        
        processed = self.state["models_processed"]
        return min(100.0, (processed / total) * 100)
    
    def get_unprocessed_models(self) -> List[str]:
        """
        Get list of models that need to be processed.
        
        Returns:
            List of model IDs that haven't been processed yet
        """
        with self.lock:
            if not self.state["current_batch"]:
                return []
            
            return [
                model_id for model_id in self.state["current_batch"]
                if model_id not in self.state["processed_model_ids"]
            ]
    
    def get_errored_models(self) -> Dict[str, str]:
        """
        Get all models that encountered errors.
        
        Returns:
            Dictionary mapping model IDs to error messages
        """
        with self.lock:
            return self.state["errored_model_ids"].copy()
    
    def get_completed_models(self) -> Set[str]:
        """
        Get all successfully completed models.
        
        Returns:
            Set of completed model IDs
        """
        with self.lock:
            return self.state["completed_model_ids"].copy()
    
    def reset(self) -> None:
        """Reset the state to start fresh."""
        with self.lock:
            # Save config before reset
            config = self.state["config"].copy()
            
            # Reset to initial state
            self.state = {
                "started_at": time.time(),
                "last_updated": time.time(),
                "total_models_discovered": 0,
                "models_processed": 0,
                "models_completed": 0,
                "models_errored": 0,
                "processed_model_ids": set(),
                "completed_model_ids": set(),
                "errored_model_ids": {},
                "current_batch": [],
                "current_position": 0,
                "is_paused": False,
                "is_completed": False,
                "config": config  # Restore config
            }
            
            self._save_state()
            
    def create_checkpoint(self) -> None:
        """Manually create a checkpoint of the current state."""
        with self.lock:
            self._save_state(is_checkpoint=True)
            logging.info(f"Manual checkpoint created at position {self.state['current_position']}")