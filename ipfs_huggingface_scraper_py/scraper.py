import os
import json
import logging
import time
from huggingface_hub import list_models, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HFValidationError

# --- Configuration ---
OUTPUT_DIR = "hf_model_data"  # Directory to save configs and metadata
MAX_MODELS_TO_FETCH = 50      # Limit the number of models to process (set to None for no limit, but be careful!)
SAVE_METADATA = True          # Set to True to save model metadata as metadata.json
FILENAME_TO_DOWNLOAD = "config.json" # The specific config file we want
RETRY_DELAY_SECONDS = 5       # Delay before retrying a download after certain errors
MAX_RETRIES = 2               # Max retries for downloads

# --- Setup Logging ---
# Create output directory early if it doesn't exist, so log file can be placed there
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logging.basicConfig(
    level=logging.INFO, # Change to logging.DEBUG for more verbose serialization logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "scraper.log")),
        logging.StreamHandler()
    ]
)

# --- Main Function ---
def scrape_hf_models_and_configs(output_dir, max_models=None, save_metadata=True, filename=FILENAME_TO_DOWNLOAD):
    """
    Lists models from Hugging Face Hub and downloads their config file and optionally metadata.

    Args:
        output_dir (str): The directory where data will be saved.
        max_models (int, optional): Maximum number of models to process. Defaults to None (no limit).
        save_metadata (bool): Whether to save model metadata as metadata.json. Defaults to True.
        filename (str): The specific file to download from each model repo (e.g., "config.json").
    """
    logging.info(f"Starting Hugging Face Hub scraper. Output directory: '{output_dir}'")

    config_download_count = 0
    metadata_save_count = 0
    models_processed = 0
    skipped_no_config = 0
    skipped_repo_not_found = 0
    skipped_other_error = 0

    try:
        logging.info(f"Fetching list of models (limit: {max_models if max_models else 'None'})...")
        model_iterable = list_models(limit=max_models, full=save_metadata, cardData=save_metadata, fetch_config=False)

        total_models_str = str(max_models) if max_models else '?'
        for model_info in model_iterable:
            models_processed += 1
            model_id = model_info.id
            logging.info(f"--- Processing model {models_processed}/{total_models_str} ('{model_id}') ---")

            safe_model_dirname = model_id.replace("/", "__")
            model_save_path = os.path.join(output_dir, safe_model_dirname)
            os.makedirs(model_save_path, exist_ok=True)

            # 1. Save Metadata (Optional)
            if save_metadata:
                metadata_filepath = os.path.join(model_save_path, "metadata.json")
                metadata_dict = {} # Initialize fresh dict for each model
                try:
                    # Convert ModelInfo object to a dictionary for JSON serialization
                    for attr, value in model_info.__dict__.items():
                         if attr == 'siblings' and value is not None:
                             # Handle siblings specifically: extract rfilename (serializable string)
                             try:
                                 metadata_dict[attr] = [
                                     sib.rfilename
                                     for sib in value
                                     if hasattr(sib, 'rfilename')
                                 ]
                                 logging.debug(f" Serialized 'siblings' for {model_id} using rfilename.")
                             except Exception as e:
                                 logging.warning(f" Could not serialize 'siblings' for {model_id} using rfilename: {e}. Trying string conversion.")
                                 try:
                                     metadata_dict[attr] = [str(sib) for sib in value]
                                 except Exception as e2:
                                     logging.error(f"  String conversion failed for siblings of {model_id}: {e2}. Skipping attribute.")
                                     # Optionally set to None or skip adding the key
                                     # metadata_dict[attr] = None

                         elif hasattr(value, 'isoformat'): # Datetime objects
                              metadata_dict[attr] = value.isoformat()
                         elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                              # Basic serializable types
                              metadata_dict[attr] = value
                         else:
                              # Attempt to convert other unknown objects to string as a fallback
                              try:
                                  metadata_dict[attr] = str(value)
                                  logging.debug(f" Converted attribute '{attr}' of type {type(value)} to string for {model_id}")
                              except Exception:
                                  logging.debug(f" Skipping non-serializable attribute '{attr}' of type {type(value)} for {model_id}")


                    with open(metadata_filepath, 'w', encoding='utf-8') as f:
                        json.dump(metadata_dict, f, indent=4, ensure_ascii=False)
                    metadata_save_count += 1
                    logging.info(f"  Saved metadata to {metadata_filepath}")

                except TypeError as e:
                    # This block might still be hit if another unexpected type occurs
                    logging.error(f"  Failed to serialize metadata for {model_id} due to TypeError: {e}. Trying to save partial basic data...")
                    partial_metadata = {
                        k: v for k, v in metadata_dict.items()
                        if isinstance(v, (str, int, float, bool, list, dict, type(None))) # Strict basic types filter
                    }
                    try:
                        partial_filepath = metadata_filepath + ".partial"
                        with open(partial_filepath, 'w', encoding='utf-8') as f:
                             json.dump(partial_metadata, f, indent=4, ensure_ascii=False)
                        logging.info(f"  Saved partial basic metadata to {partial_filepath}")
                    except Exception as dump_e:
                         logging.error(f"   Could not save even partial basic metadata for {model_id}: {dump_e}")

                except Exception as e:
                    logging.error(f"  An unexpected error occurred during metadata saving for {model_id}: {type(e).__name__} - {e}", exc_info=True)


            # 2. Download the specified config file (e.g., config.json)
            config_filepath = os.path.join(model_save_path, filename)
            retries = 0
            while retries <= MAX_RETRIES:
                try:
                    logging.info(f"  Attempting to download '{filename}' (Attempt {retries + 1}/{MAX_RETRIES + 1})...")
                    # Suppress the specific UserWarning about local_dir_use_symlinks if desired
                    # import warnings
                    # with warnings.catch_warnings():
                    #     warnings.filterwarnings("ignore", message="`local_dir_use_symlinks` parameter is deprecated")
                    downloaded_path = hf_hub_download(
                            repo_id=model_id,
                            filename=filename,
                            repo_type="model",
                            local_dir=model_save_path,
                            local_dir_use_symlinks=False, # Keep False even if deprecated warning shows
                        )
                    config_download_count += 1
                    logging.info(f"  Successfully downloaded '{filename}' to {config_filepath}")
                    break

                except RepositoryNotFoundError:
                    logging.warning(f"  Repository '{model_id}' not found (maybe private or deleted). Skipping.")
                    skipped_repo_not_found += 1
                    break
                except EntryNotFoundError:
                    logging.warning(f"  '{filename}' not found in repository '{model_id}'. Skipping.")
                    skipped_no_config += 1
                    break
                except HFValidationError as e:
                     logging.error(f"  Validation Error for {model_id}/{filename}: {e}. Skipping.")
                     skipped_other_error += 1
                     break
                except Exception as e:
                    # Consider adding specific checks for common network errors if needed
                    # e.g. requests.exceptions.ConnectionError, requests.exceptions.Timeout
                    logging.error(f"  Error downloading '{filename}' for {model_id}: {type(e).__name__} - {e}")
                    retries += 1
                    if retries <= MAX_RETRIES:
                        logging.info(f"  Retrying in {RETRY_DELAY_SECONDS} seconds...")
                        time.sleep(RETRY_DELAY_SECONDS)
                    else:
                        logging.error(f"  Max retries reached for {model_id}. Skipping file.")
                        skipped_other_error += 1
                        break

            # Optional delay
            # time.sleep(0.1)

    except Exception as e:
        logging.error(f"An critical error occurred during the main scraping loop: {e}", exc_info=True)

    finally:
        logging.info("--- Scraping Summary ---")
        logging.info(f"Total models processed: {models_processed}")
        if save_metadata:
            logging.info(f"Metadata files saved successfully: {metadata_save_count}")
            # Add count of partial saves if that logic exists and is used
        logging.info(f"'{filename}' files downloaded: {config_download_count}")
        logging.info(f"Models skipped (no '{filename}'): {skipped_no_config}")
        logging.info(f"Models skipped (repo not found): {skipped_repo_not_found}")
        logging.info(f"Models skipped (other download errors): {skipped_other_error}")
        logging.info(f"Data saved in directory: '{output_dir}'")
        logging.info(f"Log file saved to: '{os.path.join(output_dir, 'scraper.log')}'")
        logging.info("------------------------")

# --- Run the Scraper ---
if __name__ == "__main__":
    scrape_hf_models_and_configs(
        output_dir=OUTPUT_DIR,
        max_models=MAX_MODELS_TO_FETCH,
        save_metadata=SAVE_METADATA,
        filename=FILENAME_TO_DOWNLOAD
    )
