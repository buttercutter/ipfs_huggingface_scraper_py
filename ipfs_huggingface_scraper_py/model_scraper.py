import os
import json
import logging
import time
from pathlib import Path
import shutil
import glob
from datetime import datetime
from huggingface_hub import list_models
from huggingface_hub.utils import HFValidationError

# --- Configuration --- (Same as before)
GDRIVE_MOUNT_POINT = "/content/drive/MyDrive"; GDRIVE_FOLDER_NAME = "hf_metadata_dataset_collection"; LOCAL_FOLDER_NAME = "hf_metadata_dataset_local_fallback"; LOCAL_WORK_DIR = Path("/content/hf_scraper_work"); LOCAL_TEMP_DATA_FILENAME = "temp_metadata_output.jsonl"; LOCAL_TEMP_LOG_FILENAME = "temp_metadata_scraper.log"; FINAL_MERGED_FILENAME = "all_models_metadata.jsonl"; FINAL_LOG_FILENAME = "metadata_scraper.log"; BACKUP_DATA_FILE_PREFIX = "metadata_backup_"; BACKUP_DATA_FILE_GLOB = f"{BACKUP_DATA_FILE_PREFIX}*.jsonl"; MAX_MODELS_TO_FETCH = None; FETCH_CARD_DATA = True; DELAY_BETWEEN_MODELS = 0.05; LOAD_RETRIES = 2; LOAD_RETRY_DELAY = 3; BACKUP_EVERY_N_RECORDS = 200

# --- Helper Functions ---
# (make_serializable, safe_serialize_dict, load_processed_ids, sync_local_to_drive
#  remain EXACTLY the same as the previous FULL version)
def make_serializable(obj):
    """Converts common non-serializable types found in ModelInfo."""
    if hasattr(obj, 'isoformat'): return obj.isoformat()
    if hasattr(obj, 'rfilename'): return obj.rfilename
    try: return str(obj)
    except Exception: return None

def safe_serialize_dict(data_dict):
    """Attempts to serialize a dictionary, handling non-serializable items."""
    serializable_dict = {}
    if not isinstance(data_dict, dict): logging.warning(f"safe_serialize_dict non-dict input: {type(data_dict)}"); return {}
    for key, value in data_dict.items():
        if isinstance(value, (list, tuple)): serializable_dict[key] = [make_serializable(item) for item in value]
        elif isinstance(value, dict): serializable_dict[key] = safe_serialize_dict(value)
        elif isinstance(value, (str, int, float, bool, type(None))): serializable_dict[key] = value
        else: serializable_dict[key] = make_serializable(value)
    return {k: v for k, v in serializable_dict.items() if v is not None or (k in data_dict and data_dict[k] is None)}


def load_processed_ids(main_filepath: Path, backup_dir: Path, backup_glob: str, retries: int = LOAD_RETRIES, delay: int = LOAD_RETRY_DELAY) -> set:
    """Reads the main JSON Lines file AND all backup files to get processed IDs."""
    processed_ids = set(); total_lines_read = 0; total_files_read = 0; files_to_read = []
    if main_filepath.is_file(): files_to_read.append(main_filepath); logging.info(f"Found main file: {main_filepath}")
    else: logging.info(f"Main file not found: {main_filepath}. Relying on backups.")
    backup_pattern = str(backup_dir / backup_glob); backup_files = sorted(glob.glob(backup_pattern))
    if backup_files: logging.info(f"Found {len(backup_files)} backup file(s)."); files_to_read.extend([Path(f) for f in backup_files])
    else: logging.info(f"No backup files found matching '{backup_pattern}'.")
    if not files_to_read: logging.info("No main/backup files found. Starting fresh."); return set()
    logging.info(f"Loading IDs from {len(files_to_read)} file(s)...")
    for filepath in files_to_read:
        logging.info(f"Reading: {filepath.name}"); file_lines_read = 0; skipped_lines = 0; loaded_from_this_file = 0; initial_set_size = len(processed_ids)
        for attempt in range(retries):
             try:
                with filepath.open('r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        file_lines_read = i + 1; line = line.strip();
                        if not line: continue
                        try: data = json.loads(line)
                        except json.JSONDecodeError: skipped_lines += 1; continue
                        if 'id' in data and isinstance(data['id'], str): processed_ids.add(data['id'])
                        else: skipped_lines += 1
                    loaded_from_this_file = len(processed_ids) - initial_set_size; total_lines_read += file_lines_read; total_files_read += 1
                    logging.info(f"  -> Read {file_lines_read} lines, loaded {loaded_from_this_file} new IDs (skipped {skipped_lines}). Total IDs: {len(processed_ids)}"); break
             except FileNotFoundError: logging.warning(f"Attempt {attempt + 1}: File not found: {filepath}");
             except IOError as e: logging.error(f"IOError reading {filepath}: {e}. Skip file.", exc_info=True); break
             except Exception as e: logging.error(f"Error reading {filepath}: {e}. Skip file.", exc_info=True); break
             if attempt < retries - 1: time.sleep(delay);
             else: logging.error(f"Failed opening {filepath} after {retries} attempts. Skip file."); break
    logging.info(f"Finished loading. Total unique IDs: {len(processed_ids)} from {total_files_read} file(s).")
    return processed_ids

def sync_local_to_drive(local_data_path: Path, local_log_path: Path, final_backup_dir: Path, final_log_path: Path, backup_data_prefix: str):
    """Copies local temp data to backup on Drive. Copies local log to overwrite final log on Drive."""
    data_backup_success = False; log_copy_success = False
    # 1. Backup Data File
    if local_data_path.is_file() and local_data_path.stat().st_size > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f"); backup_filename = f"{backup_data_prefix}{timestamp}.jsonl"; backup_filepath = final_backup_dir / backup_filename
        try:
            logging.info(f"Backing up '{local_data_path}' ({local_data_path.stat().st_size / 1024**2:.2f} MB) to '{backup_filepath}'...")
            final_backup_dir.mkdir(parents=True, exist_ok=True); shutil.copyfile(local_data_path, backup_filepath); logging.info(f"Data backup successful: '{backup_filepath}'."); data_backup_success = True
            try: logging.info(f"Clearing local temp data file '{local_data_path}'."); local_data_path.unlink()
            except Exception as e_unlink: logging.warning(f"Could not clear local temp data file after backup: {e_unlink}")
        except Exception as e_data: logging.error(f"CRITICAL: Failed to back up data '{local_data_path}' to '{backup_filepath}': {e_data}", exc_info=True); logging.warning("Local temp data file NOT cleared.")
    else: logging.debug("Local temp data empty/non-existent. Skip data backup."); data_backup_success = True
    # 2. Copy/Overwrite Log File
    if local_log_path.is_file() and local_log_path.stat().st_size > 0:
        try:
            logging.info(f"Copying local log '{local_log_path}' ({local_log_path.stat().st_size / 1024**2:.2f} MB) to overwrite '{final_log_path}'...")
            final_log_path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(local_log_path, final_log_path); logging.info(f"Log file copy successful to '{final_log_path}'."); log_copy_success = True
        except Exception as e_log: logging.error(f"CRITICAL: Failed to copy log '{local_log_path}' to '{final_log_path}': {e_log}", exc_info=True)
    else: logging.debug("Local temp log empty/non-existent. Skip log copy."); log_copy_success = True
    return data_backup_success and log_copy_success


# --- Main Scraping Function ---
def scrape_all_hf_metadata(
    final_output_filepath: Path,
    final_backup_dir: Path,
    local_temp_data_path: Path,
    local_temp_log_path: Path,
    max_models=None,
    fetch_card_data=True,
    delay=0.05,
    backup_every=BACKUP_EVERY_N_RECORDS
    ):
    # --- Logging Setup (Points to LOCAL temp log path initially) ---
    log_file_handler = logging.FileHandler(local_temp_log_path)
    log_stream_handler = logging.StreamHandler()
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[log_file_handler, log_stream_handler])
    logging.getLogger('huggingface_hub.repocard_data').setLevel(logging.ERROR)
    final_log_filepath = final_backup_dir / FINAL_LOG_FILENAME # Define final log path

    # --- Start & Config Logging ---
    logging.info(f"--- Starting Aggregation Run ---")
    logging.info(f"Final backup/log directory: '{final_backup_dir}'")
    logging.info(f"Using local temp data file: '{local_temp_data_path}'")
    logging.info(f"Using local temp log file: '{local_temp_log_path}'")
    logging.info(f"Creating backup file every {backup_every} records.")
    # ... other config logs ...

    # --- Load processed IDs ---
    processed_ids = load_processed_ids(final_output_filepath, final_backup_dir, BACKUP_DATA_FILE_GLOB)
    initial_processed_count = len(processed_ids); logging.info(f"Resuming. {initial_processed_count} models previously processed.")

    # --- Initialize Counters & Fetch List ---
    models_processed_total_iterator = 0; models_saved_this_run_in_temp = 0; models_failed_serialization = 0; models_skipped_resume = 0;
    total_models_str = '?'; model_iterator = None; start_time = None; log_file_descriptor = -1;

    try:
        # Get log file descriptor
        try: log_file_descriptor = log_file_handler.stream.fileno()
        except Exception as log_fd_err: logging.warning(f"Could not get log file descriptor: {log_fd_err}")

        # --- Fetch Model List ---
        logging.info("Fetching model list...");
        try: model_iterator = list_models(limit=max_models, full=True, cardData=fetch_card_data, iterator=True); logging.info("Using iterator.")
        except TypeError as e:
             if 'iterator' in str(e): logging.warning("No iterator support. Falling back."); model_iterable_fallback = list_models(limit=max_models, full=True, cardData=fetch_card_data); model_iterator = model_iterable_fallback; logging.info("Fetched using fallback.")
             else: raise e
        if model_iterator is None: logging.error("Failed to get model list. Aborting."); return

        # --- Main Processing Loop ---
        start_time = time.time() # Start timer before the loop begins
        f_local_data = None     # Initialize file handle outside loop

        for model_info in model_iterator:
            # <<< Open/Reopen local temp file if needed >>>
            if f_local_data is None or f_local_data.closed:
                logging.info(f"Opening local temp data file '{local_temp_data_path}' for append ('a').")
                try:
                    f_local_data = local_temp_data_path.open('a', encoding='utf-8')
                    logging.info("Opened local temp data file successfully.")
                except IOError as e:
                    logging.error(f"CRITICAL IO error opening local temp data file '{local_temp_data_path}': {e}", exc_info=True)
                    break # Cannot continue without temp file

            # Process model
            models_processed_total_iterator += 1; model_id = model_info.id if hasattr(model_info, 'id') else 'UNKNOWN_ID'
            if model_id != 'UNKNOWN_ID' and model_id in processed_ids:
                models_skipped_resume += 1;
                if models_skipped_resume % 10000 == 1 : logging.info(f"Skipping '{model_id}' (processed). Total skipped: {models_skipped_resume}.")
                continue

            # Log Progress
            current_processed_count = models_processed_total_iterator - models_skipped_resume
            if current_processed_count % 250 == 1 and current_processed_count > 1:
                 if start_time: elapsed_time = time.time() - start_time; rate = (current_processed_count / elapsed_time) * 60 if elapsed_time > 0 else 0
                 logging.info(f"--- Progress: {current_processed_count} new (Iter: {models_processed_total_iterator}/{total_models_str}, Skip: {models_skipped_resume}) | In Temp: {models_saved_this_run_in_temp % backup_every} | Rate: {rate:.1f} new/min ---") # Show count *in current temp batch*

            # Process and Save to LOCAL TEMP
            try:
                # ... (Serialization logic - same as before) ...
                raw_metadata = model_info.__dict__; serializable_metadata = safe_serialize_dict(raw_metadata); json_record = json.dumps(serializable_metadata, ensure_ascii=False)

                f_local_data.write(json_record + '\n');
                models_saved_this_run_in_temp += 1 # Increment total count this run

                # <<< Periodic Backup/Sync (with close/reopen) >>>
                # Check using total count this run
                if models_saved_this_run_in_temp > 0 and models_saved_this_run_in_temp % backup_every == 0:
                    logging.info(f"Reached {backup_every} records in current batch ({models_saved_this_run_in_temp} total this run). Triggering backup/sync.");

                    # <<< Close local file BEFORE sync >>>
                    logging.debug("Closing local temp data file before sync...")
                    f_local_data.close()
                    logging.debug("Local temp data file closed.")

                    # Flush local log buffer
                    log_file_handler.flush()

                    # Perform the backup/sync
                    sync_successful = sync_local_to_drive(
                        local_temp_data_path,
                        local_temp_log_path,
                        final_backup_dir,
                        final_log_filepath,
                        BACKUP_DATA_FILE_PREFIX
                    )

                    if sync_successful:
                        logging.info(f"Periodic backup/sync successful. Local temp data file was cleared by sync function.")
                        # File handle `f_local_data` is already closed and file unlinked
                        # It will be reopened at the start of the next iteration
                    else:
                        logging.warning("Periodic backup/sync failed. Local temp files NOT cleared. Will retry next interval.")
                        # <<< Reopen local file immediately if sync failed to continue writing >>>
                        logging.info("Reopening local temp data file after failed sync...")
                        try:
                             f_local_data = local_temp_data_path.open('a', encoding='utf-8')
                             logging.info("Reopened local temp data file.")
                        except IOError as e_reopen:
                             logging.error(f"CRITICAL: Failed to reopen local temp file after failed sync: {e_reopen}. Aborting loop.", exc_info=True)
                             break # Exit loop if we can't reopen

                    # Optional: Sync log file descriptor if available
                    if log_file_descriptor != -1:
                         try: os.fsync(log_file_descriptor)
                         except Exception as log_sync_err: logging.error(f"Error syncing log file descriptor: {log_sync_err}")

            # Handle processing/write errors
            except Exception as e:
                models_failed_serialization += 1; logging.error(f"Processing/write local error for '{model_id}': {type(e).__name__}. Skip.", exc_info=False)

            # Delay
            if delay > 0: time.sleep(delay)
        # --- End For Loop ---

    # Handle interruptions/errors
    except KeyboardInterrupt: logging.warning("Scraping interrupted by user.")
    except Exception as e: logging.error(f"CRITICAL error during scraping: {e}", exc_info=True)

    finally:
        logging.info("--- Aggregation Summary (End of Run) ---")

        # <<< Close open file handle if loop exited >>>
        if f_local_data is not None and not f_local_data.closed:
            logging.info("Closing local temp data file handle before final sync.")
            f_local_data.close()

        # --- Final Backup/Sync ---
        logging.info("Attempting final backup/sync of local files...")
        final_sync_successful = sync_local_to_drive(
            local_temp_data_path,
            local_temp_log_path,
            final_backup_dir,
            final_log_filepath,
            BACKUP_DATA_FILE_PREFIX
        )
        # ... (rest of finally block and summary logging - same as before) ...
        if final_sync_successful: logging.info("Final backup/sync successful.")
        else: logging.warning("Final backup/sync FAILED.")
        final_total_count = initial_processed_count + models_saved_this_run_in_temp # Count based on temp file writes
        try: log_file_handler.flush(); # Only flush, fsync might fail if already closed
        except Exception as final_log_flush_e: logging.error(f"Error during final log flush: {final_log_flush_e}")
        if start_time:
             end_time = time.time(); total_time = end_time - start_time;
             logging.info(f"Total models encountered: {models_processed_total_iterator}"); logging.info(f"Models skipped (resume): {models_skipped_resume}"); logging.info(f"Models attempted this run: {models_processed_total_iterator - models_skipped_resume}");
             remaining_in_temp = local_temp_data_path.stat().st_size if local_temp_data_path.is_file() else 0
             logging.info(f"Records written locally this run: {models_saved_this_run_in_temp} ({remaining_in_temp} bytes left in temp if final sync failed)")
             logging.info(f"Total records saved to Drive (estimated): {final_total_count if final_sync_successful else initial_processed_count}") # More accurate estimate
             logging.info(f"Models skipped (errors): {models_failed_serialization}"); logging.info(f"Total time this run: {total_time:.2f} seconds");
        else: logging.info("Scraping did not reach main processing loop.")
        logging.info(f"Final Log file destination: '{final_log_filepath}'"); logging.info("--------------------------");
        log_file_handler.close();
        try:
             if local_temp_data_path.is_file(): local_temp_data_path.unlink(); logging.info(f"Cleaned up local temp data: {local_temp_data_path}")
             if local_temp_log_path.is_file(): local_temp_log_path.unlink(); logging.info(f"Cleaned up local temp log: {local_temp_log_path}")
        except Exception as clean_e: logging.warning(f"Could not clean up local temp files: {clean_e}")


# --- Script Execution (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    # --- Determine Output Paths ---
    print("--- Determining Output Paths ---")
    # ... (Path determination logic remains the same) ...
    gdrive_base = Path(GDRIVE_MOUNT_POINT); gdrive_target_dir = gdrive_base / GDRIVE_FOLDER_NAME; local_target_dir = Path(LOCAL_FOLDER_NAME); effective_final_dir = None;
    print(f"Checking for GDrive: {gdrive_base}");
    if gdrive_base.is_dir() and gdrive_base.exists():
        print(f"Mount OK. Checking target: {gdrive_target_dir}");
        if gdrive_target_dir.is_dir():
             print(f"Target Google Drive directory found. Using Google Drive.")
             effective_final_dir = gdrive_target_dir
        else:
             print(f"Target Google Drive directory '{gdrive_target_dir}' not found. Will attempt to create.")
             try:
                  gdrive_target_dir.mkdir(parents=True, exist_ok=True)
                  print(f"Successfully created Google Drive directory.")
                  effective_final_dir = gdrive_target_dir
             except Exception as e:
                  print(f"Error creating Google Drive directory '{gdrive_target_dir}': {e}")
                  print("Falling back to local directory.")
                  effective_final_dir = local_target_dir

    effective_final_dir.mkdir(parents=True, exist_ok=True); LOCAL_WORK_DIR.mkdir(parents=True, exist_ok=True); print(f"Effective final destination directory: {effective_final_dir}");
    final_output_filepath = effective_final_dir / FINAL_MERGED_FILENAME; final_log_filepath = effective_final_dir / FINAL_LOG_FILENAME; local_temp_data_path = LOCAL_WORK_DIR / LOCAL_TEMP_DATA_FILENAME; local_temp_log_path = LOCAL_WORK_DIR / LOCAL_TEMP_LOG_FILENAME;
    print(f"Main data file path (for loading IDs): {final_output_filepath}"); print(f"Backup data files pattern: {effective_final_dir / BACKUP_DATA_FILE_GLOB}"); print(f"Final log file path: {final_log_filepath}"); print(f"Local temp data file path: {local_temp_data_path}"); print(f"Local temp log file path: {local_temp_log_path}"); print("-" * 30);

    # Remove existing local temp files before start
    if local_temp_data_path.exists():
      print(f"Removing existing local temp data file: {local_temp_data_path}")
      try:
          local_temp_data_path.unlink()
      except OSError as e: print(f"Warning: Could not remove temp data file: {e}")

    if local_temp_log_path.exists():
      print(f"Removing existing local temp log file: {local_temp_log_path}")
      try:
          local_temp_log_path.unlink()
      except OSError as e: print(f"Warning: Could not remove temp log file: {e}")

    # --- Run the Scraper ---
    scrape_all_hf_metadata(
        final_output_filepath=final_output_filepath,
        final_backup_dir=effective_final_dir, # Corrected parameter name
        local_temp_data_path=local_temp_data_path,
        local_temp_log_path=local_temp_log_path,
        max_models=MAX_MODELS_TO_FETCH,
        fetch_card_data=FETCH_CARD_DATA,
        delay=DELAY_BETWEEN_MODELS,
        backup_every=BACKUP_EVERY_N_RECORDS
    )


# --- Optional: Post-Scraping Merge ---
print("\n--- Optional: Merging backup files ---")
backup_files_to_merge = sorted(glob.glob(str(effective_final_dir / BACKUP_DATA_FILE_GLOB)))
if backup_files_to_merge:
    print(f"Found {len(backup_files_to_merge)} backup files to merge into {final_output_filepath}")
    try:
      # Open the main file in append mode for merging
      with final_output_filepath.open('a', encoding='utf-8') as f_main:
          for backup_file in backup_files_to_merge:
              backup_path = Path(backup_file)
              print(f"Merging: {backup_path.name}...")
              # Open backup file in read mode
              with backup_path.open('r', encoding='utf-8') as f_backup:
                  shutil.copyfileobj(f_backup, f_main) # Efficiently copy content
                  # Delete the backup file after successful merge
                  backup_path.unlink()
                  print(f"Merged and deleted {backup_path.name}")
                  print("All backup files merged successfully.")
    except Exception as merge_e:
        print(f"Error during merge process: {merge_e}. Backup files were not deleted.")
else:
    print("No backup files found to merge.")
