import os
import json
import logging
import time
import traceback
from pathlib import Path
import shutil
import glob
from datetime import datetime
from tqdm.auto import tqdm
from typing import Optional, Union, Set, Dict, List, Tuple

# Hugging Face related
from huggingface_hub import list_models, hf_hub_download, HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HFValidationError

# Data handling and Parquet
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Embeddings
from sentence_transformers import SentenceTransformer
import torch

# --- IPFS CID Generation Code (from provided ipfs_multiformats.py) ---
import hashlib
from multiformats import CID, multihash
import tempfile
import sys

class ipfs_multiformats_py:
    def __init__(self, resources=None, metadata=None):
        self.multihash = multihash
        # Added error handling for multihash version/import
        if not hasattr(self.multihash, 'wrap') or not hasattr(self.multihash, 'decode'):
             logging.warning("Multihash library structure might have changed. CID generation may fail.")
        return None

    def get_file_sha256(self, file_path):
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.digest()
        except Exception as e:
            logging.error(f"Error hashing file {file_path}: {e}")
            return None

    # Takes bytes input directly
    def get_bytes_sha256(self, data_bytes: bytes):
        hasher = hashlib.sha256()
        hasher.update(data_bytes)
        return hasher.digest()

    def get_multihash_sha256(self, content_hash):
        if content_hash is None:
            return None
        try:
            # Try using multihash.digest instead of wrap
            mh = self.multihash.digest(content_hash, 'sha2-256')
            return mh
        except Exception as e:
            logging.error(f"Error creating multihash: {e}")
            return None

    def get_multihash_sha256_old(self, content_hash):
        if content_hash is None:
            return None
        try:
            # Use 'sha2-256' which corresponds to code 0x12
            #mh = self.multihash.wrap(code='sha2-256', digest=content_hash)
            mh = self.multihash.wrap('sha2-256', content_hash)
            return mh
        except Exception as e:
            logging.error(f"Error wrapping hash in multihash: {e}")
            return None

    def get_cid_old(self, data):
        """Generates CID v1 base32 for bytes data or file path."""
        mh = None
        try:
            if isinstance(data, (str, Path)) and os.path.isfile(data):
                # logging.debug(f"Calculating CID for file: {data}")
                file_content_hash = self.get_file_sha256(data)
                mh = self.get_multihash_sha256(file_content_hash)
            elif isinstance(data, bytes):
                # logging.debug(f"Calculating CID for bytes (length: {len(data)})")
                bytes_hash = self.get_bytes_sha256(data)
                mh = self.get_multihash_sha256(bytes_hash)
            elif isinstance(data, str):
                # logging.debug(f"Calculating CID for string (length: {len(data)})")
                # Treat string as UTF-8 bytes
                bytes_hash = self.get_bytes_sha256(data.encode('utf-8'))
                mh = self.get_multihash_sha256(bytes_hash)
            else:
                logging.warning(f"Unsupported data type for CID generation: {type(data)}. Skipping CID.")
                return None

            if mh:
                # CIDv1, base32, raw codec (0x55)
                cid = CID(base='base32', version=1, codec='raw', multihash=mh)
                return str(cid)
            else:
                return None
        except Exception as e:
            logging.error(f"Error generating CID: {e}", exc_info=False)
            return None

    def get_cid(self, data):
        """Generates CID v1 base32 for bytes data or file path."""
        try:
            # Get the hash first
            content_hash = None
            if isinstance(data, (str, Path)) and os.path.isfile(data):
                content_hash = self.get_file_sha256(data)
            elif isinstance(data, bytes):
                content_hash = self.get_bytes_sha256(data)
            elif isinstance(data, str):
                content_hash = self.get_bytes_sha256(data.encode('utf-8'))
            else:
                logging.warning(f"Unsupported data type for CID generation: {type(data)}. Skipping CID.")
                return None

            if not content_hash:
                return None

            # Try the new CID API format
            try:
                # Version 1 of multiformats may use from_digest or other method instead of passing multihash directly
                from multiformats import multihash
                digest = multihash.digest(content_hash, 'sha2-256')
                cid = CID.from_digest(digest, 'raw')  # Try this format first
                return str(cid)
            except (AttributeError, TypeError):
                try:
                    # Try alternate creation method
                    mh = self.get_multihash_sha256(content_hash)
                    cid = CID(version=1, codec='raw', hash=mh)  # Try with hash parameter
                    return str(cid)
                except:
                    # Fallback to simple base64 encoding if CID creation fails
                    import base64
                    b64_hash = base64.b64encode(content_hash).decode('ascii')
                    return f"sha256:{b64_hash}"

        except Exception as e:
            logging.error(f"Error generating CID: {e}", exc_info=False)
            # Fallback to a simple hash representation
            try:
                if isinstance(data, (str, Path)) and os.path.isfile(data):
                    content_hash = self.get_file_sha256(data)
                elif isinstance(data, bytes):
                    content_hash = self.get_bytes_sha256(data)
                elif isinstance(data, str):
                    content_hash = self.get_bytes_sha256(data.encode('utf-8'))
                else:
                    return None

                import base64
                return f"sha256:{base64.b64encode(content_hash).decode('ascii')}"
            except:
                return None
# --- End IPFS CID Code ---


# --- Configuration ---
# --- Paths ---
GDRIVE_MOUNT_POINT = "/content/drive/MyDrive"
GDRIVE_FOLDER_NAME = "hf_metadata_dataset_collection"
LOCAL_FOLDER_NAME = "./hf_metadata_dataset_local_fallback"
LOCAL_WORK_DIR = Path("./hf_embedding_work")

# Input JSONL File
INPUT_JSONL_FILENAME = "all_models_metadata.jsonl" # Assumed in final dir

# --- Output File Names ---
# Final Destination (Drive/Local Fallback)
FINAL_METADATA_PARQUET_FILENAME = "model_metadata.parquet" # Metadata + CIDs
FINAL_EMBEDDINGS_PARQUET_FILENAME = "model_embeddings.parquet" # CIDs + Embeddings
FINAL_LOG_FILENAME = "embedding_generator.log"

# Local Temporary Files (in LOCAL_WORK_DIR)
LOCAL_TEMP_METADATA_PARQUET = "temp_model_metadata.parquet"
LOCAL_TEMP_EMBEDDINGS_PARQUET = "temp_model_embeddings.parquet"
LOCAL_TEMP_LOG_FILENAME = "temp_embedding_generator.log"

# --- Batch Configuration ---
BATCH_SAVE_THRESHOLD = 1000  # Save after processing this many records
BATCH_SAVE_DIR_NAME = "batch_files"  # Subdirectory for batch files
PERIODIC_MERGE_FREQUENCY = 5  # Merge to Google Drive every X batches (0 to disable)
CLEAN_AFTER_PERIODIC_MERGE = True  # Whether to clean up batch files after periodic merge

# --- Processing Config ---
MAX_RECORDS_TO_PROCESS = None # Limit records from JSONL (for testing), None for all
BATCH_SIZE = 1024
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# Control what gets embedded and CID generated
PROCESS_CONFIG_JSON = True
PROCESS_README_CONTENT = True

# --- Hub Upload Config ---
UPLOAD_TO_HUB = True
TARGET_REPO_ID = "YourUsername/your-dataset-repo-name" # CHANGE THIS
TARGET_REPO_TYPE = "dataset"
METADATA_FILENAME_IN_REPO = "model_metadata.parquet"
EMBEDDINGS_FILENAME_IN_REPO = "model_embeddings.parquet"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Functions ---
def make_serializable(obj):
    """Converts common non-serializable types found in ModelInfo."""
    if hasattr(obj, 'isoformat'): return obj.isoformat()
    if hasattr(obj, 'rfilename'): return obj.rfilename
    try: return str(obj)
    except Exception: return None

def safe_serialize_dict(data_dict):
    """Attempts to serialize a dictionary, handling non-serializable items."""
    # This function might not be needed if we read directly from JSONL,
    # but keep it for potential future use or if handling raw ModelInfo objects.
    serializable_dict = {}
    if not isinstance(data_dict, dict): logging.warning(f"safe_serialize_dict non-dict input: {type(data_dict)}"); return {}
    for key, value in data_dict.items():
        if isinstance(value, (list, tuple)): serializable_dict[key] = [make_serializable(item) for item in value]
        elif isinstance(value, dict): serializable_dict[key] = safe_serialize_dict(value)
        elif isinstance(value, (str, int, float, bool, type(None))): serializable_dict[key] = value
        else: serializable_dict[key] = make_serializable(value)
    return {k: v for k, v in serializable_dict.items() if v is not None or (k in data_dict and data_dict[k] is None)}

# --- NEW: Generate Record CID Function ---
def generate_record_cid(cid_generator, model_id: str, config_cid: Optional[str] = None, readme_cid: Optional[str] = None) -> str:
    """
    Generate a primary record CID from model_id and available content CIDs.
    This will be used as the primary key for both Parquet files.
    """
    # Create a base string that combines all available IDs
    cid_parts = [f"model:{model_id}"]
    if config_cid:
        cid_parts.append(f"config:{config_cid}")
    if readme_cid:
        cid_parts.append(f"readme:{readme_cid}")

    # Join all parts and generate a CID from the combined string
    combined_string = "|".join(cid_parts)
    return cid_generator.get_cid(combined_string)

# --- Safe Parquet Saving Function ---
def save_dataframe_to_parquet_safely(df, filepath):
    """Saves DataFrame to Parquet with explicit schema handling for mixed types."""
    try:
        # First attempt: Convert known problematic columns to string
        df_safe = df.copy()

        # Handle the 'gated' column specifically which caused the original error
        if 'gated' in df_safe.columns:
            df_safe['gated'] = df_safe['gated'].astype(str)

        # Convert all object columns except model_id and record_cid to string to be safe
        for col in df_safe.select_dtypes(include=['object']).columns:
            if col not in ['model_id', 'record_cid', 'config_cid', 'readme_cid']:  # Keep IDs as is
                df_safe[col] = df_safe[col].astype(str)

        # Try saving with pandas
        df_safe.to_parquet(filepath, index=False)
        return True

    except Exception as e:
        logging.warning(f"First attempt to save Parquet failed: {e}")

        try:
            # Second attempt: Use PyArrow with explicit schema
            schema = pa.Schema.from_pandas(df)
            fields = list(schema)

            # Convert all string/binary fields to string type except IDs
            for i, field in enumerate(fields):
                if (pa.types.is_string(field.type) or pa.types.is_binary(field.type)) and \
                   field.name not in ['model_id', 'record_cid', 'config_cid', 'readme_cid']:
                    fields[i] = pa.field(field.name, pa.string())

            new_schema = pa.schema(fields)

            # Force conversion of problematic columns
            df_safe = df.copy()
            for col in df_safe.select_dtypes(include=['object']).columns:
                if col not in ['model_id', 'record_cid', 'config_cid', 'readme_cid']:
                    df_safe[col] = df_safe[col].astype(str)

            # Convert to table with schema and write
            table = pa.Table.from_pandas(df_safe, schema=new_schema)
            pq.write_table(table, filepath)
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

# --- UPDATED: Load Processed CIDs from EMBEDDINGS Parquet and Batch Files ---
def load_processed_cids_from_parquet(filepath: Path, batch_dir: Optional[Path] = None) -> set:
    """
    Reads the record_cid column from:
    1. The final EMBEDDINGS Parquet file
    2. Any batch files in the batch_dir, if provided
    3. Also checks for CSV fallback files

    Returns a set of processed record_cids.
    """
    processed_cids = set()

    # 1. Load from final Parquet if it exists
    if filepath.is_file():
        logging.info(f"Found existing EMBEDDINGS Parquet: {filepath}. Loading processed CIDs...")
        try:
            # Only load the record_cid column for efficiency
            df_existing = pd.read_parquet(filepath, columns=['record_cid'])
            file_cids = set(df_existing['record_cid'].tolist())
            processed_cids.update(file_cids)
            logging.info(f"Loaded {len(file_cids)} CIDs from existing Embeddings Parquet.")
        except Exception as e:
            logging.warning(f"Could not load 'record_cid' from '{filepath}': {e}. Will check for CSV fallback.")
            # Check for CSV fallback
            csv_filepath = filepath.with_suffix('.csv')
            if csv_filepath.is_file():
                try:
                    df_csv = pd.read_csv(csv_filepath, usecols=['record_cid'])
                    csv_cids = set(df_csv['record_cid'].tolist())
                    processed_cids.update(csv_cids)
                    logging.info(f"Loaded {len(csv_cids)} CIDs from CSV fallback: {csv_filepath}")
                except Exception as csv_e:
                    logging.warning(f"Could not load CIDs from CSV fallback: {csv_e}")

    # 2. Load from batch files if provided
    if batch_dir and batch_dir.is_dir():
        # Check both Parquet and CSV batch files
        batch_files_parquet = list(batch_dir.glob("embeddings_batch_*.parquet"))
        batch_files_csv = list(batch_dir.glob("embeddings_batch_*.csv"))

        if batch_files_parquet:
            logging.info(f"Found {len(batch_files_parquet)} embedding batch Parquet files.")
            batch_cids_count = 0

            for batch_file in batch_files_parquet:
                try:
                    df_batch = pd.read_parquet(batch_file, columns=['record_cid'])
                    batch_cids = set(df_batch['record_cid'].tolist())
                    batch_cids_count += len(batch_cids)
                    processed_cids.update(batch_cids)
                except Exception as e:
                    logging.warning(f"Error loading CIDs from batch file {batch_file}: {e}")

            logging.info(f"Loaded {batch_cids_count} additional CIDs from Parquet batch files.")

        if batch_files_csv:
            logging.info(f"Found {len(batch_files_csv)} embedding batch CSV files.")
            csv_batch_cids_count = 0

            for batch_file in batch_files_csv:
                try:
                    df_batch = pd.read_csv(batch_file, usecols=['record_cid'])
                    batch_cids = set(df_batch['record_cid'].tolist())
                    csv_batch_cids_count += len(batch_cids)
                    processed_cids.update(batch_cids)
                except Exception as e:
                    logging.warning(f"Error loading CIDs from CSV batch file {batch_file}: {e}")

            logging.info(f"Loaded {csv_batch_cids_count} additional CIDs from CSV batch files.")

    total_cids = len(processed_cids)
    if total_cids > 0:
        logging.info(f"Total of {total_cids} unique record CIDs loaded for resume.")
    else:
        logging.info(f"No existing processed CIDs found. Will process all records.")

    return processed_cids

# --- Sync Local Files to Final Destination ---
def sync_local_files_to_final(
    local_metadata_path: Path,
    local_embeddings_path: Path,
    local_log_path: Path,
    final_metadata_path: Path,
    final_embeddings_path: Path,
    final_log_path: Path
    ):
    """
    Copies local Parquet/log files to overwrite final destination files.
    Returns True if all necessary copies succeeded.
    """
    success = True # Assume success initially

    # Copy Metadata Parquet or CSV
    if local_metadata_path.is_file():
        try:
            logging.info(f"Copying local Metadata '{local_metadata_path}' to '{final_metadata_path}'...")
            final_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_metadata_path, final_metadata_path)
            logging.info("Metadata file copy successful.")
        except Exception as e:
            logging.error(f"Failed to copy Metadata file: {e}", exc_info=True)
            success = False

        # Also check for CSV fallback
        csv_path = local_metadata_path.with_suffix('.csv')
        if csv_path.is_file():
            try:
                csv_dest = final_metadata_path.with_suffix('.csv')
                logging.info(f"Copying CSV fallback: {csv_path} to {csv_dest}")
                shutil.copyfile(csv_path, csv_dest)
            except Exception as e:
                logging.error(f"Failed to copy CSV fallback: {e}")
                # Don't affect overall success status for CSV fallback
    else:
        logging.debug("Local Metadata file non-existent. Skipping copy.")

    # Copy Embeddings Parquet or CSV
    if local_embeddings_path.is_file():
        try:
            logging.info(f"Copying local Embeddings '{local_embeddings_path}' to '{final_embeddings_path}'...")
            final_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_embeddings_path, final_embeddings_path)
            logging.info("Embeddings file copy successful.")
        except Exception as e:
            logging.error(f"Failed to copy Embeddings file: {e}", exc_info=True)
            success = False

        # Also check for CSV fallback
        csv_path = local_embeddings_path.with_suffix('.csv')
        if csv_path.is_file():
            try:
                csv_dest = final_embeddings_path.with_suffix('.csv')
                logging.info(f"Copying CSV fallback: {csv_path} to {csv_dest}")
                shutil.copyfile(csv_path, csv_dest)
            except Exception as e:
                logging.error(f"Failed to copy CSV fallback: {e}")
                # Don't affect overall success status for CSV fallback
    else:
        logging.debug("Local Embeddings file non-existent. Skipping copy.")

    # Copy Log File
    if local_log_path.is_file() and local_log_path.stat().st_size > 0:
        try:
            logging.info(f"Copying local log '{local_log_path}' to overwrite '{final_log_path}'...")
            final_log_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_log_path, final_log_path)
            logging.info("Log file copy successful.")
        except Exception as e:
            logging.error(f"Failed to copy log file: {e}", exc_info=True)
            success = False # Log copy fail is less critical but still indicate
    else:
        logging.debug("Local temp log empty/non-existent. Skipping log copy.")

    return success

# --- Periodic Merge Function ---
def perform_periodic_merge(
    batch_save_dir: Path,
    merged_batch_tracker: Set[str],
    local_temp_metadata_path: Path,
    local_temp_embeddings_path: Path,
    final_metadata_path: Path,
    final_embeddings_path: Path,
    final_log_path: Path,
    local_temp_log_path: Path
):
    """
    Merges unprocessed batch files and syncs to final destination.
    Returns number of batches merged.
    """
    # Find all batch files that haven't been merged yet
    meta_batch_files = []
    embed_batch_files = []

    # Check Parquet files
    for batch_file in batch_save_dir.glob("metadata_batch_*.parquet"):
        if batch_file.name not in merged_batch_tracker:
            meta_batch_files.append(batch_file)

    for batch_file in batch_save_dir.glob("embeddings_batch_*.parquet"):
        if batch_file.name not in merged_batch_tracker:
            embed_batch_files.append(batch_file)

    # Check CSV fallback files
    for batch_file in batch_save_dir.glob("metadata_batch_*.csv"):
        if batch_file.name not in merged_batch_tracker:
            meta_batch_files.append(batch_file)

    for batch_file in batch_save_dir.glob("embeddings_batch_*.csv"):
        if batch_file.name not in merged_batch_tracker:
            embed_batch_files.append(batch_file)

    if not meta_batch_files or not embed_batch_files:
        logging.info("No new batches to merge periodically.")
        return 0

    logging.info(f"Performing periodic merge of {len(meta_batch_files)} metadata files and {len(embed_batch_files)} embedding files")

    # Process metadata files
    try:
        # Load new batch files
        dfs_meta = []
        for batch_file in meta_batch_files:
            try:
                if batch_file.suffix.lower() == '.parquet':
                    df_batch = pd.read_parquet(batch_file)
                else:  # CSV
                    df_batch = pd.read_csv(batch_file)
                dfs_meta.append(df_batch)
                # Mark as processed
                merged_batch_tracker.add(batch_file.name)
            except Exception as e:
                logging.error(f"Error loading batch file {batch_file}: {e}")

        if dfs_meta:
            # Merge new batch data
            df_meta_new = pd.concat(dfs_meta, ignore_index=True)

            # Load existing file if it exists
            df_meta_merged = None
            if final_metadata_path.exists():
                try:
                    df_meta_existing = pd.read_parquet(final_metadata_path)
                    # Combine with new data, ensuring no duplicate record_cids
                    existing_cids = set(df_meta_existing['record_cid'])
                    df_meta_unique = df_meta_new[~df_meta_new['record_cid'].isin(existing_cids)]
                    df_meta_merged = pd.concat([df_meta_existing, df_meta_unique], ignore_index=True)
                except Exception as e:
                    logging.warning(f"Could not load existing metadata file: {e}. Using only new data.")
                    df_meta_merged = df_meta_new
            else:
                df_meta_merged = df_meta_new

            # Save merged metadata
            if df_meta_merged is not None:
                local_temp_metadata_path.parent.mkdir(parents=True, exist_ok=True)
                save_success = save_dataframe_to_parquet_safely(df_meta_merged, local_temp_metadata_path)
                if not save_success:
                    logging.warning("Metadata periodic merge saved as CSV fallback instead of Parquet")
    except Exception as e:
        logging.error(f"Error in periodic metadata merge: {e}", exc_info=True)

    # Process embeddings files
    try:
        # Load new batch files
        dfs_embed = []
        for batch_file in embed_batch_files:
            try:
                if batch_file.suffix.lower() == '.parquet':
                    df_batch = pd.read_parquet(batch_file)
                else:  # CSV
                    df_batch = pd.read_csv(batch_file)
                dfs_embed.append(df_batch)
                # Mark as processed
                merged_batch_tracker.add(batch_file.name)
            except Exception as e:
                logging.error(f"Error loading batch file {batch_file}: {e}")

        if dfs_embed:
            # Merge new batch data
            df_embed_new = pd.concat(dfs_embed, ignore_index=True)

            # Load existing file if it exists
            df_embed_merged = None
            if final_embeddings_path.exists():
                try:
                    df_embed_existing = pd.read_parquet(final_embeddings_path)
                    # Combine with new data, ensuring no duplicate record_cids
                    existing_cids = set(df_embed_existing['record_cid'])
                    df_embed_unique = df_embed_new[~df_embed_new['record_cid'].isin(existing_cids)]
                    df_embed_merged = pd.concat([df_embed_existing, df_embed_unique], ignore_index=True)
                except Exception as e:
                    logging.warning(f"Could not load existing embeddings file: {e}. Using only new data.")
                    df_embed_merged = df_embed_new
            else:
                df_embed_merged = df_embed_new

            # Save merged embeddings
            if df_embed_merged is not None:
                local_temp_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                save_success = save_dataframe_to_parquet_safely(df_embed_merged, local_temp_embeddings_path)
                if not save_success:
                    logging.warning("Embeddings periodic merge saved as CSV fallback instead of Parquet")
    except Exception as e:
        logging.error(f"Error in periodic embeddings merge: {e}", exc_info=True)

    # Sync to final destination
    sync_success = sync_local_files_to_final(
        local_metadata_path=local_temp_metadata_path,
        local_embeddings_path=local_temp_embeddings_path,
        local_log_path=local_temp_log_path,
        final_metadata_path=final_metadata_path,
        final_embeddings_path=final_embeddings_path,
        final_log_path=final_log_path
    )

    if sync_success:
        logging.info(f"Periodic merge: Successfully merged and synced {len(meta_batch_files)} batches to final destination")
        # Optionally clean up batch files that have been merged
        if CLEAN_AFTER_PERIODIC_MERGE:
            try:
                for batch_file in meta_batch_files + embed_batch_files:
                    if batch_file.exists():
                        batch_file.unlink()
                logging.info("Cleaned up batch files after periodic merge")
            except Exception as e:
                logging.warning(f"Could not clean up some batch files: {e}")
    else:
        logging.error("Periodic merge: Failed to sync to final destination")

    return len(meta_batch_files)

# --- UPDATED: Main Embedding Generation Function with CID-based Primary Key ---
def create_embedding_dataset(
    input_jsonl_filepath: Path,
    final_metadata_parquet_path: Path,
    final_embeddings_parquet_path: Path,
    local_temp_metadata_path: Path,
    local_temp_embeddings_path: Path,
    local_temp_log_path: Path,
    final_log_filepath: Path,
    max_records: Optional[int] = None,
    batch_size: int = 32,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    process_config: bool = PROCESS_CONFIG_JSON,
    process_readme: bool = PROCESS_README_CONTENT,
    ):
    """
    Reads metadata, generates CIDs & embeddings, combines, saves to TWO Parquet files locally,
    then copies to final destination. Now uses record_cid as the primary key in both Parquet files.

    Features incremental batch saving, safe Parquet saving, and periodic merge to prevent losing progress.
    """
    # --- Setup batch directory ---
    batch_save_dir = LOCAL_WORK_DIR / BATCH_SAVE_DIR_NAME
    batch_save_dir.mkdir(parents=True, exist_ok=True)

    # --- Configure logging to use the local temp log file ---
    log_file_handler = logging.FileHandler(local_temp_log_path)
    log_stream_handler = logging.StreamHandler()
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[log_file_handler, log_stream_handler])
    logging.getLogger('huggingface_hub.repocard_data').setLevel(logging.ERROR)

    # --- Log configuration ---
    logging.info(f"--- Starting Embedding Generation with CID as Primary Key ---")
    logging.info(f"Input JSONL: '{input_jsonl_filepath}'")
    logging.info(f"Final Metadata Parquet Output: '{final_metadata_parquet_path}'")
    logging.info(f"Final Embeddings Parquet Output: '{final_embeddings_parquet_path}'")
    logging.info(f"Local Temp Metadata: '{local_temp_metadata_path}'")
    logging.info(f"Local Temp Embeddings: '{local_temp_embeddings_path}'")
    logging.info(f"Batch Save Directory: '{batch_save_dir}'")
    logging.info(f"Batch Save Threshold: {BATCH_SAVE_THRESHOLD}")
    logging.info(f"Periodic Merge Frequency: {PERIODIC_MERGE_FREQUENCY} batches")
    logging.info(f"Clean After Periodic Merge: {CLEAN_AFTER_PERIODIC_MERGE}")
    logging.info(f"Local Temp Log: '{local_temp_log_path}'")
    logging.info(f"Final Log Output: '{final_log_filepath}'")
    logging.info(f"Embedding Model: '{embedding_model_name}', Batch Size: {batch_size}")
    logging.info(f"Process Config: {process_config}, Process README: {process_readme}")
    logging.info(f"Max Records: {'All' if max_records is None else max_records}")

    # --- Load Embedding Model ---
    try:
        logging.info(f"Loading embedding model: {embedding_model_name}")

        # Check for MPS (Apple Silicon GPU) availability first, then CUDA, then fall back to CPU
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            logging.info(f"Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = 'cuda'
            logging.info(f"Using NVIDIA GPU (CUDA)")
        else:
            device = 'cpu'
            logging.info(f"Using CPU (no GPU acceleration available)")

        model = SentenceTransformer(embedding_model_name, device=device)
        cid_generator = ipfs_multiformats_py() # Initialize CID generator
        logging.info("Embedding model & CID generator loaded.")
    except Exception as e:
        logging.error(f"Failed to load embedding model or init CID generator: {e}", exc_info=True)
        return None, None # Return None for both paths

    # --- Load processed CIDs (instead of model_ids) from EMBEDDINGS destination AND batch files ---
    processed_cids = load_processed_cids_from_parquet(
        filepath=final_embeddings_parquet_path,
        batch_dir=batch_save_dir
    )
    initial_processed_count = len(processed_cids)
    logging.info(f"Resuming from {initial_processed_count} records already processed.")

    # --- Batch saving and periodic merge setup ---
    batch_counter = 0
    records_since_last_save = 0
    merged_batch_tracker = set()  # Track which batch files have been merged during periodic merges

    # Keep a lookup of model_id to record_cid for this session
    # This helps us avoid recomputing CIDs for models we've seen but not yet saved
    model_id_to_record_cid = {}

    # --- Process JSONL File ---
    metadata_records_list = [] # Holds dicts for metadata parquet
    embeddings_records_list = [] # Holds dicts for embeddings parquet
    batch_inputs = []            # Holds tuples: (original_dict, config_text, readme_text)
    record_count_from_jsonl = 0; processed_count_this_run = 0; skipped_resume_count = 0; skipped_error_count = 0
    start_time = None

    try:
        logging.info(f"Opening input JSONL file: {input_jsonl_filepath}")
        start_time = time.time()
        with input_jsonl_filepath.open('r', encoding='utf-8') as f_jsonl:
            pbar = tqdm(f_jsonl, desc="Processing JSONL", unit="record")
            for line in pbar:
                record_count_from_jsonl += 1
                if max_records is not None and processed_count_this_run >= max_records: logging.info(f"Reached max_records limit ({max_records}). Stopping."); break

                try:
                    line = line.strip();
                    if not line: continue
                    data = json.loads(line) # Original metadata dictionary
                    model_id = data.get('id')
                    if not model_id or not isinstance(model_id, str): logging.warning(f"Skip record {record_count_from_jsonl}: missing/invalid 'id'."); skipped_error_count += 1; continue

                    # --- Extract text for embedding & CID generation ---
                    config_text = ""; config_cid = None; config_dict_or_str = data.get('config')
                    if process_config and config_dict_or_str is not None:
                        if isinstance(config_dict_or_str, dict):
                            try: config_text = json.dumps(config_dict_or_str, separators=(',', ':')); config_cid = cid_generator.get_cid(config_text) # Use compact string for CID
                            except TypeError: logging.warning(f"Cannot serialize config for {model_id}. Skip CID/embed.")
                        elif isinstance(config_dict_or_str, str): # Handle if config is already a string
                             config_text = config_dict_or_str; config_cid = cid_generator.get_cid(config_text)
                        else: logging.warning(f"Config for {model_id} type {type(config_dict_or_str)}. Skip CID/embed.")

                    readme_text = ""; readme_cid = None
                    if process_readme:
                        card_data = data.get('cardData')
                        if isinstance(card_data, dict): readme_text = card_data.get('text', '') or ''
                        elif isinstance(card_data, str): readme_text = card_data # If cardData itself is the string
                        if not readme_text and isinstance(data.get('description'), str): readme_text = data['description'] # Fallback
                        if readme_text: readme_cid = cid_generator.get_cid(readme_text)

                    # --- Generate record_cid (primary key) ---
                    record_cid = generate_record_cid(cid_generator, model_id, config_cid, readme_cid)

                    # Store in lookup for future reference
                    model_id_to_record_cid[model_id] = record_cid

                    # Skip if this record_cid has already been processed
                    if record_cid in processed_cids:
                        skipped_resume_count += 1
                        continue

                    processed_count_this_run += 1
                    pbar.set_postfix_str(f"Batching: {model_id}", refresh=True)

                    # Add to batch for embedding
                    batch_inputs.append((data, config_text, readme_text, config_cid, readme_cid, record_cid))

                    # --- Process Batch when full ---
                    if len(batch_inputs) >= batch_size:
                        pbar.set_postfix_str(f"Embedding batch ({len(batch_inputs)})...", refresh=True)
                        try:
                            original_data_batch = [item[0] for item in batch_inputs]
                            config_texts_batch = [item[1] for item in batch_inputs]
                            readme_texts_batch = [item[2] for item in batch_inputs]
                            config_cids_batch = [item[3] for item in batch_inputs]
                            readme_cids_batch = [item[4] for item in batch_inputs]
                            record_cids_batch = [item[5] for item in batch_inputs]

                            # Generate embeddings
                            config_embeddings = model.encode(config_texts_batch, batch_size=batch_size, show_progress_bar=False) if process_config else [None] * len(batch_inputs)
                            readme_embeddings = model.encode(readme_texts_batch, batch_size=batch_size, show_progress_bar=False) if process_readme else [None] * len(batch_inputs)

                            # --- Create records for BOTH Parquet files ---
                            for i, original_data in enumerate(original_data_batch):
                                current_model_id = original_data.get('id')
                                current_record_cid = record_cids_batch[i]
                                if not current_model_id or not current_record_cid: continue

                                # 1. Metadata Record
                                metadata_record = original_data.copy() # Start with all original metadata
                                # Remove bulky/embedded fields if they exist, keep CIDs
                                metadata_record.pop('config_embedding', None)
                                metadata_record.pop('readme_embedding', None)
                                # Add CIDs
                                metadata_record['record_cid'] = current_record_cid  # Primary key
                                if process_config: metadata_record['config_cid'] = config_cids_batch[i]
                                if process_readme: metadata_record['readme_cid'] = readme_cids_batch[i]
                                metadata_records_list.append(metadata_record)

                                # 2. Embedding Record
                                embedding_record = {
                                    'record_cid': current_record_cid,  # Primary key
                                    'model_id': current_model_id  # Keep model_id for reference
                                }
                                if process_config: embedding_record['config_embedding'] = config_embeddings[i].tolist() if config_texts_batch[i] else None
                                if process_readme: embedding_record['readme_embedding'] = readme_embeddings[i].tolist() if readme_texts_batch[i] else None
                                embeddings_records_list.append(embedding_record)

                                # Mark this record as processed to avoid reprocessing if script restarts
                                processed_cids.add(current_record_cid)

                                # Increment counter for batch saving
                                records_since_last_save += 1

                            logging.debug(f"Processed batch. Metadata size: {len(metadata_records_list)}, Embeddings size: {len(embeddings_records_list)}")

                            # --- Save batch if we've reached the threshold ---
                            if records_since_last_save >= BATCH_SAVE_THRESHOLD:
                                batch_counter += 1
                                timestamp = int(time.time())

                                # Save metadata batch with safe Parquet saving
                                meta_batch_file = batch_save_dir / f"metadata_batch_{batch_counter}_{timestamp}.parquet"
                                df_meta_batch = pd.DataFrame(metadata_records_list)
                                success_meta = save_dataframe_to_parquet_safely(df_meta_batch, meta_batch_file)

                                # Save embeddings batch
                                embed_batch_file = batch_save_dir / f"embeddings_batch_{batch_counter}_{timestamp}.parquet"
                                df_embed_batch = pd.DataFrame(embeddings_records_list)
                                success_embed = save_dataframe_to_parquet_safely(df_embed_batch, embed_batch_file)

                                if success_meta and success_embed:
                                    logging.info(f"Saved batch {batch_counter} with {len(embeddings_records_list)} records")
                                else:
                                    logging.warning(f"Batch {batch_counter} save had issues. Check logs.")

                                # Clear the lists to start a new batch and reset counter
                                metadata_records_list = []
                                embeddings_records_list = []
                                records_since_last_save = 0

                                # --- Periodic merge to final destination ---
                                if PERIODIC_MERGE_FREQUENCY > 0 and batch_counter % PERIODIC_MERGE_FREQUENCY == 0:
                                    pbar.set_postfix_str(f"Periodic merge to Google Drive...", refresh=True)
                                    batches_merged = perform_periodic_merge(
                                        batch_save_dir=batch_save_dir,
                                        merged_batch_tracker=merged_batch_tracker,
                                        local_temp_metadata_path=local_temp_metadata_path,
                                        local_temp_embeddings_path=local_temp_embeddings_path,
                                        final_metadata_path=final_metadata_parquet_path,
                                        final_embeddings_path=final_embeddings_parquet_path,
                                        final_log_path=final_log_filepath,
                                        local_temp_log_path=local_temp_log_path
                                    )
                                    pbar.set_postfix_str(f"Merged {batches_merged} batches to Google Drive", refresh=True)

                        except Exception as e_embed:
                             logging.error(f"Error embedding batch: {e_embed}", exc_info=True)
                             skipped_error_count += len(batch_inputs) # Count whole batch as skipped

                        batch_inputs = [] # Clear batch

                # Handle line processing errors
                except json.JSONDecodeError: logging.warning(f"Skip record {record_count_from_jsonl}: JSON decode error."); skipped_error_count += 1
                except Exception as e_line: logging.error(f"Skip record {record_count_from_jsonl}: Error - {e_line}", exc_info=False); skipped_error_count += 1
        # --- End reading JSONL file ---

        # --- Process Final Remaining Batch ---
        if batch_inputs:
            pbar.set_postfix_str(f"Embedding final batch ({len(batch_inputs)})...", refresh=True)
            try:
                 # (Similar embedding and record creation logic as in the main batch processing)
                 original_data_batch = [item[0] for item in batch_inputs]
                 config_texts_batch = [item[1] for item in batch_inputs]
                 readme_texts_batch = [item[2] for item in batch_inputs]
                 config_cids_batch = [item[3] for item in batch_inputs]
                 readme_cids_batch = [item[4] for item in batch_inputs]
                 record_cids_batch = [item[5] for item in batch_inputs]

                 config_embeddings = model.encode(config_texts_batch, batch_size=batch_size, show_progress_bar=False) if process_config else [None] * len(batch_inputs)
                 readme_embeddings = model.encode(readme_texts_batch, batch_size=batch_size, show_progress_bar=False) if process_readme else [None] * len(batch_inputs)

                 for i, original_data in enumerate(original_data_batch):
                     current_model_id = original_data.get('id')
                     current_record_cid = record_cids_batch[i]
                     if not current_model_id or not current_record_cid: continue

                     metadata_record = original_data.copy()
                     metadata_record.pop('config_embedding', None)
                     metadata_record.pop('readme_embedding', None)
                     metadata_record['record_cid'] = current_record_cid  # Primary key
                     if process_config: metadata_record['config_cid'] = config_cids_batch[i]
                     if process_readme: metadata_record['readme_cid'] = readme_cids_batch[i]
                     metadata_records_list.append(metadata_record)

                     embedding_record = {
                         'record_cid': current_record_cid,  # Primary key
                         'model_id': current_model_id  # Keep model_id for reference
                     }
                     if process_config: embedding_record['config_embedding'] = config_embeddings[i].tolist() if config_texts_batch[i] else None
                     if process_readme: embedding_record['readme_embedding'] = readme_embeddings[i].tolist() if readme_texts_batch[i] else None
                     embeddings_records_list.append(embedding_record)

                     # Mark as processed
                     processed_cids.add(current_record_cid)
                     records_since_last_save += 1

                 logging.debug(f"Processed final batch. Metadata size: {len(metadata_records_list)}, Embeddings size: {len(embeddings_records_list)}")
            except Exception as e_embed_final:
                logging.error(f"Error embedding final batch: {e_embed_final}", exc_info=True)
                skipped_error_count += len(batch_inputs)
        # --- End processing batches ---

        # --- Save any remaining records as a final batch ---
        if metadata_records_list:
            batch_counter += 1
            timestamp = int(time.time())

            # Save final metadata batch with safe Parquet saving
            meta_batch_file = batch_save_dir / f"metadata_batch_{batch_counter}_{timestamp}.parquet"
            df_meta_batch = pd.DataFrame(metadata_records_list)
            success_meta = save_dataframe_to_parquet_safely(df_meta_batch, meta_batch_file)

            # Save final embeddings batch
            embed_batch_file = batch_save_dir / f"embeddings_batch_{batch_counter}_{timestamp}.parquet"
            df_embed_batch = pd.DataFrame(embeddings_records_list)
            success_embed = save_dataframe_to_parquet_safely(df_embed_batch, embed_batch_file)

            if success_meta and success_embed:
                logging.info(f"Saved final batch {batch_counter} with {len(embeddings_records_list)} records")
            else:
                logging.warning(f"Final batch {batch_counter} save had issues. Check logs.")

            # Clear lists
            metadata_records_list = []
            embeddings_records_list = []

        pbar.close()
        logging.info("Finished processing records from JSONL.")

        # --- Merge all remaining batches into the final files ---
        saved_metadata_path = None
        saved_embeddings_path = None

        # Process any remaining batches that haven't been merged during periodic merges
        if PERIODIC_MERGE_FREQUENCY > 0:
            logging.info("Performing final merge of any remaining batches...")
            batches_merged = perform_periodic_merge(
                batch_save_dir=batch_save_dir,
                merged_batch_tracker=merged_batch_tracker,
                local_temp_metadata_path=local_temp_metadata_path,
                local_temp_embeddings_path=local_temp_embeddings_path,
                final_metadata_path=final_metadata_parquet_path,
                final_embeddings_path=final_embeddings_parquet_path,
                final_log_path=final_log_filepath,
                local_temp_log_path=local_temp_log_path
            )
            logging.info(f"Final merge: processed {batches_merged} remaining batches")

            # Set return paths to the actual files we're using
            if local_temp_metadata_path.exists():
                saved_metadata_path = local_temp_metadata_path
            elif local_temp_metadata_path.with_suffix('.csv').exists():
                saved_metadata_path = local_temp_metadata_path.with_suffix('.csv')

            if local_temp_embeddings_path.exists():
                saved_embeddings_path = local_temp_embeddings_path
            elif local_temp_embeddings_path.with_suffix('.csv').exists():
                saved_embeddings_path = local_temp_embeddings_path.with_suffix('.csv')

            return saved_metadata_path, saved_embeddings_path

        # --- Only do full manual merge if periodic merges were disabled ---
        # Find all batch files
        meta_batch_files = list(batch_save_dir.glob("metadata_batch_*.parquet"))
        embed_batch_files = list(batch_save_dir.glob("embeddings_batch_*.parquet"))

        # Also check for CSV fallbacks
        meta_batch_csv_files = list(batch_save_dir.glob("metadata_batch_*.csv"))
        embed_batch_csv_files = list(batch_save_dir.glob("embeddings_batch_*.csv"))

        meta_batch_files.extend(meta_batch_csv_files)
        embed_batch_files.extend(embed_batch_csv_files)

        if meta_batch_files:
            logging.info(f"Merging {len(meta_batch_files)} metadata batch files...")
            try:
                dfs_meta = []
                for batch_file in tqdm(meta_batch_files, desc="Loading metadata batches"):
                    try:
                        if batch_file.suffix.lower() == '.parquet':
                            df_batch = pd.read_parquet(batch_file)
                        else:  # CSV
                            df_batch = pd.read_csv(batch_file)
                        dfs_meta.append(df_batch)
                    except Exception as e_batch:
                        logging.error(f"Error loading batch file {batch_file}: {e_batch}")

                if dfs_meta:
                    df_meta_merged = pd.concat(dfs_meta, ignore_index=True)
                    logging.info(f"Saving merged Metadata DataFrame ({df_meta_merged.shape}) to: {local_temp_metadata_path}")
                    local_temp_metadata_path.parent.mkdir(parents=True, exist_ok=True)

                    # Use safe Parquet saving for the merged file
                    save_success = save_dataframe_to_parquet_safely(df_meta_merged, local_temp_metadata_path)
                    if save_success:
                        saved_metadata_path = local_temp_metadata_path
                        logging.info("Merged metadata Parquet saved locally.")
                    else:
                        logging.warning("Could not save merged metadata as Parquet. Check for CSV fallback.")
                        # The save_dataframe_to_parquet_safely function will have created a CSV fallback if Parquet failed
                        csv_path = local_temp_metadata_path.with_suffix('.csv')
                        if csv_path.exists():
                            saved_metadata_path = csv_path
                            logging.info(f"Using CSV fallback for metadata: {csv_path}")
            except Exception as e_merge_meta:
                logging.error(f"Failed to merge metadata batch files: {e_merge_meta}", exc_info=True)
        else:
            logging.warning("No metadata batch files found to merge.")

        if embed_batch_files:
            logging.info(f"Merging {len(embed_batch_files)} embeddings batch files...")
            try:
                dfs_embed = []
                for batch_file in tqdm(embed_batch_files, desc="Loading embedding batches"):
                    try:
                        if batch_file.suffix.lower() == '.parquet':
                            df_batch = pd.read_parquet(batch_file)
                        else:  # CSV
                            df_batch = pd.read_csv(batch_file)
                        dfs_embed.append(df_batch)
                    except Exception as e_batch:
                        logging.error(f"Error loading batch file {batch_file}: {e_batch}")

                if dfs_embed:
                    df_embed_merged = pd.concat(dfs_embed, ignore_index=True)
                    logging.info(f"Saving merged Embeddings DataFrame ({df_embed_merged.shape}) to: {local_temp_embeddings_path}")
                    local_temp_embeddings_path.parent.mkdir(parents=True, exist_ok=True)

                    # Use safe Parquet saving for the merged file
                    save_success = save_dataframe_to_parquet_safely(df_embed_merged, local_temp_embeddings_path)
                    if save_success:
                        saved_embeddings_path = local_temp_embeddings_path
                        logging.info("Merged embeddings Parquet saved locally.")
                    else:
                        logging.warning("Could not save merged embeddings as Parquet. Check for CSV fallback.")
                        # The save_dataframe_to_parquet_safely function will have created a CSV fallback if Parquet failed
                        csv_path = local_temp_embeddings_path.with_suffix('.csv')
                        if csv_path.exists():
                            saved_embeddings_path = csv_path
                            logging.info(f"Using CSV fallback for embeddings: {csv_path}")
            except Exception as e_merge_embed:
                logging.error(f"Failed to merge embeddings batch files: {e_merge_embed}", exc_info=True)
        else:
            logging.warning("No embeddings batch files found to merge.")

        # If we already have a final embeddings file, combine with it
        if saved_embeddings_path and final_embeddings_parquet_path.exists():
            try:
                logging.info(f"Found existing final embeddings file. Merging with new data...")
                # Load both datasets
                try:
                    df_existing = pd.read_parquet(final_embeddings_parquet_path)
                except Exception:
                    # Try CSV fallback
                    csv_path = final_embeddings_parquet_path.with_suffix('.csv')
                    if csv_path.exists():
                        df_existing = pd.read_csv(csv_path)
                    else:
                        raise

                # Load new data
                if saved_embeddings_path.suffix.lower() == '.parquet':
                    df_new = pd.read_parquet(saved_embeddings_path)
                else:  # CSV
                    df_new = pd.read_csv(saved_embeddings_path)

                # Get unique record_cids from both
                existing_cids = set(df_existing['record_cid'].tolist())
                new_cids = set(df_new['record_cid'].tolist())

                # Only keep rows from df_new that aren't in df_existing
                df_new_unique = df_new[~df_new['record_cid'].isin(existing_cids)]

                if len(df_new_unique) > 0:
                    # Combine datasets
                    df_combined = pd.concat([df_existing, df_new_unique], ignore_index=True)
                    logging.info(f"Combined {len(df_existing)} existing and {len(df_new_unique)} new records")

                    # Save combined dataset using safe Parquet saving
                    save_success = save_dataframe_to_parquet_safely(df_combined, saved_embeddings_path)
                    if save_success:
                        logging.info(f"Saved combined embeddings with {len(df_combined)} total records")
                    else:
                        logging.warning("Could not save combined embeddings as Parquet. Check for CSV fallback.")
                else:
                    logging.info("No new unique embeddings to add to existing file")
            except Exception as e_combine:
                logging.error(f"Error combining with existing embeddings: {e_combine}", exc_info=True)

        # Also do the same for metadata
        if saved_metadata_path and final_metadata_parquet_path.exists():
            try:
                logging.info(f"Found existing final metadata file. Merging with new data...")
                # Load both datasets
                try:
                    df_existing = pd.read_parquet(final_metadata_parquet_path)
                except Exception:
                    # Try CSV fallback
                    csv_path = final_metadata_parquet_path.with_suffix('.csv')
                    if csv_path.exists():
                        df_existing = pd.read_csv(csv_path)
                    else:
                        raise

                # Load new data
                if saved_metadata_path.suffix.lower() == '.parquet':
                    df_new = pd.read_parquet(saved_metadata_path)
                else:  # CSV
                    df_new = pd.read_csv(saved_metadata_path)

                # Get unique record_cids from both
                existing_cids = set(df_existing['record_cid'].tolist())
                new_cids = set(df_new['record_cid'].tolist())

                # Only keep rows from df_new that aren't in df_existing
                df_new_unique = df_new[~df_new['record_cid'].isin(existing_cids)]

                if len(df_new_unique) > 0:
                    # Combine datasets
                    df_combined = pd.concat([df_existing, df_new_unique], ignore_index=True)
                    logging.info(f"Combined {len(df_existing)} existing and {len(df_new_unique)} new metadata records")

                    # Save combined dataset using safe Parquet saving
                    save_success = save_dataframe_to_parquet_safely(df_combined, saved_metadata_path)
                    if save_success:
                        logging.info(f"Saved combined metadata with {len(df_combined)} total records")
                    else:
                        logging.warning("Could not save combined metadata as Parquet. Check for CSV fallback.")
                else:
                    logging.info("No new unique metadata to add to existing file")
            except Exception as e_combine:
                logging.error(f"Error combining with existing metadata: {e_combine}", exc_info=True)

        # Clean up batch files after successful merge if periodic merges were disabled
        if PERIODIC_MERGE_FREQUENCY == 0 and saved_metadata_path and saved_embeddings_path:
            try:
                logging.info("Cleaning up batch files after final merge...")
                for batch_file in list(batch_save_dir.glob("*_batch_*.*")):
                    batch_file.unlink()
                logging.info("Batch files cleaned up.")
            except Exception as e_cleanup:
                logging.warning(f"Error cleaning up batch files: {e_cleanup}")

        return saved_metadata_path, saved_embeddings_path # Return paths to locally saved files

    # Handle file/main processing errors
    except FileNotFoundError: logging.error(f"CRITICAL: Input JSONL file not found: {input_jsonl_filepath}."); return None, None
    except Exception as e_main: logging.error(f"CRITICAL error: {e_main}", exc_info=True); return None, None

    # --- Final Summary ---
    finally:
        total_processed_in_run = processed_count_this_run
        total_batches_saved = batch_counter
        total_batches_merged = len(merged_batch_tracker)

        logging.info("--- Embedding Generation Summary ---")
        logging.info(f"Records read from JSONL: {record_count_from_jsonl}")
        logging.info(f"Records skipped (resume): {skipped_resume_count}")
        logging.info(f"Records processed this run: {total_processed_in_run}")
        logging.info(f"Records skipped (errors): {skipped_error_count}")
        logging.info(f"Total batches saved: {total_batches_saved}")
        logging.info(f"Total batches merged to Google Drive: {total_batches_merged}")
        logging.info(f"Total unique records processed (including previous runs): {len(processed_cids)}")
        logging.info(f"Local temp Metadata path: {local_temp_metadata_path}")
        logging.info(f"Local temp Embeddings path: {local_temp_embeddings_path}")
        logging.info(f"Local temp Log path: {local_temp_log_path}")
        if start_time: logging.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        logging.info("------------------------------------")
        # Final sync/cleanup happens after this function returns

# --- Upload Function (Modified for two files) ---
def upload_files_to_hub(
    local_metadata_path: Path,
    local_embeddings_path: Path,
    repo_id: str,
    repo_type: str = "dataset",
    metadata_path_in_repo: Optional[str] = None,
    embeddings_path_in_repo: Optional[str] = None,
    hf_token: Union[str, bool, None] = None
    ):
    """Uploads the generated Parquet files to the Hugging Face Hub."""
    api = HfApi(token=hf_token)
    uploaded_meta = False
    uploaded_embed = False

    # Upload Metadata (Parquet or CSV)
    if local_metadata_path and local_metadata_path.exists():
        path_in_repo_meta = metadata_path_in_repo or local_metadata_path.name
        logging.info(f"Uploading Metadata: {local_metadata_path} to {repo_id} as {path_in_repo_meta}...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_metadata_path), path_in_repo=path_in_repo_meta, repo_id=repo_id, repo_type=repo_type,
                commit_message=f"Update metadata ({local_metadata_path.suffix}) {time.strftime('%Y-%m-%d %H:%M:%S')}"
            ); logging.info("Metadata upload successful."); uploaded_meta = True
        except Exception as e: logging.error(f"Metadata upload failed: {e}", exc_info=True)
    else: logging.warning("Local metadata file not found or not specified. Skipping metadata upload.")

    # Upload Embeddings (Parquet or CSV)
    if local_embeddings_path and local_embeddings_path.exists():
        path_in_repo_embed = embeddings_path_in_repo or local_embeddings_path.name
        logging.info(f"Uploading Embeddings: {local_embeddings_path} to {repo_id} as {path_in_repo_embed}...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_embeddings_path), path_in_repo=path_in_repo_embed, repo_id=repo_id, repo_type=repo_type,
                commit_message=f"Update embeddings ({local_embeddings_path.suffix}) {time.strftime('%Y-%m-%d %H:%M:%S')}"
            ); logging.info("Embeddings upload successful."); uploaded_embed = True
        except Exception as e: logging.error(f"Embeddings upload failed: {e}", exc_info=True)
    else: logging.warning("Local embeddings file not found or not specified. Skipping embeddings upload.")

    return uploaded_meta and uploaded_embed # Return overall success

# --- Script Execution (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    # --- Determine Paths ---
    print("--- Determining Output Paths ---")
    gdrive_base = Path(GDRIVE_MOUNT_POINT); gdrive_target_dir = gdrive_base / GDRIVE_FOLDER_NAME
    local_fallback_dir = Path(LOCAL_FOLDER_NAME); effective_final_dir = None;
    print(f"Checking GDrive: {gdrive_base}");
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

    else:
        local_fallback_dir.mkdir(parents=True, exist_ok=True)
        print(f"Mount not found. Using local fallback: {local_fallback_dir}")
        effective_final_dir = local_fallback_dir

    effective_final_dir.mkdir(parents=True, exist_ok=True); LOCAL_WORK_DIR.mkdir(parents=True, exist_ok=True); print(f"Effective final destination directory: {effective_final_dir}");

    # Define final destination paths
    final_metadata_filepath = effective_final_dir / FINAL_METADATA_PARQUET_FILENAME
    final_embeddings_filepath = effective_final_dir / FINAL_EMBEDDINGS_PARQUET_FILENAME
    final_log_filepath = effective_final_dir / FINAL_LOG_FILENAME
    input_jsonl_filepath = effective_final_dir / INPUT_JSONL_FILENAME # Assume input is also in final dir

    # Define local working paths
    local_temp_metadata_path = LOCAL_WORK_DIR / LOCAL_TEMP_METADATA_PARQUET
    local_temp_embeddings_path = LOCAL_WORK_DIR / LOCAL_TEMP_EMBEDDINGS_PARQUET
    local_temp_log_path = LOCAL_WORK_DIR / LOCAL_TEMP_LOG_FILENAME

    print(f"Input JSONL path: {input_jsonl_filepath}")
    print(f"Final Metadata Parquet path: {final_metadata_filepath}")
    print(f"Final Embeddings Parquet path: {final_embeddings_filepath}")
    print(f"Final log file path: {final_log_filepath}")
    print(f"Local temp Metadata path: {local_temp_metadata_path}")
    print(f"Local temp Embeddings path: {local_temp_embeddings_path}")
    print(f"Local temp log file path: {local_temp_log_path}")
    print("-" * 30)

    # Remove existing local temp files before start
    if local_temp_metadata_path.exists():
        print(f"Removing local temp metadata: {local_temp_metadata_path}")
        try:
            local_temp_metadata_path.unlink()
        except OSError as e: print(f"Warn: {e}")

    if local_temp_embeddings_path.exists():
        print(f"Removing local temp embeddings: {local_temp_embeddings_path}")
        try:
            local_temp_embeddings_path.unlink()
        except OSError as e: print(f"Warn: {e}")

    if local_temp_log_path.exists():
        print(f"Removing local temp log: {local_temp_log_path}")
        try:
            local_temp_log_path.unlink()
        except OSError as e: print(f"Warn: {e}")


    # --- Run the Embedding Generation ---
    # Returns paths to the *local* temp parquet files if successful
    local_meta_path, local_embed_path = create_embedding_dataset(
        input_jsonl_filepath=input_jsonl_filepath,
        final_metadata_parquet_path=final_metadata_filepath,   # For loading resume
        final_embeddings_parquet_path=final_embeddings_filepath, # For loading resume
        local_temp_metadata_path=local_temp_metadata_path,   # Local save dest
        local_temp_embeddings_path=local_temp_embeddings_path, # Local save dest
        local_temp_log_path=local_temp_log_path,             # Local log dest
        final_log_filepath=final_log_filepath,               # Final log for logging clarity
        max_records=MAX_RECORDS_TO_PROCESS,
        batch_size=BATCH_SIZE,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        process_config=PROCESS_CONFIG_JSON,
        process_readme=PROCESS_README_CONTENT,
    )

    # --- Sync final local files to Drive/Destination ---
    if local_meta_path or local_embed_path: # Check if at least one file was created
         logging.info("Attempting to sync final local files to destination...")
         sync_success = sync_local_files_to_final(
              local_metadata_path=local_temp_metadata_path, # Use the defined local path vars
              local_embeddings_path=local_temp_embeddings_path,
              local_log_path=local_temp_log_path,
              final_metadata_path=final_metadata_filepath,
              final_embeddings_path=final_embeddings_filepath,
              final_log_path=final_log_filepath
         )
         if sync_success:
              logging.info("Final sync to destination successful.")
              # --- Upload final Parquet from Destination to Hub (Optional) ---
              if UPLOAD_TO_HUB:
                   upload_files_to_hub(
                        local_metadata_path=final_metadata_filepath, # Upload from final dest
                        local_embeddings_path=final_embeddings_filepath,
                        repo_id=TARGET_REPO_ID,
                        repo_type=TARGET_REPO_TYPE,
                        metadata_path_in_repo=METADATA_FILENAME_IN_REPO,
                        embeddings_path_in_repo=EMBEDDINGS_FILENAME_IN_REPO,
                        hf_token=None # Uses login
                   )
              else: logging.info("Hub upload skipped by configuration.")
         else: logging.error("Final sync to destination FAILED. Cannot upload to Hub.")
    else: logging.warning("Local Parquet file creation failed or no data processed. Skipping final sync and Hub upload.")

    # --- Clean up local temp files ---
    logging.info("Attempting final cleanup of local temp files...")
    try:
        if local_temp_metadata_path.is_file(): local_temp_metadata_path.unlink(); logging.info(f"Cleaned {local_temp_metadata_path}")
        if local_temp_embeddings_path.is_file(): local_temp_embeddings_path.unlink(); logging.info(f"Cleaned {local_temp_embeddings_path}")
        if local_temp_log_path.is_file(): local_temp_log_path.unlink(); logging.info(f"Cleaned {local_temp_log_path}")
    except Exception as clean_e: logging.warning(f"Could not clean up local temp files: {clean_e}")

    logging.info("Script finished.")
