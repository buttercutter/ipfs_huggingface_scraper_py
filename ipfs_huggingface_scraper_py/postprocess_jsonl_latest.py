import os
import json
import logging
import time
import traceback
from pathlib import Path
import shutil
import psutil
import glob
import gc
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
# https://github.com/endomorphosis/ipfs_accelerate_py/blob/5f88e36551b626e99b05bd4bd8a3e043c5c0e8c9/ipfs_accelerate_py/ipfs_multiformats.py#L25
import hashlib
from multiformats import CID, multihash
import tempfile
import os
import sys
class ipfs_multiformats_py:
    def __init__(self, resources, metadata):
        self.multihash = multihash
        return None

    # Step 1: Hash the file content with SHA-256
    def get_file_sha256(self, file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.digest()

    # Step 2: Wrap the hash in Multihash format
    def get_multihash_sha256(self, file_content_hash):
        mh = self.multihash.wrap(file_content_hash, 'sha2-256')
        return mh

    # Step 3: Generate CID from Multihash (CIDv1)
    def get_cid(self, file_data):
        if os.path.isfile(file_data) == True:
            absolute_path = os.path.abspath(file_data)
            file_content_hash = self.get_file_sha256(file_data)
            mh = self.get_multihash_sha256(file_content_hash)
            cid = CID('base32', 'raw', mh)
        else:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                filename = f.name
                with open(filename, 'w') as f_new:
                    f_new.write(file_data)
                file_content_hash = self.get_file_sha256(filename)
                mh = self.get_multihash_sha256(file_content_hash)
                cid = CID('base32', 1, 'raw', mh)
                os.remove(filename)
        return str(cid)
# --- End IPFS CID Code ---


# --- Configuration ---
# --- Paths ---
GDRIVE_MOUNT_POINT = "/content/drive/MyDrive"
GDRIVE_FOLDER_NAME = "hf_metadata_dataset_collection"
LOCAL_FOLDER_NAME = "./hf_metadata_dataset_local_fallback"
LOCAL_WORK_DIR = Path(os.path.abspath("./hf_embedding_work"))

# Input JSONL File
INPUT_JSONL_FILENAME = "all_models_metadata.jsonl" # Assumed in final dir

# --- Output File Names ---
# Final Destination (Drive/Local Fallback)
FINAL_METADATA_PARQUET_FILENAME = "model_metadata.parquet" # Metadata + CIDs
FINAL_EMBEDDINGS_PARQUET_FILENAME = "model_embeddings.parquet" # CIDs + Embeddings
FINAL_LOG_FILENAME = "embedding_generator.log"

# Local Temporary Files (in LOCAL_WORK_DIR)
LOCAL_TEMP_METADATA_JSONL = "temp_model_metadata.jsonl"
LOCAL_TEMP_EMBEDDINGS_JSONL = "temp_model_embeddings.jsonl"
LOCAL_TEMP_LOG_FILENAME = "temp_embedding_generator.log"

# --- Batch Configuration ---
BATCH_SAVE_THRESHOLD = 1000  # Save after processing this many records
BATCH_SAVE_DIR_NAME = "batch_files"  # Subdirectory for batch files
PERIODIC_MERGE_FREQUENCY = 5  # Merge to Google Drive every X batches (0 to disable)
CLEAN_AFTER_PERIODIC_MERGE = True  # Whether to clean up batch files after periodic merge

# --- Memory Management Configuration ---
MEMORY_CLEANUP_THRESHOLD_MB = 1000  # Force extra cleanup if memory growth exceeds this

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
        df_safe.to_parquet(filepath, index=False, compression='gzip')
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


# final conversion from JSONL to Parquet would happen only once at the end of all processing.
def convert_jsonl_to_parquet(
    meta_jsonl_path: Path,
    embed_jsonl_path: Path,
    local_temp_metadata_path: Path,
    local_temp_embeddings_path: Path,
    chunk_size: int = 50000,
    max_memory_mb: int = 2000  # Memory threshold for adaptive processing
):
    """
    Convert very large JSONL files to Parquet using a streaming approach with minimal memory usage.

    Args:
        meta_jsonl_path: Path to metadata JSONL file
        embed_jsonl_path: Path to embeddings JSONL file
        local_temp_metadata_path: Output path for metadata Parquet file
        local_temp_embeddings_path: Output path for embeddings Parquet file
        chunk_size: Initial number of records to process at once (will adapt based on memory usage)
        max_memory_mb: Maximum memory usage threshold in MB
    """
    import json
    import os
    import gc
    import time
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm
    import psutil

    logging.info("Starting optimized streaming conversion from JSONL to Parquet")

    def get_memory_usage_mb():
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def estimate_total_lines(file_path, sample_size=1000000):
        """Estimate total lines in file without reading entire file"""
        try:
            # Get file size
            file_size = os.path.getsize(file_path)

            # If file is small enough, just count lines directly
            if file_size < 100 * 1024 * 1024:  # 100 MB
                with open(file_path, 'r') as f:
                    return sum(1 for _ in f)

            # Sample beginning of file to estimate line size
            line_count = 0
            bytes_read = 0
            with open(file_path, 'r') as f:
                for _ in range(sample_size):
                    line = f.readline()
                    if not line:
                        break
                    bytes_read += len(line.encode('utf-8'))
                    line_count += 1

            if line_count == 0:
                return 0

            # Calculate average line size and estimate total
            avg_line_size = bytes_read / line_count
            estimated_lines = int(file_size / avg_line_size)

            logging.info(f"Estimated lines in {file_path.name}: {estimated_lines:,} (based on avg line size: {avg_line_size:.1f} bytes)")
            return estimated_lines

        except Exception as e:
            logging.error(f"Error estimating lines in file: {e}")
            return 0

    def infer_schema_from_samples(file_path, num_samples=1000):
        """Infer schema by sampling from beginning, middle, and end of file"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return None

            samples = []
            with open(file_path, 'r') as f:
                # Read samples from beginning
                for _ in range(num_samples // 3):
                    line = f.readline()
                    if not line:
                        break
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

                # Read samples from middle
                middle_pos = file_size // 2
                f.seek(middle_pos)
                f.readline()  # Skip partial line
                for _ in range(num_samples // 3):
                    line = f.readline()
                    if not line:
                        break
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

                # Read samples from end
                end_pos = max(0, file_size - 100000)  # 100 KB from end
                f.seek(end_pos)
                f.readline()  # Skip partial line
                for _ in range(num_samples // 3):
                    line = f.readline()
                    if not line:
                        break
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            if not samples:
                logging.error(f"No valid JSON samples found in {file_path}")
                return None

            # Convert samples to pyarrow schema
            import pandas as pd
            sample_df = pd.DataFrame(samples)

            # Convert all columns to string type to avoid type mismatches
            for col in sample_df.columns:
                if col != 'embedding':  # Keep embedding as is since it's numeric
                    sample_df[col] = sample_df[col].astype(str)

            # Handle embedding field specially if it exists
            if 'embedding' in sample_df.columns:
                # Ensure embedding is a list of float
                if sample_df['embedding'].dtype != 'object':
                    # If not already a list, convert to string
                    sample_df['embedding'] = sample_df['embedding'].astype(str)

            # Convert to PyArrow Table and extract schema
            table = pa.Table.from_pandas(sample_df)
            logging.info(f"Inferred schema with {len(table.schema.names)} fields")
            return table.schema

        except Exception as e:
            logging.error(f"Error inferring schema: {e}", exc_info=True)
            return None

    def stream_jsonl_to_parquet(jsonl_path, parquet_path, file_type, initial_chunk_size):
        """Process a JSONL file in a streaming fashion with adaptive chunk sizing"""
        if not jsonl_path.exists():
            logging.warning(f"{file_type} JSONL file not found: {jsonl_path}")
            return False

        logging.info(f"Starting streaming conversion of {file_type} JSONL: {jsonl_path} -> {parquet_path}")
        start_time = time.time()

        # Get schema by sampling
        schema = infer_schema_from_samples(jsonl_path)
        if schema is None:
            logging.error(f"Failed to infer schema for {file_type}")
            return False

        # Estimate total for progress reporting
        estimated_total = estimate_total_lines(jsonl_path)

        # Track current chunk size - will adapt based on memory usage
        current_chunk_size = initial_chunk_size
        records_processed = 0
        chunk_count = 0

        try:
            # Create parquet writer with inferred schema
            with pq.ParquetWriter(parquet_path, schema) as writer:
                # Process in chunks to limit memory usage
                buffer = []

                with tqdm(total=estimated_total, desc=f"Converting {file_type}") as pbar:
                    with open(jsonl_path, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            try:
                                record = json.loads(line)

                                # Convert all string fields to ensure type consistency
                                for key, value in record.items():
                                    if key != 'embedding' and value is not None and not isinstance(value, (list, dict)):
                                        record[key] = str(value)

                                buffer.append(record)

                                # When buffer reaches chunk size, write to parquet
                                if len(buffer) >= current_chunk_size:
                                    # Convert buffer to PyArrow table
                                    import pandas as pd
                                    chunk_df = pd.DataFrame(buffer)

                                    # Handle embedding field specially if it exists
                                    if 'embedding' in chunk_df.columns:
                                        # Ensure embedding is a list of float
                                        if chunk_df['embedding'].dtype != 'object':
                                            # If not already a list, convert to string
                                            chunk_df['embedding'] = chunk_df['embedding'].astype(str)

                                    # Convert non-embedding fields to string
                                    for col in chunk_df.columns:
                                        if col != 'embedding':
                                            chunk_df[col] = chunk_df[col].astype(str)

                                    # Write chunk
                                    table = pa.Table.from_pandas(chunk_df, schema=schema)
                                    writer.write_table(table)

                                    # Update progress
                                    records_processed += len(buffer)
                                    pbar.update(len(buffer))
                                    chunk_count += 1

                                    # Clear buffer and force garbage collection
                                    buffer = []
                                    del chunk_df, table
                                    gc.collect()

                                    # Adaptive chunk sizing based on memory usage
                                    current_memory = get_memory_usage_mb()
                                    if current_memory > max_memory_mb:
                                        # Reduce chunk size if memory usage is too high
                                        new_chunk_size = max(1000, int(current_chunk_size * 0.8))
                                        logging.info(f"Memory usage high ({current_memory:.1f} MB). Reducing chunk size from {current_chunk_size} to {new_chunk_size}")
                                        current_chunk_size = new_chunk_size
                                    elif current_memory < max_memory_mb * 0.5 and current_chunk_size < initial_chunk_size:
                                        # Increase chunk size if memory usage is low
                                        new_chunk_size = min(initial_chunk_size, int(current_chunk_size * 1.2))
                                        logging.info(f"Memory usage low ({current_memory:.1f} MB). Increasing chunk size from {current_chunk_size} to {new_chunk_size}")
                                        current_chunk_size = new_chunk_size

                                    # Log progress periodically
                                    if chunk_count % 10 == 0:
                                        elapsed = time.time() - start_time
                                        rate = records_processed / elapsed if elapsed > 0 else 0
                                        logging.info(f"Processed {records_processed:,} records ({rate:.1f} records/sec), memory: {current_memory:.1f} MB")

                            except json.JSONDecodeError:
                                logging.warning(f"Invalid JSON at line {line_num}")
                                continue
                            except Exception as e:
                                logging.warning(f"Error processing line {line_num}: {e}")
                                continue

                    # Write any remaining records
                    if buffer:
                        try:
                            import pandas as pd
                            chunk_df = pd.DataFrame(buffer)

                            # Handle embedding field specially if it exists
                            if 'embedding' in chunk_df.columns:
                                # Ensure embedding is a list of float
                                if chunk_df['embedding'].dtype != 'object':
                                    # If not already a list, convert to string
                                    chunk_df['embedding'] = chunk_df['embedding'].astype(str)

                            # Convert non-embedding fields to string
                            for col in chunk_df.columns:
                                if col != 'embedding':
                                    chunk_df[col] = chunk_df[col].astype(str)

                            # Write final chunk
                            table = pa.Table.from_pandas(chunk_df, schema=schema)
                            writer.write_table(table)

                            # Update progress
                            records_processed += len(buffer)
                            pbar.update(len(buffer))

                        except Exception as e:
                            logging.error(f"Error writing final chunk: {e}")

            # Report final stats
            elapsed = time.time() - start_time
            rate = records_processed / elapsed if elapsed > 0 else 0
            logging.info(f"Successfully converted {records_processed:,} {file_type} records in {elapsed:.1f} seconds ({rate:.1f} records/sec)")
            logging.info(f"Created {file_type} Parquet file: {parquet_path} ({os.path.getsize(parquet_path) / (1024*1024):.1f} MB)")
            return True

        except Exception as e:
            logging.error(f"Error during {file_type} conversion: {e}", exc_info=True)
            return False

    # Convert metadata file
    meta_success = stream_jsonl_to_parquet(meta_jsonl_path, local_temp_metadata_path, "metadata", chunk_size)

    # Force garbage collection before processing embeddings
    gc.collect()

    # Convert embeddings file
    embed_success = stream_jsonl_to_parquet(embed_jsonl_path, local_temp_embeddings_path, "embeddings", chunk_size)

    if meta_success and embed_success:
        logging.info("JSONL to Parquet conversion completed successfully")
        return True
    else:
        logging.error("JSONL to Parquet conversion encountered errors")
        return False



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


# Track memory across perform_periodic_merge() function calls
last_merge_memory_usage = 0

def perform_periodic_merge(
    batch_save_dir: Path,
    merged_batch_tracker: Set[str],
    local_temp_metadata_path: Path,
    local_temp_embeddings_path: Path,
    final_log_path: Path,
    local_temp_log_path: Path
):
    """
    100% JSONL-only periodic merge with NO Parquet operations whatsoever.
    Only merges to JSONL files, conversion to Parquet happens separately at the end.
    """
    global last_merge_memory_usage

    # Track memory at function start
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 * 1024)
    logging.info(f"Starting JSONL-only periodic merge. Current memory: {start_memory:.2f} MB")

    # Define paths for working JSONL files (strip .parquet suffix if present)
    meta_jsonl_path = Path(str(local_temp_metadata_path).replace('.parquet', '.jsonl'))
    embed_jsonl_path = Path(str(local_temp_embeddings_path).replace('.parquet', '.jsonl'))

    # Find all JSONL batch files that haven't been merged yet
    meta_batch_files = []
    embed_batch_files = []

    # Only look for JSONL batch files
    for batch_file in batch_save_dir.glob("metadata_batch_*.jsonl"):
        if batch_file.name not in merged_batch_tracker:
            meta_batch_files.append(batch_file)

    for batch_file in batch_save_dir.glob("embeddings_batch_*.jsonl"):
        if batch_file.name not in merged_batch_tracker:
            embed_batch_files.append(batch_file)

    if not meta_batch_files and not embed_batch_files:
        logging.info("No new JSONL batches to merge periodically.")
        return 0

    logging.info(f"Performing JSONL-only merge of {len(meta_batch_files)} metadata files and {len(embed_batch_files)} embedding files")

    # --- Process metadata files ---
    if meta_batch_files:
        try:
            # Load existing record CIDs from JSONL to avoid duplicates
            existing_cids = set()

            # Check if JSONL exists from previous run and load CIDs
            if meta_jsonl_path.exists():
                logging.info(f"Scanning existing JSONL for CIDs: {meta_jsonl_path}")
                with open(meta_jsonl_path, 'r') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if 'record_cid' in record:
                                existing_cids.add(record['record_cid'])
                        except:
                            pass
                logging.info(f"Found {len(existing_cids)} existing CIDs in metadata JSONL")

            # Open JSONL in append mode
            with open(meta_jsonl_path, 'a') as jsonl_out:
                # Process each batch file
                for batch_file in meta_batch_files:
                    try:
                        logging.info(f"Processing metadata batch: {batch_file.name}")

                        # Process the JSONL batch file line by line
                        new_records_count = 0
                        total_records_count = 0

                        with open(batch_file, 'r') as batch_in:
                            for line in batch_in:
                                total_records_count += 1
                                try:
                                    record = json.loads(line)
                                    # Filter out records with CIDs that already exist
                                    if 'record_cid' in record and record['record_cid'] not in existing_cids:
                                        # Write new record to output JSONL
                                        jsonl_out.write(line)
                                        # Add to existing CIDs to avoid future duplicates
                                        existing_cids.add(record['record_cid'])
                                        new_records_count += 1
                                except json.JSONDecodeError:
                                    logging.warning(f"Could not parse JSON line in {batch_file.name}")

                        # Log stats
                        logging.info(f"Batch has {total_records_count} records, {new_records_count} are new")

                        # Mark batch as processed
                        merged_batch_tracker.add(batch_file.name)

                        # Clean up batch file if enabled
                        if CLEAN_AFTER_PERIODIC_MERGE:
                            try:
                                batch_file.unlink()
                                logging.debug(f"Removed processed batch file: {batch_file}")
                            except Exception as e:
                                logging.warning(f"Could not remove batch file: {e}")

                        # Force memory cleanup after each batch
                        gc.collect()

                    except Exception as e:
                        logging.error(f"Error processing batch file {batch_file}: {e}")
        except Exception as e:
            logging.error(f"Error in metadata merge process: {e}", exc_info=True)

    # Force memory cleanup between metadata and embeddings
    gc.collect()

    # --- Process embeddings files (similar approach) ---
    if embed_batch_files:
        try:
            # Load existing record CIDs from JSONL to avoid duplicates
            existing_cids = set()

            # Check if JSONL exists from previous run and load CIDs
            if embed_jsonl_path.exists():
                logging.info(f"Scanning existing JSONL for CIDs: {embed_jsonl_path}")
                with open(embed_jsonl_path, 'r') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if 'record_cid' in record:
                                existing_cids.add(record['record_cid'])
                        except:
                            pass
                logging.info(f"Found {len(existing_cids)} existing CIDs in embeddings JSONL")

            # Open JSONL in append mode
            with open(embed_jsonl_path, 'a') as jsonl_out:
                # Process each batch file
                for batch_file in embed_batch_files:
                    try:
                        logging.info(f"Processing embeddings batch: {batch_file.name}")

                        # Process the JSONL batch file line by line
                        new_records_count = 0
                        total_records_count = 0

                        with open(batch_file, 'r') as batch_in:
                            for line in batch_in:
                                total_records_count += 1
                                try:
                                    record = json.loads(line)
                                    # Filter out records with CIDs that already exist
                                    if 'record_cid' in record and record['record_cid'] not in existing_cids:
                                        # Write new record to output JSONL
                                        jsonl_out.write(line)
                                        # Add to existing CIDs to avoid future duplicates
                                        existing_cids.add(record['record_cid'])
                                        new_records_count += 1
                                except json.JSONDecodeError:
                                    logging.warning(f"Could not parse JSON line in {batch_file.name}")

                        # Log stats
                        logging.info(f"Batch has {total_records_count} records, {new_records_count} are new")

                        # Mark batch as processed
                        merged_batch_tracker.add(batch_file.name)

                        # Clean up batch file if enabled
                        if CLEAN_AFTER_PERIODIC_MERGE:
                            try:
                                batch_file.unlink()
                                logging.debug(f"Removed processed batch file: {batch_file}")
                            except Exception as e:
                                logging.warning(f"Could not remove batch file: {e}")

                        # Force memory cleanup after each batch
                        gc.collect()

                    except Exception as e:
                        logging.error(f"Error processing batch file {batch_file}: {e}")
        except Exception as e:
            logging.error(f"Error in embeddings merge process: {e}", exc_info=True)

    # -- Only sync the log file, no Parquet files during runtime --
    try:
        if local_temp_log_path.is_file() and local_temp_log_path.stat().st_size > 0:
            final_log_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_temp_log_path, final_log_path)
            logging.info("Log file sync successful.")
    except Exception as e:
        logging.error(f"Failed to sync log file: {e}")

    # Final cleanup
    for _ in range(3):
        gc.collect()

    # Update memory tracking for next call
    end_memory = process.memory_info().rss / (1024 * 1024)
    logging.info(f"Memory at end of merge: {end_memory:.2f} MB (Change: {end_memory - start_memory:.2f} MB)")
    last_merge_memory_usage = end_memory

    return len(meta_batch_files) + len(embed_batch_files)



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
    JSONL-only workflow that reads metadata, generates CIDs & embeddings,
    and saves all outputs as JSONL until the very end.
    """
    # --- Setup batch directory ---
    batch_save_dir = LOCAL_WORK_DIR / BATCH_SAVE_DIR_NAME
    batch_save_dir.mkdir(parents=True, exist_ok=True)

    # --- Define JSONL paths by converting Parquet paths ---
    meta_jsonl_path = Path(str(local_temp_metadata_path).replace('.parquet', '.jsonl'))
    embed_jsonl_path = Path(str(local_temp_embeddings_path).replace('.parquet', '.jsonl'))

    # --- Configure logging to use the local temp log file ---
    log_file_handler = logging.FileHandler(local_temp_log_path)
    log_stream_handler = logging.StreamHandler()
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[log_file_handler, log_stream_handler])
    logging.getLogger('huggingface_hub.repocard_data').setLevel(logging.ERROR)

    # --- Log configuration ---
    logging.info(f"--- Starting Embedding Generation with JSONL-only workflow ---")
    logging.info(f"Input JSONL: '{input_jsonl_filepath}'")
    logging.info(f"Metadata JSONL Output: '{meta_jsonl_path}'")
    logging.info(f"Embeddings JSONL Output: '{embed_jsonl_path}'")
    logging.info(f"Final Metadata Parquet Output (post-processing): '{final_metadata_parquet_path}'")
    logging.info(f"Final Embeddings Parquet Output (post-processing): '{final_embeddings_parquet_path}'")
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
        cid_generator = ipfs_multiformats_py(resources=None, metadata=None) # Initialize CID generator
        logging.info("Embedding model & CID generator loaded.")
    except Exception as e:
        logging.error(f"Failed to load embedding model or init CID generator: {e}", exc_info=True)
        return None, None # Return None for both paths

    # --- Load processed CIDs from existing JSONL files ---
    processed_cids = set()

    # 1. Check the main embeddings JSONL file
    if embed_jsonl_path.exists():
        logging.info(f"Found existing embeddings JSONL: {embed_jsonl_path}. Loading processed CIDs...")
        try:
            with open(embed_jsonl_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if 'record_cid' in record:
                            processed_cids.add(record['record_cid'])
                    except:
                        pass
            logging.info(f"Loaded {len(processed_cids)} CIDs from existing embeddings JSONL.")
        except Exception as e:
            logging.warning(f"Could not load CIDs from '{embed_jsonl_path}': {e}")

    # 2. Check batch files
    batch_files = list(batch_save_dir.glob("embeddings_batch_*.jsonl"))
    if batch_files:
        logging.info(f"Found {len(batch_files)} embedding batch JSONL files.")
        batch_cids_count = 0

        for batch_file in batch_files:
            try:
                with open(batch_file, 'r') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if 'record_cid' in record:
                                processed_cids.add(record['record_cid'])
                                batch_cids_count += 1
                        except:
                            pass
            except Exception as e:
                logging.warning(f"Error loading CIDs from batch file {batch_file}: {e}")

        logging.info(f"Loaded {batch_cids_count} additional CIDs from JSONL batch files.")

    initial_processed_count = len(processed_cids)
    logging.info(f"Resuming from {initial_processed_count} records already processed.")

    # --- Batch saving and periodic merge setup ---
    batch_counter = 0
    records_since_last_save = 0
    merged_batch_tracker = set()  # Track which batch files have been merged

    # Keep a lookup of model_id to record_cid for this session
    model_id_to_record_cid = {}

    # --- Process JSONL File ---
    metadata_records_list = [] # Holds dicts for metadata
    embeddings_records_list = [] # Holds dicts for embeddings
    batch_inputs = []           # Holds tuples for batch processing
    record_count_from_jsonl = 0; processed_count_this_run = 0; skipped_resume_count = 0; skipped_error_count = 0
    start_time = None

    try:
        logging.info(f"Opening input JSONL file: {input_jsonl_filepath}")
        start_time = time.time()
        with input_jsonl_filepath.open('r', encoding='utf-8') as f_jsonl:
            pbar = tqdm(f_jsonl, desc="Processing JSONL", unit="record")
            for line in pbar:
                record_count_from_jsonl += 1
                if max_records is not None and processed_count_this_run >= max_records:
                    logging.info(f"Reached max_records limit ({max_records}). Stopping.");
                    break

                try:
                    line = line.strip()
                    if not line: continue
                    data = json.loads(line) # Original metadata dictionary
                    model_id = data.get('id')
                    if not model_id or not isinstance(model_id, str):
                        logging.warning(f"Skip record {record_count_from_jsonl}: missing/invalid 'id'.");
                        skipped_error_count += 1;
                        continue

                    # --- Extract text for embedding & CID generation ---
                    config_text = ""; config_cid = None; config_dict_or_str = data.get('config')
                    if process_config and config_dict_or_str is not None:
                        if isinstance(config_dict_or_str, dict):
                            try:
                                config_text = json.dumps(config_dict_or_str, separators=(',', ':'));
                                config_cid = cid_generator.get_cid(config_text) # Use compact string for CID
                            except TypeError:
                                logging.warning(f"Cannot serialize config for {model_id}. Skip CID/embed.")
                        elif isinstance(config_dict_or_str, str): # Handle if config is already a string
                             config_text = config_dict_or_str;
                             config_cid = cid_generator.get_cid(config_text)
                        else:
                            logging.warning(f"Config for {model_id} type {type(config_dict_or_str)}. Skip CID/embed.")

                    readme_text = ""; readme_cid = None
                    if process_readme:
                        card_data = data.get('cardData')
                        if isinstance(card_data, dict):
                            readme_text = card_data.get('text', '') or ''
                        elif isinstance(card_data, str):
                            readme_text = card_data # If cardData itself is the string
                        if not readme_text and isinstance(data.get('description'), str):
                            readme_text = data['description'] # Fallback
                        if readme_text:
                            readme_cid = cid_generator.get_cid(readme_text)

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

                            # --- Create records for BOTH data formats ---
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
                                if process_config:
                                    embedding_record['config_embedding'] = config_embeddings[i].tolist() if config_texts_batch[i] else None
                                if process_readme:
                                    embedding_record['readme_embedding'] = readme_embeddings[i].tolist() if readme_texts_batch[i] else None
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

                                # Save metadata batch as JSONL
                                meta_batch_file = batch_save_dir / f"metadata_batch_{batch_counter}_{timestamp}.jsonl"
                                success_meta = True
                                try:
                                    with open(meta_batch_file, 'w') as f:
                                        for record in metadata_records_list:
                                            f.write(json.dumps(safe_serialize_dict(record)) + '\n')
                                    logging.info(f"Saved metadata batch {batch_counter} as JSONL with {len(metadata_records_list)} records")
                                except Exception as e:
                                    logging.error(f"Error saving metadata batch as JSONL: {e}")
                                    success_meta = False

                                # Save embeddings batch as JSONL
                                embed_batch_file = batch_save_dir / f"embeddings_batch_{batch_counter}_{timestamp}.jsonl"
                                success_embed = True
                                try:
                                    with open(embed_batch_file, 'w') as f:
                                        for record in embeddings_records_list:
                                            f.write(json.dumps(safe_serialize_dict(record)) + '\n')
                                    logging.info(f"Saved embeddings batch {batch_counter} as JSONL with {len(embeddings_records_list)} records")
                                except Exception as e:
                                    logging.error(f"Error saving embeddings batch as JSONL: {e}")
                                    success_embed = False

                                if success_meta and success_embed:
                                    logging.info(f"Saved batch {batch_counter} with {len(embeddings_records_list)} records")
                                else:
                                    logging.warning(f"Batch {batch_counter} save had issues. Check logs.")

                                # Clear the lists to start a new batch and reset counter
                                metadata_records_list = []
                                embeddings_records_list = []
                                records_since_last_save = 0

                                # --- Periodic merge to final JSONL files ---
                                if PERIODIC_MERGE_FREQUENCY > 0 and batch_counter % PERIODIC_MERGE_FREQUENCY == 0:
                                    pbar.set_postfix_str(f"Periodic merge to JSONL...", refresh=True)
                                    batches_merged = perform_periodic_merge(
                                        batch_save_dir=batch_save_dir,
                                        merged_batch_tracker=merged_batch_tracker,
                                        local_temp_metadata_path=local_temp_metadata_path,
                                        local_temp_embeddings_path=local_temp_embeddings_path,
                                        final_log_path=final_log_filepath,
                                        local_temp_log_path=local_temp_log_path
                                    )
                                    pbar.set_postfix_str(f"Merged {batches_merged} batches to JSONL", refresh=True)

                        except Exception as e_embed:
                             logging.error(f"Error embedding batch: {e_embed}", exc_info=True)
                             skipped_error_count += len(batch_inputs) # Count whole batch as skipped

                        batch_inputs = [] # Clear batch

                # Handle line processing errors
                except json.JSONDecodeError:
                    logging.warning(f"Skip record {record_count_from_jsonl}: JSON decode error.");
                    skipped_error_count += 1
                except Exception as e_line:
                    logging.error(f"Skip record {record_count_from_jsonl}: Error - {e_line}", exc_info=False);
                    skipped_error_count += 1
        # --- End reading JSONL file ---

        # --- Process Final Remaining Batch ---
        if batch_inputs:
            pbar.set_postfix_str(f"Embedding final batch ({len(batch_inputs)})...", refresh=True)
            try:
                # Process just like the main batch
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
                    if process_config:
                        embedding_record['config_embedding'] = config_embeddings[i].tolist() if config_texts_batch[i] else None
                    if process_readme:
                        embedding_record['readme_embedding'] = readme_embeddings[i].tolist() if readme_texts_batch[i] else None
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

            # Save final metadata batch as JSONL
            meta_batch_file = batch_save_dir / f"metadata_batch_{batch_counter}_{timestamp}.jsonl"
            success_meta = True
            try:
                with open(meta_batch_file, 'w') as f:
                    for record in metadata_records_list:
                        f.write(json.dumps(safe_serialize_dict(record)) + '\n')
                logging.info(f"Saved final metadata batch as JSONL with {len(metadata_records_list)} records")
            except Exception as e:
                logging.error(f"Error saving final metadata batch as JSONL: {e}")
                success_meta = False

            # Save final embeddings batch as JSONL
            embed_batch_file = batch_save_dir / f"embeddings_batch_{batch_counter}_{timestamp}.jsonl"
            success_embed = True
            try:
                with open(embed_batch_file, 'w') as f:
                    for record in embeddings_records_list:
                        f.write(json.dumps(safe_serialize_dict(record)) + '\n')
                logging.info(f"Saved final embeddings batch as JSONL with {len(embeddings_records_list)} records")
            except Exception as e:
                logging.error(f"Error saving final embeddings batch as JSONL: {e}")
                success_embed = False

            if success_meta and success_embed:
                logging.info(f"Saved final batch {batch_counter} with {len(embeddings_records_list)} records")
            else:
                logging.warning(f"Final batch {batch_counter} save had issues. Check logs.")

            # Clear lists
            metadata_records_list = []
            embeddings_records_list = []

        pbar.close()
        logging.info("Finished processing records from JSONL.")

        # --- Merge all remaining batches into the final JSONL files ---
        # Process any remaining batches that haven't been merged
        if PERIODIC_MERGE_FREQUENCY > 0:
            logging.info("Performing final merge of any remaining batches...")
            batches_merged = perform_periodic_merge(
                batch_save_dir=batch_save_dir,
                merged_batch_tracker=merged_batch_tracker,
                local_temp_metadata_path=local_temp_metadata_path,
                local_temp_embeddings_path=local_temp_embeddings_path,
                final_log_path=final_log_filepath,
                local_temp_log_path=local_temp_log_path
            )
            logging.info(f"Final merge: processed {batches_merged} remaining batches")

        # Return the JSONL paths for final conversion
        return meta_jsonl_path, embed_jsonl_path

    # Handle file/main processing errors
    except FileNotFoundError:
        logging.error(f"CRITICAL: Input JSONL file not found: {input_jsonl_filepath}.");
        return None, None
    except Exception as e_main:
        logging.error(f"CRITICAL error: {e_main}", exc_info=True);
        return None, None

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
        logging.info(f"Total batches merged: {total_batches_merged}")
        logging.info(f"Total unique records processed (including previous runs): {len(processed_cids)}")
        if start_time:
            logging.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        logging.info("------------------------------------")



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
    local_temp_metadata_path = LOCAL_WORK_DIR / LOCAL_TEMP_METADATA_JSONL
    local_temp_embeddings_path = LOCAL_WORK_DIR / LOCAL_TEMP_EMBEDDINGS_JSONL
    local_temp_log_path = LOCAL_WORK_DIR / LOCAL_TEMP_LOG_FILENAME

    print(f"Input JSONL path: {input_jsonl_filepath}")
    print(f"Final Metadata Parquet path: {final_metadata_filepath}")
    print(f"Final Embeddings Parquet path: {final_embeddings_filepath}")
    print(f"Final log file path: {final_log_filepath}")
    print(f"Local temp Metadata path: {local_temp_metadata_path}")
    print(f"Local temp Embeddings path: {local_temp_embeddings_path}")
    print(f"Local temp log file path: {local_temp_log_path}")
    print("-" * 30)


    # Check for existing local temp files (for resumption)
    resuming_from_previous_run = False
    if local_temp_metadata_path.exists() and local_temp_embeddings_path.exists():
        file_size_meta = local_temp_metadata_path.stat().st_size
        file_size_embed = local_temp_embeddings_path.stat().st_size

        if file_size_meta > 0 and file_size_embed > 0:
            print(f"Found existing temp files, will resume processing:")
            print(f"  - Metadata file: {local_temp_metadata_path} ({file_size_meta} bytes)")
            print(f"  - Embeddings file: {local_temp_embeddings_path} ({file_size_embed} bytes)")
            resuming_from_previous_run = True
        else:
            print(f"Found existing temp files but they're empty, removing them:")
            if file_size_meta == 0:
                print(f"  - Removing empty metadata file: {local_temp_metadata_path}")
                local_temp_metadata_path.unlink()
            if file_size_embed == 0:
                print(f"  - Removing empty embeddings file: {local_temp_embeddings_path}")
                local_temp_embeddings_path.unlink()
    else:
        print(f"No existing temp files found, starting fresh processing run.")


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

        # After all processing is complete
        meta_jsonl_path = LOCAL_WORK_DIR / LOCAL_TEMP_METADATA_JSONL
        embed_jsonl_path = LOCAL_WORK_DIR / LOCAL_TEMP_EMBEDDINGS_JSONL

        # After all processing is complete
        # Define Parquet output paths
        local_temp_metadata_parquet = LOCAL_WORK_DIR / FINAL_METADATA_PARQUET_FILENAME
        local_temp_embeddings_parquet = LOCAL_WORK_DIR / FINAL_EMBEDDINGS_PARQUET_FILENAME

        # One-time conversion from JSONL to Parquet at the very end
        convert_jsonl_to_parquet(
            meta_jsonl_path=meta_jsonl_path,
            embed_jsonl_path=embed_jsonl_path,
            local_temp_metadata_path=local_temp_metadata_parquet,
            local_temp_embeddings_path=local_temp_embeddings_parquet,
            chunk_size=50000,       # Starting chunk size (will adapt)
            max_memory_mb=2000      # Memory threshold in MB
        )

        sync_success = sync_local_files_to_final(
            local_metadata_path=local_temp_metadata_parquet, # Use the defined local path vars
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

    '''
    # --- Clean up local temp files ---
    logging.info("Attempting final cleanup of local temp files...")
    try:
        if local_temp_metadata_path.is_file(): local_temp_metadata_path.unlink(); logging.info(f"Cleaned {local_temp_metadata_path}")
        if local_temp_embeddings_path.is_file(): local_temp_embeddings_path.unlink(); logging.info(f"Cleaned {local_temp_embeddings_path}")
        if local_temp_log_path.is_file(): local_temp_log_path.unlink(); logging.info(f"Cleaned {local_temp_log_path}")
    except Exception as clean_e: logging.warning(f"Could not clean up local temp files: {clean_e}")
    '''
    logging.info("Script finished.")

