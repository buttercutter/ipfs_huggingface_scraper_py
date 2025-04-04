import os
import json
import logging
import time
from pathlib import Path
from tqdm.auto import tqdm

# Hugging Face related
from huggingface_hub import list_models, hf_hub_download, HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HFValidationError

# Data handling
import pandas as pd

# Configure paths
DATA_DIR = Path.home() / "Downloads/hf_metadata_dataset_local_fallback"  # Change to your local path
INPUT_JSONL = DATA_DIR / "all_models_metadata.jsonl"
ENHANCED_JSONL = DATA_DIR / "enhanced_models_metadata.jsonl"

# HF Hub settings
TARGET_REPO_ID = "buttercutter/models-metadata-dataset"  # Change this!
TARGET_REPO_TYPE = "dataset"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_readme_content(repo_id, token=HF_TOKEN):
    """Downloads a model's README.md file and returns its content as text."""
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="model",
            token=token,
            library_name="hf_dataset_enhancer"
        )
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            logging.warning(f"Could not decode README.md for {repo_id} as UTF-8.")
            return None
        except Exception as e:
            logging.error(f"Error reading README.md for {repo_id}: {e}")
            return None

    except EntryNotFoundError:
        logging.info(f"README.md not found in {repo_id}.")
        return None
    except Exception as e:
        logging.error(f"Error downloading README.md for {repo_id}: {e}")
        return None


def get_config_json(repo_id, token=HF_TOKEN):
    """Downloads a model's config.json file and returns its content as a dictionary."""
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            repo_type="model",
            token=token,
            library_name="hf_dataset_enhancer"
        )
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            return content
        except json.JSONDecodeError:
            logging.warning(f"Could not parse config.json for {repo_id} as valid JSON.")
            return None
        except UnicodeDecodeError:
            logging.warning(f"Could not decode config.json for {repo_id} as UTF-8.")
            return None
        except Exception as e:
            logging.error(f"Error reading config.json for {repo_id}: {e}")
            return None

    except EntryNotFoundError:
        logging.info(f"config.json not found in {repo_id}.")
        return None
    except Exception as e:
        logging.error(f"Error downloading config.json for {repo_id}: {e}")
        return None


def enhance_dataset():
    """Reads the input JSONL, adds README content for each model, and saves enhanced data."""
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if input file exists
    if not INPUT_JSONL.exists():
        logging.error(f"Input file not found: {INPUT_JSONL}")
        return False

    # Process the input file
    logging.info(f"Processing {INPUT_JSONL}...")

    # Count total records for progress bar
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # Process each record
    with open(INPUT_JSONL, 'r', encoding='utf-8') as infile, open(ENHANCED_JSONL, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Enhancing models"):
            try:
                # Parse the JSON record
                record = json.loads(line.strip())

                # Get model ID
                model_id = record.get('id')
                if not model_id:
                    logging.warning(f"Skipping record without model ID: {record}")
                    continue

                # 1. Fetch README.md if not already present
                if 'readme' not in record:
                    # Fetch README.md content
                    readme_content = get_readme_content(model_id)
                    # Add README content to the record
                    record['readme'] = readme_content

                # 2. Fetch config.json if not already present
                if 'config_json' not in record:
                    config_content = get_config_json(model_id)
                    record['config_json'] = config_content

                # Write the enhanced record
                outfile.write(json.dumps(record) + '\n')

            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line: {line[:100]}...")
            except Exception as e:
                logging.error(f"Error processing record: {e}")

    logging.info(f"Enhanced dataset saved to {ENHANCED_JSONL}")
    return True

def upload_to_hub():
    """Uploads the enhanced dataset to Hugging Face Hub."""
    if not ENHANCED_JSONL.exists():
        logging.error(f"Enhanced dataset file not found: {ENHANCED_JSONL}")
        return False

    logging.info(f"Uploading dataset to Hugging Face Hub: {TARGET_REPO_ID}")

    try:
        api = HfApi()

        # Create the repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=TARGET_REPO_ID,
                repo_type=TARGET_REPO_TYPE,
                exist_ok=True
            )
            logging.info(f"Repository {TARGET_REPO_ID} ready.")
        except Exception as e:
            logging.warning(f"Could not create/check repository: {e}")

        # Upload the JSONL file
        api.upload_file(
            path_or_fileobj=str(ENHANCED_JSONL),
            path_in_repo="enhanced_models_metadata.jsonl",
            repo_id=TARGET_REPO_ID,
            repo_type=TARGET_REPO_TYPE,
            commit_message=f"Upload enhanced models metadata with README content"
        )
        logging.info("Dataset successfully uploaded to Hugging Face Hub!")

        # Convert to Parquet and upload as well
        try:
            parquet_path = ENHANCED_JSONL.with_suffix('.parquet')
            logging.info(f"Converting to Parquet format: {parquet_path}")

            # Read JSONL and save as Parquet
            df = pd.read_json(ENHANCED_JSONL, lines=True)
            df.to_parquet(parquet_path, index=False)

            # Upload Parquet file
            api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo="enhanced_models_metadata.parquet",
                repo_id=TARGET_REPO_ID,
                repo_type=TARGET_REPO_TYPE,
                commit_message=f"Add Parquet version of dataset"
            )
            logging.info("Parquet file successfully uploaded to Hugging Face Hub!")
        except Exception as e:
            logging.error(f"Error converting/uploading Parquet file: {e}")

        return True

    except Exception as e:
        logging.error(f"Error uploading to Hugging Face Hub: {e}")
        return False

if __name__ == "__main__":
    # Make sure Hugging Face is configured
    print("Make sure you're logged in to Hugging Face (`huggingface-cli login`)")
    print(f"Target repository: {TARGET_REPO_ID}")

    # Enhance the dataset with README content
    if enhance_dataset():
        # Upload the enhanced dataset to Hugging Face Hub
        upload_to_hub()

    print("Process complete!")
