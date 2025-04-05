# API Reference

This document provides a comprehensive reference for the IPFS HuggingFace Scraper's Python API and CLI.

## Python API

### Enhanced Scraper

The main scraper class for models from HuggingFace Hub.

```python
from ipfs_huggingface_scraper_py import EnhancedScraper
```

#### Constructor

```python
EnhancedScraper(config_path=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str` or `None` | Path to the configuration file. If None, uses default config. |

#### Methods

##### `scrape_models`

```python
scrape_models(max_models=None)
```

Start the model scraping process.

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_models` | `int` or `None` | Maximum number of models to scrape. If provided, overrides the config value. |

##### `resume`

```python
resume()
```

Resume a paused scraping operation.

#### Properties

- `state_manager`: The `StateManager` instance for tracking progress
- `rate_limiter`: The `RateLimiter` instance for controlling API usage
- `ipfs_storage`: The `IpfsStorage` instance for IPFS integration
- `config`: The `Config` instance with configuration values

#### Example

```python
from ipfs_huggingface_scraper_py import EnhancedScraper

# Create with default config
scraper = EnhancedScraper()

# Start scraping with max 100 models
scraper.scrape_models(max_models=100)

# Resume a paused operation
scraper.resume()
```

### DatasetsScraper

Scraper class for datasets from HuggingFace Hub.

```python
from ipfs_huggingface_scraper_py import DatasetsScraper
```

#### Constructor

```python
DatasetsScraper(config_path=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str` or `None` | Path to the configuration file. If None, uses default config. |

#### Methods

##### `scrape_datasets`

```python
scrape_datasets(max_datasets=None)
```

Start the dataset scraping process.

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_datasets` | `int` or `None` | Maximum number of datasets to scrape. If provided, overrides the config value. |

##### `resume`

```python
resume()
```

Resume a paused dataset scraping operation.

#### Example

```python
from ipfs_huggingface_scraper_py import DatasetsScraper

# Create with default config
scraper = DatasetsScraper()

# Start scraping with max 50 datasets
scraper.scrape_datasets(max_datasets=50)

# Resume a paused operation
scraper.resume()
```

### SpacesScraper

Scraper class for spaces from HuggingFace Hub.

```python
from ipfs_huggingface_scraper_py import SpacesScraper
```

#### Constructor

```python
SpacesScraper(config_path=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str` or `None` | Path to the configuration file. If None, uses default config. |

#### Methods

##### `scrape_spaces`

```python
scrape_spaces(max_spaces=None)
```

Start the spaces scraping process.

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_spaces` | `int` or `None` | Maximum number of spaces to scrape. If provided, overrides the config value. |

##### `resume`

```python
resume()
```

Resume a paused spaces scraping operation.

#### Example

```python
from ipfs_huggingface_scraper_py import SpacesScraper

# Create with default config
scraper = SpacesScraper()

# Start scraping with max 25 spaces
scraper.scrape_spaces(max_spaces=25)

# Resume a paused operation
scraper.resume()
```

### Configuration

Manages the configuration for the scraper.

```python
from ipfs_huggingface_scraper_py import Config, get_config
```

#### Constructor

```python
Config(config_path=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str` or `None` | Path to the configuration file. If None, looks for 'config.toml' in the current directory. |

#### Methods

##### `get`

```python
get(section, key, default=None)
```

Get a configuration value.

| Parameter | Type | Description |
|-----------|------|-------------|
| `section` | `str` | Configuration section |
| `key` | `str` | Configuration key |
| `default` | `any` | Default value if not found |

##### `set`

```python
set(section, key, value)
```

Set a configuration value.

| Parameter | Type | Description |
|-----------|------|-------------|
| `section` | `str` | Configuration section |
| `key` | `str` | Configuration key |
| `value` | `any` | Value to set |

##### `save`

```python
save(path=None)
```

Save current configuration to a TOML file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` or `None` | Path to save to. If None, uses the path from initialization. |

##### `export_config_template`

```python
export_config_template(path)
```

Export a configuration template with default values and comments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to save the template |

#### Global Config Function

```python
get_config(config_path=None)
```

Get the global configuration instance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str` or `None` | Optional path to config file to use |

#### Example

```python
from ipfs_huggingface_scraper_py import Config

# Create a new config
config = Config()

# Set values for different entity types
config.set('scraper', 'entity_types', ['models', 'datasets', 'spaces'])
config.set('scraper', 'max_models', 100)
config.set('scraper', 'max_datasets', 50)
config.set('scraper', 'max_spaces', 25)
config.set('scraper', 'track_provenance', True)
config.set('api', 'authenticated', True)
config.set('storage', 'enable_knowledge_graph', True)

# Get values
entity_types = config.get('scraper', 'entity_types')
max_datasets = config.get('scraper', 'max_datasets')

# Save config
config.save('my_config.toml')

# Export template
config.export_config_template('config_template.toml')
```

### State Manager

Manages the state of scraping operations.

```python
from ipfs_huggingface_scraper_py import StateManager
```

#### Constructor

```python
StateManager(state_dir, state_file="scraper_state.json")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `state_dir` | `str` | Directory to store state files |
| `state_file` | `str` | Filename for the main state file |

#### Methods

##### `mark_model_processed`

```python
mark_model_processed(model_id)
```

Mark a model as processed (attempted).

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | The Hugging Face model ID |

##### `mark_model_completed`

```python
mark_model_completed(model_id)
```

Mark a model as successfully completed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | The Hugging Face model ID |

##### `mark_model_errored`

```python
mark_model_errored(model_id, error_message)
```

Mark a model as failed with an error.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | The Hugging Face model ID |
| `error_message` | `str` | Description of the error |

##### `is_model_processed`

```python
is_model_processed(model_id)
```

Check if a model has been processed (attempted).

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | The Hugging Face model ID |

##### `is_model_completed`

```python
is_model_completed(model_id)
```

Check if a model has been successfully completed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | The Hugging Face model ID |

##### `get_model_error`

```python
get_model_error(model_id)
```

Get the error message for a model if it failed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | The Hugging Face model ID |

##### `set_current_batch`

```python
set_current_batch(batch)
```

Set the current batch of models being processed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch` | `list` | List of model IDs in the current batch |

##### `update_position`

```python
update_position(position)
```

Update the current position in the overall model list.

| Parameter | Type | Description |
|-----------|------|-------------|
| `position` | `int` | Current position |

##### `set_total_discovered`

```python
set_total_discovered(count)
```

Set the total number of models discovered.

| Parameter | Type | Description |
|-----------|------|-------------|
| `count` | `int` | Total count of models |

##### `pause`

```python
pause()
```

Pause the scraping operation.

##### `resume`

```python
resume()
```

Resume the scraping operation.

##### `complete`

```python
complete()
```

Mark the scraping operation as completed.

##### `is_paused`

```python
is_paused()
```

Check if scraping is paused.

##### `is_completed`

```python
is_completed()
```

Check if scraping is completed.

##### `get_progress`

```python
get_progress()
```

Get a summary of current progress.

##### `reset`

```python
reset()
```

Reset the state to start fresh.

##### `create_checkpoint`

```python
create_checkpoint()
```

Manually create a checkpoint of the current state.

#### Example

```python
from ipfs_huggingface_scraper_py import StateManager

# Create a state manager
state_manager = StateManager(".scraper_state")

# Mark models as processed
state_manager.mark_model_processed("bert-base-uncased")
state_manager.mark_model_completed("bert-base-uncased")
state_manager.mark_model_errored("invalid-model", "Model not found")

# Check state
if state_manager.is_model_completed("bert-base-uncased"):
    print("Model has been processed successfully")

# Get progress
progress = state_manager.get_progress()
print(f"Processed {progress['models_processed']} models")
```

### Rate Limiter

Implements rate limiting for API calls.

```python
from ipfs_huggingface_scraper_py import RateLimiter
```

#### Constructor

```python
RateLimiter(default_rate=5.0, daily_quota=300000, authenticated_quota=1000000, max_retries=5)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `default_rate` | `float` | Default requests per second |
| `daily_quota` | `int` | Default daily quota for anonymous requests |
| `authenticated_quota` | `int` | Daily quota for authenticated requests |
| `max_retries` | `int` | Maximum number of retries for rate limited requests |

#### Methods

##### `set_authenticated`

```python
set_authenticated(is_authenticated=True)
```

Set authentication status to determine quota.

| Parameter | Type | Description |
|-----------|------|-------------|
| `is_authenticated` | `bool` | Whether to use authenticated quota |

##### `set_rate_limit`

```python
set_rate_limit(endpoint, rate)
```

Set custom rate limit for a specific endpoint.

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | API endpoint identifier |
| `rate` | `float` | Maximum requests per second |

##### `wait_if_needed`

```python
wait_if_needed(endpoint)
```

Wait if necessary to respect rate limits.

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | API endpoint identifier |

##### `record_success`

```python
record_success(endpoint, quota_cost=1)
```

Record a successful API call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | API endpoint identifier |
| `quota_cost` | `int` | Cost against the daily quota |

##### `record_rate_limited`

```python
record_rate_limited(endpoint)
```

Record a rate limited API call and apply backoff.

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | API endpoint identifier |

##### `execute_with_rate_limit`

```python
execute_with_rate_limit(endpoint, func, *args, **kwargs)
```

Execute a function with rate limiting applied.

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | API endpoint identifier |
| `func` | `callable` | Function to execute |
| `*args` | | Arguments to pass to the function |
| `**kwargs` | | Keyword arguments to pass to the function |

#### Example

```python
from ipfs_huggingface_scraper_py import RateLimiter
import requests

# Create a rate limiter
rate_limiter = RateLimiter(default_rate=5.0)
rate_limiter.set_authenticated(True)

# Set endpoint-specific rate
rate_limiter.set_rate_limit("list_models", 2.0)

# Execute with rate limit
def fetch_data(url):
    response = requests.get(url)
    return response.json()

data = rate_limiter.execute_with_rate_limit(
    "list_models", 
    fetch_data, 
    "https://huggingface.co/api/models"
)

# Manual rate limiting
rate_limiter.wait_if_needed("list_models")
response = requests.get("https://huggingface.co/api/models")
rate_limiter.record_success("list_models")
```

### IPFS Storage

Manages IPFS integration for storing model metadata and files.

```python
from ipfs_huggingface_scraper_py import IpfsStorage
```

#### Constructor

```python
IpfsStorage(config)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `dict` | Storage configuration dictionary |

#### Methods

##### `is_ipfs_available`

```python
is_ipfs_available()
```

Check if IPFS is available.

##### `add_file_to_ipfs`

```python
add_file_to_ipfs(file_path)
```

Add a file to IPFS.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to the file to add |

##### `add_directory_to_ipfs`

```python
add_directory_to_ipfs(dir_path)
```

Add a directory to IPFS recursively.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dir_path` | `str` | Path to the directory to add |

##### `pin_cid`

```python
pin_cid(cid)
```

Pin a CID to ensure it's not garbage collected.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cid` | `str` | Content identifier to pin |

##### `store_metadata_as_json`

```python
store_metadata_as_json(metadata, output_path=None)
```

Store metadata as JSON and add to IPFS.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metadata` | `dict` | Metadata dictionary to store |
| `output_path` | `str` or `None` | Path to save the JSON file (optional) |

##### `store_metadata_as_parquet`

```python
store_metadata_as_parquet(metadata_list, output_path=None)
```

Store metadata as Parquet and add to IPFS.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metadata_list` | `list` | List of metadata dictionaries to store |
| `output_path` | `str` or `None` | Path to save the Parquet file (optional) |

##### `store_model_files`

```python
store_model_files(model_dir)
```

Store all model files in IPFS.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_dir` | `str` | Directory containing model files |

##### `convert_jsonl_to_parquet`

```python
convert_jsonl_to_parquet(jsonl_path, parquet_path=None)
```

Convert JSONL file to Parquet format.

| Parameter | Type | Description |
|-----------|------|-------------|
| `jsonl_path` | `str` | Path to the JSONL file |
| `parquet_path` | `str` or `None` | Path to save the Parquet file (optional) |

##### `create_car_file`

```python
create_car_file(content_path, car_path=None)
```

Create a CAR (Content Addressable aRchive) file from content.

| Parameter | Type | Description |
|-----------|------|-------------|
| `content_path` | `str` | Path to the content (file or directory) |
| `car_path` | `str` or `None` | Path to save the CAR file (optional) |

##### `get_file_from_ipfs`

```python
get_file_from_ipfs(cid, output_path)
```

Retrieve a file from IPFS.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cid` | `str` | Content identifier of the file |
| `output_path` | `str` | Path to save the retrieved file |

##### `calculate_cid`

```python
calculate_cid(file_path)
```

Calculate the CID for a file without adding it to IPFS.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to the file |

#### Example

```python
from ipfs_huggingface_scraper_py import IpfsStorage

# Create storage with config
config = {
    "use_ipfs": True,
    "ipfs_add_options": {
        "pin": True,
        "wrap_with_directory": True
    }
}
storage = IpfsStorage(config)

# Store a file in IPFS
if storage.is_ipfs_available():
    cid = storage.add_file_to_ipfs("path/to/file.json")
    print(f"File added with CID: {cid}")
    
    # Pin the CID
    storage.pin_cid(cid)
    
    # Store metadata
    metadata = {"name": "Model Name", "description": "Model description"}
    json_path, json_cid = storage.store_metadata_as_json(metadata)
    
    # Store directory
    dir_cid = storage.store_model_files("path/to/model_dir")
```

## ProvenanceTracker

Tracks and manages provenance relationships between entities.

```python
from ipfs_huggingface_scraper_py import ProvenanceTracker
```

### Constructor

```python
ProvenanceTracker(storage_dir)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `storage_dir` | `str` | Directory for storing provenance information |

### Methods

#### `add_model_base_relationship`

```python
add_model_base_relationship(model_id, base_model_id)
```

Add a relationship indicating a model is derived from a base model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | ID of the derived model |
| `base_model_id` | `str` | ID of the base model |

#### `add_model_dataset_relationship`

```python
add_model_dataset_relationship(model_id, dataset_id, relationship_type="trained_on")
```

Add a relationship between a model and a dataset.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | ID of the model |
| `dataset_id` | `str` | ID of the dataset |
| `relationship_type` | `str` | Type of relationship (e.g., "trained_on", "evaluated_on") |

#### `add_space_entity_relationship`

```python
add_space_entity_relationship(space_id, entity_id, entity_type, relationship_type="uses")
```

Add a relationship between a space and another entity.

| Parameter | Type | Description |
|-----------|------|-------------|
| `space_id` | `str` | ID of the space |
| `entity_id` | `str` | ID of the related entity |
| `entity_type` | `str` | Type of entity ("model" or "dataset") |
| `relationship_type` | `str` | Type of relationship |

#### `get_model_base_models`

```python
get_model_base_models(model_id)
```

Get base models for a given model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | ID of the model |

#### `get_model_derived_models`

```python
get_model_derived_models(model_id)
```

Get models derived from a given model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | ID of the base model |

#### `get_model_datasets`

```python
get_model_datasets(model_id, relationship_type=None)
```

Get datasets related to a model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | ID of the model |
| `relationship_type` | `str` or `None` | Type of relationship to filter by (optional) |

#### `extract_relationships_from_metadata`

```python
extract_relationships_from_metadata(metadata, entity_id, entity_type)
```

Extract and store relationships from entity metadata.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metadata` | `dict` | Entity metadata dictionary |
| `entity_id` | `str` | ID of the entity |
| `entity_type` | `str` | Type of entity ("model", "dataset", or "space") |

#### `generate_provenance_graph`

```python
generate_provenance_graph(output_file)
```

Generate a graph representation of provenance relationships.

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_file` | `str` | Path to save the graph data |

### Example

```python
from ipfs_huggingface_scraper_py import ProvenanceTracker

# Initialize tracker
tracker = ProvenanceTracker("./provenance_data")

# Add relationships
tracker.add_model_base_relationship(
    "distilbert-base-uncased-finetuned-sst-2-english", 
    "distilbert-base-uncased"
)

tracker.add_model_dataset_relationship(
    "distilbert-base-uncased-finetuned-sst-2-english", 
    "sst-2", 
    "trained_on"
)

# Extract relationships from metadata
metadata = {"modelId": "bert-base-uncased", "tags": ["dataset:squad"]}
tracker.extract_relationships_from_metadata(metadata, "bert-large-squad", "model")

# Generate graph
tracker.generate_provenance_graph("provenance_network.json")
```

### UnifiedExport

Manages export of scraped entity data to Parquet files.

```python
from ipfs_huggingface_scraper_py import UnifiedExport
```

#### Constructor

```python
UnifiedExport(config=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `dict` or `None` | Configuration dictionary with export settings |

#### Methods

##### `normalize_schema`

```python
normalize_schema(entity_list, entity_type)
```

Normalize schema across different entity types to ensure consistency.

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity_list` | `list` | List of entity metadata dictionaries |
| `entity_type` | `str` | Type of entity ('model', 'dataset', or 'space') |

##### `save_dataframe_to_parquet_safely`

```python
save_dataframe_to_parquet_safely(df, filepath)
```

Saves DataFrame to Parquet with explicit schema handling for mixed types.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pandas.DataFrame` | DataFrame to save |
| `filepath` | `str` or `Path` | Path to save the Parquet file |

##### `load_existing_data`

```python
load_existing_data(filepath)
```

Load existing data from a Parquet file with fallback to CSV.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `Path` | Path to the Parquet file |

##### `store_entity_data`

```python
store_entity_data(entity_list, entity_type, output_path=None, merge_with_existing=True)
```

Store entity data as Parquet and add to IPFS if available.

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity_list` | `list` | List of entity metadata dictionaries |
| `entity_type` | `str` | Type of entity ('model', 'dataset', or 'space') |
| `output_path` | `str` or `None` | Path to save the Parquet file (optional) |
| `merge_with_existing` | `bool` | Whether to merge with existing data (if any) |

##### `store_unified_data`

```python
store_unified_data(models_list=None, datasets_list=None, spaces_list=None, output_path=None)
```

Store all entity types in a unified Parquet file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `models_list` | `list` or `None` | List of model metadata dictionaries (optional) |
| `datasets_list` | `list` or `None` | List of dataset metadata dictionaries (optional) |
| `spaces_list` | `list` or `None` | List of space metadata dictionaries (optional) |
| `output_path` | `str` or `None` | Path to save the unified Parquet file (optional) |

##### `get_entity_statistics`

```python
get_entity_statistics(filepath=None)
```

Get statistics about the stored entity data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str` or `None` | Path to the Parquet file (optional, uses default if not provided) |

#### Example

```python
from ipfs_huggingface_scraper_py import UnifiedExport
import json
import os

# Initialize the exporter
exporter = UnifiedExport({
    "data_dir": "./data",
    "use_ipfs": True  # Enable IPFS integration
})

# Load metadata from scraped data
def load_entity_metadata(base_dir, entity_type):
    entity_list = []
    entity_dir = os.path.join(base_dir, entity_type)
    
    for entity_name in os.listdir(entity_dir):
        metadata_path = os.path.join(entity_dir, entity_name, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                entity_list.append(metadata)
    
    return entity_list

# Load data for each entity type
scraped_dir = "./hf_data"
models_data = load_entity_metadata(scraped_dir, "models")
datasets_data = load_entity_metadata(scraped_dir, "datasets")
spaces_data = load_entity_metadata(scraped_dir, "spaces")

# Export to unified Parquet file
output_path, cid = exporter.store_unified_data(
    models_list=models_data,
    datasets_list=datasets_data,
    spaces_list=spaces_data
)

print(f"Exported data to: {output_path}")
if cid:
    print(f"Added to IPFS with CID: {cid}")

# Or export each entity type separately
models_path, _ = exporter.store_entity_data(models_data, 'model')
datasets_path, _ = exporter.store_entity_data(datasets_data, 'dataset')
spaces_path, _ = exporter.store_entity_data(spaces_data, 'space')

# Get statistics about the exported data
stats = exporter.get_entity_statistics(output_path)
print(f"Total entities: {stats['total_entities']}")
print(f"Entity types: {stats['entity_types']}")
```

## Command Line Interface

The IPFS HuggingFace Scraper provides a command-line interface for common operations.

### Global Options

| Option | Description |
|--------|-------------|
| `--log-level` | Set logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO |
| `--config` | Path to configuration file |

### Commands

#### `export-config`

Export a configuration template.

```bash
hf-scraper export-config --output CONFIG_PATH
```

| Option | Description |
|--------|-------------|
| `--output` | Path to save the configuration template. Required. |

#### `models`

Scrape models from HuggingFace Hub.

```bash
hf-scraper models [--max MAX] [--output-dir DIR] [--resume] [--token TOKEN]
```

| Option | Description |
|--------|-------------|
| `--max` | Maximum number of models to scrape |
| `--output-dir` | Directory to save scraped data |
| `--resume` | Resume a paused scraping operation |
| `--token` | HuggingFace API token |

#### `datasets`

Scrape datasets from HuggingFace Hub.

```bash
hf-scraper datasets [--max MAX] [--output-dir DIR] [--resume] [--token TOKEN]
```

| Option | Description |
|--------|-------------|
| `--max` | Maximum number of datasets to scrape |
| `--output-dir` | Directory to save scraped data |
| `--resume` | Resume a paused scraping operation |
| `--token` | HuggingFace API token |

#### `spaces`

Scrape spaces from HuggingFace Hub.

```bash
hf-scraper spaces [--max MAX] [--output-dir DIR] [--resume] [--token TOKEN]
```

| Option | Description |
|--------|-------------|
| `--max` | Maximum number of spaces to scrape |
| `--output-dir` | Directory to save scraped data |
| `--resume` | Resume a paused scraping operation |
| `--token` | HuggingFace API token |

#### `all`

Scrape models, datasets, and spaces from HuggingFace Hub.

```bash
hf-scraper all [--max-models MAX] [--max-datasets MAX] [--max-spaces MAX] [--output-dir DIR] [--token TOKEN]
```

| Option | Description |
|--------|-------------|
| `--max-models` | Maximum number of models to scrape |
| `--max-datasets` | Maximum number of datasets to scrape |
| `--max-spaces` | Maximum number of spaces to scrape |
| `--output-dir` | Base directory to save entities |
| `--token` | HuggingFace API token |

#### `export`

Export scraped data to a Parquet file.

```bash
hf-scraper export --input-dir DIR [--output-file FILE] [--entity-types TYPES] [--separate-files]
```

| Option | Description |
|--------|-------------|
| `--input-dir` | Directory containing scraped data. Required. |
| `--output-file` | Output Parquet file path (default: in data directory) |
| `--entity-types` | Entity types to include in the export (default: all) |
| `--separate-files` | Export each entity type to a separate file |

### Examples

```bash
# Export a configuration template
hf-scraper export-config --output my_config.toml

# Scrape models
hf-scraper models --max 100 --output-dir ./hf_data

# Scrape datasets
hf-scraper datasets --max 50 --output-dir ./hf_data

# Scrape spaces
hf-scraper spaces --max 25 --output-dir ./hf_data

# Scrape all entity types
hf-scraper all --max-models 100 --max-datasets 50 --max-spaces 25 --output-dir ./hf_data

# Use authentication
HF_API_TOKEN="your_token" hf-scraper models --max 100

# Resume a paused operation
hf-scraper models --resume --config my_config.toml

# Set logging level
hf-scraper models --max 100 --log-level DEBUG

# Export scraped data to a unified Parquet file
hf-scraper export --input-dir ./hf_data

# Export only models and datasets to separate files
hf-scraper export --input-dir ./hf_data --entity-types models datasets --separate-files

# Export to a specific output file
hf-scraper export --input-dir ./hf_data --output-file ./data/custom_export.parquet
```

### Environment Variables

The following environment variables can be used to configure the CLI:

| Variable | Description |
|----------|-------------|
| `HF_API_TOKEN` | HuggingFace API token for authenticated requests |
| `HF_OUTPUT_DIR` | Directory to save scraped data |
| `HF_MAX_MODELS` | Maximum number of models to scrape |
| `HF_MAX_DATASETS` | Maximum number of datasets to scrape |
| `HF_MAX_SPACES` | Maximum number of spaces to scrape |
| `HF_LOG_LEVEL` | Logging level (INFO, DEBUG, WARNING, ERROR) |
| `HF_ENTITY_TYPES` | Comma-separated list of entity types to scrape (models,datasets,spaces) |