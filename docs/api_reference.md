# API Reference

This document provides a comprehensive reference for the IPFS HuggingFace Scraper's Python API and CLI.

## Python API

### Enhanced Scraper

The main scraper class that orchestrates the entire scraping process.

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

Start the scraping process.

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

# Set values
config.set('scraper', 'max_models', 100)
config.set('api', 'authenticated', True)

# Get values
max_models = config.get('scraper', 'max_models')
is_authenticated = config.get('api', 'authenticated')

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

## Command Line Interface

The IPFS HuggingFace Scraper provides a command-line interface for common operations.

### Global Options

| Option | Description |
|--------|-------------|
| `--log-level` | Set logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO |

### Commands

#### `init`

Initialize a new configuration file.

```bash
hf-scraper init --output CONFIG_PATH
```

| Option | Description |
|--------|-------------|
| `--output` | Path to save the configuration template. Default: config.toml |

#### `scrape`

Scrape models from HuggingFace Hub.

```bash
hf-scraper scrape [--config CONFIG_PATH] [--max-models MAX] [--output-dir DIR] [--api-token TOKEN]
```

| Option | Description |
|--------|-------------|
| `--config` | Path to configuration file |
| `--max-models` | Maximum number of models to scrape |
| `--output-dir` | Directory to save scraped data |
| `--api-token` | HuggingFace API token |

#### `resume`

Resume a paused scraping operation.

```bash
hf-scraper resume [--config CONFIG_PATH]
```

| Option | Description |
|--------|-------------|
| `--config` | Path to configuration file |

#### `status`

Show status of scraping operation.

```bash
hf-scraper status [--config CONFIG_PATH]
```

| Option | Description |
|--------|-------------|
| `--config` | Path to configuration file |

### Examples

```bash
# Initialize a configuration file
hf-scraper init --output my_config.toml

# Start scraping with max 100 models
hf-scraper scrape --config my_config.toml --max-models 100

# Use API token
hf-scraper scrape --api-token "hf_xxxxxxxxxxx"

# Resume a paused operation
hf-scraper resume --config my_config.toml

# Check status
hf-scraper status --config my_config.toml

# Set logging level
hf-scraper scrape --config my_config.toml --log-level DEBUG
```