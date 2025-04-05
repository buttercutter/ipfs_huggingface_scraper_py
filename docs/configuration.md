# Configuration

The IPFS HuggingFace Scraper is highly configurable through TOML configuration files. This document details all available configuration options and provides examples for common scenarios.

## Configuration Methods

Configuration can be provided through several methods (in order of precedence):

1. Command-line arguments
2. Environment variables
3. Configuration file (TOML)
4. Default values

## Configuration File

The configuration file uses the TOML format. Here's a complete example:

```toml
# HuggingFace Scraper Configuration

# Scraper settings
[scraper]
output_dir = "hf_data"                # Directory to save scraped data
max_models = 100                      # Maximum models to scrape (null for no limit)
max_datasets = 50                     # Maximum datasets to scrape (null for no limit)
max_spaces = 25                       # Maximum spaces to scrape (null for no limit)
entity_types = ["models", "datasets", "spaces"]  # Entity types to scrape
track_provenance = true               # Track provenance information
save_metadata = true                  # Whether to save entity metadata
filename_to_download = "config.json"  # Specific file to download from each model
dataset_preview_max_rows = 100        # Maximum rows for dataset preview
batch_size = 100                      # Number of entities to process in each batch
skip_existing = true                  # Skip already processed entities
retry_delay_seconds = 5               # Delay before retrying failed downloads
max_retries = 2                       # Maximum retries for failed operations
log_level = "INFO"                    # Logging level (INFO, DEBUG, WARNING, ERROR)
metadata_fields = [                   # Metadata fields to extract
    "id", 
    "modelId", 
    "sha", 
    "lastModified", 
    "tags", 
    "pipeline_tag", 
    "siblings", 
    "config"
]

# API settings
[api]
base_url = "https://huggingface.co"                # Base URL for HuggingFace API
api_token = ""                                     # API token (leave empty for anonymous)
authenticated = false                              # Whether to use authentication
anonymous_rate_limit = 5.0                         # Requests per second for anonymous
authenticated_rate_limit = 10.0                    # Requests per second for authenticated
daily_anonymous_quota = 300000                     # Daily quota for anonymous (300K)
daily_authenticated_quota = 1000000                # Daily quota for authenticated (1M)
max_retries = 5                                    # Maximum retries for rate limit errors
timeout = 30                                       # Request timeout in seconds

# Storage settings
[storage]
use_ipfs = true                                    # Whether to use IPFS storage
local_cache_max_size_gb = 10                       # Maximum local cache size in GB
local_cache_retention_days = 30                    # Days to retain files in local cache
metadata_format = "parquet"                        # Metadata storage format (json, parquet)
enable_knowledge_graph = true                      # Enable relationship knowledge graph

# IPFS add options
[storage.ipfs_add_options]
pin = true                                         # Whether to pin files in IPFS
wrap_with_directory = true                         # Wrap content in directory
chunker = "size-262144"                            # Chunking strategy
hash = "sha2-256"                                  # Hash algorithm

# State management settings
[state]
state_dir = ".scraper_state"                       # Directory for state files
checkpoint_interval = 50                           # Create checkpoint every N entities
auto_resume = true                                 # Automatically resume interrupted operations

# Provenance settings
[provenance]
extract_base_models = true                         # Extract base model information
extract_dataset_relationships = true               # Extract dataset relationships
extract_evaluation_datasets = true                 # Extract evaluation dataset info
track_version_history = true                       # Track entity version history
max_relationship_depth = 3                         # Max depth for relationship traversal
```

## Environment Variables

The following environment variables can be used to override configuration:

- `HF_API_TOKEN`: HuggingFace API token for authenticated requests
- `HF_OUTPUT_DIR`: Directory to save scraped data
- `HF_MAX_MODELS`: Maximum number of models to scrape
- `HF_MAX_DATASETS`: Maximum number of datasets to scrape
- `HF_MAX_SPACES`: Maximum number of spaces to scrape
- `HF_LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `HF_ENTITY_TYPES`: Comma-separated list of entity types to scrape (models,datasets,spaces)

## Configuration Sections

### Scraper Settings

| Option | Description | Default | Type |
|--------|-------------|---------|------|
| `output_dir` | Directory to save scraped data | `"hf_data"` | string |
| `max_models` | Maximum models to scrape (null for no limit) | `null` | integer or null |
| `max_datasets` | Maximum datasets to scrape (null for no limit) | `null` | integer or null |
| `max_spaces` | Maximum spaces to scrape (null for no limit) | `null` | integer or null |
| `entity_types` | Entity types to scrape | `["models"]` | list of strings |
| `track_provenance` | Track provenance information | `true` | boolean |
| `save_metadata` | Whether to save entity metadata | `true` | boolean |
| `filename_to_download` | Specific file to download from each model | `"config.json"` | string |
| `dataset_preview_max_rows` | Maximum rows for dataset preview | `100` | integer |
| `batch_size` | Number of entities to process in each batch | `100` | integer |
| `skip_existing` | Skip already processed entities | `true` | boolean |
| `retry_delay_seconds` | Delay before retrying failed downloads | `5` | integer |
| `max_retries` | Maximum retries for failed operations | `2` | integer |
| `log_level` | Logging level | `"INFO"` | string |
| `metadata_fields` | Metadata fields to extract | `[...]` | list of strings |

### API Settings

| Option | Description | Default | Type |
|--------|-------------|---------|------|
| `base_url` | Base URL for HuggingFace API | `"https://huggingface.co"` | string |
| `api_token` | API token (leave empty for anonymous) | `""` | string |
| `authenticated` | Whether to use authentication | `false` | boolean |
| `anonymous_rate_limit` | Requests per second for anonymous | `5.0` | float |
| `authenticated_rate_limit` | Requests per second for authenticated | `10.0` | float |
| `daily_anonymous_quota` | Daily quota for anonymous | `300000` | integer |
| `daily_authenticated_quota` | Daily quota for authenticated | `1000000` | integer |
| `max_retries` | Maximum retries for rate limit errors | `5` | integer |
| `timeout` | Request timeout in seconds | `30` | integer |

### Storage Settings

| Option | Description | Default | Type |
|--------|-------------|---------|------|
| `use_ipfs` | Whether to use IPFS storage | `true` | boolean |
| `local_cache_max_size_gb` | Maximum local cache size in GB | `10` | integer |
| `local_cache_retention_days` | Days to retain files in local cache | `30` | integer |
| `metadata_format` | Metadata storage format | `"parquet"` | string |
| `enable_knowledge_graph` | Enable relationship knowledge graph | `true` | boolean |

#### IPFS Add Options

| Option | Description | Default | Type |
|--------|-------------|---------|------|
| `pin` | Whether to pin files in IPFS | `true` | boolean |
| `wrap_with_directory` | Wrap content in directory | `true` | boolean |
| `chunker` | Chunking strategy | `"size-262144"` | string |
| `hash` | Hash algorithm | `"sha2-256"` | string |

### State Management Settings

| Option | Description | Default | Type |
|--------|-------------|---------|------|
| `state_dir` | Directory for state files | `".scraper_state"` | string |
| `checkpoint_interval` | Create checkpoint every N entities | `50` | integer |
| `auto_resume` | Automatically resume interrupted operations | `true` | boolean |

### Provenance Settings

| Option | Description | Default | Type |
|--------|-------------|---------|------|
| `extract_base_models` | Extract base model information | `true` | boolean |
| `extract_dataset_relationships` | Extract dataset relationships | `true` | boolean |
| `extract_evaluation_datasets` | Extract evaluation dataset info | `true` | boolean |
| `track_version_history` | Track entity version history | `true` | boolean |
| `max_relationship_depth` | Max depth for relationship traversal | `3` | integer |

## Configuration Examples

### Minimal Models-Only Scraping

```toml
[scraper]
output_dir = "hf_models_only"
max_models = 10
entity_types = ["models"]
save_metadata = true

[storage]
use_ipfs = false
```

### Datasets-Only Configuration

```toml
[scraper]
output_dir = "hf_datasets_only"
max_datasets = 20
entity_types = ["datasets"]
dataset_preview_max_rows = 50

[api]
api_token = "YOUR_HF_TOKEN"
authenticated = true

[storage]
use_ipfs = true
metadata_format = "parquet"
```

### Comprehensive Multi-Entity Configuration

```toml
[scraper]
output_dir = "hf_complete"
entity_types = ["models", "datasets", "spaces"]
max_models = 100
max_datasets = 50
max_spaces = 25
track_provenance = true
batch_size = 50
skip_existing = true

[api]
api_token = "YOUR_HF_TOKEN"
authenticated = true
authenticated_rate_limit = 10.0

[storage]
use_ipfs = true
enable_knowledge_graph = true
metadata_format = "parquet"

[provenance]
extract_base_models = true
extract_dataset_relationships = true
extract_evaluation_datasets = true
max_relationship_depth = 3
```

## Using with Python API

```python
from ipfs_huggingface_scraper_py import (
    Config, 
    EnhancedScraper, 
    DatasetsScraper, 
    SpacesScraper,
    ProvenanceTracker
)

# Load existing config
config = Config('path/to/config.toml')

# Or create a new config
config = Config()
config.set('scraper', 'entity_types', ['models', 'datasets', 'spaces'])
config.set('scraper', 'max_models', 50)
config.set('scraper', 'max_datasets', 25)
config.set('scraper', 'max_spaces', 10)
config.set('api', 'authenticated', True)
config.set('api', 'api_token', 'YOUR_HF_TOKEN')
config.set('provenance', 'extract_base_models', True)

# Save config
config.save('my_config.toml')

# Create model scraper with config
model_scraper = EnhancedScraper('my_config.toml')
model_scraper.scrape_models()

# Create dataset scraper with same config
dataset_scraper = DatasetsScraper('my_config.toml')
dataset_scraper.scrape_datasets()

# Create space scraper with same config
space_scraper = SpacesScraper('my_config.toml')
space_scraper.scrape_spaces()

# Work with provenance information
tracker = ProvenanceTracker('/path/to/provenance/storage')
tracker.extract_relationships_from_metadata(metadata, 'model-id', 'model')
```

## Generating a Template

To generate a configuration template file:

```bash
hf-scraper export-config --output config.toml
```

Then edit the generated file as needed.