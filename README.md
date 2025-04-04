# IPFS HuggingFace Scraper

A specialized module for scraping and processing model metadata from HuggingFace Hub with robust IPFS integration.

## Features

- Discovers and scrapes model metadata from HuggingFace
- Processes and structures this metadata into appropriate formats (JSON, Parquet)
- Stores the metadata and model files in content-addressable storage via IPFS
- Provides robust state management for resumable operations
- Implements rate limiting with adaptive backoff for API quotas
- Supports concurrent processing for improved performance

## Installation

```bash
pip install ipfs_huggingface_scraper_py
```

## Quick Start

### Command Line Usage

```bash
# Initialize a configuration file
hf-scraper init --output config.toml

# Edit the configuration file as needed

# Start scraping
hf-scraper scrape --config config.toml --max-models 100

# Resume a paused scraping operation
hf-scraper resume --config config.toml

# Check scraping status
hf-scraper status --config config.toml
```

### Python API Usage

```python
from ipfs_huggingface_scraper_py import EnhancedScraper, Config

# Create configuration
config = Config()
config.set("scraper", "max_models", 100)
config.set("scraper", "output_dir", "hf_data")
config.save("config.toml")

# Create and run scraper
scraper = EnhancedScraper("config.toml")
scraper.scrape_models()

# Resume a paused scraping operation
scraper.resume()
```

## Components

The scraper consists of several key components:

1. **Model Discovery Service**
   - Finds and enumerates models on HuggingFace Hub
   - Implements paginated listing of models
   - Filters models based on configurable criteria

2. **Metadata Collector**
   - Retrieves detailed metadata for each model
   - Extracts information from model cards, config files, etc.
   - Normalizes metadata into consistent format

3. **State Manager**
   - Tracks overall scraping progress
   - Maintains persistent state for resumable operations
   - Records success/failure status of individual operations

4. **Rate Limiter**
   - Enforces HuggingFace API rate limits
   - Implements adaptive backoff strategies
   - Provides quota management across distributed scrapers

5. **IPFS Storage**
   - Interfaces with IPFS via ipfs_kit_py
   - Manages content-addressed storage of metadata and files
   - Handles efficient conversion between formats

## Configuration

The scraper is highly configurable through a TOML configuration file. Main configuration sections:

### Scraper Settings

```toml
[scraper]
output_dir = "hf_model_data"
max_models = 100
save_metadata = true
filename_to_download = "config.json"
batch_size = 100
skip_existing = true
retry_delay_seconds = 5
max_retries = 2
log_level = "INFO"
```

### API Settings

```toml
[api]
base_url = "https://huggingface.co"
api_token = ""
authenticated = false
anonymous_rate_limit = 5.0
authenticated_rate_limit = 10.0
daily_anonymous_quota = 300000
daily_authenticated_quota = 1000000
max_retries = 5
timeout = 30
```

### Storage Settings

```toml
[storage]
use_ipfs = true
local_cache_max_size_gb = 10
local_cache_retention_days = 30
metadata_format = "parquet"

[storage.ipfs_add_options]
pin = true
wrap_with_directory = true
chunker = "size-262144"
hash = "sha2-256"
```

### State Management

```toml
[state]
state_dir = ".scraper_state"
checkpoint_interval = 50
auto_resume = true
```

## Dependencies

This module integrates with several other components:

- **ipfs_datasets_py**: Provides dataset management, conversion, and GraphRAG capabilities
- **ipfs_kit_py**: Handles low-level IPFS operations and integration
- **ipfs_model_manager_py**: Manages model deployment and serving

## License

GNU Affero General Public License v3 or later (AGPLv3+)