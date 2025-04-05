# IPFS HuggingFace Scraper

A specialized module for scraping and processing content from HuggingFace Hub, including models, datasets, and spaces. This tool stores the metadata and content in content-addressable storage via IPFS.

## Features

- **Multiple Content Type Support**: Scrape models, datasets, and spaces from HuggingFace Hub
- **Robust State Management**: Resume interrupted scraping operations seamlessly
- **Rate Limiting**: Adaptive rate limiting to respect API quotas
- **IPFS Integration**: Store content in distributed IPFS storage
- **Provenance Tracking**: Track relationships and lineage between entities
- **Metadata Extraction**: Extract rich metadata from all entity types

## Installation

```bash
pip install ipfs_huggingface_scraper_py
```

## Dependencies

The scraper depends on the following core modules:
- `ipfs_datasets_py`: For dataset operations
- `ipfs_kit_py`: For IPFS interaction
- `huggingface_hub`: For interacting with the HuggingFace Hub API

## Usage

### Command Line Interface

```bash
# Scrape models
hf-scraper models --max 100 --output-dir ./hf_data

# Scrape datasets
hf-scraper datasets --max 50 --output-dir ./hf_data

# Scrape spaces
hf-scraper spaces --max 25 --output-dir ./hf_data

# Scrape all entity types
hf-scraper all --max-models 100 --max-datasets 50 --max-spaces 25 --output-dir ./hf_data

# Export configuration template
hf-scraper export-config --output config_template.toml
```

### Python API

```python
from ipfs_huggingface_scraper_py import EnhancedScraper, DatasetsScraper, SpacesScraper

# Scrape models
models_scraper = EnhancedScraper()
models_scraper.scrape_models(max_models=100)

# Scrape datasets
datasets_scraper = DatasetsScraper()
datasets_scraper.scrape_datasets(max_datasets=50)

# Scrape spaces
spaces_scraper = SpacesScraper()
spaces_scraper.scrape_spaces(max_spaces=25)

# Track provenance
from ipfs_huggingface_scraper_py import ProvenanceTracker
tracker = ProvenanceTracker("./provenance_data")
tracker.add_model_base_relationship("distilbert-base-uncased-finetuned-sst-2-english", "distilbert-base-uncased")
```

## Configuration

The scraper can be configured via a TOML file. You can generate a template configuration file with:

```bash
hf-scraper export-config --output config.toml
```

Key configuration sections include:

### Scraper Configuration

```toml
[scraper]
# Output directory for scraped data
output_dir = "hf_model_data"

# Maximum number of entities to scrape (None means unlimited)
max_models = 100
max_datasets = 50
max_spaces = 25

# Entity types to scrape (can include "models", "datasets", "spaces")
entity_types = ["models", "datasets", "spaces"]

# Track provenance information between entities
track_provenance = true

# Save metadata for each entity
save_metadata = true

# File to download for models (usually config.json)
filename_to_download = "config.json"

# Maximum rows for dataset preview
dataset_preview_max_rows = 100

# Batch size for processing entities
batch_size = 100

# Skip entities that have already been processed
skip_existing = true
```

### API Configuration

```toml
[api]
# Base URL for the HuggingFace API
base_url = "https://huggingface.co"

# API token for authentication (more rate limits with authentication)
api_token = null

# Rate limits
anonymous_rate_limit = 5.0  # requests per second
authenticated_rate_limit = 10.0  # requests per second
```

### Storage Configuration

```toml
[storage]
# Use IPFS for storage
use_ipfs = true

# IPFS add options
ipfs_add_options = { pin = true, wrap_with_directory = true }

# Local cache settings
local_cache_max_size_gb = 10
local_cache_retention_days = 30

# Enable knowledge graph for storing relationships
enable_knowledge_graph = true
```

### Provenance Configuration

```toml
[provenance]
# Extract base model information
extract_base_models = true

# Extract dataset relationships
extract_dataset_relationships = true

# Extract evaluation dataset information
extract_evaluation_datasets = true

# Track entity version history
track_version_history = true

# Maximum depth for relationship traversal
max_relationship_depth = 3
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)