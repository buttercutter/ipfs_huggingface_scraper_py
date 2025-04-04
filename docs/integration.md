# Integration

This document explains how the IPFS HuggingFace Scraper integrates with IPFS and other modules.

## IPFS Integration

The scraper integrates with IPFS through the `ipfs_kit_py` module, which provides a high-level API for interacting with IPFS.

### IPFS Architecture

The scraper uses the following IPFS components:

- **IPFS Daemon**: The core IPFS node that provides content-addressed storage
- **IPFS HTTP API**: REST interface for IPFS operations
- **IPFS CLI**: Command-line interface for IPFS (used via subprocess)
- **IPFS Gateway**: HTTP gateway for accessing IPFS content

### Content Addressing

All data stored by the scraper is content-addressed using IPFS Content Identifiers (CIDs):

1. **Model Metadata**: Stored as JSON or Parquet files with unique CIDs
2. **Model Files**: Each file (e.g., config.json) is stored with its own CID
3. **Model Directories**: Each model gets a directory in IPFS with its own CID

### Content Storage

The scraper uses several mechanisms for storing content in IPFS:

1. **Direct Add**: Files are added directly to IPFS
2. **Directory Add**: Model directories are added recursively
3. **Pinning**: CIDs are pinned to ensure content persistence
4. **CAR Files**: Content-addressable archives for efficient transport

### IPFS Storage Structure

The storage structure in IPFS follows this pattern:

```
/ipfs/<root-cid>/
└── models/
    ├── model1/
    │   ├── metadata.json
    │   ├── config.json
    │   └── ipfs_cid.txt
    ├── model2/
    │   ├── metadata.json
    │   ├── config.json
    │   └── ipfs_cid.txt
    └── ...
```

## Integration with ipfs_datasets_py

The scraper integrates with `ipfs_datasets_py` for dataset operations:

### Dataset Conversion

The scraper can convert between different dataset formats:

1. **JSONL to Parquet**: Conversion for efficient columnar storage
2. **Parquet to CAR**: Creation of content-addressable archives
3. **Metadata Embedding**: Vector embedding generation for semantic search

### GraphRAG Capabilities

Integration with the GraphRAG capabilities of `ipfs_datasets_py` enables:

1. **Knowledge Graph Creation**: Relationships between models
2. **Semantic Search**: Finding models by similarity
3. **Model Lineage Tracking**: Tracking model derivation and fine-tuning

## Integration with ipfs_model_manager_py

The scraper provides data to `ipfs_model_manager_py` for model deployment:

### Model Management

1. **Model Discovery**: Finding available models in the scraped data
2. **CID-based Retrieval**: Retrieving models by their CIDs
3. **Model Metadata**: Providing detailed information about models

### Model Deployment

1. **Content Verification**: Ensuring model file integrity via CIDs
2. **Efficient Distribution**: Using IPFS for content-addressed distribution
3. **Version Management**: Tracking model versions and updates

## HuggingFace API Integration

The scraper interacts with several HuggingFace API endpoints:

### HuggingFace Hub API

1. **Model Listing**: `/api/models` for listing available models
2. **Model Info**: `/api/models/{model_id}` for detailed model information
3. **File Access**: Specific file downloads and raw file access

### Authentication

1. **API Tokens**: Using HuggingFace API tokens for authenticated requests
2. **Rate Limits**: Respecting different rate limits for anonymous vs. authenticated requests
3. **Token Security**: Secure handling of API tokens via environment variables

## Integration Patterns

The scraper uses several integration patterns:

### Content-Addressed Integration

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  HuggingFace  │────▶│     Scraper   │────▶│     IPFS      │
│     API       │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
                             │                     ▲
                             │                     │
                             ▼                     │
                      ┌───────────────┐     ┌───────────────┐
                      │  Local Files  │────▶│ ipfs_datasets │
                      │               │     │               │
                      └───────────────┘     └───────────────┘
```

### Service Integration

```
┌─────────────────────────────────────────────────────────────┐
│                       Scraper                               │
│                                                             │
│  ┌────────────────┐   ┌────────────────┐   ┌──────────────┐ │
│  │ Model Discovery│──▶│ Data Processing│──▶│ IPFS Storage │ │
│  └────────────────┘   └────────────────┘   └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
           │                                        │
           ▼                                        ▼
┌─────────────────────┐                 ┌─────────────────────┐
│   HuggingFace API   │                 │   ipfs_kit_py       │
└─────────────────────┘                 └─────────────────────┘
                                                   │
                                                   ▼
                                        ┌─────────────────────┐
                                        │   IPFS Network      │
                                        └─────────────────────┘
```

## Integration Examples

### Scraping Models and Storing in IPFS

```python
from ipfs_huggingface_scraper_py import EnhancedScraper, Config

# Configure scraper with IPFS enabled
config = Config()
config.set("storage", "use_ipfs", True)
config.save("ipfs_config.toml")

# Create and run scraper
scraper = EnhancedScraper("ipfs_config.toml")
scraper.scrape_models(max_models=10)

# Access the ipfs_storage component directly
ipfs_storage = scraper.ipfs_storage

# Store additional metadata as Parquet
metadata_list = [{"model_id": "model1", "task": "text-classification"}, 
                 {"model_id": "model2", "task": "translation"}]
parquet_path, parquet_cid = ipfs_storage.store_metadata_as_parquet(metadata_list)
print(f"Metadata stored in IPFS with CID: {parquet_cid}")
```

### Integrating with ipfs_datasets_py

```python
from ipfs_huggingface_scraper_py import IpfsStorage
from ipfs_datasets_py import IpfsDataset

# Create IPFS storage
config = {"use_ipfs": True}
ipfs_storage = IpfsStorage(config)

# Store model files in IPFS
model_cid = ipfs_storage.store_model_files("path/to/model")

# Create IPFS dataset
dataset = IpfsDataset()

# Add model to dataset
dataset.add_record({
    "type": "model",
    "name": "My Model",
    "cid": model_cid,
    "metadata": {
        "task": "classification",
        "size": "base"
    }
})

# Save dataset to IPFS
dataset_cid = dataset.save()
print(f"Dataset stored in IPFS with CID: {dataset_cid}")
```

### Using the Scraper with ipfs_model_manager_py

```python
from ipfs_huggingface_scraper_py import EnhancedScraper
from ipfs_model_manager_py import IpfsModelManager

# Scrape models
scraper = EnhancedScraper()
scraper.scrape_models(max_models=5)

# Initialize model manager
model_manager = IpfsModelManager()

# Deploy models from scraped data
output_dir = scraper.config.get("scraper", "output_dir")
models_path = f"{output_dir}/models"

for model_dir in os.listdir(models_path):
    model_path = os.path.join(models_path, model_dir)
    
    # Read the stored CID
    with open(os.path.join(model_path, "ipfs_cid.txt"), "r") as f:
        model_cid = f.read().strip()
    
    # Deploy model from IPFS
    model_manager.deploy_model_from_ipfs(model_cid, model_dir.replace("__", "/"))
```

## Configuration for Integration

The scraper can be configured for different integration scenarios:

### IPFS-Only Integration

```toml
[storage]
use_ipfs = true
local_cache_max_size_gb = 5
metadata_format = "json"

[storage.ipfs_add_options]
pin = true
wrap_with_directory = true
```

### Dataset Integration

```toml
[storage]
use_ipfs = true
metadata_format = "parquet"

[scraper]
output_dir = "dataset_models"
```

### Model Manager Integration

```toml
[scraper]
save_metadata = true
filename_to_download = "config.json"

[storage]
use_ipfs = true

[storage.ipfs_add_options]
pin = true
```