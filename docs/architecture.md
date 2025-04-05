# Architecture

This document provides an overview of the IPFS HuggingFace Scraper architecture, its components, and their interactions.

## System Architecture

The scraper is built with a modular architecture that separates concerns into distinct components:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  Scraper Components                                      │
└───────────────────────────────────────┬─────────────────────────────────────────────────┘
                                        │
    ┌───────────────────────────────────┼───────────────────────────────────────────┐
    ▼                                   ▼                                           ▼
┌────────────────────┐          ┌────────────────────┐                  ┌─────────────────────┐
│  Enhanced Scraper  │          │ Datasets Scraper   │                  │   Spaces Scraper    │
│ (models)           │          │                    │                  │                     │
└─────────┬──────────┘          └──────────┬─────────┘                  └──────────┬──────────┘
          │                               │                                        │
          └───────────────────────────────┼────────────────────────────────────────┘
                                          │
    ┌───────────────────────────────────────────────────────────────────┐
    ▼                       ▼                       ▼                    ▼
┌─────────────┐         ┌───────────────┐       ┌────────────────┐  ┌────────────────┐
│    Config   │         │  State Manager │       │  Rate Limiter  │  │ ProvenanceTracker │
└─────────────┘         └───────────────┘       └────────────────┘  └────────────────┘
                            │                       │                    │
                            │                       │                    │
    ┌───────────────────────┼───────────────────────┼────────────────────┘
    ▼                       ▼                       ▼
┌─────────────┐         ┌───────────────┐       ┌────────────────┐
│ IPFS Storage│         │HuggingFace API│       │     CLI        │
└──────┬──────┘         └───────────────┘       └────────────────┘
       │
       ▼
┌─────────────┐
│  ipfs_kit   │
└─────────────┘
```

## Components

### Entity-Specific Scrapers

#### 1. Enhanced Scraper (`enhanced_scraper.py`)

The main orchestrator component for scraping models from HuggingFace Hub.

**Responsibilities:**
- Initializes all required components
- Discovers models from HuggingFace Hub
- Processes models in batches
- Handles errors and interruptions
- Provides resumable operations

#### 2. Datasets Scraper (`datasets/datasets_scraper.py`)

Specialized component for scraping datasets from HuggingFace Hub.

**Responsibilities:**
- Discovers datasets from HuggingFace Hub
- Processes datasets in batches
- Extracts dataset metadata and preview information
- Downloads dataset samples or preview content
- Creates appropriate directory structure for datasets

#### 3. Spaces Scraper (`spaces/spaces_scraper.py`)

Specialized component for scraping spaces from HuggingFace Hub.

**Responsibilities:**
- Discovers spaces from HuggingFace Hub
- Processes spaces in batches
- Extracts space metadata, thumbnails, and configuration
- Identifies related models and datasets used by spaces
- Creates appropriate directory structure for spaces

### Core Components

#### 4. State Manager (`state_manager.py`)

Manages the state of scraping operations, enabling resumable scraping.

**Responsibilities:**
- Checkpointing for long-running operations
- State persistence through JSON files
- Tracking of processed entities and their statuses
- Resumption of interrupted operations
- Thread-safety for concurrent operations

#### 5. Rate Limiter (`rate_limiter.py`)

Implements rate limiting for API calls with quota management.

**Responsibilities:**
- Enforces rate limits for different API endpoints
- Implements adaptive backoff for rate limit errors
- Provides quota distribution across multiple operations
- Tracks rate limit usage and remaining quota

#### 6. IPFS Storage (`ipfs_integration.py`)

Manages IPFS integration for storing entity metadata and files.

**Responsibilities:**
- Content-addressed storage of metadata
- Storage of entity files in IPFS
- Efficient conversion between formats (JSON, Parquet, CAR)
- Pin management for persistence in IPFS

#### 7. Configuration (`config.py`)

Configuration manager for the HuggingFace scraper.

**Responsibilities:**
- Loading and saving configuration from TOML files
- Falling back to environment variables and default values
- Providing a unified interface for configuration access
- Validating configuration values

#### 8. CLI (`cli.py`)

Command-line interface for the scraper.

**Responsibilities:**
- Parsing command-line arguments
- Executing appropriate scraper operations for different entity types
- Supporting the "all" command for scraping all entity types
- Supporting the "export" command for exporting to Parquet
- Displaying status and progress information
- Providing help and documentation

#### 9. Provenance Tracker (`provenance.py`)

Tracks relationships and provenance information between entities.

**Responsibilities:**
- Maintaining a graph of relationships between entities
- Tracking base model relationships
- Identifying dataset-model relationships
- Tracking spaces and their component usages
- Generating provenance graphs for visualization
- Extracting relationships from metadata

#### 10. Unified Export (`unified_export.py`)

Manages the export of scraped entity data to centralized Parquet files.

**Responsibilities:**
- Aggregating metadata from different entity types
- Normalizing schema across entity types
- Safely handling Parquet conversion with fallback mechanisms
- Merging new data with existing data
- IPFS integration for storage of Parquet files
- Providing statistics about stored entities

## Data Flow

The typical data flow through the scraper follows this pattern:

1. **Discovery**: The appropriate scraper identifies candidate entities via HuggingFace API
2. **Filtering**: Entities are filtered based on criteria (size, task, popularity, etc.)
3. **Processing**: Each entity is processed:
   - Metadata is collected and normalized
   - Files are downloaded and validated
   - State Manager tracks progress
   - Rate Limiter controls API usage
   - Provenance information is extracted
4. **Storage**: Processed data is stored via IPFS Storage
5. **Provenance**: Relationships between entities are recorded
6. **Completion**: State Manager records successful completion

## Threading Model

The scraper uses a multi-threaded approach for concurrency:

1. The main thread orchestrates the overall process
2. A thread pool executes entity processing in parallel
3. Each component uses thread-safe operations:
   - State Manager uses an RLock for concurrent access
   - Rate Limiter tracks per-endpoint request timing
   - IPFS Storage handles concurrent file operations
   - ProvenanceTracker manages concurrent relationship updates

## Error Handling

The scraper implements robust error handling at multiple levels:

1. **Entity-level recovery**: Errors processing individual entities don't stop the scraper
2. **Endpoint-level backoff**: Rate limiting errors trigger adaptive backoff
3. **Global error recovery**: Critical errors are caught, logged, and state is preserved
4. **Resumable operations**: Interrupted scraping can be resumed from the last checkpoint

## Integration Points

The scraper integrates with several external systems:

1. **HuggingFace Hub API**:
   - Entity discovery and metadata retrieval
   - File downloads for models, datasets, and spaces
   - Rate-limited for quota compliance
   - Support for different entity types (models, datasets, spaces)

2. **IPFS**:
   - Content addressing for entity metadata and files
   - Distributed storage across IPFS network
   - Pin management for content persistence
   - Knowledge graph storage for provenance information

3. **Local filesystem**:
   - Temporary storage for downloaded files
   - Persistent state storage
   - Configurable local cache
   - Relationship data storage

## Entity Storage Structure

Each entity type is stored with a consistent directory structure:

### Models
```
output_dir/
└── models/
    ├── owner__model_name/
    │   ├── metadata.json       # Model metadata
    │   ├── config.json         # Downloaded model configuration
    │   ├── ipfs_cid.txt        # IPFS CID for the model (if IPFS enabled)
    │   └── ...                 # Other model files
```

### Datasets
```
output_dir/
└── datasets/
    ├── owner__dataset_name/
    │   ├── metadata.json       # Dataset metadata
    │   ├── preview_info.json   # Dataset preview information
    │   ├── ipfs_cid.txt        # IPFS CID for the dataset (if IPFS enabled)
    │   └── ...                 # Dataset sample files
```

### Spaces
```
output_dir/
└── spaces/
    ├── owner__space_name/
    │   ├── metadata.json       # Space metadata
    │   ├── thumbnail_info.json # Space thumbnail information
    │   ├── relationships.json  # Space entity relationships
    │   ├── ipfs_cid.txt        # IPFS CID for the space (if IPFS enabled)
    │   └── ...                 # Space configuration files
```

### Provenance
```
output_dir/
└── provenance/
    ├── entity_relationships.json  # All entity relationships
    ├── provenance_graph.json      # Generated provenance graph
    ├── model_relationships/       # Model-specific relationships
    ├── dataset_relationships/     # Dataset-specific relationships
    └── space_relationships/       # Space-specific relationships
```

### Unified Data Export
```
data/
└── huggingface_hub_metadata/
    ├── all_entities_metadata.parquet  # Unified Parquet file with all entity types
    ├── all_models_metadata.parquet    # Models-only Parquet file
    ├── all_datasets_metadata.parquet  # Datasets-only Parquet file
    ├── all_spaces_metadata.parquet    # Spaces-only Parquet file
    └── README.md                      # Information about the data
```

The Parquet files provide a consolidated view of all scraped entities with the following benefits:
- Efficient storage with columnar compression
- Fast querying and filtering capabilities
- Schema normalization across entity types
- Unified access to all entity metadata
- Content-addressed storage with IPFS integration
- Optimized for data analysis with tools like pandas and PyArrow