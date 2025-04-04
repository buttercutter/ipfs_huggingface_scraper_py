# Architecture

This document provides an overview of the IPFS HuggingFace Scraper architecture, its components, and their interactions.

## System Architecture

The scraper is built with a modular architecture that separates concerns into distinct components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Enhanced Scraper                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
    ┌───────────────────────────┼───────────────────────┐
    ▼                           ▼                       ▼
┌─────────────┐         ┌───────────────┐       ┌────────────────┐
│    Config   │         │  State Manager │       │  Rate Limiter  │
└─────────────┘         └───────────────┘       └────────────────┘
                                │                       │
                                │                       │
    ┌───────────────────────────┼───────────────────────┐
    ▼                           ▼                       ▼
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

### 1. Enhanced Scraper (`enhanced_scraper.py`)

The main orchestrator component that coordinates the entire scraping process.

**Responsibilities:**
- Initializes all other components
- Discovers models from HuggingFace Hub
- Processes models in batches
- Handles errors and interruptions
- Provides resumable operations

### 2. State Manager (`state_manager.py`)

Manages the state of scraping operations, enabling resumable scraping.

**Responsibilities:**
- Checkpointing for long-running operations
- State persistence through JSON files
- Tracking of processed models and their statuses
- Resumption of interrupted operations
- Thread-safety for concurrent operations

### 3. Rate Limiter (`rate_limiter.py`)

Implements rate limiting for API calls with quota management.

**Responsibilities:**
- Enforces rate limits for different API endpoints
- Implements adaptive backoff for rate limit errors
- Provides quota distribution across multiple operations
- Tracks rate limit usage and remaining quota

### 4. IPFS Storage (`ipfs_integration.py`)

Manages IPFS integration for storing model metadata and files.

**Responsibilities:**
- Content-addressed storage of model metadata
- Storage of model files in IPFS
- Efficient conversion between formats (JSON, Parquet, CAR)
- Pin management for persistence in IPFS

### 5. Configuration (`config.py`)

Configuration manager for the HuggingFace scraper.

**Responsibilities:**
- Loading and saving configuration from TOML files
- Falling back to environment variables and default values
- Providing a unified interface for configuration access
- Validating configuration values

### 6. CLI (`cli.py`)

Command-line interface for the scraper.

**Responsibilities:**
- Parsing command-line arguments
- Executing appropriate scraper operations
- Displaying status and progress information
- Providing help and documentation

## Data Flow

The typical data flow through the scraper follows this pattern:

1. **Discovery**: The Enhanced Scraper identifies candidate models via HuggingFace API
2. **Filtering**: Models are filtered based on criteria (size, task, popularity, etc.)
3. **Processing**: Each model is processed:
   - Metadata is collected and normalized
   - Files are downloaded and validated
   - State Manager tracks progress
   - Rate Limiter controls API usage
4. **Storage**: Processed data is stored via IPFS Storage
5. **Completion**: State Manager records successful completion

## Threading Model

The scraper uses a multi-threaded approach for concurrency:

1. The main thread orchestrates the overall process
2. A thread pool executes model processing in parallel
3. Each component uses thread-safe operations:
   - State Manager uses an RLock for concurrent access
   - Rate Limiter tracks per-endpoint request timing
   - IPFS Storage handles concurrent file operations

## Error Handling

The scraper implements robust error handling at multiple levels:

1. **Model-level recovery**: Errors processing individual models don't stop the scraper
2. **Endpoint-level backoff**: Rate limiting errors trigger adaptive backoff
3. **Global error recovery**: Critical errors are caught, logged, and state is preserved
4. **Resumable operations**: Interrupted scraping can be resumed from the last checkpoint

## Integration Points

The scraper integrates with several external systems:

1. **HuggingFace Hub API**:
   - Model discovery and metadata retrieval
   - File downloads
   - Rate-limited for quota compliance

2. **IPFS**:
   - Content addressing for model metadata and files
   - Distributed storage across IPFS network
   - Pin management for content persistence

3. **Local filesystem**:
   - Temporary storage for downloaded files
   - Persistent state storage
   - Configurable local cache