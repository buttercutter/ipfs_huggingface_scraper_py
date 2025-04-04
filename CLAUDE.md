# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The `ipfs_huggingface_scraper_py` is a specialized module for scraping and processing model metadata from HuggingFace Hub. This module is responsible for:

1. Discovering and scraping model metadata from HuggingFace
2. Processing and structuring this metadata into appropriate formats
3. Storing the metadata in content-addressable storage via IPFS
4. Providing efficient lookup and search capabilities

This module serves as an integration layer between HuggingFace Hub and a distributed IPFS-based model registry. It works alongside several other modules (imported as dependencies):

- **ipfs_datasets_py**: Provides dataset management, conversion, and GraphRAG capabilities
- **ipfs_kit_py**: Handles low-level IPFS operations and integration
- **ipfs_model_manager_py**: Manages model deployment and serving


## Development Environment

### Build & Test Commands
- **Install**: `pip install -e .`
- **Build**: `python setup.py build`
- **Run all tests**: `python -m test.test`
- **Run single test**: `python -m test.test_ipfs_kit` or `python -m test.test_storacha_kit`
- **Run API server**: `uvicorn ipfs_kit_py.api:app --reload --port 8000`
- **Generate AST**: `python -m astroid ipfs_kit_py > ast_analysis.json`
- **Check for duplications**: `pylint --disable=all --enable=duplicate-code ipfs_kit_py`

### Development Guidelines
- **Test-First Development**: All new features must first be developed in the test/ folder
- **Feature Isolation**: Do not modify code outside of test/ until fully debugged
- **API Exposure**: All functionality should be exposed via FastAPI endpoints
- **Performance Focus**: Use memory-mapped structures and Arrow C Data Interface for low-latency IPC
- **Code Analysis**: Maintain an abstract syntax tree (AST) of the project to identify and prevent code duplication
- **DRY Principle**: Use the AST to enforce Don't Repeat Yourself by detecting similar code structures

### Testing Strategy

The project follows a comprehensive testing approach to ensure reliability and maintainability:

#### Test Organization
- **Unit Tests**: Located in the `test/` directory with file naming pattern `test_*.py`
- **Integration Tests**: Also in `test/` but focused on component interactions
- **Performance Tests**: Specialized tests for measuring throughput and latency

#### Recent Test Improvements
- **Mock Integration**: Fixed PyArrow mocking for cluster state helpers
- **Role-Based Architecture**: Improved fixtures for master/worker/leecher node testing
- **Gateway Compatibility**: Enhanced testing with proper filesystem interface mocking
- **LibP2P Integration**: Fixed tests to work without external dependencies
- **Parameter Validation**: Corrected constructor argument handling in tests
- **Interface Focus**: Made tests more resilient to implementation changes by focusing on behaviors rather than implementation details

#### Test Patterns
1. **Fixture-Based Testing**: Use pytest fixtures for test setup and teardown
2. **Mocking IPFS Daemon**: Use subprocess mocking to avoid actual daemon dependency
3. **Property-Based Testing**: Use hypothesis for edge case discovery
4. **Snapshot Testing**: For configuration and schema verification
5. **Parallelized Test Execution**: For faster feedback cycles
6. **PyArrow Patching**: Special handling for PyArrow Schema objects and Table methods
7. **Logging Suppression**: Context managers to control test output noise

#### Continuous Integration Integration
- Tests are run on every PR and commit to main branch
- Test reports and coverage metrics are generated automatically
- Performance regression tests compare against baseline benchmarks

## Code Style Guidelines
- **Imports**: Standard library first, third-party next, project imports last
- **Variables/Functions**: Use snake_case
- **Classes**: Use snake_case (this is project-specific, differs from PEP 8)
- **Indentation**: 4 spaces, no tabs
- **Error Handling**: Use try/except blocks, catch specific exceptions when possible
- **No Type Annotations**: Project doesn't use typing hints
- **Docstrings**: Not consistently used

The project is a wrapper around HuggingFace Transformers that adds IPFS model management capabilities, allowing models to be downloaded from HTTP/S3/IPFS based on availability and speed.

### API Integration Points
- **IPFS HTTP API**: REST interface (localhost:5001/api/v0) for core IPFS operations
- **IPFS Cluster API**: REST interface (localhost:9094/api/v0) for cluster coordination
- **IPFS Cluster Proxy**: Proxied IPFS API (localhost:9095/api/v0)
- **IPFS Gateway**: Content retrieval via HTTP (localhost:8080/ipfs/[cid])
- **IPFS Socket Interface**: Unix socket for high-performance local communication (/ip4/127.0.0.1/tcp/4001)
- **IPFS Unix Socket API**: On Linux, Kubo can be configured to expose its API via a Unix domain socket instead of HTTP, providing lower-latency communication for local processes. This can be configured in the IPFS config file by modifying the `API.Addresses` field to include a Unix socket path (e.g., `/unix/path/to/socket`).

These APIs enable creating "swarms of swarms" by allowing distributed clusters to communicate across networks and coordinate content pinning, replication, and routing across organizational boundaries. Socket interfaces provide lower-latency communication for high-performance local operations, with Unix domain sockets being particularly efficient for inter-process communication on the same machine.

## IPFS Core Concepts

The IPFS (InterPlanetary File System) architecture is built on several key concepts and components that are essential to understand for effective implementation:

### Content Addressing
- **Content Identifiers (CIDs)**: Unique fingerprints of content based on cryptographic hashes
- **Multihash Format**: Extensible hashing format supporting multiple hash algorithms (default: SHA-256)
- **Base32/Base58 Encoding**: Human-readable representations of binary CIDs
- **Version Prefixes**: CIDv0 (base58btc-encoded SHA-256) vs CIDv1 (self-describing, supports multicodec)

### Data Structures
- **Merkle DAG (Directed Acyclic Graph)**: Core data structure for content-addressed storage
- **IPLD (InterPlanetary Linked Data)**: Framework for creating data models with content-addressable linking
- **UnixFS**: File system abstraction built on IPLD for representing traditional files/directories
- **Blocks**: Raw data chunks that form the atomic units of the Merkle DAG

### Network Components
- **DHT (Distributed Hash Table)**: Distributed key-value store for content routing
- **Bitswap**: Protocol for exchanging blocks between peers
- **libp2p**: Modular networking stack powering IPFS peer-to-peer communication
- **MultiFormats**: Self-describing protocols, formats, and addressing schemes
- **IPNS (InterPlanetary Name System)**: Mutable naming system for content addressing

### Node Types
- **Full Nodes**: Store and serve content, participate in DHT
- **Gateway Nodes**: Provide HTTP access to IPFS content
- **Client Nodes**: Lightweight nodes that rely on others for content routing/storage
- **Bootstrap Nodes**: Well-known nodes that help new nodes join the network
- **Relay Nodes**: Assist with NAT traversal and indirect connections

### Key Operations
- **Adding Content**: Hash-based deduplication and chunking strategies
- **Retrieving Content**: Resolution process from CID to data
- **Pinning**: Mechanism to prevent content from being garbage collected
- **Publishing**: Making content discoverable through DHT/IPNS
- **Garbage Collection**: Process for reclaiming storage from unpinned content


### Dependencies
The project depends on the following key libraries:
- **requests**: For HTTP requests to HuggingFace API
- **tqdm**: For progress tracking during scraping
- **pandas**: For data manipulation and export to Parquet
- **pyarrow**: For working with Arrow and Parquet formats
- **multiformats**: For CID generation and IPFS integration
- **pyyaml/toml**: For configuration management

These dependencies are imported from the core modules:
- **ipfs_datasets_py**: Imported for dataset operations
- **ipfs_kit_py**: Imported for IPFS interaction
- **ipfs_model_manager_py**: Imported for model deployment

### Module-Specific Guidelines
- **Focus on Scraping Logic**: Keep scraping logic separate from storage concerns
- **Delegate Storage Operations**: Use ipfs_datasets_py and ipfs_kit_py for storage operations
- **Maintain State**: Implement robust state tracking for resumable operations
- **Respect Rate Limits**: Implement appropriate rate limiting for HuggingFace API
- **Data Redundancy**: Ensure data is properly replicated and persisted in storage layer
- **Separation of Concerns**: Keep scraping, processing, and storage operations modular
- **Defensive Programming**: Handle API changes and unexpected responses gracefully
- **Configurable Behavior**: Make scraping parameters configurable (batch size, retry counts, etc.)

### Testing Strategy

The project should follow a comprehensive testing approach specific to scraping operations:

#### Test Organization
- **Unit Tests**: Test individual scraper components in isolation
- **Integration Tests**: Test integration with HuggingFace API
- **Storage Tests**: Test proper storage of scraped data via integrated modules
- **Recovery Tests**: Test resumption of interrupted scraping operations
- **Rate Limit Tests**: Test adherence to API rate limits
- **Mock Tests**: Use HuggingFace API mocks to avoid excessive API calls during testing

#### Test Patterns
1. **Fixture-Based Testing**: Use pytest fixtures for test setup and teardown
2. **API Mocking**: Mock HuggingFace API responses for reproducible tests
3. **State Testing**: Test proper state management across scraping sessions
4. **Error Handling**: Test recovery from network errors and API failures
5. **Incremental Scraping**: Test detection of already-scraped models
6. **Performance Testing**: Benchmark scraping operations for large model sets

## Code Style Guidelines
- **Imports**: Standard library first, third-party next, project imports last
- **Variables/Functions**: Use snake_case
- **Classes**: Use PascalCase
- **Indentation**: 4 spaces, no tabs
- **Error Handling**: Use try/except blocks, catch specific exceptions when possible
- **Type Annotations**: Use type hints for function parameters and return values
- **Docstrings**: Include description, parameters, returns, and examples
- **Constants**: Define constants at module level in UPPER_CASE
- **Line Length**: Maximum 100 characters per line
- **Batched Operations**: Use tqdm for progress bars, implement batch saving
- **Logging**: Use the structured logging approach with appropriate levels
- **Configuration**: Use TOML files for configuration, with sensible defaults

## IPFS Core Concepts

Understanding the following IPFS concepts is crucial for effective implementation:

### Content Addressing
- **Content Identifiers (CIDs)**: Unique fingerprints of content based on cryptographic hashes
- **Multihash Format**: Extensible hashing format supporting multiple hash algorithms (default: SHA-256)
- **Base32/Base58 Encoding**: Human-readable representations of binary CIDs
- **Version Prefixes**: CIDv0 (base58btc-encoded SHA-256) vs CIDv1 (self-describing, supports multicodec)

### Data Structures
- **Merkle DAG (Directed Acyclic Graph)**: Core data structure for content-addressed storage
- **IPLD (InterPlanetary Linked Data)**: Framework for creating data models with content-addressable linking
- **UnixFS**: File system abstraction built on IPLD for representing traditional files/directories
- **Blocks**: Raw data chunks that form the atomic units of the Merkle DAG

## Scraper Architecture

### Component Overview

The scraper consists of several key components with distinct responsibilities:

1. **Model Discovery Service**
   - Finds and enumerates models on HuggingFace Hub
   - Implements paginated listing of models
   - Filters models based on configurable criteria
   - Tracks already-discovered models

2. **Metadata Collector**
   - Retrieves detailed metadata for each model
   - Extracts information from model cards, config files, etc.
   - Normalizes metadata into consistent format
   - Validates metadata completeness and correctness

3. **File Acquisition System**
   - Downloads model-related files (config.json, tokenizer.json, etc.)
   - Implements resumable downloads for large files
   - Verifies file integrity using checksums
   - Organizes files in appropriate directory structure

4. **State Manager**
   - Tracks overall scraping progress
   - Maintains persistent state for resumable operations
   - Records success/failure status of individual operations
   - Provides checkpointing capabilities

5. **Rate Limiter**
   - Enforces HuggingFace API rate limits
   - Implements adaptive backoff strategies
   - Provides quota management across distributed scrapers
   - Logs rate limit events for monitoring

6. **Storage Adapter**
   - Interfaces with IPFS via ipfs_kit_py
   - Manages content-addressed storage of metadata and files
   - Handles efficient conversion between formats (via ipfs_datasets_py)
   - Ensures proper replication and persistence

### Data Flow

The typical data flow through the scraper follows this pattern:

1. **Discovery**: The Model Discovery Service identifies candidate models
2. **Filtering**: Models are filtered based on criteria (size, task, popularity, etc.)
3. **Metadata Collection**: The Metadata Collector retrieves detailed information
4. **File Acquisition**: Necessary files are downloaded and validated
5. **Processing**: Metadata and files are processed into appropriate formats
6. **Storage**: Processed data is stored in IPFS via the Storage Adapter
7. **Indexing**: Content is indexed for efficient lookup and search
8. **State Update**: The State Manager records successful completion

### Integration with Other Modules

The scraper leverages other modules for specialized functions:

1. **ipfs_datasets_py Integration**
   - Used for structured dataset management
   - Provides conversion between formats (Parquet, CAR, etc.)
   - Enables vector embedding for semantic search
   - Offers knowledge graph capabilities for model relationships

2. **ipfs_kit_py Integration**
   - Handles low-level IPFS operations
   - Manages content addressing and CID generation
   - Provides pin management for persistence
   - Enables distributed storage across IPFS network

3. **ipfs_model_manager_py Integration**
   - Manages deployment of scraped models
   - Provides model serving capabilities
   - Enables model version tracking
   - Connects models to applications

### Storage Strategy

The scraper maintains redundant storage to ensure data availability:

1. **Local File Cache**
   - Maintains local copies of downloaded files
   - Implements cleanup policies based on space constraints
   - Provides fast access for frequently used models
   - Serves as fallback if network storage is unavailable

2. **IPFS Content-Addressed Storage**
   - Stores all metadata and files with content addressing
   - Ensures data integrity through cryptographic verification
   - Enables global deduplication of model files
   - Provides distributed access to model data

3. **Structured Data Storage**
   - Stores metadata in efficient formats (Parquet)
   - Enables SQL-like queries on model properties
   - Facilitates efficient filtering and search
   - Supports batch operations on metadata

### Error Handling and Recovery

The scraper implements robust error handling and recovery:

1. **Network Failures**
   - Retries with exponential backoff for transient failures
   - Circuit breaker pattern for persistent failures
   - Logs detailed error information for troubleshooting
   - Preserves partial progress for later resumption

2. **API Limitations**
   - Respects rate limits with adaptive throttling
   - Handles API changes with flexible parsing
   - Gracefully degrades functionality when endpoints are unavailable
   - Caches responses when appropriate

3. **Storage Failures**
   - Implements multi-tiered fallback storage options
   - Verifies successful storage before updating state
   - Allows manual recovery and retry of failed operations
   - Maintains integrity of stored data

## Implementation Plan

### Phase 1: Core Scraper Implementation (1-2 months)
1. **Basic Scraper Framework**
   - Implement HuggingFace API client with authentication
   - Create model discovery and listing capabilities
   - Develop metadata extraction from various file types
   - Implement basic file download functionality
   - Create scraper state management system

2. **Resilient Operations**
   - Add retry mechanisms with exponential backoff
   - Implement rate limiting with proper API quota management
   - Create checkpointing for long-running operations
   - Develop resumable downloads for large files
   - Add robust error handling and recovery

3. **Storage Integration**
   - Create integration with ipfs_kit_py for IPFS storage
   - Implement CID generation for all downloaded files
   - Develop storage adapters for Parquet and CAR formats
   - Ensure proper pinning and replication of content
   - Create metadata indexes for efficient lookup

4. **Configuration and Monitoring**
   - Develop configuration system with TOML files
   - Create logging framework for operation tracking
   - Implement progress reporting with tqdm
   - Add telemetry for scraper performance
   - Create health check capabilities

### Phase 2: Advanced Features (2-3 months)
1. **Metadata Enrichment**
   - Extract structured data from model cards
   - Generate embeddings for semantic search
   - Compute model statistics and metrics
   - Create categorization and tagging system
   - Develop model relationship extraction

2. **Distributed Scraping**
   - Implement coordinator-worker architecture
   - Create task distribution system
   - Develop synchronization between scrapers
   - Add work stealing for load balancing
   - Implement distributed progress tracking

3. **Advanced Storage**
   - Integrate with IPFS Cluster for better replication
   - Implement tiered storage (hot/cold storage)
   - Create efficient chunking strategies
   - Develop storage optimization techniques
   - Add automatic garbage collection

4. **Search and Discovery**
   - Implement metadata-based search
   - Create vector similarity search
   - Develop faceted search capabilities
   - Add relationship-based model discovery
   - Implement query optimization

### Phase 3: Integration and Scaling (2-3 months)
1. **Complete System Integration**
   - Integrate deeply with ipfs_datasets_py for GraphRAG
   - Connect with ipfs_model_manager_py for deployment
   - Develop unified API across components
   - Create event-based communication between modules
   - Implement cross-component monitoring

2. **Performance Optimization**
   - Optimize scraping throughput
   - Implement parallel processing capabilities
   - Create caching strategies
   - Optimize storage operations
   - Add performance benchmarking

3. **Advanced Features**
   - Implement incremental updates for model changes
   - Create differential storage for model versions
   - Develop model comparison capabilities
   - Add webhook notifications for model updates
   - Implement subscriptions for model changes

## Data Formats and Schemas

### Metadata Schema
```json
{
  "model_id": "string",  // Original HuggingFace model ID
  "name": "string",  // Model name
  "description": "string",  // Model description
  "architecture": "string",  // Model architecture
  "task_type": ["string"],  // List of supported tasks
  "language": ["string"],  // List of supported languages
  "license": "string",  // License information
  "downloads": "number",  // Download count
  "last_updated": "string",  // ISO timestamp
  "tags": ["string"],  // List of tags
  "author": "string",  // Model author
  "files": [  // List of model files
    {
      "filename": "string",
      "size_bytes": "number",
      "cid": "string",  // IPFS Content ID
      "sha256": "string"  // File hash
    }
  ],
  "config": {  // Extracted from config.json
    "hidden_size": "number",
    "num_attention_heads": "number",
    "num_hidden_layers": "number",
    // Other architecture-specific config
  },
  "performance": {  // Performance metrics
    "task": "string",
    "dataset": "string",
    "metric": "string",
    "value": "number"
  },
  "relationships": [  // Model relationships
    {
      "relationship_type": "string",  // e.g., "base_model", "fine_tuned_from"
      "target_model_id": "string"
    }
  ],
  "embedding_vector": "bytes",  // Binary vector representation
  "cid": "string"  // IPFS Content ID for the metadata record
}
```

### Storage Organization
1. **IPFS Structure**
   - Each model gets a directory in IPFS
   - Metadata stored as individual IPLD objects
   - Files stored with content addressing
   - Directory structure preserved in UnixFS

2. **Parquet Schema**
   - Efficient columnar storage for metadata
   - Partitioned by model categories
   - Optimized for filtering and search
   - Includes CID references to IPFS content

3. **Local Cache Structure**
   - Organized by model ID for easy lookup
   - Contains original files and processed metadata
   - Implements eviction policies based on access frequency
   - Keeps track of what's already been uploaded to IPFS

## Development Workflow

1. **Feature Development**
   - Create feature branch from main
   - Develop and test in isolation
   - Write comprehensive tests for new functionality
   - Review code for style and performance

2. **Integration Testing**
   - Test integration with other modules
   - Verify proper data flow through system
   - Test with realistic HuggingFace data
   - Benchmark performance with production-like load

3. **Deployment**
   - Create Docker containers for deployment
   - Deploy to testing environment
   - Verify functionality in integrated setting
   - Release to production with proper monitoring

## Documentation Guidelines
- Document all public APIs with clear docstrings
- Create examples for common usage patterns
- Maintain architecture documentation
- Update schema documentation when data formats change
- Document integration points with other modules
- Create troubleshooting guides for common issues
- Document configuration options with examples
- Keep a changelog of scraper behavior changes

## HuggingFace API Integration

The scraper interacts with several HuggingFace API endpoints:

1. **Model Hub API**
   - `/api/models`: Lists available models with pagination
   - `/api/models/{model_id}`: Retrieves detailed model information
   - `/api/datasets`: Lists available datasets for model evaluation

2. **Model Files API**
   - Direct file downloads from `huggingface.co/{owner}/{model_id}/resolve/{revision}/{filename}`
   - Raw file viewing from `huggingface.co/{owner}/{model_id}/raw/{revision}/{filename}`

3. **Authentication**
   - Uses API tokens for authenticated requests
   - Respects token scopes and permissions
   - Implements secure token handling

4. **Rate Limiting**
   - Adapts to API rate limits (currently 300K requests/day for anonymous, 1M for authenticated)
   - Implements exponential backoff when limits are reached
   - Provides quota distribution across multiple scrapers