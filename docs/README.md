# IPFS HuggingFace Scraper Documentation

This directory contains comprehensive documentation for the IPFS HuggingFace Scraper module.

## Contents

- [Architecture](architecture.md) - System architecture and component interactions
- [Configuration](configuration.md) - Detailed configuration options and examples
- [API Reference](api_reference.md) - Python API and CLI reference
- [Integration](integration.md) - Integration with IPFS and other modules
- [Tutorials](tutorials/README.md) - Step-by-step tutorials for common operations
- [Development](development.md) - Development guidelines and workflow

## Overview

The IPFS HuggingFace Scraper is a specialized module for scraping and processing model metadata from HuggingFace Hub. This module is responsible for:

1. Discovering and scraping model metadata from HuggingFace
2. Processing and structuring this metadata into appropriate formats
3. Storing the metadata in content-addressable storage via IPFS
4. Providing efficient lookup and search capabilities

This module serves as an integration layer between HuggingFace Hub and a distributed IPFS-based model registry, working alongside several other modules:

- **ipfs_datasets_py**: Provides dataset management, conversion, and GraphRAG capabilities
- **ipfs_kit_py**: Handles low-level IPFS operations and integration
- **ipfs_model_manager_py**: Manages model deployment and serving

## Quick Start

### Installation

```bash
pip install ipfs_huggingface_scraper_py
```

### Basic Usage

```python
from ipfs_huggingface_scraper_py import EnhancedScraper

# Create and run scraper with default configuration
scraper = EnhancedScraper()
scraper.scrape_models(max_models=100)
```

### Command Line Usage

```bash
# Initialize a configuration file
hf-scraper init --output config.toml

# Start scraping
hf-scraper scrape --config config.toml --max-models 100
```

## Key Features

- **Robust State Management**: Supports resumable operations with checkpointing
- **Rate Limiting**: Implements adaptive rate limiting for API quotas
- **IPFS Integration**: Stores metadata and files in content-addressable storage
- **Configurable**: Extensive configuration options via TOML files
- **Concurrent Processing**: Leverages multi-threading for improved performance
- **Command Line Interface**: User-friendly CLI for common operations