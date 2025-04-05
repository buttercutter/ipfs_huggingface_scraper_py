"""
IPFS HuggingFace Scraper
------------------------

A specialized module for scraping and processing content from HuggingFace Hub, including
models, datasets, and spaces. This module is responsible for:

1. Discovering and scraping content metadata from HuggingFace
2. Processing and structuring this metadata into appropriate formats
3. Storing the metadata and content in content-addressable storage via IPFS
4. Tracking provenance and relationships between entities
5. Providing efficient lookup and search capabilities
"""

__version__ = "0.2.0"

from .config import Config, get_config
from .state_manager import StateManager
from .rate_limiter import RateLimiter
from .ipfs_integration import IpfsStorage
from .enhanced_scraper import EnhancedScraper
from .scraper import scrape_hf_models_and_configs
from .provenance import ProvenanceTracker
from .datasets import DatasetsScraper
from .spaces import SpacesScraper
from .unified_export import UnifiedExport

__all__ = [
    "Config",
    "get_config",
    "StateManager",
    "RateLimiter",
    "IpfsStorage",
    "EnhancedScraper",
    "scrape_hf_models_and_configs",
    "ProvenanceTracker",
    "DatasetsScraper",
    "SpacesScraper",
    "UnifiedExport",
]