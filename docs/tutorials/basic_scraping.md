# Basic Scraping Tutorial

This tutorial demonstrates how to use the IPFS HuggingFace Scraper for basic content scraping from Hugging Face Hub, including models, datasets, and spaces.

## Prerequisites

- Python 3.7 or later
- IPFS HuggingFace Scraper installed (`pip install ipfs_huggingface_scraper_py`)
- Optional: HuggingFace API token for authenticated requests

## Step 1: Initialize a Models Scraper

First, let's create a basic script to initialize and run the models scraper:

```python
# basic_models_scraper.py
from ipfs_huggingface_scraper_py import EnhancedScraper, Config

# Create a configuration
config = Config()

# Configure basic settings
config.set("scraper", "output_dir", "hf_data")
config.set("scraper", "max_models", 10)  # Start with a small number for testing
config.set("scraper", "entity_types", ["models"])
config.set("storage", "use_ipfs", False)  # Disable IPFS for now

# Save configuration
config.save("scraper_config.toml")

# Initialize and run the scraper
scraper = EnhancedScraper("scraper_config.toml")
print("Starting models scraper...")
scraper.scrape_models()
print("Models scraping completed!")
```

Run the script to start scraping:

```bash
python basic_models_scraper.py
```

## Step 2: Scrape Datasets

Let's now try scraping some datasets from Hugging Face:

```python
# basic_datasets_scraper.py
from ipfs_huggingface_scraper_py import DatasetsScraper, Config

# Create a configuration
config = Config()

# Configure basic settings
config.set("scraper", "output_dir", "hf_data")
config.set("scraper", "max_datasets", 5)  # Start with a small number
config.set("scraper", "entity_types", ["datasets"])
config.set("scraper", "dataset_preview_max_rows", 10)  # Limit preview size
config.set("storage", "use_ipfs", False)

# Save configuration
config.save("datasets_config.toml")

# Initialize and run the datasets scraper
scraper = DatasetsScraper("datasets_config.toml")
print("Starting datasets scraper...")
scraper.scrape_datasets()
print("Datasets scraping completed!")
```

Run the datasets script:

```bash
python basic_datasets_scraper.py
```

## Step 3: Scrape Spaces

Now let's scrape some Hugging Face Spaces:

```python
# basic_spaces_scraper.py
from ipfs_huggingface_scraper_py import SpacesScraper, Config

# Create a configuration
config = Config()

# Configure basic settings
config.set("scraper", "output_dir", "hf_data")
config.set("scraper", "max_spaces", 5)  # Start with a small number
config.set("scraper", "entity_types", ["spaces"])
config.set("storage", "use_ipfs", False)

# Save configuration
config.save("spaces_config.toml")

# Initialize and run the spaces scraper
scraper = SpacesScraper("spaces_config.toml")
print("Starting spaces scraper...")
scraper.scrape_spaces()
print("Spaces scraping completed!")
```

Run the spaces script:

```bash
python basic_spaces_scraper.py
```

## Step 4: Examine the Results

After the scrapers complete, you can examine the results in the output directory:

```python
# examine_results.py
import os
import json

def explore_entity_directory(base_dir, entity_type):
    entity_dir = os.path.join(base_dir, entity_type)
    if not os.path.exists(entity_dir):
        print(f"No {entity_type} directory found")
        return
        
    entities = os.listdir(entity_dir)
    print(f"\nFound {len(entities)} {entity_type}:")
    
    # Show first 3 entities
    for entity_name in entities[:3]:
        entity_path = os.path.join(entity_dir, entity_name)
        print(f"\n--- {entity_type.capitalize()[:-1]}: {entity_name} ---")
        
        # List files in the entity directory
        files = os.listdir(entity_path)
        print(f"Files: {', '.join(files)}")
        
        # Check for metadata
        metadata_path = os.path.join(entity_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                try:
                    metadata = json.load(f)
                    print(f"ID: {metadata.get('id', 'Unknown')}")
                    
                    # Show entity-specific metadata
                    if entity_type == "models":
                        print(f"Pipeline: {metadata.get('pipeline_tag', 'Unknown')}")
                    elif entity_type == "datasets":
                        print(f"Size: {metadata.get('dataset_size', 'Unknown')}")
                    elif entity_type == "spaces":
                        print(f"SDK: {metadata.get('sdk', 'Unknown')}")
                        
                    if 'tags' in metadata:
                        print(f"Tags: {', '.join(metadata.get('tags', []))[:100]}...")
                except:
                    print("Error reading metadata file")

# Main script
output_dir = "hf_data"
print("Exploring scraped data in:", output_dir)

# Explore each entity type
explore_entity_directory(output_dir, "models")
explore_entity_directory(output_dir, "datasets")
explore_entity_directory(output_dir, "spaces")
```

Run the examination script:

```bash
python examine_results.py
```

## Step 5: Scrape All Entity Types at Once

Instead of running separate scrapers, you can use the all-in-one approach with the "all" command:

```python
# all_entities_scraper.py
import os
from ipfs_huggingface_scraper_py import Config

# Create a configuration
config = Config()

# Configure basic settings
config.set("scraper", "output_dir", "hf_all_data")
config.set("scraper", "entity_types", ["models", "datasets", "spaces"])
config.set("scraper", "max_models", 10)
config.set("scraper", "max_datasets", 5)
config.set("scraper", "max_spaces", 3) 
config.set("scraper", "track_provenance", True)  # Enable provenance tracking
config.set("storage", "use_ipfs", False)

# Save configuration
config.save("all_entities_config.toml")

# Run from command line (easier than Python API for multi-entity scraping)
import subprocess
subprocess.run(["hf-scraper", "all", "--config", "all_entities_config.toml"])
```

Run the all-in-one script:

```bash
python all_entities_scraper.py
```

## Step 6: Configure Authentication

For better rate limits, let's update our script to use authentication:

```python
# authenticated_scraper.py
import os
from ipfs_huggingface_scraper_py import Config

# Get API token from environment variable or set directly
api_token = os.environ.get("HF_API_TOKEN", "")

# Create a configuration
config = Config()

# Configure basic settings
config.set("scraper", "output_dir", "hf_authenticated_data")
config.set("scraper", "entity_types", ["models", "datasets", "spaces"])
config.set("scraper", "max_models", 10)
config.set("scraper", "max_datasets", 5)
config.set("scraper", "max_spaces", 3)
config.set("scraper", "track_provenance", True)
config.set("storage", "use_ipfs", False)

# Configure authentication
config.set("api", "api_token", api_token)
config.set("api", "authenticated", bool(api_token))

# Save configuration
config.save("auth_config.toml")

# Run from command line
import subprocess
subprocess.run(["hf-scraper", "all", "--config", "auth_config.toml"])
```

Set your API token and run the script:

```bash
export HF_API_TOKEN="your_token_here"
python authenticated_scraper.py
```

## Step 7: Track Provenance

Let's examine and enhance the provenance relationships between entities:

```python
# provenance_example.py
import os
import json
from ipfs_huggingface_scraper_py import ProvenanceTracker

# Initialize the provenance tracker (use the same directory as configured in scraper)
tracker = ProvenanceTracker("./hf_authenticated_data/provenance")

# Add some explicit relationships
# Example 1: Model derived from a base model
tracker.add_model_base_relationship(
    "distilbert-base-uncased-finetuned-sst-2-english", 
    "distilbert-base-uncased"
)

# Example 2: Model trained on a dataset
tracker.add_model_dataset_relationship(
    "distilbert-base-uncased-finetuned-sst-2-english", 
    "sst-2", 
    "trained_on"
)

# Example 3: Space using a model
tracker.add_space_entity_relationship(
    "sentence-transformers/sentence-similarity", 
    "sentence-transformers/all-MiniLM-L6-v2", 
    "model", 
    "uses"
)

# Extract relationships from entity metadata
def process_entity_metadata(base_dir, entity_type):
    entity_dir = os.path.join(base_dir, entity_type)
    if not os.path.exists(entity_dir):
        return
        
    for entity_name in os.listdir(entity_dir):
        entity_path = os.path.join(entity_dir, entity_name)
        metadata_path = os.path.join(entity_path, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                try:
                    metadata = json.load(f)
                    entity_id = metadata.get('id', entity_name.replace("__", "/"))
                    print(f"Processing {entity_type} metadata: {entity_id}")
                    tracker.extract_relationships_from_metadata(metadata, entity_id, entity_type[:-1])  # Remove 's' from end
                except Exception as e:
                    print(f"Error processing {entity_type} {entity_name}: {e}")

# Process all entities in the output directory
output_dir = "hf_authenticated_data"
process_entity_metadata(output_dir, "models")
process_entity_metadata(output_dir, "datasets")
process_entity_metadata(output_dir, "spaces")

# Generate a provenance graph
tracker.generate_provenance_graph("provenance_graph.json")
print("Generated provenance graph: provenance_graph.json")

# Query for relationships
print("\nQuerying relationships:")
bert_base = "bert-base-uncased"
bert_derivatives = tracker.get_model_derived_models(bert_base)
if bert_derivatives:
    print(f"Models derived from {bert_base}:")
    for derived in bert_derivatives:
        print(f"- {derived}")

glue_dataset = "glue"
models_using_glue = tracker.get_dataset_models(glue_dataset)
if models_using_glue:
    print(f"\nModels using {glue_dataset} dataset:")
    for model in models_using_glue:
        print(f"- {model}")
```

Run the provenance script:

```bash
python provenance_example.py
```

## Step 8: Using the CLI

Instead of writing Python scripts, you can use the command-line interface:

1. Export a configuration template:
   ```bash
   hf-scraper export-config --output my_config.toml
   ```

2. Edit the configuration file to customize settings:
   ```bash
   # Edit the file with your favorite editor
   nano my_config.toml
   ```

3. Run the scrapers individually:
   ```bash
   # Scrape models
   hf-scraper models --max 10 --output-dir ./hf_cli_data
   
   # Scrape datasets
   hf-scraper datasets --max 5 --output-dir ./hf_cli_data
   
   # Scrape spaces
   hf-scraper spaces --max 3 --output-dir ./hf_cli_data
   ```

4. Or scrape all entity types at once:
   ```bash
   hf-scraper all --max-models 10 --max-datasets 5 --max-spaces 3 --output-dir ./hf_cli_data
   ```

5. With authentication:
   ```bash
   export HF_API_TOKEN="your_token_here"
   hf-scraper all --config my_config.toml
   ```

## Step 9: Export to Unified Parquet File

Let's use the new UnifiedExport functionality to store all scraped entity data in a centralized Parquet file:

```python
# unified_export_example.py
import os
from ipfs_huggingface_scraper_py import Config, UnifiedExport
import json
from pathlib import Path

# Define the directory where scraped data is stored
scraped_data_dir = "hf_all_data"

# Initialize the exporter 
exporter = UnifiedExport({
    "data_dir": "./data",
    "use_ipfs": False  # Set to True if you want IPFS integration
})

# Load scraped data for each entity type
models_data = []
datasets_data = []
spaces_data = []

# Helper function to load entity metadata
def load_entity_metadata(base_dir, entity_type):
    entity_list = []
    entity_dir = os.path.join(base_dir, entity_type)
    
    if not os.path.exists(entity_dir):
        print(f"No {entity_type} directory found in {base_dir}")
        return entity_list
        
    for entity_name in os.listdir(entity_dir):
        metadata_path = os.path.join(entity_dir, entity_name, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    entity_list.append(metadata)
            except Exception as e:
                print(f"Error loading {entity_type} metadata for {entity_name}: {e}")
    
    print(f"Loaded {len(entity_list)} {entity_type} metadata records")
    return entity_list

# Load metadata for each entity type
models_data = load_entity_metadata(scraped_data_dir, "models")
datasets_data = load_entity_metadata(scraped_data_dir, "datasets")
spaces_data = load_entity_metadata(scraped_data_dir, "spaces")

# Export all entity data to a unified Parquet file
output_path, cid = exporter.store_unified_data(
    models_list=models_data,
    datasets_list=datasets_data,
    spaces_list=spaces_data
)

if output_path:
    print(f"Successfully exported unified data to: {output_path}")
    if cid:
        print(f"Added to IPFS with CID: {cid}")
    
    # Get statistics about the exported data
    stats = exporter.get_entity_statistics(output_path)
    print("\nEntity Statistics:")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Entity types: {stats['entity_types']}")
    print(f"File size: {stats['file_size_bytes'] / 1024:.1f} KB")
    print(f"Last updated: {stats['last_updated']}")
else:
    print("Failed to export unified data")

# You can also export each entity type to separate Parquet files
models_path, _ = exporter.store_entity_data(models_data, 'model')
datasets_path, _ = exporter.store_entity_data(datasets_data, 'dataset')
spaces_path, _ = exporter.store_entity_data(spaces_data, 'space')

if models_path:
    print(f"\nModels data exported to: {models_path}")
if datasets_path:
    print(f"Datasets data exported to: {datasets_path}")
if spaces_path:
    print(f"Spaces data exported to: {spaces_path}")
```

Run the unified export script:

```bash
python unified_export_example.py
```

The exported Parquet files will be stored in the `data/huggingface_hub_metadata/` directory, making it easy to work with the data using tools like pandas, PyArrow, or data science libraries.

## Conclusion

In this tutorial, you've learned how to:

1. Initialize and configure the IPFS HuggingFace Scraper for different entity types
2. Scrape models, datasets, and spaces from Hugging Face Hub
3. Examine the scraped results for each entity type
4. Scrape all entity types at once with a unified configuration
5. Configure authentication for better rate limits
6. Track and query provenance information between entities
7. Use the command-line interface for simplified operations
8. Export all scraped entity data to a unified Parquet file for efficient storage and analysis

The next tutorial will cover advanced topics like resumable operations, IPFS integration, and more complex provenance tracking for large-scale data governance.