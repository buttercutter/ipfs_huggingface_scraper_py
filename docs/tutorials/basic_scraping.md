# Basic Scraping Tutorial

This tutorial demonstrates how to use the IPFS HuggingFace Scraper for basic model scraping.

## Prerequisites

- Python 3.7 or later
- IPFS HuggingFace Scraper installed (`pip install ipfs_huggingface_scraper_py`)
- Optional: HuggingFace API token for authenticated requests

## Step 1: Initialize the Scraper

First, let's create a basic script to initialize the scraper:

```python
# basic_scraper.py
from ipfs_huggingface_scraper_py import EnhancedScraper, Config

# Create a configuration
config = Config()

# Configure basic settings
config.set("scraper", "output_dir", "hf_scraped_models")
config.set("scraper", "max_models", 10)  # Start with a small number for testing
config.set("storage", "use_ipfs", False)  # Disable IPFS for now

# Save configuration
config.save("scraper_config.toml")

# Initialize the scraper
scraper = EnhancedScraper("scraper_config.toml")
```

## Step 2: Run the Scraper

Now, let's add code to run the scraper:

```python
# basic_scraper.py
from ipfs_huggingface_scraper_py import EnhancedScraper, Config

# Create a configuration
config = Config()

# Configure basic settings
config.set("scraper", "output_dir", "hf_scraped_models")
config.set("scraper", "max_models", 10)
config.set("storage", "use_ipfs", False)

# Save configuration
config.save("scraper_config.toml")

# Initialize the scraper
scraper = EnhancedScraper("scraper_config.toml")

# Run the scraper
print("Starting scraper...")
scraper.scrape_models()
print("Scraping completed!")
```

Run the script to start scraping:

```bash
python basic_scraper.py
```

## Step 3: Examine the Results

After the scraper completes, you can examine the results in the output directory (`hf_scraped_models`):

```python
# examine_results.py
import os
import json

output_dir = "hf_scraped_models"
model_dirs = os.listdir(output_dir)

print(f"Found {len(model_dirs)} model directories:")
for model_dir in model_dirs[:5]:  # Show first 5 models
    model_path = os.path.join(output_dir, model_dir)
    print(f"\n--- Model: {model_dir} ---")
    
    # List files in the model directory
    files = os.listdir(model_path)
    print(f"Files: {', '.join(files)}")
    
    # Check for metadata
    metadata_path = os.path.join(model_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            print(f"Model ID: {metadata.get('id', 'Unknown')}")
            print(f"Model Type: {metadata.get('pipeline_tag', 'Unknown')}")
            if 'tags' in metadata:
                print(f"Tags: {', '.join(metadata.get('tags', []))}")
```

Run the examination script:

```bash
python examine_results.py
```

## Step 4: Configure Authentication

For better rate limits, let's update our script to use authentication:

```python
# authenticated_scraper.py
import os
from ipfs_huggingface_scraper_py import EnhancedScraper, Config

# Get API token from environment variable or set directly
api_token = os.environ.get("HF_API_TOKEN", "")

# Create a configuration
config = Config()

# Configure basic settings
config.set("scraper", "output_dir", "hf_scraped_models")
config.set("scraper", "max_models", 20)  # Increased to 20 models
config.set("storage", "use_ipfs", False)

# Configure authentication
config.set("api", "api_token", api_token)
config.set("api", "authenticated", bool(api_token))

# Save configuration
config.save("scraper_config.toml")

# Initialize the scraper
scraper = EnhancedScraper("scraper_config.toml")

# Run the scraper
print("Starting authenticated scraper...")
scraper.scrape_models()
print("Scraping completed!")
```

Set your API token and run the script:

```bash
export HF_API_TOKEN="your_token_here"
python authenticated_scraper.py
```

## Step 5: Customize Scraping Parameters

Now, let's customize what we're scraping:

```python
# custom_scraper.py
from ipfs_huggingface_scraper_py import EnhancedScraper, Config

# Create a configuration
config = Config()

# Configure basic settings
config.set("scraper", "output_dir", "hf_custom_models")
config.set("scraper", "max_models", 50)
config.set("scraper", "save_metadata", True)
config.set("scraper", "filename_to_download", "pytorch_model.bin.index.json")  # Different file
config.set("scraper", "batch_size", 10)
config.set("storage", "use_ipfs", False)

# Save configuration
config.save("custom_config.toml")

# Initialize the scraper
scraper = EnhancedScraper("custom_config.toml")

# Run the scraper
print("Starting custom scraper...")
scraper.scrape_models()
print("Scraping completed!")
```

Run the customized scraper:

```bash
python custom_scraper.py
```

## Step 6: Using the CLI

Instead of writing Python scripts, you can use the command-line interface:

1. Initialize a configuration file:
   ```bash
   hf-scraper init --output my_config.toml
   ```

2. Edit the configuration file to customize settings:
   ```bash
   # Edit the file with your favorite editor
   nano my_config.toml
   ```

3. Run the scraper:
   ```bash
   hf-scraper scrape --config my_config.toml --max-models 30
   ```

4. Check the status:
   ```bash
   hf-scraper status --config my_config.toml
   ```

## Conclusion

In this tutorial, you've learned how to:

1. Initialize and configure the IPFS HuggingFace Scraper
2. Run basic scraping operations
3. Examine the scraped results
4. Configure authentication for better rate limits
5. Customize scraping parameters
6. Use the command-line interface

The next tutorial, [Resumable Operations](resumable_operations.md), will show you how to handle long-running scraping operations with pausing and resuming capabilities.