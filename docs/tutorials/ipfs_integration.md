# IPFS Integration Tutorial

This tutorial demonstrates how to use the IPFS integration features of the IPFS HuggingFace Scraper to store scraped models in IPFS.

## Prerequisites

- Python 3.7 or later
- IPFS HuggingFace Scraper installed (`pip install ipfs_huggingface_scraper_py`)
- IPFS daemon installed and running (`ipfs daemon`)
- ipfs_kit_py and ipfs_datasets_py installed
- Optional: HuggingFace API token

## Step 1: Ensure IPFS is Running

Before we start, make sure the IPFS daemon is running:

```bash
# Start IPFS daemon in a separate terminal
ipfs daemon

# Verify IPFS is working in another terminal
ipfs id
```

You should see your IPFS node information. If not, check your IPFS installation.

## Step 2: Configure the Scraper for IPFS

Let's create a script that configures the scraper to use IPFS for storage:

```python
# ipfs_scraper.py
from ipfs_huggingface_scraper_py import EnhancedScraper, Config

# Create a configuration
config = Config()

# Configure basic settings
config.set("scraper", "output_dir", "hf_ipfs_models")
config.set("scraper", "max_models", 5)  # Start small for testing

# Configure IPFS storage
config.set("storage", "use_ipfs", True)
config.set("storage", "metadata_format", "parquet")

# Configure IPFS add options
config.set("storage.ipfs_add_options", "pin", True)
config.set("storage.ipfs_add_options", "wrap_with_directory", True)
config.set("storage.ipfs_add_options", "chunker", "size-262144")

# Save configuration
config.save("ipfs_config.toml")

# Initialize the scraper
scraper = EnhancedScraper("ipfs_config.toml")

# Run the scraper
print("Starting IPFS-enabled scraper...")
scraper.scrape_models()
print("Scraping completed!")
```

Run the script:

```bash
python ipfs_scraper.py
```

## Step 3: Verify Content in IPFS

After scraping, let's verify that the content has been stored in IPFS:

```python
# verify_ipfs.py
import os
import json

output_dir = "hf_ipfs_models"
model_dirs = os.listdir(output_dir)

print(f"Found {len(model_dirs)} model directories:")
for model_dir in model_dirs:
    model_path = os.path.join(output_dir, model_dir)
    cid_file_path = os.path.join(model_path, "ipfs_cid.txt")
    
    if os.path.exists(cid_file_path):
        with open(cid_file_path, 'r') as f:
            cid = f.read().strip()
            print(f"\nModel: {model_dir}")
            print(f"IPFS CID: {cid}")
            
            # Verify CID exists in IPFS
            import subprocess
            result = subprocess.run(["ipfs", "ls", cid], 
                                   capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Content verified in IPFS with {len(result.stdout.strip().split(os.linesep))} files")
                print(f"To view: http://localhost:8080/ipfs/{cid}")
            else:
                print(f"Failed to verify content: {result.stderr}")
```

Run the verification script:

```bash
python verify_ipfs.py
```

## Step 4: Working with IPFS Storage Directly

You can work with the `IpfsStorage` class directly for more control:

```python
# direct_ipfs.py
from ipfs_huggingface_scraper_py import IpfsStorage

# Create storage with config
config = {
    "use_ipfs": True,
    "ipfs_add_options": {
        "pin": True,
        "wrap_with_directory": True
    }
}
storage = IpfsStorage(config)

# Check if IPFS is available
if not storage.is_ipfs_available():
    print("IPFS is not available. Please check your IPFS daemon.")
    exit(1)

# Add a file to IPFS
file_path = "ipfs_config.toml"
cid = storage.add_file_to_ipfs(file_path)
print(f"Added file to IPFS: {cid}")

# Store some metadata
metadata = {
    "name": "Example Model",
    "description": "This is an example model",
    "tags": ["example", "test"],
    "parameters": 1000000
}

# Store as JSON
json_path, json_cid = storage.store_metadata_as_json(metadata)
print(f"Stored metadata as JSON: {json_cid}")

# Store multiple metadata records as Parquet
metadata_list = [
    {"name": "Model 1", "parameters": 1000000},
    {"name": "Model 2", "parameters": 2000000}
]
parquet_path, parquet_cid = storage.store_metadata_as_parquet(metadata_list)
print(f"Stored metadata as Parquet: {parquet_cid}")

# Pin CIDs for persistence
storage.pin_cid(cid)
storage.pin_cid(json_cid)
storage.pin_cid(parquet_cid)
```

Run the direct IPFS script:

```bash
python direct_ipfs.py
```

## Step 5: Converting Between Formats

The IPFS storage integration also supports converting between formats:

```python
# format_conversion.py
from ipfs_huggingface_scraper_py import IpfsStorage

# Create storage with config
config = {
    "use_ipfs": True,
    "ipfs_add_options": {
        "pin": True
    }
}
storage = IpfsStorage(config)

# Create a sample JSONL file
import json
with open("sample.jsonl", "w") as f:
    for i in range(10):
        model = {
            "model_id": f"model-{i}",
            "parameters": i * 1000000,
            "tags": ["sample", f"tag-{i}"]
        }
        f.write(json.dumps(model) + "\n")

# Convert JSONL to Parquet
parquet_path = storage.convert_jsonl_to_parquet("sample.jsonl")
print(f"Converted JSONL to Parquet: {parquet_path}")

# Add Parquet to IPFS
parquet_cid = storage.add_file_to_ipfs(parquet_path)
print(f"Added Parquet to IPFS: {parquet_cid}")

# Create a CAR file
car_path = storage.create_car_file(parquet_path)
if car_path:
    print(f"Created CAR file: {car_path}")
    
    # Add CAR to IPFS
    car_cid = storage.add_file_to_ipfs(car_path)
    print(f"Added CAR to IPFS: {car_cid}")
```

Run the format conversion script:

```bash
python format_conversion.py
```

## Step 6: Integrating with ipfs_datasets_py

For deeper integration with ipfs_datasets_py:

```python
# dataset_integration.py
from ipfs_huggingface_scraper_py import IpfsStorage
import os

try:
    from ipfs_datasets_py import IpfsDataset
except ImportError:
    print("ipfs_datasets_py not installed. Install with: pip install ipfs_datasets_py")
    exit(1)

# Create storage with config
config = {
    "use_ipfs": True,
}
storage = IpfsStorage(config)

# Create dataset
dataset = IpfsDataset()

# Process models from our scraped directory
output_dir = "hf_ipfs_models"
if not os.path.exists(output_dir):
    print(f"Directory {output_dir} not found. Run the ipfs_scraper.py script first.")
    exit(1)

model_dirs = os.listdir(output_dir)
print(f"Adding {len(model_dirs)} models to dataset...")

for model_dir in model_dirs:
    model_path = os.path.join(output_dir, model_dir)
    cid_file_path = os.path.join(model_path, "ipfs_cid.txt")
    metadata_path = os.path.join(model_path, "metadata.json")
    
    if os.path.exists(cid_file_path) and os.path.exists(metadata_path):
        # Read CID
        with open(cid_file_path, 'r') as f:
            cid = f.read().strip()
        
        # Read metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Add to dataset
        dataset.add_record({
            "type": "model",
            "name": model_dir.replace("__", "/"),
            "cid": cid,
            "metadata": {
                "id": metadata.get("id", ""),
                "task": metadata.get("pipeline_tag", "unknown"),
                "tags": metadata.get("tags", []),
                "description": metadata.get("description", "")
            }
        })
        print(f"Added model {model_dir} to dataset")

# Save dataset
dataset_cid = dataset.save()
print(f"Dataset saved with CID: {dataset_cid}")
print(f"Access via: http://localhost:8080/ipfs/{dataset_cid}")
```

Run the dataset integration script:

```bash
python dataset_integration.py
```

## Step 7: Using the CLI with IPFS

You can also use the CLI with IPFS enabled:

1. Initialize a configuration file:
   ```bash
   hf-scraper init --output ipfs_cli_config.toml
   ```

2. Edit the configuration file to enable IPFS:
   ```toml
   # ipfs_cli_config.toml
   [storage]
   use_ipfs = true
   
   [storage.ipfs_add_options]
   pin = true
   wrap_with_directory = true
   ```

3. Run the scraper with IPFS:
   ```bash
   hf-scraper scrape --config ipfs_cli_config.toml --max-models 10
   ```

## Conclusion

In this tutorial, you've learned how to:

1. Configure the scraper to use IPFS for storage
2. Verify content storage in IPFS
3. Work directly with the IpfsStorage class
4. Convert between different formats (JSONL, Parquet, CAR)
5. Integrate with ipfs_datasets_py
6. Use the CLI with IPFS enabled

IPFS integration provides several benefits:
- Content-addressed storage ensures data integrity
- Distributed storage for resilience
- Deduplication of identical content
- Content can be retrieved from any IPFS node
- Efficient sharing of model files