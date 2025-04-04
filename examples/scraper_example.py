#!/usr/bin/env python3
"""
Example using the enhanced HuggingFace scraper.

This example demonstrates:
1. Setting up the scraper with a custom configuration
2. Scraping models using different parameters
3. Resuming a paused scraping operation
"""

import os
import logging
import time
import argparse
from ipfs_huggingface_scraper_py import EnhancedScraper, Config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def example_scrape():
    """Run an example scraping operation."""
    # Create a custom configuration
    config = Config()
    
    # Update configuration
    config.set("scraper", "max_models", 10)
    config.set("scraper", "output_dir", "example_output")
    config.set("api", "authenticated", False)  # Set to True if you have an API token
    config.set("storage", "use_ipfs", False)   # Set to True if you have IPFS installed
    
    # Save configuration for reference
    os.makedirs("example_output", exist_ok=True)
    config.save("example_output/config.toml")
    
    # Create and run scraper
    print("Starting scraper...")
    scraper = EnhancedScraper("example_output/config.toml")
    scraper.scrape_models()
    
    # Show a pause and resume example
    print("\nExample: Pausing and resuming scraper...")
    
    # Create a scraper with more models
    config.set("scraper", "max_models", 20)
    config.save("example_output/config.toml")
    
    # Create new scraper
    scraper = EnhancedScraper("example_output/config.toml")
    
    # Reset state to start fresh
    scraper.state_manager.reset()
    
    # Start scraping in a way we can simulate an interruption
    try:
        # Define a scraping function that will be interrupted
        def interruptible_scrape():
            counter = 0
            for model_id in scraper._discover_models(20):
                if counter >= 5:
                    # Simulate interruption after processing 5 models
                    raise KeyboardInterrupt()
                
                # Process model
                print(f"Processing model: {model_id}")
                scraper._process_model(model_id)
                counter += 1
        
        # Run the function and catch the interruption
        try:
            interruptible_scrape()
        except KeyboardInterrupt:
            print("Scraping interrupted!")
            # Pause is automatically handled in EnhancedScraper.scrape_models when KeyboardInterrupt
            scraper.state_manager.pause()
        
        # Show pause status
        print(f"Scraper paused: {scraper.state_manager.is_paused()}")
        print(f"Models processed so far: {scraper.state_manager.state['models_processed']}")
        
        # Wait a moment
        print("Waiting 2 seconds before resuming...")
        time.sleep(2)
        
        # Resume scraping
        print("Resuming scraping...")
        scraper.resume()
        
        # Show completion status
        print(f"Scraper completed: {scraper.state_manager.is_completed()}")
        print(f"Total models processed: {scraper.state_manager.state['models_processed']}")
        
    except Exception as e:
        print(f"Error in example: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HuggingFace Scraper Example")
    parser.add_argument("--api-token", type=str, help="HuggingFace API token")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set API token if provided
    if args.api_token:
        os.environ["HF_API_TOKEN"] = args.api_token
    
    # Set up logging
    setup_logging()
    
    # Run example
    example_scrape()

if __name__ == "__main__":
    main()