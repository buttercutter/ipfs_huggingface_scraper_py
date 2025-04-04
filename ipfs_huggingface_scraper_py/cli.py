#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from typing import Optional

from .enhanced_scraper import EnhancedScraper
from .config import Config, get_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HuggingFace Model Scraper with IPFS integration"
    )
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape models from HuggingFace Hub")
    scrape_parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    scrape_parser.add_argument(
        "--max-models", 
        type=int, 
        help="Maximum number of models to scrape"
    )
    scrape_parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Directory to save scraped data"
    )
    scrape_parser.add_argument(
        "--api-token", 
        type=str, 
        help="HuggingFace API token"
    )
    
    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume a paused scraping operation")
    resume_parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new configuration file")
    init_parser.add_argument(
        "--output", 
        type=str, 
        default="config.toml", 
        help="Path to save the configuration template"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show status of scraping operation")
    status_parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    # Common options
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Set logging level"
    )
    
    return parser.parse_args()

def setup_logging(level_name: str = "INFO"):
    """Set up logging configuration."""
    level = getattr(logging, level_name)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def init_config(output_path: str) -> None:
    """Initialize a new configuration file."""
    try:
        config = Config()
        config.export_config_template(output_path)
        print(f"Configuration template created at: {output_path}")
        print("Edit this file to customize your scraper settings.")
    except Exception as e:
        print(f"Error creating configuration template: {e}")
        sys.exit(1)

def show_status(config_path: Optional[str] = None) -> None:
    """Show the status of the scraping operation."""
    try:
        # Get config to determine state directory
        config = get_config(config_path)
        state_dir = config.config["state"]["state_dir"]
        
        # Import here to avoid circular imports
        from .state_manager import StateManager
        
        # Create state manager pointing to the same state dir
        state_manager = StateManager(state_dir)
        
        # Get progress
        progress = state_manager.get_progress()
        
        # Print status
        print("=== Scraper Status ===")
        print(f"Status: {'Completed' if progress['is_completed'] else 'Paused' if progress['is_paused'] else 'Running'}")
        print(f"Total models discovered: {progress['total_models_discovered']}")
        print(f"Models processed: {progress['models_processed']}")
        print(f"Models completed: {progress['models_completed']}")
        print(f"Models with errors: {progress['models_errored']}")
        print(f"Completion: {progress['completion_percentage']:.2f}%")
        
        # Calculate elapsed time in a human-readable format
        elapsed_seconds = progress['elapsed_time']
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Show current position
        print(f"Current position: {progress['current_position']}")
        
        # Show state file location
        print(f"State file: {os.path.join(state_dir, 'scraper_state.json')}")
        print("=====================")
        
    except Exception as e:
        print(f"Error showing status: {e}")
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Process commands
    if args.command == "scrape":
        # Set API token from args or environment
        if args.api_token:
            os.environ["HF_API_TOKEN"] = args.api_token
        
        # Set output directory from args
        if args.output_dir:
            os.environ["HF_OUTPUT_DIR"] = args.output_dir
        
        # Create and run scraper
        scraper = EnhancedScraper(args.config)
        scraper.scrape_models(args.max_models)
        
    elif args.command == "resume":
        # Create scraper and resume
        scraper = EnhancedScraper(args.config)
        scraper.resume()
        
    elif args.command == "init":
        # Initialize a new configuration file
        init_config(args.output)
        
    elif args.command == "status":
        # Show status
        show_status(args.config)
        
    else:
        # No command provided - show help
        print("No command specified. Use one of: scrape, resume, init, status")
        print("Run with --help for more information.")
        sys.exit(1)

if __name__ == "__main__":
    main()