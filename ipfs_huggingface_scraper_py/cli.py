import argparse
import logging
import sys
import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from .enhanced_scraper import EnhancedScraper
from .datasets.datasets_scraper import DatasetsScraper
from .spaces.spaces_scraper import SpacesScraper
from .config import get_config, Config
from .unified_export import UnifiedExport

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="HuggingFace Hub scraper for models, datasets, and spaces")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--config", type=str, help="Path to configuration file")
    common_parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                              help="Logging level")
    
    # Models command
    models_parser = subparsers.add_parser("models", parents=[common_parser], 
                                         help="Scrape models from HuggingFace Hub")
    models_parser.add_argument("--max", type=int, help="Maximum number of models to scrape")
    models_parser.add_argument("--output-dir", type=str, help="Directory to save models")
    models_parser.add_argument("--resume", action="store_true", help="Resume a paused scraping operation")
    models_parser.add_argument("--token", type=str, help="HuggingFace API token")
    
    # Datasets command
    datasets_parser = subparsers.add_parser("datasets", parents=[common_parser], 
                                           help="Scrape datasets from HuggingFace Hub")
    datasets_parser.add_argument("--max", type=int, help="Maximum number of datasets to scrape")
    datasets_parser.add_argument("--output-dir", type=str, help="Directory to save datasets")
    datasets_parser.add_argument("--resume", action="store_true", help="Resume a paused scraping operation")
    datasets_parser.add_argument("--token", type=str, help="HuggingFace API token")
    
    # Spaces command
    spaces_parser = subparsers.add_parser("spaces", parents=[common_parser], 
                                         help="Scrape spaces from HuggingFace Hub")
    spaces_parser.add_argument("--max", type=int, help="Maximum number of spaces to scrape")
    spaces_parser.add_argument("--output-dir", type=str, help="Directory to save spaces")
    spaces_parser.add_argument("--resume", action="store_true", help="Resume a paused scraping operation")
    spaces_parser.add_argument("--token", type=str, help="HuggingFace API token")
    
    # All command (scrape all entity types)
    all_parser = subparsers.add_parser("all", parents=[common_parser], 
                                      help="Scrape models, datasets, and spaces from HuggingFace Hub")
    all_parser.add_argument("--max-models", type=int, help="Maximum number of models to scrape")
    all_parser.add_argument("--max-datasets", type=int, help="Maximum number of datasets to scrape")
    all_parser.add_argument("--max-spaces", type=int, help="Maximum number of spaces to scrape")
    all_parser.add_argument("--output-dir", type=str, help="Base directory to save entities")
    all_parser.add_argument("--token", type=str, help="HuggingFace API token")
    
    # Export config command
    export_config_parser = subparsers.add_parser("export-config", 
                                               help="Export a configuration template")
    export_config_parser.add_argument("--output", type=str, required=True, 
                                    help="Path to save the configuration template")
    
    # Export to Parquet command
    export_parser = subparsers.add_parser("export", parents=[common_parser],
                                        help="Export scraped data to a Parquet file")
    export_parser.add_argument("--input-dir", type=str, required=True,
                             help="Directory containing scraped data")
    export_parser.add_argument("--output-file", type=str,
                             help="Output Parquet file path (default: in data directory)")
    export_parser.add_argument("--entity-types", nargs="+", choices=["models", "datasets", "spaces"],
                             default=["models", "datasets", "spaces"],
                             help="Entity types to include in the export")
    export_parser.add_argument("--separate-files", action="store_true",
                             help="Export each entity type to a separate file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle export-config command
    if args.command == "export-config":
        config = Config()
        config.export_config_template(args.output)
        print(f"Configuration template exported to {args.output}")
        sys.exit(0)
    
    # Setup based on args
    config_path = args.config if hasattr(args, "config") else None
    
    # Configure logging
    if hasattr(args, "log_level") and args.log_level:
        log_level = getattr(logging, args.log_level)
        logging.basicConfig(level=log_level, 
                           format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Handle API token
    if hasattr(args, "token") and args.token:
        import os
        os.environ["HF_API_TOKEN"] = args.token
    
    # Handle output directory
    if hasattr(args, "output_dir") and args.output_dir:
        import os
        os.environ["HF_OUTPUT_DIR"] = args.output_dir
    
    # Helper function to load entity metadata
    def load_entity_metadata(base_dir: str, entity_type: str) -> List[Dict[str, Any]]:
        entity_list = []
        entity_dir = os.path.join(base_dir, entity_type)
        
        if not os.path.exists(entity_dir):
            logging.warning(f"No {entity_type} directory found in {base_dir}")
            return entity_list
            
        for entity_name in os.listdir(entity_dir):
            metadata_path = os.path.join(entity_dir, entity_name, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        entity_list.append(metadata)
                except Exception as e:
                    logging.error(f"Error loading {entity_type} metadata for {entity_name}: {e}")
        
        logging.info(f"Loaded {len(entity_list)} {entity_type} metadata records")
        return entity_list
    
    # Handle commands
    if args.command == "models":
        scraper = EnhancedScraper(config_path)
        if args.resume:
            scraper.resume()
        else:
            max_models = args.max if hasattr(args, "max") else None
            scraper.scrape_models(max_models)
    
    elif args.command == "datasets":
        scraper = DatasetsScraper(config_path)
        if args.resume:
            scraper.resume()
        else:
            max_datasets = args.max if hasattr(args, "max") else None
            scraper.scrape_datasets(max_datasets)
    
    elif args.command == "spaces":
        scraper = SpacesScraper(config_path)
        if args.resume:
            scraper.resume()
        else:
            max_spaces = args.max if hasattr(args, "max") else None
            scraper.scrape_spaces(max_spaces)
    
    elif args.command == "all":
        # Scrape all entity types
        logging.info("Scraping models, datasets, and spaces from HuggingFace Hub")
        
        # Update config
        config = get_config(config_path)
        config.config["scraper"]["entity_types"] = ["models", "datasets", "spaces"]
        
        # Set max counts from arguments if provided
        if hasattr(args, "max_models") and args.max_models is not None:
            config.config["scraper"]["max_models"] = args.max_models
        if hasattr(args, "max_datasets") and args.max_datasets is not None:
            config.config["scraper"]["max_datasets"] = args.max_datasets
        if hasattr(args, "max_spaces") and args.max_spaces is not None:
            config.config["scraper"]["max_spaces"] = args.max_spaces
        
        # Scrape models
        logging.info("Starting model scraping")
        models_scraper = EnhancedScraper(config_path)
        models_scraper.scrape_models(config.config["scraper"]["max_models"])
        
        # Scrape datasets
        logging.info("Starting dataset scraping")
        datasets_scraper = DatasetsScraper(config_path)
        datasets_scraper.scrape_datasets(config.config["scraper"]["max_datasets"])
        
        # Scrape spaces
        logging.info("Starting space scraping")
        spaces_scraper = SpacesScraper(config_path)
        spaces_scraper.scrape_spaces(config.config["scraper"]["max_spaces"])
        
        logging.info("All scraping operations completed")
    
    elif args.command == "export":
        # Export scraped data to Parquet
        logging.info("Exporting scraped data to Parquet file(s)")
        
        if not os.path.exists(args.input_dir):
            logging.error(f"Input directory not found: {args.input_dir}")
            sys.exit(1)
        
        # Get configuration
        if config_path:
            config = get_config(config_path)
        else:
            config = Config()
        
        # Create exporter
        exporter = UnifiedExport(config.config)
        
        # Load data for each specified entity type
        models_data = []
        datasets_data = []
        spaces_data = []
        
        if "models" in args.entity_types:
            models_data = load_entity_metadata(args.input_dir, "models")
        
        if "datasets" in args.entity_types:
            datasets_data = load_entity_metadata(args.input_dir, "datasets")
        
        if "spaces" in args.entity_types:
            spaces_data = load_entity_metadata(args.input_dir, "spaces")
        
        # Export data
        if args.separate_files:
            # Export each entity type to a separate file
            if models_data:
                models_path, models_cid = exporter.store_entity_data(models_data, 'model')
                if models_path:
                    logging.info(f"Models data exported to: {models_path}")
                    if models_cid:
                        logging.info(f"Models data CID: {models_cid}")
            
            if datasets_data:
                datasets_path, datasets_cid = exporter.store_entity_data(datasets_data, 'dataset')
                if datasets_path:
                    logging.info(f"Datasets data exported to: {datasets_path}")
                    if datasets_cid:
                        logging.info(f"Datasets data CID: {datasets_cid}")
            
            if spaces_data:
                spaces_path, spaces_cid = exporter.store_entity_data(spaces_data, 'space')
                if spaces_path:
                    logging.info(f"Spaces data exported to: {spaces_path}")
                    if spaces_cid:
                        logging.info(f"Spaces data CID: {spaces_cid}")
        else:
            # Export all data to a single unified file
            output_path = args.output_file
            output_path, cid = exporter.store_unified_data(
                models_list=models_data,
                datasets_list=datasets_data,
                spaces_list=spaces_data,
                output_path=output_path
            )
            
            if output_path:
                logging.info(f"All data exported to: {output_path}")
                if cid:
                    logging.info(f"Data CID: {cid}")
                
                # Show statistics
                stats = exporter.get_entity_statistics(output_path)
                logging.info(f"Total records: {stats['total_entities']}")
                if 'entity_types' in stats:
                    for entity_type, count in stats['entity_types'].items():
                        logging.info(f"{entity_type}: {count} records")
            else:
                logging.error("Failed to export data")
                sys.exit(1)

if __name__ == "__main__":
    main()