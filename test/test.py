"""Main test runner for the IPFS HuggingFace Scraper module."""

import os
import sys
import json
import pytest
import time

def main():
    """Run all tests and generate a report."""
    start_time = time.time()
    
    print("Running IPFS HuggingFace Scraper tests...")
    
    # Add project directory to path
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Define test files to run (in order of dependencies)
    test_files = [
        "test_config.py",
        "test_rate_limiter.py",
        "test_state_manager.py",
        "test_ipfs_integration.py",
        "test_enhanced_scraper.py",
        "test_datasets_scraper.py",
        "test_spaces_scraper.py",
        "test_provenance.py",
        "test_unified_export.py",
        "test_cli.py"
    ]
    
    # Run tests
    args = [
        "-v",  # Verbose output
        "--junitxml=test-results.xml"  # JUnit XML report
    ]
    
    # Check if specific test file is specified
    if len(sys.argv) > 1:
        # Run specific test file
        args.append(sys.argv[1])
    else:
        # Run all test files in order
        args.extend([os.path.join(os.path.dirname(__file__), tf) for tf in test_files])
    
    result = pytest.main(args)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Generate test summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration,
        "exit_code": result,
        "status": "Success" if result == 0 else "Failed",
        "test_files": test_files,
        "entity_types_tested": ["models", "datasets", "spaces"],
        "components_tested": [
            "config", "rate_limiter", "state_manager", "ipfs_integration", 
            "enhanced_scraper", "datasets_scraper", "spaces_scraper", 
            "provenance", "unified_export", "cli"
        ]
    }
    
    with open(os.path.join(project_dir, "test", "test_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Tests completed in {duration:.2f} seconds with status: {summary['status']}")
    print(f"Entity types tested: {', '.join(summary['entity_types_tested'])}")
    
    return result

if __name__ == "__main__":
    sys.exit(main())