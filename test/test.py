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
    
    # Run tests
    args = [
        "-v",  # Verbose output
        os.path.dirname(__file__),  # Test directory
        "--junitxml=test-results.xml"  # JUnit XML report
    ]
    
    # Check if specific test file is specified
    if len(sys.argv) > 1:
        # Replace test module with specific test file
        args[1] = sys.argv[1]
    
    result = pytest.main(args)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Generate test summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration,
        "exit_code": result,
        "status": "Success" if result == 0 else "Failed"
    }
    
    with open(os.path.join(project_dir, "test", "test_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Tests completed in {duration:.2f} seconds with status: {summary['status']}")
    
    return result

if __name__ == "__main__":
    sys.exit(main())