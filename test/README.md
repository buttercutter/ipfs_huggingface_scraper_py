# IPFS HuggingFace Scraper Tests

This directory contains tests for the IPFS HuggingFace Scraper module.

## Running Tests

To run all tests:

```bash
python -m test.test
```

To run a specific test:

```bash
python -m test.test test/test_config.py
python -m test.test test/test_datasets_scraper.py
```

## Test Organization

The tests are organized into the following categories:

- **Unit Tests**: Testing individual components in isolation
- **Integration Tests**: Testing interactions between components
- **Functional Tests**: Testing end-to-end functionality
- **Entity Tests**: Testing different entity types (models, datasets, spaces)

## Test Files

Core components:
- `test_state_manager.py`: Tests for the state management system
- `test_rate_limiter.py`: Tests for the rate limiter component
- `test_ipfs_integration.py`: Tests for IPFS integration (requires IPFS daemon)
- `test_enhanced_scraper.py`: Tests for the enhanced model scraper
- `test_config.py`: Tests for the configuration system
- `test_cli.py`: Tests for the command-line interface

Entity types:
- `test_datasets_scraper.py`: Tests for the datasets scraper
- `test_spaces_scraper.py`: Tests for the spaces scraper
- `test_provenance.py`: Tests for the provenance tracking system

## New Entity Tests

To run tests for the new entity types and provenance tracking:

```bash
python -m test.test test/test_datasets_scraper.py
python -m test.test test/test_spaces_scraper.py
python -m test.test test/test_provenance.py
```

## Mock Tests

For HuggingFace API tests, we use mocked responses to avoid excessive API calls during testing:

```python
@patch("ipfs_huggingface_scraper_py.datasets.datasets_scraper.list_datasets")
def test_discover_datasets(self, mock_list_datasets):
    # Set up mock
    mock_list_datasets.return_value = [mock_dataset1, mock_dataset2]
    
    # Test code here
```

## Writing Tests

When writing new tests, follow these guidelines:

1. Use descriptive test names that indicate what is being tested
2. Use fixture-based testing with pytest
3. Mock external dependencies
4. Include both positive and negative test cases
5. Test error handling and edge cases

## Test Results

Test results are saved to `test_results.json` with summary statistics about test execution.

## Documentation

See `TEST_SUMMARY.md` for a comprehensive overview of the test suite.