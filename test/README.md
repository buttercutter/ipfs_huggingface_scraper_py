# IPFS HuggingFace Scraper Tests

This directory contains tests for the IPFS HuggingFace Scraper module.

## Running Tests

To run all tests:

```bash
python -m test.test
```

To run a specific test:

```bash
python -m test.test_state_manager
python -m test.test_rate_limiter
```

## Test Organization

The tests are organized into the following categories:

- **Unit Tests**: Testing individual components in isolation
- **Integration Tests**: Testing interactions between components
- **Functional Tests**: Testing end-to-end functionality

## Test Files

- `test_state_manager.py`: Tests for the state management system
- `test_rate_limiter.py`: Tests for the rate limiter component
- `test_ipfs_integration.py`: Tests for IPFS integration (requires IPFS daemon)
- `test_enhanced_scraper.py`: Tests for the enhanced scraper (in progress)
- `test_config.py`: Tests for the configuration system (in progress)
- `test_cli.py`: Tests for the command-line interface (in progress)

## Mock Tests

For HuggingFace API tests, we use mocked responses to avoid excessive API calls during testing.

## Writing Tests

When writing new tests, follow these guidelines:

1. Use descriptive test names that indicate what is being tested
2. Use fixture-based testing with pytest
3. Mock external dependencies
4. Include both positive and negative test cases
5. Test error handling and edge cases

## Test Development

The project follows a test-first development approach:

1. Write tests for new functionality first
2. Implement the minimum code required to pass the tests
3. Refactor and optimize while keeping tests passing