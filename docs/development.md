# Development

This document provides guidelines for developing the IPFS HuggingFace Scraper module.

## Development Environment

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/endomorphosis/ipfs_huggingface_scraper_py.git
   cd ipfs_huggingface_scraper_py
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

3. Install development dependencies:
   ```bash
   pip install pytest pytest-cov pylint black
   ```

### Build & Test Commands

- **Install**: `pip install -e .`
- **Build**: `python setup.py build`
- **Run all tests**: `python -m test.test`
- **Run single test**: `python -m test.test_state_manager`
- **Check for duplications**: `pylint --disable=all --enable=duplicate-code ipfs_huggingface_scraper_py`

## Development Guidelines

### Test-First Development

- All new features must first be developed in the `test/` folder
- Start by writing tests for the new functionality
- Implement the minimum code required to pass the tests
- Refactor and optimize code while keeping tests passing

### Feature Isolation

- Do not modify code outside of `test/` until fully debugged
- Develop new components in isolation before integration
- Use mocks and stubs to test component interactions

### API Exposure

- All functionality should be exposed via FastAPI endpoints
- Define clear API contracts with proper schema validation
- Document API endpoints with examples and response schemas

### Performance Focus

- Use memory-mapped structures for large datasets
- Implement concurrent processing for I/O-bound operations
- Minimize memory usage when processing large numbers of models
- Use streaming patterns for large file operations

### Code Style

- **Imports**: Standard library first, third-party next, project imports last
- **Variables/Functions**: Use `snake_case`
- **Classes**: Use `PascalCase`
- **Indentation**: 4 spaces, no tabs
- **Error Handling**: Use try/except blocks, catch specific exceptions when possible
- **Type Annotations**: Use type hints for function parameters and return values
- **Docstrings**: Include description, parameters, returns, and examples
- **Constants**: Define constants at module level in `UPPER_CASE`
- **Line Length**: Maximum 100 characters per line

Format your code using Black:
```bash
black ipfs_huggingface_scraper_py
```

### Testing Strategy

#### Test Organization

- **Unit Tests**: Located in the `test/` directory with file naming pattern `test_*.py`
- **Integration Tests**: Test interactions between components
- **Mocked Tests**: Use mocks for external dependencies
- **Recovery Tests**: Test resumption of interrupted operations
- **Rate Limit Tests**: Test adherence to API rate limits

#### Test Patterns

1. **Fixture-Based Testing**: Use pytest fixtures for test setup and teardown
2. **API Mocking**: Mock HuggingFace API responses for reproducible tests
3. **State Testing**: Test proper state management across scraping sessions
4. **Error Handling**: Test recovery from network errors and API failures
5. **Incremental Scraping**: Test detection of already-scraped models
6. **Performance Testing**: Benchmark scraping operations for large model sets

## Git Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/new-feature
   ```

2. Commit changes with descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: description of the feature"
   ```

3. Push branch to remote:
   ```bash
   git push origin feature/new-feature
   ```

4. Create pull request from feature branch to main

5. After review and approval, merge pull request

## Implementation Plan

The development roadmap is divided into phases:

### Phase 1: Core Scraper Implementation

- [x] Basic Scraper Framework
- [x] Resilient Operations
- [x] Storage Integration
- [x] Configuration and Monitoring

### Phase 2: Advanced Features

- [ ] Metadata Enrichment
- [ ] Distributed Scraping
- [ ] Advanced Storage
- [ ] Search and Discovery

### Phase 3: Integration and Scaling

- [ ] Complete System Integration
- [ ] Performance Optimization
- [ ] Advanced Features

## Adding New Components

When adding new components to the scraper:

1. **Create a new module file** in `ipfs_huggingface_scraper_py/`
2. **Write tests** in `test/`
3. **Update `__init__.py`** to expose the new component
4. **Update documentation** in `docs/`
5. **Update README.md** with any new features

## Documentation Guidelines

- Document all public APIs with clear docstrings
- Create examples for common usage patterns
- Maintain architecture documentation
- Update schema documentation when data formats change
- Document integration points with other modules
- Create troubleshooting guides for common issues
- Document configuration options with examples
- Keep a changelog of scraper behavior changes

## Release Process

1. Update version number in:
   - `setup.py`
   - `pyproject.toml`
   - `ipfs_huggingface_scraper_py/__init__.py`

2. Update CHANGELOG.md with version changes

3. Create a git tag:
   ```bash
   git tag -a vX.Y.Z -m "Version X.Y.Z"
   git push origin vX.Y.Z
   ```

4. Build the distribution:
   ```bash
   python -m build
   ```

5. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

## Troubleshooting Development Issues

### IPFS Connectivity

If you encounter issues with IPFS connectivity:

1. Ensure IPFS daemon is running:
   ```bash
   ipfs daemon
   ```

2. Check IPFS API port (default: 5001):
   ```bash
   curl http://localhost:5001/api/v0/version
   ```

3. Reset IPFS connections:
   ```bash
   ipfs swarm disconnect --all
   ipfs bootstrap add --default
   ```

### HuggingFace API Issues

If you encounter issues with the HuggingFace API:

1. Validate your API token:
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" https://huggingface.co/api/whoami
   ```

2. Check rate limits:
   ```bash
   # Anonymous: 300K requests/day
   # Authenticated: 1M requests/day
   ```

3. Use exponential backoff for retries

### State Management Issues

If you encounter issues with state management:

1. Check state file permissions
2. Validate JSON format
3. Create backups before modifications
4. Use checkpoint files for recovery