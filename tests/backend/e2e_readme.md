# End-to-End Testing for TaxPilot Search API

This directory contains end-to-end tests for the TaxPilot search API. These tests verify the entire pipeline from data processing to API serving.

## Components Tested

1. **Data Processing Pipeline**
   - XML file parsing and database storage
   - Law and section entity creation

2. **Embedding Generation**
   - Vector embedding generation (using mock embedder for tests)
   - Vector database storage and retrieval

3. **API Serving**
   - Search endpoint functionality
   - Article-based grouping
   - Filtering and sorting

## Running the Tests

### Using pytest

```bash
poetry run pytest tests/backend/test_e2e_search.py
```

### Using the Manual Script

For interactive testing and debugging:

```bash
poetry run python run_e2e_test.py
```

## Test Architecture

### `EndToEndTestEnvironment` Class

The `EndToEndTestEnvironment` class manages the complete testing environment:

- Database initialization (in-memory or file-based)
- Data pipeline execution on test XML files
- Embedding generation and vector database setup
- API server startup

### Mock Components

- `MockTextEmbedder`: Provides deterministic mock embeddings for testing
- Test FastAPI app: Simplified mock endpoints for testing

## Example Test Data

The tests include a simple test dataset (estg_sample.xml) containing excerpts from the German Income Tax Law (EStG).

## Extending the Tests

Add more test cases by:

1. Extending the test XML data
2. Adding more test methods to the `TestEndToEndSearch` class
3. Testing additional API endpoints as they're developed