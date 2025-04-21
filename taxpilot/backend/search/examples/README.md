# TaxPilot Search Examples

This directory contains end-to-end examples and demonstration scripts for the TaxPilot search functionality. These examples serve as integration tests and practical demonstrations of how to use the search components.

## Available Examples

### 1. Basic Search Example (`example.py`)

A comprehensive end-to-end demonstration of the TaxPilot search workflow:

1. Indexing laws from the DuckDB database
2. Running searches on the indexed content
3. Displaying search results with highlighting

Usage:
```bash
# Run with full indexing
python -m taxpilot.backend.search.examples.example

# Skip indexing and only run searches on existing data
python -m taxpilot.backend.search.examples.example --search-only

# Enable debug output for more detailed result information
python -m taxpilot.backend.search.examples.example --debug
```

### 2. Article-Based Search Demo (`article_search_demo.py`)

Demonstrates the enhanced article-based search functionality, comparing it with traditional segment-based search:

1. Executes the same queries with both search methods
2. Shows how article-based search groups results by legal article
3. Highlights the benefits of article context versus disconnected segments

Usage:
```bash
python -m taxpilot.backend.search.examples.article_search_demo
```

## Technical Documentation

For detailed technical documentation about the search workflow and implementation, see the [search_example.md](./search_example.md) file, which provides:

1. Complete data flow explanation
2. Component interactions
3. Search algorithm details
4. Article-based search implementation

## Integration Testing

These examples serve as integration tests that verify the complete search pipeline:

1. Database connectivity
2. Text segmentation
3. Embedding generation
4. Vector database operations
5. Search result retrieval and formatting
6. Hierarchical article grouping

Running these examples regularly helps ensure the search system works correctly as a whole, beyond what unit tests can validate.