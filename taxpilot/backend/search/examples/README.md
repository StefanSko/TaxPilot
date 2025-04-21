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
```

Note: This example requires Qdrant server running locally. For a simplified version that doesn't require Qdrant, see the simplified_demo.py example.

### 2. Article-Based Search Demo (`article_search_demo.py`)

Demonstrates the enhanced article-based search functionality, comparing it with traditional segment-based search:

1. Executes the same queries with both search methods
2. Shows how article-based search groups results by legal article
3. Highlights the benefits of article context versus disconnected segments

Usage:
```bash
python -m taxpilot.backend.search.examples.article_search_demo
```

Note: This example also requires Qdrant server running locally. For a version that doesn't require Qdrant, use the enhanced_simplified_demo.py example.

### 3. Simplified Database Demo (`simplified_demo.py` and `enhanced_simplified_demo.py`)

Demonstrates TaxPilot's search capabilities using direct database queries without requiring a vector database:

1. Connects directly to the DuckDB database
2. Performs simple keyword searches to showcase functionality
3. Displays results with highlighting
4. Shows article-based grouping

Usage:
```bash
# Run basic simplified demo with keyword search
python -m taxpilot.backend.search.examples.simplified_demo

# Run enhanced demo with article grouping comparison
python -m taxpilot.backend.search.examples.enhanced_simplified_demo
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