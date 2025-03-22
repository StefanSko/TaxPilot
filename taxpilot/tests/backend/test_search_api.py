"""
Tests for the search API module.

This module tests the search API functionality, including:
- Query processing
- Search execution
- Result formatting and highlighting
- Caching
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

from taxpilot.backend.search.search_api import (
    SearchService,
    QueryResult,
    SearchResults
)
from taxpilot.backend.search.vector_db import SearchResult as VectorSearchResult


@pytest.fixture
def mock_vector_db():
    """Create a mock vector database."""
    mock_db = MagicMock()
    
    # Mock search method to return sample results
    def mock_search(params):
        # Return sample vector search results
        results = [
            VectorSearchResult(
                segment_id="estg_1_1",
                score=0.95,
                metadata={
                    "law_id": "estg",
                    "section_id": "estg_1",
                    "section_number": "1",
                    "title": "Steuerpflicht",
                    "embedding_model": "gbert-base"
                }
            ),
            VectorSearchResult(
                segment_id="estg_2_1",
                score=0.85,
                metadata={
                    "law_id": "estg",
                    "section_id": "estg_2",
                    "section_number": "2",
                    "title": "Einkunftsarten",
                    "embedding_model": "gbert-base"
                }
            ),
            VectorSearchResult(
                segment_id="kstg_1_1",
                score=0.75,
                metadata={
                    "law_id": "kstg",
                    "section_id": "kstg_1",
                    "section_number": "1",
                    "title": "Steuerpflicht",
                    "embedding_model": "gbert-base"
                }
            ),
        ]
        
        # Apply limit and offset if specified
        start = params.offset if params.offset else 0
        end = start + params.limit if params.limit else len(results)
        
        # Return results based on limit and offset
        return results[start:end]
    
    mock_db.search = mock_search
    return mock_db


@pytest.fixture
def mock_embedder():
    """Create a mock text embedder."""
    mock_emb = MagicMock()
    
    # Mock embed_text method to return a fixed vector
    mock_emb.embed_text.return_value = [0.1] * 768
    
    return mock_emb


@pytest.fixture
def mock_db_connection():
    """Create a mock database connection."""
    mock_conn = MagicMock()
    
    # Create a mock connection context
    mock_context = MagicMock()
    mock_execute = MagicMock()
    
    # Mock the query result
    mock_results = [
        {
            "segment_id": "estg_1_1",
            "content": "Der Einkommensteuer unterliegen natürliche Personen.",
            "section_content": "§ 1 Steuerpflicht\n(1) Der Einkommensteuer unterliegen natürliche Personen.",
            "title": "Steuerpflicht",
            "section_number": "1",
            "law_id": "estg",
            "law_abbreviation": "EStG",
            "law_name": "Einkommensteuergesetz"
        },
        {
            "segment_id": "estg_2_1",
            "content": "Einkünfte aus Land- und Forstwirtschaft, gewerbliche Einkünfte und weitere.",
            "section_content": "§ 2 Einkunftsarten\n(1) Einkünfte aus Land- und Forstwirtschaft, gewerbliche Einkünfte und weitere.",
            "title": "Einkunftsarten",
            "section_number": "2",
            "law_id": "estg",
            "law_abbreviation": "EStG",
            "law_name": "Einkommensteuergesetz"
        },
        {
            "segment_id": "kstg_1_1",
            "content": "Der Körperschaftsteuer unterliegen juristische Personen.",
            "section_content": "§ 1 Steuerpflicht\n(1) Der Körperschaftsteuer unterliegen juristische Personen.",
            "title": "Steuerpflicht",
            "section_number": "1",
            "law_id": "kstg",
            "law_abbreviation": "KStG",
            "law_name": "Körperschaftsteuergesetz"
        }
    ]
    
    # Setup the mock query result
    mock_execute.fetchall.return_value = mock_results
    mock_context.execute.return_value = mock_execute
    mock_conn.get_connection.return_value.__enter__.return_value = mock_context
    
    return mock_conn


def test_search_service_initialization():
    """Test that the SearchService initializes correctly."""
    # Create service with default arguments
    service = SearchService()
    
    # Verify instance attributes
    assert hasattr(service, 'vector_db')
    assert hasattr(service, 'embedder')
    assert hasattr(service, 'db_connection')
    assert hasattr(service, '_cache')
    assert service._cache_size == 100
    assert service._cache_order == []


def test_search_basic(mock_vector_db, mock_embedder, mock_db_connection):
    """Test basic search functionality."""
    # Create service with mocks
    service = SearchService(
        vector_db=mock_vector_db,
        embedder=mock_embedder,
        db_connection=mock_db_connection
    )
    
    # Perform search
    results = service.search("Steuerpflicht", page=1, limit=10)
    
    # Verify results
    assert isinstance(results, SearchResults)
    assert len(results.results) == 3
    assert results.total == 3
    assert results.page == 1
    assert results.limit == 10
    assert results.query == "Steuerpflicht"
    assert results.execution_time_ms > 0
    
    # Verify first result
    first_result = results.results[0]
    assert first_result.id == "estg_1_1"
    assert first_result.law_id == "estg"
    assert first_result.section_number == "1"
    assert first_result.title == "Steuerpflicht"
    assert "Einkommensteuer" in first_result.content
    assert first_result.relevance_score == 0.95


def test_search_with_filters(mock_vector_db, mock_embedder, mock_db_connection):
    """Test search with filters."""
    # Create service with mocks
    service = SearchService(
        vector_db=mock_vector_db,
        embedder=mock_embedder,
        db_connection=mock_db_connection
    )
    
    # Mock the _process_filters method to test filter processing
    original_process_filters = service._process_filters
    
    def mock_process_filters(filters):
        # Pass through to original method for tracking
        processed = original_process_filters(filters)
        # Verify that filters are processed correctly
        assert processed.get('law_id') == 'estg'
        return processed
    
    service._process_filters = mock_process_filters
    
    # Perform search with filters
    results = service.search(
        "Steuerpflicht",
        filters={"law_id": "estg"},
        page=1,
        limit=10
    )
    
    # Verify that search was called with correct parameters
    mock_vector_db.search.assert_called()
    
    # Restore original method
    service._process_filters = original_process_filters


def test_search_pagination(mock_vector_db, mock_embedder, mock_db_connection):
    """Test search pagination."""
    # Create service with mocks
    service = SearchService(
        vector_db=mock_vector_db,
        embedder=mock_embedder,
        db_connection=mock_db_connection
    )
    
    # Perform search with pagination
    results_page_1 = service.search("Steuerpflicht", page=1, limit=2)
    results_page_2 = service.search("Steuerpflicht", page=2, limit=1)
    
    # Verify pagination
    assert len(results_page_1.results) == 2
    assert len(results_page_2.results) == 1
    assert results_page_1.page == 1
    assert results_page_2.page == 2
    assert results_page_1.limit == 2
    assert results_page_2.limit == 1
    
    # Verify different results
    assert results_page_1.results[0].id != results_page_2.results[0].id


def test_search_highlighting(mock_vector_db, mock_embedder, mock_db_connection):
    """Test result highlighting."""
    # Create service with mocks
    service = SearchService(
        vector_db=mock_vector_db,
        embedder=mock_embedder,
        db_connection=mock_db_connection
    )
    
    # Test highlight function directly
    highlighted = service._highlight_text(
        "Der Einkommensteuer unterliegen natürliche Personen.",
        "Einkommensteuer"
    )
    
    # Verify highlighting
    assert "<mark>Einkommensteuer</mark>" in highlighted
    
    # Test in full search (highlight=True is default)
    results = service.search("Einkommensteuer")
    
    # Verify highlighting in results
    assert "<mark>Einkommensteuer</mark>" in results.results[0].content_with_highlights


def test_search_caching(mock_vector_db, mock_embedder, mock_db_connection):
    """Test search result caching."""
    # Create service with mocks
    service = SearchService(
        vector_db=mock_vector_db,
        embedder=mock_embedder,
        db_connection=mock_db_connection,
        cache_size=2
    )
    
    # First search should miss cache
    service.search("Steuerpflicht")
    assert len(service._cache) == 1
    assert len(service._cache_order) == 1
    
    # Same search should hit cache
    mock_vector_db.search.reset_mock()
    service.search("Steuerpflicht")
    
    # Vector DB search should not be called again
    mock_vector_db.search.assert_not_called()
    
    # Different search should miss cache
    service.search("Einkunftsarten")
    assert len(service._cache) == 2
    assert len(service._cache_order) == 2
    
    # Cache eviction test (cache_size = 2)
    service.search("Körperschaftsteuer")
    # Should still have 2 items in cache (oldest removed)
    assert len(service._cache) == 2
    assert len(service._cache_order) == 2
    # First query should be evicted
    assert service._get_cache_key("Steuerpflicht", None, 1, 10) not in service._cache


def test_search_error_handling(mock_vector_db, mock_embedder, mock_db_connection):
    """Test error handling during search."""
    # Create service with mocks
    service = SearchService(
        vector_db=mock_vector_db,
        embedder=mock_embedder,
        db_connection=mock_db_connection
    )
    
    # Make vector_db.search raise an exception
    mock_vector_db.search.side_effect = Exception("Test error")
    
    # Search should return empty results without raising
    results = service.search("Error test")
    
    # Verify empty results
    assert len(results.results) == 0
    assert results.total == 0


def test_result_enrichment(mock_vector_db, mock_embedder, mock_db_connection):
    """Test result enrichment with content from database."""
    # Create service with mocks
    service = SearchService(
        vector_db=mock_vector_db,
        embedder=mock_embedder,
        db_connection=mock_db_connection
    )
    
    # Create mock vector results
    vector_results = [
        VectorSearchResult(
            segment_id="estg_1_1",
            score=0.95,
            metadata={
                "law_id": "estg",
                "section_id": "estg_1",
                "section_number": "1",
                "title": "Steuerpflicht"
            }
        )
    ]
    
    # Test enrichment function directly
    enriched = service._enrich_results(vector_results, "Einkommensteuer", True)
    
    # Verify enriched result
    assert len(enriched) == 1
    assert enriched[0].id == "estg_1_1"
    assert enriched[0].law_id == "estg"
    assert "Einkommensteuer" in enriched[0].content
    assert "<mark>Einkommensteuer</mark>" in enriched[0].content_with_highlights


def test_close_resources():
    """Test that resources are closed properly."""
    # Create mocks
    vector_db = MagicMock()
    embedder = MagicMock()
    db_connection = MagicMock()
    
    # Create service with mocks
    service = SearchService(
        vector_db=vector_db,
        embedder=embedder,
        db_connection=db_connection
    )
    
    # Close resources
    service.close()
    
    # Verify that resources were closed
    vector_db.close.assert_called_once()
    embedder.close.assert_called_once()
    db_connection.close.assert_called_once()