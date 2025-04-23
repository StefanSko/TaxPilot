"""
Integration tests for the search API endpoint.

This module contains tests that test the API with actual components,
focusing on how the endpoint integrates with the search service and returns results.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from taxpilot.backend.api.app import app
from taxpilot.backend.search.search_api import QueryResult, SearchResults


@pytest.fixture
def client():
    """Test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_search_results():
    """Mock search results for testing."""
    results = [
        QueryResult(
            id="estg_13_1",
            law_id="estg",
            law_abbreviation="EStG",
            section_number="13",
            title="Einkünfte aus Land- und Forstwirtschaft",
            content="Dies ist der Inhalt von § 13 EStG zu Land- und Forstwirtschaft.",
            content_with_highlights="Dies ist der Inhalt von § 13 EStG zu <mark>Land</mark>- und Forstwirtschaft.",
            relevance_score=0.92,
            metadata={"section_id": "estg_13", "segment_id": "estg_13_1"}
        ),
        QueryResult(
            id="estg_15_1",
            law_id="estg",
            law_abbreviation="EStG",
            section_number="15",
            title="Einkünfte aus Gewerbebetrieb",
            content="Dies ist der Inhalt von § 15 EStG zu Gewerbebetrieb.",
            content_with_highlights="Dies ist der Inhalt von § 15 EStG zu <mark>Gewerbebetrieb</mark>.",
            relevance_score=0.85,
            metadata={"section_id": "estg_15", "segment_id": "estg_15_1"}
        )
    ]
    
    return SearchResults(
        results=results,
        total=2,
        page=1,
        limit=10,
        query="Gewerbebetrieb",
        execution_time_ms=42.5,
        vector_results=[]
    )


class TestSearchEndpointIntegration:
    """Integration tests for the search endpoint."""

    @patch('taxpilot.backend.search.search_api.SearchService')
    @patch('taxpilot.backend.api.search_utils.highlight_text')
    def test_search_endpoint_success(self, mock_highlight_text, mock_search_service_class, client, mock_search_results):
        """Test successful search with mocked search service."""
        # Setup mocks
        mock_search_service = MagicMock()
        mock_search_service.search.return_value = mock_search_results
        mock_search_service.close = MagicMock()  # Add close method
        mock_search_service_class.return_value = mock_search_service
        
        # Set up highlight_text mock
        mock_highlight_text.return_value = "Dies ist der Inhalt von § 13 EStG zu <mark>Land</mark>- und Forstwirtschaft."
        
        # Make request
        response = client.post(
            "/api/search",
            json={
                "query": "Gewerbebetrieb", 
                "search_type": "semantic",
                "page": 1,
                "limit": 10
            }
        )
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        
        # Check basic structure
        assert "results" in data
        assert "total_results" in data
        assert "page" in data
        assert "limit" in data
        assert "query" in data
        assert "search_type" in data
        assert "group_by_article" in data
        assert "execution_time_ms" in data
        
        # Check result structure
        assert len(data["results"]) == 2
        assert data["total_results"] == 2
        
        # Check first result
        first_result = data["results"][0]
        assert first_result["id"] == "estg_13_1"
        assert first_result["law_id"] == "estg"
        assert first_result["law_abbreviation"] == "EStG"
        assert first_result["section_number"] == "13"
        assert first_result["title"] == "Einkünfte aus Land- und Forstwirtschaft"
        
        # Verify mock was called with expected parameters
        mock_search_service.search.assert_called_once_with(
            query="Gewerbebetrieb",
            filters={},
            page=1,
            limit=10,
            highlight=True,
            cache=True,
            min_score=0.5,
            group_by_article=False
        )
        
        # Verify close was called
        mock_search_service.close.assert_called_once()
    
    @patch('taxpilot.backend.search.article_search.ArticleSearchService')
    def test_search_endpoint_with_article_grouping(self, mock_article_search_class, client, mock_search_results):
        """Test search with article grouping enabled."""
        # Setup mock
        mock_article_search = MagicMock()
        mock_article_search.search.return_value = mock_search_results
        mock_article_search.close = MagicMock()  # Add close method
        mock_article_search_class.return_value = mock_article_search
        
        # Make request with article grouping
        response = client.post(
            "/api/search",
            json={
                "query": "Gewerbebetrieb", 
                "search_type": "semantic",
                "group_by_article": True,
                "page": 1,
                "limit": 10
            }
        )
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        
        # Check article grouping flag
        assert data["group_by_article"] is True
        
        # Verify the right service was called
        mock_article_search.search.assert_called_once()
        mock_article_search.close.assert_called_once()
    
    @patch('taxpilot.backend.search.search_api.SearchService')
    def test_search_endpoint_with_filters(self, mock_search_service_class, client, mock_search_results):
        """Test search with filters applied."""
        # Setup mock
        mock_search_service = MagicMock()
        mock_search_service.search.return_value = mock_search_results
        mock_search_service.close = MagicMock()  # Add close method
        mock_search_service_class.return_value = mock_search_service
        
        # Make request with filters
        response = client.post(
            "/api/search",
            json={
                "query": "Gewerbebetrieb", 
                "filters": {"law_id": "estg"},
                "page": 1,
                "limit": 10
            }
        )
        
        # Assert response
        assert response.status_code == 200
        
        # Verify mock was called with expected parameters including filters
        mock_search_service.search.assert_called_once_with(
            query="Gewerbebetrieb",
            filters={"law_id": "estg"},
            page=1,
            limit=10,
            highlight=True,
            cache=True,
            min_score=0.5,
            group_by_article=False
        )
    
    @patch('taxpilot.backend.search.search_api.SearchService')
    def test_search_endpoint_error_handling(self, mock_search_service_class, client):
        """Test error handling in search endpoint."""
        # Setup mock to raise exception
        mock_search_service = MagicMock()
        mock_search_service.search.side_effect = ValueError("Invalid search parameter")
        mock_search_service.close = MagicMock()  # Add close method
        mock_search_service_class.return_value = mock_search_service
        
        # Make request
        response = client.post(
            "/api/search",
            json={"query": "Gewerbebetrieb"}
        )
        
        # Assert response
        assert response.status_code == 422  # Validation error
        assert "Invalid search request" in response.json()["detail"]
        
        # Verify close was called even after error
        mock_search_service.close.assert_called_once()