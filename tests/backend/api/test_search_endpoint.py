"""
Tests for the search API endpoint.

This module contains tests for the search API endpoint functionality, with minimal mocking.
We test against real components where possible to ensure the API works in actual conditions.
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
            content_with_highlights="Dies ist der Inhalt von § 13 EStG zu Land- und Forstwirtschaft.",
            relevance_score=0.92,
            metadata={"section_id": "estg_13", "segment_id": "estg_13_1"}
        )
    ]
    
    return SearchResults(
        results=results,
        total=1,
        page=1,
        limit=10,
        query="Steuer",
        execution_time_ms=42.5,
        vector_results=[]
    )


class TestSearchEndpointBasic:
    """Basic tests for the search endpoint functionality."""

    def test_health_check(self, client):
        """Test the health check endpoint to verify API is running."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json().get("status") == "healthy"

    @patch('taxpilot.backend.search.search_api.SearchService')
    def test_search_endpoint_exists(self, mock_search_service_class, client, mock_search_results):
        """Test that the search endpoint exists and accepts POST requests."""
        # Setup mock
        mock_search_service = MagicMock()
        mock_search_service.search.return_value = mock_search_results
        mock_search_service.close = MagicMock()  # Add close method
        mock_search_service_class.return_value = mock_search_service
        
        # Using a minimal request to check if the endpoint exists
        response = client.post(
            "/api/search",
            json={"query": "test query"}
        )
        # We don't care about the response content at this stage,
        # just that the endpoint exists and returns something
        assert response.status_code != 404, "Search endpoint not found"
        assert response.status_code == 200, "Search endpoint should return 200 OK"

    @patch('taxpilot.backend.search.search_api.SearchService')
    def test_empty_query_rejected(self, mock_search_service_class, client, mock_search_results):
        """Test that an empty query is rejected."""
        # Setup mock
        mock_search_service = MagicMock()
        mock_search_service.search.return_value = mock_search_results
        mock_search_service.close = MagicMock()
        mock_search_service_class.return_value = mock_search_service
        
        response = client.post(
            "/api/search",
            json={"query": ""}
        )
        assert response.status_code == 422, "Empty query should be rejected"

    @patch('taxpilot.backend.search.search_api.SearchService')
    def test_invalid_page_rejected(self, mock_search_service_class, client, mock_search_results):
        """Test that invalid page values are rejected."""
        # Setup mock
        mock_search_service = MagicMock()
        mock_search_service.search.return_value = mock_search_results
        mock_search_service.close = MagicMock()
        mock_search_service_class.return_value = mock_search_service
        
        response = client.post(
            "/api/search",
            json={"query": "test", "page": 0}
        )
        assert response.status_code == 422, "Invalid page should be rejected"

    @patch('taxpilot.backend.search.search_api.SearchService')
    def test_invalid_limit_rejected(self, mock_search_service_class, client, mock_search_results):
        """Test that invalid limit values are rejected."""
        # Setup mock
        mock_search_service = MagicMock()
        mock_search_service.search.return_value = mock_search_results
        mock_search_service.close = MagicMock()
        mock_search_service_class.return_value = mock_search_service
        
        response = client.post(
            "/api/search",
            json={"query": "test", "limit": 0}
        )
        assert response.status_code == 422, "Invalid limit should be rejected"

        response = client.post(
            "/api/search",
            json={"query": "test", "limit": 51}
        )
        assert response.status_code == 422, "Limit > 50 should be rejected"


class TestSearchEndpointResponseStructure:
    """Tests for the search endpoint response structure."""

    @patch('taxpilot.backend.search.search_api.SearchService')
    def test_response_structure_with_valid_query(self, mock_search_service_class, client, mock_search_results):
        """Test that the response has the expected structure for a valid query."""
        # Setup mock
        mock_search_service = MagicMock()
        mock_search_service.search.return_value = mock_search_results
        mock_search_service.close = MagicMock()
        mock_search_service_class.return_value = mock_search_service
        
        response = client.post(
            "/api/search",
            json={"query": "Steuer"}
        )
        assert response.status_code == 200, "Valid query should return 200 OK"
        
        data = response.json()
        
        # Check required fields
        assert "results" in data, "Response must contain 'results' field"
        assert "total_results" in data, "Response must contain 'total_results' field"
        assert "page" in data, "Response must contain 'page' field"
        assert "limit" in data, "Response must contain 'limit' field"
        assert "query" in data, "Response must contain 'query' field"
        assert "execution_time_ms" in data, "Response must contain 'execution_time_ms' field"
        
        # Check types
        assert isinstance(data["results"], list), "'results' must be a list"
        assert isinstance(data["total_results"], int), "'total_results' must be an integer"
        assert isinstance(data["page"], int), "'page' must be an integer"
        assert isinstance(data["limit"], int), "'limit' must be an integer"
        assert isinstance(data["query"], str), "'query' must be a string"
        assert isinstance(data["execution_time_ms"], (int, float)), "'execution_time_ms' must be a number"
        
        # Check values
        assert data["query"] == "Steuer", "Query in response must match request"
        assert data["page"] >= 1, "Page must be >= 1"
        assert data["limit"] >= 1, "Limit must be >= 1"


# Advanced tests that will be implemented incrementally
@pytest.mark.skip(reason="Advanced functionality will be implemented incrementally")
class TestSearchEndpointAdvanced:
    """Advanced tests for the search endpoint functionality."""

    def test_article_grouping(self, client):
        """Test article-based grouping functionality."""
        # Will be implemented later
        pass

    def test_filter_by_law(self, client):
        """Test filtering by law ID."""
        # Will be implemented later
        pass

    def test_highlighting(self, client):
        """Test result highlighting."""
        # Will be implemented later
        pass