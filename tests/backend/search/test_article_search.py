"""
Unit tests for the article-based search functionality.

These tests verify the enhanced search that groups results by article
instead of returning disconnected text segments.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from taxpilot.backend.search.article_search import (
    ArticleSearchService,
    ArticleSearchAggregator,
    ArticleScoreStrategy,
    ArticleSearchResult,
    MatchingSegment
)
from taxpilot.backend.search.search_api import (
    SearchService, 
    QueryResult,
    SearchResults
)
from taxpilot.backend.search.vector_db import SearchResult


@pytest.fixture
def mock_vector_results():
    """Create mock vector search results."""
    return [
        # Article 1, Section 1, two segments
        SearchResult(
            segment_id="estg_13_p1",
            law_id="estg",
            section_id="estg_13",
            article_id="estg_13",
            hierarchy_path="estg/§13/abs1",
            segment_type="paragraph",
            position_in_parent=1,
            score=0.92,
            metadata={},
            embedding_model="test_model",
            embedding_version="1.0"
        ),
        SearchResult(
            segment_id="estg_13_p2",
            law_id="estg",
            section_id="estg_13",
            article_id="estg_13",
            hierarchy_path="estg/§13/abs2",
            segment_type="paragraph",
            position_in_parent=2,
            score=0.85,
            metadata={},
            embedding_model="test_model",
            embedding_version="1.0"
        ),
        # Article 2, Section 2, one segment
        SearchResult(
            segment_id="estg_14_p1",
            law_id="estg",
            section_id="estg_14",
            article_id="estg_14",
            hierarchy_path="estg/§14/abs1",
            segment_type="paragraph",
            position_in_parent=1,
            score=0.78,
            metadata={},
            embedding_model="test_model",
            embedding_version="1.0"
        )
    ]


@pytest.fixture
def mock_query_results():
    """Create mock query results."""
    return [
        QueryResult(
            id="estg_13_p1",
            law_id="estg",
            law_abbreviation="EStG",
            section_number="13",
            title="Einkünfte aus Land- und Forstwirtschaft",
            content="Paragraph 1 content about Landwirtschaft.",
            content_with_highlights="Paragraph 1 content about <mark>Landwirtschaft</mark>.",
            relevance_score=0.92,
            metadata={}
        ),
        QueryResult(
            id="estg_13_p2",
            law_id="estg",
            law_abbreviation="EStG",
            section_number="13",
            title="Einkünfte aus Land- und Forstwirtschaft",
            content="Paragraph 2 content about Forstwirtschaft.",
            content_with_highlights="Paragraph 2 content about <mark>Forstwirtschaft</mark>.",
            relevance_score=0.85,
            metadata={}
        ),
        QueryResult(
            id="estg_14_p1",
            law_id="estg",
            law_abbreviation="EStG",
            section_number="14",
            title="Veräußerung von Betriebsvermögen",
            content="Content about Veräußerung.",
            content_with_highlights="Content about <mark>Veräußerung</mark>.",
            relevance_score=0.78,
            metadata={}
        )
    ]


@pytest.fixture
def mock_search_results(mock_vector_results, mock_query_results):
    """Create mock search results."""
    return SearchResults(
        results=mock_query_results,
        total=3,
        page=1,
        limit=10,
        query="Landwirtschaft",
        execution_time_ms=25.5,
        vector_results=mock_vector_results
    )


@pytest.fixture
def article_search_service():
    """Create an article search service with a mock base service."""
    mock_search_service = MagicMock()
    article_service = ArticleSearchService(
        base_search_service=mock_search_service,
        score_strategy=ArticleScoreStrategy.MAX
    )
    return article_service, mock_search_service


def test_article_aggregation(mock_vector_results):
    """Test that segments are correctly aggregated into articles."""
    # Test the aggregator directly
    aggregator = ArticleSearchAggregator(
        score_strategy=ArticleScoreStrategy.MAX,
        max_segments_per_article=5
    )
    
    article_dict = aggregator.aggregate_results(mock_vector_results)
    
    # There should be two articles
    assert len(article_dict) == 2
    
    # Check article IDs are in the results
    article_keys = sorted(article_dict.keys())
    assert "estg_13" in article_keys[0] or "estg:estg_13" in article_keys[0]
    assert "estg_14" in article_keys[1] or "estg:estg_14" in article_keys[1]
    
    # Article 1 should have two segments
    article1_key = [k for k in article_dict.keys() if "estg_13" in k][0]
    article1 = article_dict[article1_key]
    assert len(article1.matching_segments) == 2
    
    # Article 1's score should be the max of its segments (0.92)
    assert article1.relevance_score == 0.92
    
    # Article 2 should have one segment
    article2_key = [k for k in article_dict.keys() if "estg_14" in k][0]
    article2 = article_dict[article2_key]
    assert len(article2.matching_segments) == 1
    
    # Article 2's score should be the score of its only segment (0.78)
    assert article2.relevance_score == 0.78


def test_group_results_by_article(article_search_service, mock_search_results):
    """Test the grouping of search results by article."""
    article_service, _ = article_search_service
    
    # Call the method under test
    article_results = article_service._group_results_by_article(
        mock_search_results,
        "Landwirtschaft",
        True
    )
    
    # Check that the right number of articles are returned
    assert len(article_results.results) == 2
    
    # Check that articles are ordered by score (highest first)
    assert article_results.results[0].section_number == "13"
    assert article_results.results[1].section_number == "14"
    
    # Check that result metadata indicates it's an article result
    assert "is_article_result" in article_results.results[0].metadata
    assert article_results.results[0].metadata["is_article_result"] is True
    
    # Check that the number of matching segments is recorded
    assert article_results.results[0].metadata["matching_segments"] == 2
    assert article_results.results[1].metadata["matching_segments"] == 1


def test_article_search(mock_vector_results, mock_query_results):
    """Test article-based search through the search interface."""
    # Create a mock that will be used for enrich_results
    with patch('taxpilot.backend.search.search_api.SearchService._enrich_results') as mock_enrich:
        # Set up the mock to return our query results
        mock_enrich.return_value = mock_query_results
        
        # Create a search service with mocked components
        search_service = SearchService(
            vector_db=MagicMock(),
            embedder=MagicMock(),
            db_connection=MagicMock()
        )
        
        # Mock the vector_db.search_similar to return our vector results
        search_service.vector_db.search_similar.return_value = mock_vector_results
        
        # Enable article-based search
        results = search_service.search(
            query="Landwirtschaft",
            group_by_article=True
        )
        
        # Verify that article grouping was used
        assert len(results.results) == 2
        
        # Verify that results are sorted by score
        assert results.results[0].relevance_score >= results.results[1].relevance_score


def test_score_strategies():
    """Test different article scoring strategies."""
    # Create test segments
    segments = [
        SearchResult(
            segment_id="s1",
            law_id="estg",
            section_id="sec1",
            article_id="a1",
            score=0.9,
            metadata={"position_in_parent": 1},
            embedding_model="test_model",
            embedding_version="1.0"
        ),
        SearchResult(
            segment_id="s2",
            law_id="estg",
            section_id="sec1",
            article_id="a1",
            score=0.8,
            metadata={"position_in_parent": 2},
            embedding_model="test_model",
            embedding_version="1.0"
        ),
        SearchResult(
            segment_id="s3",
            law_id="estg",
            section_id="sec1",
            article_id="a1",
            score=0.7,
            metadata={"position_in_parent": 3},
            embedding_model="test_model",
            embedding_version="1.0"
        )
    ]
    
    # Test MAX strategy
    aggregator_max = ArticleSearchAggregator(score_strategy=ArticleScoreStrategy.MAX)
    score_max = aggregator_max._calculate_article_score(segments)
    assert score_max == 0.9
    
    # Test AVERAGE strategy
    aggregator_avg = ArticleSearchAggregator(score_strategy=ArticleScoreStrategy.AVERAGE)
    score_avg = aggregator_avg._calculate_article_score(segments)
    assert score_avg == pytest.approx((0.9 + 0.8 + 0.7) / 3, 0.01)
    
    # Test COUNT_BOOSTED strategy
    aggregator_count = ArticleSearchAggregator(score_strategy=ArticleScoreStrategy.COUNT_BOOSTED)
    score_count = aggregator_count._calculate_article_score(segments)
    # Should be max score (0.9) multiplied by (1 + 0.5 * count_factor)
    # where count_factor is min(1.0, len(segments)/5.0) = min(1.0, 3/5) = 0.6
    expected_count_boosted = 0.9 * (1.0 + 0.5 * 0.6)
    assert score_count == pytest.approx(expected_count_boosted, 0.01)