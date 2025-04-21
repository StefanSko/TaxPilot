"""
Test the article-based search functionality.

This script tests the enhanced search that groups results by article
instead of returning disconnected text segments.
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

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


class TestArticleSearch(unittest.TestCase):
    """Test cases for article-based search."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock vector search results
        self.vector_results = [
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
        
        # Create corresponding query results
        self.query_results = [
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
        
        # Create search results object
        self.search_results = SearchResults(
            results=self.query_results,
            total=3,
            page=1,
            limit=10,
            query="Landwirtschaft",
            execution_time_ms=25.5,
            vector_results=self.vector_results
        )
        
        # Create mock search service
        self.mock_search_service = MagicMock()
        self.mock_search_service.search.return_value = self.search_results
        
        # Create article search service
        self.article_service = ArticleSearchService(
            base_search_service=self.mock_search_service,
            score_strategy=ArticleScoreStrategy.MAX
        )
    
    def test_article_aggregation(self):
        """Test that segments are correctly aggregated into articles."""
        # Test the aggregator directly
        aggregator = ArticleSearchAggregator(
            score_strategy=ArticleScoreStrategy.MAX,
            max_segments_per_article=5
        )
        
        article_dict = aggregator.aggregate_results(self.vector_results)
        
        # There should be two articles
        self.assertEqual(len(article_dict), 2)
        
        # Check article IDs
        article_keys = sorted(article_dict.keys())
        self.assertIn("estg_13", article_keys[0])
        self.assertIn("estg_14", article_keys[1])
        
        # Article 1 should have two segments
        article1 = article_dict.get(f"estg:estg_13", article_dict.get("estg_13"))
        self.assertEqual(len(article1.matching_segments), 2)
        
        # Article 1's score should be the max of its segments (0.92)
        self.assertEqual(article1.relevance_score, 0.92)
        
        # Article 2 should have one segment
        article2 = article_dict.get(f"estg:estg_14", article_dict.get("estg_14"))
        self.assertEqual(len(article2.matching_segments), 1)
        
        # Article 2's score should be the score of its only segment (0.78)
        self.assertEqual(article2.relevance_score, 0.78)
    
    def test_group_results_by_article(self):
        """Test the grouping of search results by article."""
        # Call the method under test
        article_results = self.article_service._group_results_by_article(
            self.search_results,
            "Landwirtschaft",
            True
        )
        
        # Check that the right number of articles are returned
        self.assertEqual(len(article_results.results), 2)
        
        # Check that articles are ordered by score (highest first)
        self.assertEqual(article_results.results[0].section_number, "13")
        self.assertEqual(article_results.results[1].section_number, "14")
        
        # Check that result metadata indicates it's an article result
        self.assertIn("is_article_result", article_results.results[0].metadata)
        self.assertTrue(article_results.results[0].metadata["is_article_result"])
        
        # Check that the number of matching segments is recorded
        self.assertEqual(article_results.results[0].metadata["matching_segments"], 2)
        self.assertEqual(article_results.results[1].metadata["matching_segments"], 1)
    
    def test_article_search(self):
        """Test article-based search through the search interface."""
        # Create a search service with mocked vector_db and embedder
        with patch('taxpilot.backend.search.search_api.SearchService._enrich_results') as mock_enrich:
            # Set up the mock to return our query results
            mock_enrich.return_value = self.query_results
            
            # Create a real search service with mocked components
            search_service = SearchService(
                vector_db=MagicMock(),
                embedder=MagicMock(),
                db_connection=MagicMock()
            )
            
            # Mock the vector_db.search_similar to return our vector results
            search_service.vector_db.search_similar.return_value = self.vector_results
            
            # Enable article-based search
            results = search_service.search(
                query="Landwirtschaft",
                group_by_article=True
            )
            
            # Verify that article grouping was used
            self.assertEqual(len(results.results), 2)
            
            # Verify that results are sorted by score
            self.assertGreaterEqual(
                results.results[0].relevance_score,
                results.results[1].relevance_score
            )


if __name__ == "__main__":
    unittest.main()