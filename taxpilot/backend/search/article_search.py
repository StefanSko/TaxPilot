"""
Article-based search functionality for TaxPilot.

This module provides the enhanced search functionality that groups search results
by article, enabling users to see complete articles in search results rather than
disconnected text segments.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Set
import statistics

from taxpilot.backend.search.vector_db import SearchResult, SearchParameters
from taxpilot.backend.search.search_api import QueryResult, SearchResults

# Configure logging
logger = logging.getLogger(__name__)


class ArticleScoreStrategy(Enum):
    """Strategy for calculating article-level relevance scores."""
    
    MAX = "max"  # Use the highest segment score
    AVERAGE = "average"  # Use the average of all segment scores
    WEIGHTED = "weighted"  # Weight by position and count
    COUNT_BOOSTED = "count_boosted"  # Boost by count of matching segments


@dataclass
class MatchingSegment:
    """A segment that matched the search query."""
    
    segment_id: str
    text: str
    score: float
    text_with_highlights: str = ""
    position_in_article: int = 0
    segment_type: str = ""


@dataclass
class ArticleSearchResult:
    """A search result representing a complete article with matched segments."""
    
    article_id: str
    law_id: str
    title: str
    law_abbreviation: str
    section_number: str
    content: str
    relevance_score: float
    matching_segments: List[MatchingSegment] = field(default_factory=list)
    
    @property
    def best_matching_segment(self) -> Optional[MatchingSegment]:
        """Get the highest-scoring matching segment."""
        if not self.matching_segments:
            return None
        return max(self.matching_segments, key=lambda s: s.score)
    
    @property
    def content_with_highlights(self) -> str:
        """Generate content with all matching segments highlighted."""
        # This is a placeholder - actual implementation would require 
        # merging highlights from all segments into the content
        return self.content


class ArticleSearchAggregator:
    """
    Aggregates segment-level search results into article-level results.
    
    This class takes segment-level search results and groups them by article,
    calculating aggregate relevance scores and presenting complete articles
    as search results.
    """
    
    def __init__(
        self, 
        score_strategy: ArticleScoreStrategy = ArticleScoreStrategy.WEIGHTED,
        max_segments_per_article: int = 3,
        min_article_score: float = 0.6
    ):
        """
        Initialize the article search aggregator.
        
        Args:
            score_strategy: Strategy for calculating article relevance scores
            max_segments_per_article: Maximum number of top segments to include per article
            min_article_score: Minimum score for an article to be included in results
        """
        self.score_strategy = score_strategy
        self.max_segments_per_article = max_segments_per_article
        self.min_article_score = min_article_score
    
    def aggregate_results(self, segment_results: List[SearchResult]) -> Dict[str, ArticleSearchResult]:
        """
        Aggregate segment-level search results into article-level results.
        
        Args:
            segment_results: List of search results at the segment level
            
        Returns:
            Dictionary mapping article IDs to ArticleSearchResult objects
        """
        # Group results by article
        article_groups: Dict[str, List[SearchResult]] = {}
        
        for result in segment_results:
            # Use article_id if available, otherwise use section_id
            article_key = result.article_id if result.article_id else f"{result.law_id}:{result.section_id}"
            
            if article_key not in article_groups:
                article_groups[article_key] = []
                
            article_groups[article_key].append(result)
        
        # Create article results with aggregate scores
        article_results: Dict[str, ArticleSearchResult] = {}
        
        for article_key, segments in article_groups.items():
            # Sort segments by score
            sorted_segments = sorted(segments, key=lambda s: s.score, reverse=True)
            
            # Get article information from the highest-scoring segment
            best_segment = sorted_segments[0]
            article_id = best_segment.article_id or best_segment.section_id
            
            # Calculate article score based on strategy
            article_score = self._calculate_article_score(sorted_segments)
            
            # Skip articles with low scores
            if article_score < self.min_article_score:
                continue
            
            # Create article result (content will be populated later)
            article_result = ArticleSearchResult(
                article_id=article_id,
                law_id=best_segment.law_id,
                title="",  # To be populated later
                law_abbreviation="",  # To be populated later
                section_number="",  # To be populated later
                content="",  # To be populated later
                relevance_score=article_score
            )
            
            # Add matching segments (limited to max per article)
            for segment in sorted_segments[:self.max_segments_per_article]:
                matching_segment = MatchingSegment(
                    segment_id=segment.segment_id,
                    text="",  # To be populated later
                    score=segment.score,
                    position_in_article=segment.position_in_parent,
                    segment_type=segment.segment_type
                )
                article_result.matching_segments.append(matching_segment)
            
            article_results[article_key] = article_result
        
        return article_results
    
    def _calculate_article_score(self, segments: List[SearchResult]) -> float:
        """
        Calculate article-level score based on the configured strategy.
        
        Args:
            segments: List of segment results belonging to the article
            
        Returns:
            Aggregate article score
        """
        if not segments:
            return 0.0
        
        # Extract scores
        scores = [s.score for s in segments]
        
        if self.score_strategy == ArticleScoreStrategy.MAX:
            # Maximum score strategy
            return max(scores)
            
        elif self.score_strategy == ArticleScoreStrategy.AVERAGE:
            # Average score strategy
            return sum(scores) / len(scores)
            
        elif self.score_strategy == ArticleScoreStrategy.COUNT_BOOSTED:
            # Boost by count of matching segments
            count_factor = min(1.0, (len(segments) / 5.0))  # Cap at 5 segments
            return max(scores) * (1.0 + 0.5 * count_factor)
            
        elif self.score_strategy == ArticleScoreStrategy.WEIGHTED:
            # Weighted strategy considering position and quantity
            # Weight by position (higher weight to earlier segments)
            position_weights = []
            for i, segment in enumerate(segments):
                # Default position weight if position_in_parent is 0
                if segment.position_in_parent <= 0:
                    weight = 1.0 - (i / (len(segments) + 1))
                else:
                    # Lower weight for segments deeper in the document
                    weight = 1.0 / (1.0 + 0.2 * segment.position_in_parent)
                position_weights.append(weight)
            
            # Calculate weighted average
            total_weight = sum(position_weights)
            weighted_score = sum(s * w for s, w in zip(scores, position_weights)) / total_weight
            
            # Add a boost based on number of matching segments
            count_factor = min(1.0, (len(segments) / 5.0))  # Cap at 5 segments
            boost = 0.2 * count_factor
            
            return weighted_score * (1.0 + boost)
        
        # Default to max score if strategy not recognized
        return max(scores)


class ArticleSearchService:
    """
    Service for article-based search in TaxPilot.
    
    This class extends the basic search functionality to group results by article,
    providing a more user-friendly presentation of search results.
    """
    
    def __init__(
        self,
        base_search_service=None,
        score_strategy: ArticleScoreStrategy = ArticleScoreStrategy.WEIGHTED,
        max_segments_per_article: int = 3,
        min_article_score: float = 0.6
    ):
        """
        Initialize the article search service.
        
        Args:
            base_search_service: The underlying search service to use
            score_strategy: Strategy for calculating article relevance scores
            max_segments_per_article: Maximum number of segments to include per article
            min_article_score: Minimum score for an article to be included
        """
        self.base_search_service = base_search_service
        self.aggregator = ArticleSearchAggregator(
            score_strategy=score_strategy,
            max_segments_per_article=max_segments_per_article,
            min_article_score=min_article_score
        )
    
    def _group_results_by_article(
        self,
        segment_results: SearchResults,
        query: str,
        highlight: bool = True
    ) -> SearchResults:
        """
        Group segment-level search results by article.
        
        Args:
            segment_results: Original segment-level search results
            query: Original search query
            highlight: Whether to highlight matches
            
        Returns:
            Search results with segments grouped by article
        """
        # Early return if there are no results to group
        if not segment_results.results or not segment_results.vector_results:
            return segment_results
            
        # Group segments by article
        article_dict = self.aggregator.aggregate_results(segment_results.vector_results)
        
        # Sort articles by score
        sorted_articles = sorted(
            article_dict.values(),
            key=lambda a: a.relevance_score,
            reverse=True
        )
        
        # Limit to the number of results in the original request
        articles_page = sorted_articles[:segment_results.limit]
        
        # Enrich article results with full content
        enriched_articles = self._enrich_article_results(
            articles_page,
            segment_results.results,
            query,
            highlight
        )
        
        # Create article-based search results
        article_results = SearchResults(
            results=enriched_articles,
            total=len(sorted_articles),
            page=segment_results.page,
            limit=segment_results.limit,
            query=query,
            execution_time_ms=segment_results.execution_time_ms,
            vector_results=segment_results.vector_results
        )
        
        return article_results
    
    def search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        page: int = 1,
        limit: int = 10,
        highlight: bool = True,
        cache: bool = True,
        min_score: float = 0.5,
        group_by_article: bool = True
    ) -> SearchResults:
        """
        Perform a search with article-based grouping.
        
        Args:
            query: The search query text
            filters: Optional filters (law_id, section, etc.)
            page: Page number for pagination (1-based)
            limit: Maximum number of results per page
            highlight: Whether to highlight matching text
            cache: Whether to use and update the cache
            min_score: Minimum relevance score for segment results
            group_by_article: Whether to group results by article
            
        Returns:
            SearchResults object containing the search results
        """
        # Phase 1: Perform segment-level search with increased limit
        # We need more segments to ensure good article coverage
        segment_limit = limit * 5  # Request more segments to get good article coverage
        
        # Create search parameters with group_by_article flag
        search_params = {
            "query": query,
            "filters": filters,
            "page": page,
            "limit": segment_limit,
            "highlight": highlight,
            "cache": cache,
            "min_score": min_score,
            "group_by_article": group_by_article
        }
        
        # Execute segment-level search
        segment_results = self.base_search_service.search(**search_params)
        
        # If grouping is disabled, return segment results directly
        if not group_by_article:
            # Limit to the requested number of results
            segment_results.results = segment_results.results[:limit]
            segment_results.total = min(segment_results.total, len(segment_results.results))
            return segment_results
        
        # Phase 2: Group segments by article and calculate aggregate scores
        article_dict = self.aggregator.aggregate_results(segment_results.vector_results)
        
        # Sort articles by score
        sorted_articles = sorted(
            article_dict.values(),
            key=lambda a: a.relevance_score,
            reverse=True
        )
        
        # Limit to requested number of articles
        articles_page = sorted_articles[:limit]
        
        # Phase 3: Enrich article results with full content
        enriched_articles = self._enrich_article_results(
            articles_page,
            segment_results.results,
            query,
            highlight
        )
        
        # Create final search results
        results = SearchResults(
            results=enriched_articles,
            total=len(sorted_articles),
            page=page,
            limit=limit,
            query=query,
            execution_time_ms=segment_results.execution_time_ms
        )
        
        return results
    
    def _enrich_article_results(
        self,
        article_results: List[ArticleSearchResult],
        segment_query_results: List[QueryResult],
        query: str,
        highlight: bool
    ) -> List[QueryResult]:
        """
        Enrich article results with full content and highlighting.
        
        Args:
            article_results: List of article search results
            segment_query_results: Original segment-level query results
            query: Original search query
            highlight: Whether to highlight matching text
            
        Returns:
            List of article-level QueryResult objects
        """
        # Create mapping from segment_id to QueryResult for easy lookup
        segment_dict = {result.id: result for result in segment_query_results}
        
        # Create article-level query results
        article_query_results = []
        
        for article in article_results:
            # Find matching segments with content
            matching_segments_with_content = []
            
            for segment in article.matching_segments:
                if segment.segment_id in segment_dict:
                    query_result = segment_dict[segment.segment_id]
                    segment.text = query_result.content
                    segment.text_with_highlights = query_result.content_with_highlights
                    matching_segments_with_content.append(segment)
            
            # Set article content based on highest scoring segment for now
            # This will be replaced with full article content retrieval in the future
            if matching_segments_with_content:
                best_segment = max(matching_segments_with_content, key=lambda s: s.score)
                best_query_result = segment_dict[best_segment.segment_id]
                
                article_query_result = QueryResult(
                    id=article.article_id,
                    law_id=article.law_id,
                    law_abbreviation=best_query_result.law_abbreviation,
                    section_number=best_query_result.section_number,
                    title=best_query_result.title,
                    content=best_query_result.content,
                    content_with_highlights=best_query_result.content_with_highlights,
                    relevance_score=article.relevance_score,
                    metadata={
                        "article_id": article.article_id,
                        "matching_segments": len(matching_segments_with_content),
                        "is_article_result": True
                    }
                )
                
                article_query_results.append(article_query_result)
        
        return article_query_results