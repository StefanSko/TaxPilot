"""
API models for GermanLawFinder.

This module defines Pydantic models for API request and response validation.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """
    Search request model for the search API endpoint.
    
    Attributes:
        query: The search query text
        search_type: Type of search to perform (semantic or keyword)
        group_by_article: Whether to group results by article instead of segments
        filters: Optional filters to apply (law_id, section_id, etc.)
        highlight: Whether to highlight matching text in results
        page: Page number (1-indexed)
        limit: Results per page
        min_score: Minimum relevance score threshold
    """
    query: str = Field(..., min_length=1, description="The search query text")
    search_type: Literal["semantic", "keyword"] | None = Field(
        "semantic", description="Type of search to perform"
    )
    group_by_article: bool | None = Field(
        False, description="Group results by article instead of segments"
    )
    filters: dict[str, object] | None = Field(
        {}, description="Filters to apply (law_id, section_id, etc.)"
    )
    highlight: bool | None = Field(
        True, description="Highlight matching text in results"
    )
    page: int | None = Field(
        1, ge=1, description="Page number (1-indexed)"
    )
    limit: int | None = Field(
        10, ge=1, le=50, description="Results per page"
    )
    min_score: float | None = Field(
        0.5, ge=0, le=1.0, description="Minimum relevance score threshold"
    )

    @field_validator('query')
    @classmethod
    def normalize_query(cls, v: str) -> str:
        """Normalize whitespace in query."""
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return ' '.join(v.split())


class SearchResultItem(BaseModel):
    """
    Model for a single search result item.
    
    Attributes:
        id: Unique identifier for the result
        law_id: Identifier for the law (e.g., "estg")
        law_abbreviation: Abbreviation for the law (e.g., "EStG")
        section_number: Section number in the law
        title: Title of the section
        content: Content of the section
        content_with_highlights: Content with matching text highlighted
        relevance_score: Relevance score for the result
        is_article_result: Whether this is an article-level result
        metadata: Additional metadata about the result
    """
    id: str
    law_id: str
    law_abbreviation: str
    section_number: str
    title: str
    content: str
    content_with_highlights: str
    relevance_score: float
    is_article_result: bool = False
    metadata: dict[str, object] | None = None


class SearchResponse(BaseModel):
    """
    Search response model for the search API endpoint.
    
    Attributes:
        results: List of search result items
        total_results: Total number of results matching the query
        page: Current page number
        limit: Results per page
        query: Original search query
        search_type: Type of search performed
        group_by_article: Whether results are grouped by article
        execution_time_ms: Execution time in milliseconds
    """
    results: list[SearchResultItem]
    total_results: int
    page: int
    limit: int
    query: str
    search_type: str
    group_by_article: bool
    execution_time_ms: float