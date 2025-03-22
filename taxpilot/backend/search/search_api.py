"""
Search API implementation for GermanLawFinder.

This module provides the core search functionality used by the API endpoints.
It handles:
- Query processing and conversion to embeddings
- Vector search execution
- Result formatting and highlighting
- Caching and optimization
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Tuple

from taxpilot.backend.search.embeddings import TextEmbedder
from taxpilot.backend.search.vector_db import (
    VectorDatabase, 
    VectorDatabaseManager,
    SearchParameters,
    SearchResult
)
from taxpilot.backend.data_processing.database import get_connection, close_connection, DbConfig


# Create a wrapper class for database connection
class DatabaseConnection:
    """Wrapper for database connection functions."""
    
    def __init__(self, config: DbConfig | None = None):
        """Initialize with optional configuration."""
        self.config = config or DbConfig()
    
    def get_connection(self):
        """Get a database connection."""
        return get_connection(self.config)
    
    def close(self):
        """Close the database connection."""
        close_connection()

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Represents a single search result with highlighting and metadata."""
    id: str
    law_id: str
    section_number: str
    title: str
    content: str
    content_with_highlights: str
    relevance_score: float
    metadata: Dict[str, Any]


@dataclass
class SearchResults:
    """Container for search results with pagination information."""
    results: List[QueryResult]
    total: int
    page: int
    limit: int
    query: str
    execution_time_ms: float


class SearchService:
    """
    Service for handling search queries in GermanLawFinder.
    
    This class provides high-level search functionality by combining
    vector search with metadata filtering and result formatting.
    """
    
    def __init__(
        self,
        vector_db: Optional[Union[VectorDatabase, VectorDatabaseManager]] = None,
        embedder: Optional[TextEmbedder] = None,
        db_connection: Optional[DatabaseConnection] = None,
        cache_size: int = 100
    ):
        """
        Initialize the search service.
        
        Args:
            vector_db: Vector database for embeddings search
            embedder: Text embedder for converting queries to vectors
            db_connection: Database connection for metadata retrieval
            cache_size: Size of the results cache
        """
        # Initialize or create services
        self.vector_db = vector_db or VectorDatabaseManager()
        self.embedder = embedder or TextEmbedder()
        self.db_connection = db_connection or DatabaseConnection()
        
        # Simple LRU cache for common queries
        self._cache = {}
        self._cache_size = cache_size
        self._cache_order = []
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        limit: int = 10,
        highlight: bool = True,
        cache: bool = True
    ) -> SearchResults:
        """
        Search for laws matching the query with optional filtering.
        
        Args:
            query: The search query text
            filters: Optional filters (law_id, section, etc.)
            page: Page number for pagination (1-based)
            limit: Maximum number of results per page
            highlight: Whether to highlight matching text
            cache: Whether to use and update the cache
        
        Returns:
            SearchResults object containing the search results
        """
        import time
        start_time = time.time()
        
        # Normalize query
        query = query.strip()
        
        # Check cache for common queries
        cache_key = self._get_cache_key(query, filters, page, limit)
        if cache and cache_key in self._cache:
            logger.debug(f"Cache hit for query: {query}")
            return self._cache[cache_key]
        
        # Process filters
        processed_filters = self._process_filters(filters or {})
        
        # Calculate offset for pagination
        offset = (page - 1) * limit
        
        try:
            # Generate embedding from the query text
            query_embedding = self.embedder.embed_text(query)
            
            # Prepare search parameters
            search_params = SearchParameters(
                vector=query_embedding,
                limit=limit,
                offset=offset,
                filter_conditions=processed_filters,
                min_score=0.6  # Configurable threshold
            )
            
            # Execute vector search
            vector_results = self.vector_db.search(search_params)
            
            # Enrich results with full content from database
            enriched_results = self._enrich_results(vector_results, query, highlight)
            
            # Package results
            results = SearchResults(
                results=enriched_results,
                total=len(vector_results) if offset == 0 else self._estimate_total_results(query, processed_filters),
                page=page,
                limit=limit,
                query=query,
                execution_time_ms=round((time.time() - start_time) * 1000, 2)
            )
            
            # Update cache for future queries
            if cache:
                self._update_cache(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}", exc_info=True)
            # Return empty results on error
            return SearchResults(
                results=[],
                total=0,
                page=page,
                limit=limit,
                query=query,
                execution_time_ms=0
            )
    
    def _process_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate search filters.
        
        Args:
            filters: Dictionary of filter conditions
            
        Returns:
            Processed filters ready for the vector database
        """
        processed = {}
        
        # Handle law filter
        if 'law_id' in filters and filters['law_id']:
            processed['law_id'] = filters['law_id']
            
        # Handle section filter
        if 'section_id' in filters and filters['section_id']:
            processed['section_id'] = filters['section_id']
            
        # Handle date/version filter
        if 'version' in filters and filters['version']:
            processed['version'] = filters['version']
            
        return processed
    
    def _enrich_results(
        self, 
        vector_results: List[SearchResult],
        query: str,
        highlight: bool
    ) -> List[QueryResult]:
        """
        Enrich vector search results with full content and highlighting.
        
        Args:
            vector_results: List of search results from vector database
            query: Original search query for highlighting
            highlight: Whether to highlight matching text
            
        Returns:
            List of QueryResult objects with full content
        """
        # If there are no results, return an empty list
        if not vector_results:
            return []
            
        # Get segment IDs to retrieve from database
        segment_ids = [result.segment_id for result in vector_results]
        
        # Retrieve full content for all segments
        segment_content = self._get_segments_content(segment_ids)
        
        # Process results
        enriched_results = []
        for result in vector_results:
            if result.segment_id not in segment_content:
                continue
                
            content = segment_content[result.segment_id]["content"]
            section_content = segment_content[result.segment_id].get("section_content", content)
            
            # Apply highlighting if requested
            highlighted_content = self._highlight_text(section_content, query) if highlight else section_content
            
            # Create enriched result
            enriched_result = QueryResult(
                id=result.segment_id,
                law_id=result.metadata.get("law_id", ""),
                section_number=result.metadata.get("section_number", ""),
                title=result.metadata.get("title", ""),
                content=section_content,
                content_with_highlights=highlighted_content,
                relevance_score=result.score,
                metadata=result.metadata
            )
            
            enriched_results.append(enriched_result)
            
        return enriched_results
    
    def _get_segments_content(self, segment_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve full content for segments from database.
        
        Args:
            segment_ids: List of segment IDs to retrieve
            
        Returns:
            Dictionary mapping segment IDs to content
        """
        if not segment_ids:
            return {}
            
        # Placeholder for actual database retrieval
        # In a real implementation, this would query DuckDB
        # for the full content of the segments
        
        with self.db_connection.get_connection() as conn:
            # Execute query to get segment content
            segments_data = {}
            
            query = """
            SELECT 
                s.id AS segment_id,
                s.content AS content,
                sections.content AS section_content,
                sections.title AS title,
                sections.section_number,
                laws.id AS law_id,
                laws.abbreviation AS law_abbreviation,
                laws.full_name AS law_name
            FROM 
                segments s
            JOIN 
                sections ON s.section_id = sections.id
            JOIN 
                laws ON sections.law_id = laws.id
            WHERE 
                s.id IN ({})
            """.format(",".join([f"'{id}'" for id in segment_ids]))
            
            try:
                result = conn.execute(query).fetchall()
                
                for row in result:
                    segment_id = row["segment_id"]
                    segments_data[segment_id] = {
                        "content": row["content"],
                        "section_content": row["section_content"],
                        "title": row["title"],
                        "section_number": row["section_number"],
                        "law_id": row["law_id"],
                        "law_abbreviation": row["law_abbreviation"],
                        "law_name": row["law_name"]
                    }
            except Exception as e:
                logger.error(f"Error retrieving segment content: {e}", exc_info=True)
                
            return segments_data
    
    def _highlight_text(self, text: str, query: str) -> str:
        """
        Highlight matching text in search results.
        
        Args:
            text: The text to highlight
            query: The search query
            
        Returns:
            Text with HTML highlighting applied
        """
        if not text or not query:
            return text
            
        # Simple highlighting implementation
        # For production, consider using a more sophisticated
        # highlighting algorithm that considers word boundaries
        
        import re
        
        # Split query into terms
        terms = query.lower().split()
        
        # Filter out very short terms and duplicates
        terms = list(set([term for term in terms if len(term) > 2]))
        
        # Sort by length (longest first) to avoid nested highlights
        terms.sort(key=len, reverse=True)
        
        # Create a copy of the text for highlighting
        highlighted = text
        
        # Apply highlighting for each term
        for term in terms:
            # Create a regex pattern that respects word boundaries
            pattern = r"\b{}\b".format(re.escape(term))
            replacement = f"<mark>{term}</mark>"
            
            try:
                # Case-insensitive replacement
                highlighted = re.sub(
                    pattern, 
                    replacement, 
                    highlighted, 
                    flags=re.IGNORECASE
                )
            except Exception as e:
                logger.warning(f"Regex error with term '{term}': {e}")
                
        return highlighted
    
    def _estimate_total_results(self, query: str, filters: Dict[str, Any]) -> int:
        """
        Estimate the total number of results for a query.
        
        Args:
            query: The search query
            filters: Applied filters
            
        Returns:
            Estimated total number of results
        """
        # For now, we'll do a quick search with a high limit
        # In the future, we could use a more efficient approach
        
        search_params = SearchParameters(
            text=query,
            limit=1000,  # High limit to estimate total
            filter_conditions=filters,
            min_score=0.6
        )
        
        try:
            results = self.vector_db.search(search_params)
            return len(results)
        except Exception as e:
            logger.error(f"Error estimating total results: {e}", exc_info=True)
            return 0
    
    def _get_cache_key(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]], 
        page: int, 
        limit: int
    ) -> str:
        """
        Generate a cache key for a search query.
        
        Args:
            query: The search query
            filters: Applied filters
            page: Page number
            limit: Results per page
            
        Returns:
            String cache key
        """
        # Create a deterministic string from the parameters
        filter_str = ""
        if filters:
            # Sort the keys for deterministic ordering
            for key in sorted(filters.keys()):
                filter_str += f"_{key}_{filters[key]}"
                
        return f"{query}{filter_str}_p{page}_l{limit}"
    
    def _update_cache(self, key: str, results: SearchResults) -> None:
        """
        Update the search results cache.
        
        Args:
            key: Cache key
            results: Search results to cache
        """
        # If cache is full, remove the oldest entry
        if len(self._cache) >= self._cache_size and key not in self._cache:
            oldest_key = self._cache_order.pop(0)
            self._cache.pop(oldest_key, None)
            
        # Add new results to cache
        self._cache[key] = results
        
        # Update the access order
        if key in self._cache_order:
            self._cache_order.remove(key)
        self._cache_order.append(key)
    
    def close(self) -> None:
        """
        Close all resources used by the search service.
        """
        # Close vector database connection
        if hasattr(self.vector_db, 'close'):
            self.vector_db.close()
            
        # Close text embedder
        if hasattr(self.embedder, 'close'):
            self.embedder.close()
            
        # Close database connection
        self.db_connection.close()