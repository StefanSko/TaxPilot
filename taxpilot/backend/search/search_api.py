"""
Search API implementation for GermanLawFinder.

This module provides the core search functionality used by the API endpoints.
It handles:
- Query processing and conversion to embeddings
- Vector search execution
- Result formatting and highlighting
- Caching and optimization
- Article-based search results grouping
"""

import logging
from dataclasses import dataclass
from typing import Any
import json
import pandas as pd

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
    law_abbreviation: str
    section_number: str
    title: str
    content: str
    content_with_highlights: str
    relevance_score: float
    metadata: dict[str, Any]


@dataclass
class SearchResults:
    """Container for search results with pagination information."""
    results: list[QueryResult]
    total: int
    page: int
    limit: int
    query: str
    execution_time_ms: float
    vector_results: list = None  # Raw vector search results (not visible to API)


class SearchService:
    """
    Service for handling search queries in GermanLawFinder.
    
    This class provides high-level search functionality by combining
    vector search with metadata filtering and result formatting.
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase | VectorDatabaseManager | None = None,
        embedder: TextEmbedder | None = None,
        db_connection: DatabaseConnection | None = None,
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
        if vector_db:
            self.vector_db = vector_db
        else:
            # Check for in-memory mode from environment variables
            import os
            from taxpilot.backend.search.vector_db import VectorDbConfig, VectorDbProvider
            
            if os.getenv("QDRANT_IN_MEMORY") == "true":
                logger.info("Using in-memory vector database for search")
                config = VectorDbConfig(
                    provider=VectorDbProvider.MEMORY,
                    collection_name=os.getenv("QDRANT_COLLECTION", "law_sections")
                )
                self.vector_db = VectorDatabaseManager(config)
            else:
                # Use regular Qdrant connection
                self.vector_db = VectorDatabaseManager()
                
        self.embedder = embedder or TextEmbedder()
        self.db_connection = db_connection or DatabaseConnection()
        
        # Simple LRU cache for common queries
        self._cache: dict[str, SearchResults] = {}
        self._cache_size = cache_size
        self._cache_order: list[str] = []
    
    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        page: int = 1,
        limit: int = 10,
        highlight: bool = True,
        cache: bool = True,
        min_score: float = 0.5,
        group_by_article: bool = False  # New parameter for article-based search
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
            min_score: Minimum relevance score for results
        
        Returns:
            SearchResults object containing the search results
        """
        import time
        start_time = time.time()
        
        # Normalize query
        query = query.strip()
        
        # Check cache for common queries
        cache_key = self._get_cache_key(query, filters, page, limit, group_by_article)
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
                query_vector=query_embedding,
                top_k=limit,
                offset=offset,
                min_score=min_score
            )
            
            # Execute vector search using the correct manager method
            vector_results = self.vector_db.search_similar(search_params)
            
            # Enrich results with full content from database
            enriched_results = self._enrich_results(vector_results, query, highlight)
            
            # Package results
            results = SearchResults(
                results=enriched_results,
                total=len(vector_results) if offset == 0 else self._estimate_total_results(query, processed_filters),
                page=page,
                limit=limit,
                query=query,
                execution_time_ms=round((time.time() - start_time) * 1000, 2),
                vector_results=vector_results  # Include raw vector results for article aggregation
            )
            
            # Apply article-based grouping if requested
            if group_by_article:
                # Import here to avoid circular imports
                from taxpilot.backend.search.article_search import ArticleSearchService
                
                # Create article search service using this service as base
                article_service = ArticleSearchService(base_search_service=self)
                
                # Convert results to article-based format
                results = article_service._group_results_by_article(results, query, highlight)
            
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
                execution_time_ms=0,
                vector_results=[]
            )
    
    def _process_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
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
        vector_results: list[SearchResult],
        query: str,
        highlight: bool
    ) -> list[QueryResult]:
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
            
        # Get segment IDs AND section IDs to retrieve from database
        segment_ids: list[str] = [result.segment_id for result in vector_results]
        
        # Retrieve full content for all segments using the results directly
        # This function now needs the full results to get the correct section_id
        segment_details = self._get_segments_content(vector_results)
        
        # Process results
        enriched_results = []
        for result in vector_results:
            # --- REMOVED TEMPORARY DEBUG LOGGING --- 

            # Use the details fetched by _get_segments_content
            if result.segment_id not in segment_details:
                logger.warning(f"Details not found in DB for segment {result.segment_id}. Skipping enrichment.")
                continue
                
            details = segment_details[result.segment_id]
            segment_text_content = details["segment_text_content"]
            section_content = details["section_content"]
            
            # Apply highlighting if requested - highlight the original segment text
            highlighted_segment_text = self._highlight_text(segment_text_content, query) if highlight else segment_text_content
            
            # Create enriched result using fetched details
            enriched_result = QueryResult(
                id=result.segment_id,
                law_id=details.get("law_id", result.law_id), # Prefer fetched, fallback to payload
                law_abbreviation=details.get("law_abbreviation", ""), # Populate abbreviation
                section_number=details.get("section_number", ""),
                title=details.get("title", ""),
                content=segment_text_content, # Show the specific segment's text
                content_with_highlights=highlighted_segment_text, # Show highlighted segment text
                relevance_score=result.score,
                metadata=result.metadata # Keep original metadata from Qdrant payload
            )
            
            enriched_results.append(enriched_result)
            
        return enriched_results
    
    def _get_segments_content(self, search_results: list[SearchResult]) -> dict[str, dict[str, Any]]:
        """
        Retrieve full content and metadata for segments from database.
        Uses the section_id from the search result payload for accurate lookup.
        
        Args:
            search_results: List of SearchResult objects from Qdrant
            
        Returns:
            Dictionary mapping segment IDs to content and metadata
        """
        if not search_results:
            return {}
            
        conn = self.db_connection.get_connection()
        segments_data = {}
        
        # Extract section_ids directly from the payload for accurate lookup
        # Also map segment_id to its corresponding section_id
        section_ids_to_lookup = set()
        segment_to_section_map = {}
        for res in search_results:
            # Use the section_id stored in the payload during indexing
            payload_section_id = res.metadata.get("section_id") 
            if payload_section_id:
                section_ids_to_lookup.add(payload_section_id)
                segment_to_section_map[res.segment_id] = payload_section_id
            else:
                 # Fallback: Try parsing segment_id if section_id is missing in payload (shouldn't happen ideally)
                 logger.warning(f"section_id missing in payload for segment {res.segment_id}. Attempting parse.")
                 parts = res.segment_id.split('_')
                 if len(parts) > 1:
                     parsed_section_id = parts[0]
                     section_ids_to_lookup.add(parsed_section_id)
                     segment_to_section_map[res.segment_id] = parsed_section_id
                 else:
                     logger.error(f"Cannot determine section_id for segment {res.segment_id}. Cannot enrich.")
                     continue # Cannot look up this segment

        if not section_ids_to_lookup:
             logger.warning("Could not extract any section IDs from search results.")
             return {}

        # Query to get section details based on extracted section_ids
        placeholders = ",".join(["?" for _ in section_ids_to_lookup])
        query = f"""
        SELECT 
            sec.id AS section_id,
            sec.content AS section_content,
            sec.title AS title,
            sec.section_number,
            l.id AS law_id,
            l.abbreviation AS law_abbreviation,
            l.full_name AS law_name
        FROM 
            sections sec
        JOIN 
            laws l ON sec.law_id = l.id
        WHERE 
            sec.id IN ({placeholders})
        """
        
        try:
            # Execute query using parameters
            cursor = conn.execute(query, list(section_ids_to_lookup))
            columns = [desc[0] for desc in cursor.description]
            result_tuples = cursor.fetchall()
            
            # Convert tuples to dictionaries and map by section_id
            # Find the index of the 'section_id' column
            try:
                section_id_index = columns.index('section_id')
            except ValueError:
                 logger.error("Could not find 'section_id' column in enrichment query results.")
                 # Handle error: return empty or partially filled results
                 return {seg_id: {"segment_text_content": "Error: DB column mapping failed."} for seg_id in segment_to_section_map.keys()}

            section_details = {row[section_id_index]: dict(zip(columns, row)) for row in result_tuples}

            # Combine section details with segment-specific info from Qdrant payload
            for res in search_results:
                 segment_id = res.segment_id
                 lookup_section_id = segment_to_section_map.get(segment_id)
                 
                 if lookup_section_id and lookup_section_id in section_details:
                     sec_detail = section_details[lookup_section_id]
                     segment_meta = res.metadata # Metadata from Qdrant payload
                     
                     # Extract segment text using start/end indices from Qdrant payload metadata
                     section_content = sec_detail["section_content"]
                     start_idx = segment_meta.get("start_idx", 0)
                     end_idx = segment_meta.get("end_idx", len(section_content))
                     segment_text_content = section_content[start_idx:end_idx]

                     segments_data[segment_id] = {
                        "segment_text_content": segment_text_content, # Use the specific segment text
                        "section_content": section_content,         # Full section content
                        "title": sec_detail["title"],
                        "section_number": sec_detail["section_number"],
                        "law_id": sec_detail["law_id"],
                        "law_abbreviation": sec_detail["law_abbreviation"],
                        "law_name": sec_detail["law_name"],
                        "original_qdrant_metadata": segment_meta # Keep original payload if needed
                     }
                 else:
                    # Log if lookup failed for a segment
                    if lookup_section_id:
                         logger.warning(f"Section details not found in DB for section_id {lookup_section_id} (derived from segment {segment_id})")
                    # else: handled above
                    segments_data[segment_id] = { # Provide empty default
                         "segment_text_content": "Error: Content lookup failed.",
                         "section_content": "", "title": "", "section_number": "", "law_id": "",
                         "law_abbreviation": "", "law_name": "", "original_qdrant_metadata": res.metadata
                    }

        except Exception as e:
            logger.error(f"Error retrieving segment content: {e}", exc_info=True)
            # Provide empty defaults for all requested segments on general error
            for res in search_results:
                 segments_data[res.segment_id] = {
                      "segment_text_content": "Error: Content lookup failed.",
                      "section_content": "", "title": "", "section_number": "", "law_id": "",
                      "law_abbreviation": "", "law_name": "", "original_qdrant_metadata": res.metadata
                 }
            
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
    
    def _estimate_total_results(self, query: str, filters: dict[str, Any]) -> int:
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
        filters: dict[str, Any] | None, 
        page: int, 
        limit: int,
        group_by_article: bool = False
    ) -> str:
        """
        Generate a cache key for a search query.
        
        Args:
            query: The search query
            filters: Applied filters
            page: Page number
            limit: Results per page
            group_by_article: Whether results are grouped by article
            
        Returns:
            String cache key
        """
        # Create a deterministic string from the parameters
        filter_str = ""
        if filters:
            # Sort the keys for deterministic ordering
            for key in sorted(filters.keys()):
                filter_str += f"_{key}_{filters[key]}"
        
        # Add group_by_article parameter to cache key
        article_str = "_art" if group_by_article else ""
                
        return f"{query}{filter_str}_p{page}_l{limit}{article_str}"
    
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