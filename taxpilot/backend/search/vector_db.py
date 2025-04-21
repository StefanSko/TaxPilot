"""
Vector database integration for efficient semantic search.

This module provides integration with Qdrant vector database for storing
and retrieving vector embeddings, with full metadata support and efficient
search capabilities.
"""

import os
import json
import time
import logging
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, cast
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.models import (
    CollectionInfo, 
    PayloadSchemaType, 
    PayloadIndexInfo,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    Payload,
    UpdateResult,
    Distance as QdrantDistance
)

from taxpilot.backend.search.embeddings import TextEmbedding, EmbeddingModelType
from taxpilot.backend.data_processing.database import DbConfig, get_connection


# Custom Exceptions
class VectorDBError(Exception):
    """Base class for exceptions related to vector database operations."""
    pass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("vector_db.log"), logging.StreamHandler()],
)
logger = logging.getLogger("vector_db")


class VectorDbProvider(Enum):
    """Supported vector database providers."""
    QDRANT = "qdrant"
    MEMORY = "memory"  # For testing and development


@dataclass
class VectorDbConfig:
    """Configuration for vector database connection."""
    
    provider: VectorDbProvider = VectorDbProvider.QDRANT
    collection_name: str = "law_sections"
    embedding_dim: int = 768
    distance_metric: str = "cosine"
    vectors_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key: str | None = os.environ.get("QDRANT_API_KEY", None)
    local_path: Path | None = None  # For local persistent storage
    metadata_schema: dict[str, Any] = field(default_factory=dict)
    use_batching: bool = True
    batch_size: int = 100
    optimize_interval: int = 10000  # Points after which to optimize
    timeout: float = 60.0  # Operation timeout in seconds
    db_config: DbConfig | None = None
    store_hierarchical_data: bool = True  # Whether to store hierarchical information


@dataclass
class SearchResult:
    """Result from a vector database search."""
    segment_id: str
    law_id: str
    section_id: str
    score: float
    metadata: dict[str, Any]
    embedding_model: str
    embedding_version: str
    article_id: str = ""  # ID of the article this segment belongs to
    hierarchy_path: str = ""  # Full hierarchical path
    segment_type: str = ""  # Type of segment (article, section, paragraph, etc.)
    position_in_parent: int = 0  # Position within parent
    
    @property
    def combined_score(self) -> float:
        """
        Calculate a combined score based on vector similarity and other factors.
        
        This can be extended to include other ranking signals like date freshness,
        section importance, etc.
        """
        return self.score
    
    @property
    def article_scope(self) -> str:
        """
        Get the article scope (law and article ID) for grouping.
        """
        return f"{self.law_id}:{self.article_id}" if self.article_id else f"{self.law_id}:{self.section_id}"


@dataclass
class SearchParameters:
    """Parameters for vector search."""
    
    query_text: str = ""  # Textual query to search for
    query_vector: np.ndarray | None = None  # Vector to search for (if pre-computed)
    law_id: str | None = None  # Filter by law ID
    section_id: str | None = None  # Filter by section ID
    article_id: str | None = None  # Filter by article ID
    top_k: int = 10  # Number of results to return
    min_score: float = 0.7  # Minimum similarity score to include
    embedding_model: str = EmbeddingModelType.DEFAULT.value  # Model to use for embedding
    embedding_version: str | None = None  # Specific version to filter by
    offset: int = 0  # For pagination
    include_metadata: bool = True  # Whether to include metadata in results
    group_by_article: bool = False  # Whether to group results by article
    max_segments_per_article: int = 3  # Maximum number of segments to return per article


class VectorDatabase:
    """Vector database for storing and retrieving embeddings."""
    
    def __init__(self, config: VectorDbConfig | None = None):
        """
        Initialize the vector database connection.
        
        Args:
            config: Optional configuration for the vector database
        """
        self.config = config or VectorDbConfig()
        self.client: Any = self._initialize_client()
        
        # Ensure DB config is set
        if self.config.db_config is None:
            self.config.db_config = DbConfig()
        
        # Initialize the collection if it doesn't exist
        self._initialize_collection()
    
    def _initialize_client(self) -> Any:
        """
        Initialize the vector database client based on the provider.
        
        Returns:
            The initialized client
        """
        if self.config.provider == VectorDbProvider.QDRANT:
            if self.config.local_path:
                # Use local persistent storage
                self.config.local_path.mkdir(parents=True, exist_ok=True)
                return QdrantClient(
                    path=str(self.config.local_path),
                    timeout=self.config.timeout
                )
            else:
                # Use remote Qdrant instance
                return QdrantClient(
                    url=self.config.vectors_url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout
                )
        elif self.config.provider == VectorDbProvider.MEMORY:
            # Memory vector store for testing
            return QdrantClient(":memory:")
        else:
            # Raise custom exception
            raise VectorDBError(f"Unsupported provider: {self.config.provider}")
    
    def _initialize_collection(self) -> None:
        """Initialize the vector collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.config.collection_name not in collection_names:
                logger.info(f"Creating collection {self.config.collection_name}")
                
                # Define the metadata schema
                # Default schema for law metadata if not provided
                if not self.config.metadata_schema:
                    self.config.metadata_schema = {
                        "law_id": PayloadSchemaType.KEYWORD,
                        "section_id": PayloadSchemaType.KEYWORD,
                        "segment_id": PayloadSchemaType.KEYWORD,
                        "embedding_model": PayloadSchemaType.KEYWORD,
                        "embedding_version": PayloadSchemaType.KEYWORD,
                        "date_created": PayloadSchemaType.DATETIME,
                        "text_length": PayloadSchemaType.INTEGER
                    }
                    
                    # Add hierarchical fields if enabled
                    if self.config.store_hierarchical_data:
                        hierarchical_fields = {
                            "article_id": PayloadSchemaType.KEYWORD,
                            "hierarchy_path": PayloadSchemaType.KEYWORD,
                            "segment_type": PayloadSchemaType.KEYWORD,
                            "position_in_parent": PayloadSchemaType.INTEGER
                        }
                        self.config.metadata_schema.update(hierarchical_fields)
                
                # Create the collection
                distance = Distance.COSINE
                if self.config.distance_metric.lower() == "dot":
                    distance = Distance.DOT
                elif self.config.distance_metric.lower() == "euclidean":
                    distance = Distance.EUCLID
                
                # Create collection with basic configuration (without metadata_schema)
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dim,
                        distance=distance
                    ),
                    on_disk_payload=True  # Store payload on disk for large collections
                )
                
                # Create indexes separately after collection creation
                self._create_indexes()
                
                logger.info(
                    f"Collection {self.config.collection_name} created with "
                    f"{self.config.embedding_dim} dimensions and {self.config.distance_metric} metric"
                )
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            # Raise custom exception
            raise VectorDBError(f"Failed to initialize Qdrant collection: {str(e)}") from e
    
    def _create_indexes(self) -> None:
        """Create indexes for efficient filtering."""
        # Key fields to create indexes for
        key_fields = ["law_id", "section_id", "embedding_model"]
        
        # Add hierarchical fields if enabled
        if self.config.store_hierarchical_data:
            key_fields.extend(["article_id", "hierarchy_path", "segment_type"])
        
        for field in key_fields:
            try:
                # Determine schema type based on field name
                field_schema = PayloadSchemaType.KEYWORD
                if field == "position_in_parent":
                    field_schema = PayloadSchemaType.INTEGER
                
                self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field,
                    field_schema=field_schema
                )
                logger.info(f"Created index on {field}")
            except Exception as e:
                logger.warning(f"Error creating index on {field}: {str(e)}")
                # Optionally raise if index creation is critical
                # raise VectorDBError(f"Failed to create index on {field}: {str(e)}\") from e
    
    def store_embedding(self, embedding: TextEmbedding) -> str:
        """
        Store a single embedding in the vector database.
        
        Args:
            embedding: The embedding to store
            
        Returns:
            The ID of the stored embedding
        """
        # Generate a unique ID for the embedding
        point_id = str(uuid.uuid4())
        
        # Prepare metadata
        metadata = {
            "law_id": embedding.law_id,
            "section_id": embedding.section_id, 
            "segment_id": embedding.segment_id,
            "embedding_model": embedding.embedding_model,
            "embedding_version": embedding.embedding_version,
            "date_created": time.time(),
            **embedding.metadata
        }
        
        # Add hierarchical data if present in metadata and enabled in config
        if self.config.store_hierarchical_data and embedding.metadata:
            if "article_id" in embedding.metadata:
                metadata["article_id"] = embedding.metadata["article_id"]
            if "hierarchy_path" in embedding.metadata:
                metadata["hierarchy_path"] = embedding.metadata["hierarchy_path"]
            if "segment_type" in embedding.metadata:
                metadata["segment_type"] = embedding.metadata["segment_type"]
            if "position_in_parent" in embedding.metadata:
                metadata["position_in_parent"] = embedding.metadata["position_in_parent"]
        
        # Upsert the embedding
        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding.vector.tolist(),
                        payload=metadata
                    )
                ]
            )
            logger.debug(f"Stored embedding with ID {point_id}")
            
            # No longer need to update DuckDB reference here, it's handled during embedding storage
            # self._update_duckdb_reference(embedding, point_id)
            
            return point_id
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            # Raise custom exception
            raise VectorDBError(f"Failed to store embedding {embedding.segment_id}: {str(e)}") from e
    
    def store_embeddings_batch(self, embeddings: list[TextEmbedding]) -> list[str]:
        """
        Store multiple embeddings in the vector database efficiently.
        
        Args:
            embeddings: List of embeddings to store
            
        Returns:
            List of IDs for the stored embeddings
        """
        if not embeddings:
            return []
        
        # Check if we should use batching
        if not self.config.use_batching or len(embeddings) <= 1:
            # Store each embedding individually
            return [self.store_embedding(embedding) for embedding in embeddings]
        
        point_ids = []
        points = []
        
        # Prepare all points
        for embedding in embeddings:
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            metadata = {
                "law_id": embedding.law_id,
                "section_id": embedding.section_id,
                "segment_id": embedding.segment_id,
                "embedding_model": embedding.embedding_model,
                "embedding_version": embedding.embedding_version,
                "date_created": time.time(),
                **embedding.metadata
            }
            
            # Add hierarchical data if present in metadata and enabled in config
            if self.config.store_hierarchical_data and embedding.metadata:
                if "article_id" in embedding.metadata:
                    metadata["article_id"] = embedding.metadata["article_id"]
                if "hierarchy_path" in embedding.metadata:
                    metadata["hierarchy_path"] = embedding.metadata["hierarchy_path"]
                if "segment_type" in embedding.metadata:
                    metadata["segment_type"] = embedding.metadata["segment_type"]
                if "position_in_parent" in embedding.metadata:
                    metadata["position_in_parent"] = embedding.metadata["position_in_parent"]
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.vector.tolist(),
                    payload=metadata
                )
            )
        
        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_ids = point_ids[i:i + batch_size]
            
            try:
                self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=batch,
                    wait=True # Wait for operation to complete for accuracy
                )
                logger.info(f"Stored batch of {len(batch)} embeddings")
                
                # Update DuckDB references for this batch
                # This update is also redundant now
                # for j, point_id in enumerate(batch_ids):
                #     self._update_duckdb_reference(embeddings[i + j], point_id)
                
            except Exception as e:
                logger.error(f"Error storing batch {i//batch_size}: {str(e)}")
                # Raise custom exception or handle partial failure
                raise VectorDBError(f"Failed to store embedding batch: {str(e)}") from e
                # Try to store each embedding individually
                for j, point in enumerate(batch):
                    try:
                        self.client.upsert(
                            collection_name=self.config.collection_name,
                            points=[point]
                        )
                        # No need to update reference here either
                        # self._update_duckdb_reference(embeddings[i + j], batch_ids[j])
                    except Exception as inner_e:
                        logger.error(f"Error storing individual point: {str(inner_e)}")
                        point_ids[i + j] = ""  # Mark as failed
                        # Optionally, collect errors instead of raising immediately
                        # raise VectorDBError(f"Failed to store individual embedding {embeddings[i+j].segment_id}: {str(inner_e)}\") from inner_e
        
        return [pid for pid in point_ids if pid]
    
    def search(self, params: SearchParameters) -> list[SearchResult]:
        """
        Search for similar vectors in the database.
        
        Args:
            params: Search parameters
            
        Returns:
            List of search results with similarity scores
        """
        try:
            # Prepare filters
            filter_dict = None
            conditions = []
            
            if params.law_id:
                conditions.append(
                    FieldCondition(
                        key="law_id",
                        match=MatchValue(value=params.law_id)
                    )
                )
                
            if params.section_id:
                conditions.append(
                    FieldCondition(
                        key="section_id",
                        match=MatchValue(value=params.section_id)
                    )
                )
                
            # Add filter for article_id if provided and we're using hierarchical data
            if params.article_id and self.config.store_hierarchical_data:
                conditions.append(
                    FieldCondition(
                        key="article_id",
                        match=MatchValue(value=params.article_id)
                    )
                )
                
            if params.embedding_model:
                conditions.append(
                    FieldCondition(
                        key="embedding_model",
                        match=MatchValue(value=params.embedding_model)
                    )
                )
                
            if params.embedding_version:
                conditions.append(
                    FieldCondition(
                        key="embedding_version",
                        match=MatchValue(value=params.embedding_version)
                    )
                )
            
            if conditions:
                filter_dict = Filter(must=conditions)
            
            # Construct basic search parameters
            search_kwargs = {
                "collection_name": self.config.collection_name,
                "query_vector": params.query_vector.tolist() if params.query_vector is not None else None,
                "limit": params.top_k,
                "with_payload": params.include_metadata,
                "score_threshold": params.min_score,
                "offset": params.offset,
            }
            
            # Handle filters differently based on local/memory mode or server mode
            if filter_dict is not None:
                if self.config.provider == VectorDbProvider.MEMORY or self.config.local_path is not None:
                    # For local/memory mode, we perform the search without filter first
                    unfiltered_result = self.client.search(**search_kwargs)
                    
                    # Then manually filter the results
                    filtered_result = []
                    for hit in unfiltered_result:
                        if hit.payload:
                            # Check all filter conditions
                            matches_all = True
                            for condition in filter_dict.must:
                                if isinstance(condition, FieldCondition) and isinstance(condition.match, MatchValue):
                                    field_value = hit.payload.get(condition.key)
                                    if field_value != condition.match.value:
                                        matches_all = False
                                        break
                            
                            if matches_all:
                                filtered_result.append(hit)
                    
                    search_result = filtered_result
                else:
                    # For server mode, use the filter directly
                    search_kwargs["filter"] = filter_dict
                    search_result = self.client.search(**search_kwargs)
            else:
                # No filter, perform normal search
                search_result = self.client.search(**search_kwargs)
            
            # Convert to our result format
            results = []
            for hit in search_result:
                if hit.payload:
                    # Extract hierarchical information if available
                    article_id = hit.payload.get("article_id", "")
                    hierarchy_path = hit.payload.get("hierarchy_path", "")
                    segment_type = hit.payload.get("segment_type", "")
                    position_in_parent = hit.payload.get("position_in_parent", 0)
                    
                    # Construct the result
                    result = SearchResult(
                        segment_id=hit.payload.get("segment_id", ""),
                        law_id=hit.payload.get("law_id", ""),
                        section_id=hit.payload.get("section_id", ""),
                        score=float(hit.score) if hit.score is not None else 0.0,
                        metadata={
                            k: v for k, v in hit.payload.items()
                            if k not in [
                                "law_id", "section_id", "segment_id", 
                                "embedding_model", "embedding_version",
                                "article_id", "hierarchy_path", "segment_type", "position_in_parent"
                            ]
                        },
                        embedding_model=hit.payload.get("embedding_model", ""),
                        embedding_version=hit.payload.get("embedding_version", ""),
                        article_id=article_id,
                        hierarchy_path=hierarchy_path,
                        segment_type=segment_type,
                        position_in_parent=position_in_parent
                    )
                    results.append(result)
            
            # Rank results by combined score
            results.sort(key=lambda x: x.combined_score, reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            # Raise custom exception
            raise VectorDBError(f"Search operation failed: {str(e)}") from e
    
    def get_by_segment_id(self, segment_id: str, embedding_model: str | None = None) -> list[SearchResult]:
        """
        Get embeddings for a specific segment.
        
        Args:
            segment_id: The segment ID to retrieve
            embedding_model: Optional model name to filter by
            
        Returns:
            List of search results matching the segment ID
        """
        try:
            # In memory mode, we need to scroll through all points
            # and filter manually since filter param might not be supported
            if self.config.provider == VectorDbProvider.MEMORY or self.config.local_path is not None:
                scroll_results = self.client.scroll(
                    collection_name=self.config.collection_name,
                    with_payload=True,
                    limit=100
                )
                
                # Points from scroll_results
                all_points = scroll_results[0]
                
                # Filter manually
                filtered_points = []
                for point in all_points:
                    if point.payload and point.payload.get("segment_id") == segment_id:
                        if embedding_model is None or point.payload.get("embedding_model") == embedding_model:
                            filtered_points.append(point)
                            
                # Replace scroll_results with filtered version
                scroll_results = (filtered_points, None)
            else:
                # Prepare filter for server mode
                conditions = [
                    FieldCondition(
                        key="segment_id",
                        match=MatchValue(value=segment_id)
                    )
                ]
                
                if embedding_model:
                    conditions.append(
                        FieldCondition(
                            key="embedding_model",
                            match=MatchValue(value=embedding_model)
                        )
                    )
                
                filter_dict = Filter(must=conditions)
                
                # Perform the search with filter
                scroll_results = self.client.scroll(
                    collection_name=self.config.collection_name,
                    filter=filter_dict,
                    with_payload=True,
                    limit=100
                )
            
            points = scroll_results[0]
            
            # Convert to our result format
            results = []
            for point in points:
                if point.payload:
                    # Extract hierarchical information if available
                    article_id = point.payload.get("article_id", "")
                    hierarchy_path = point.payload.get("hierarchy_path", "")
                    segment_type = point.payload.get("segment_type", "")
                    position_in_parent = point.payload.get("position_in_parent", 0)
                    
                    result = SearchResult(
                        segment_id=point.payload.get("segment_id", ""),
                        law_id=point.payload.get("law_id", ""),
                        section_id=point.payload.get("section_id", ""),
                        score=1.0,  # Not a search result, so use perfect score
                        metadata={
                            k: v for k, v in point.payload.items()
                            if k not in [
                                "law_id", "section_id", "segment_id", 
                                "embedding_model", "embedding_version",
                                "article_id", "hierarchy_path", "segment_type", "position_in_parent"
                            ]
                        },
                        embedding_model=point.payload.get("embedding_model", ""),
                        embedding_version=point.payload.get("embedding_version", ""),
                        article_id=article_id,
                        hierarchy_path=hierarchy_path,
                        segment_type=segment_type,
                        position_in_parent=position_in_parent
                    )
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving embeddings for segment {segment_id}: {str(e)}")
            # Raise custom exception
            raise VectorDBError(f"Failed to retrieve embeddings for segment {segment_id}: {str(e)}") from e
    
    def delete_by_law_id(self, law_id: str) -> int:
        """
        Delete all embeddings for a specific law.
        
        Args:
            law_id: ID of the law to delete
            
        Returns:
            Number of points deleted
        """
        try:
            # Prepare filter
            filter_dict = Filter(
                must=[
                    FieldCondition(
                        key="law_id",
                        match=MatchValue(value=law_id)
                    )
                ]
            )
            
            # Delete the points
            result = self.client.delete(
                collection_name=self.config.collection_name,
                points_filter=filter_dict
            )
            
            if hasattr(result, "deleted"):
                deleted_count = result.deleted
                logger.info(f"Deleted {deleted_count} embeddings for law {law_id}")
                return cast(int, deleted_count)
            else:
                logger.warning(f"Deletion for law {law_id} completed but count unknown")
                return 0
        except Exception as e:
            logger.error(f"Error deleting embeddings for law {law_id}: {str(e)}")
            # Raise custom exception
            raise VectorDBError(f"Failed to delete embeddings for law {law_id}: {str(e)}") from e
    
    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the vector collection.
        
        Returns:
            Dictionary of collection statistics
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.config.collection_name)
            
            # Get counts by model and law
            model_counts = {}
            law_counts = {}
            
            # Due to potential size, we'll use scrolling to get counts
            scroll_results = self.client.scroll(
                collection_name=self.config.collection_name,
                with_payload=True,
                limit=1000
            )
            
            total_points = 0
            
            # Process all points in batches
            while scroll_results[0]:
                points, next_offset = scroll_results
                
                for point in points:
                    total_points += 1
                    
                    if point.payload:
                        # Count by model
                        model = point.payload.get("embedding_model", "unknown")
                        model_counts[model] = model_counts.get(model, 0) + 1
                        
                        # Count by law
                        law_id = point.payload.get("law_id", "unknown")
                        law_counts[law_id] = law_counts.get(law_id, 0) + 1
                
                if next_offset is None:
                    break
                    
                # Get next batch
                scroll_results = self.client.scroll(
                    collection_name=self.config.collection_name,
                    with_payload=True,
                    limit=1000,
                    offset=next_offset
                )
            
            # Compile statistics
            stats = {
                "collection_name": self.config.collection_name,
                "vector_size": self.config.embedding_dim,
                "distance_metric": self.config.distance_metric,
                "total_vectors": total_points,
                "vectors_by_model": model_counts,
                "vectors_by_law": law_counts,
                "optimization_status": "unknown"
            }
            
            # Try to get optimization status if available
            if collection_info and hasattr(collection_info, "optimization"):
                stats["optimization_status"] = collection_info.optimization
            
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            # Raise custom exception instead of returning error dict
            raise VectorDBError(f"Failed to get collection stats: {str(e)}") from e
    
    def delete_collection(self) -> bool:
        """Delete the entire vector collection."""
        try:
            logger.warning(f"Deleting Qdrant collection: {self.config.collection_name}")
            self.client.delete_collection(collection_name=self.config.collection_name)
            logger.info(f"Collection {self.config.collection_name} deleted successfully.")
            # Re-initialize to ensure it's recreated cleanly on next use if needed
            self._initialize_collection()
            return True
        except UnexpectedResponse as e:
            # Qdrant might return 404 if collection doesn't exist, which is fine
            if e.status_code == 404:
                logger.info(f"Collection {self.config.collection_name} did not exist, nothing to delete.")
                # Still ensure it's initialized for subsequent operations
                self._initialize_collection()
                return True
            else:
                logger.error(f"Error deleting collection {self.config.collection_name}: {str(e)} ({e.status_code})")
                # Raise custom exception
                raise VectorDBError(f"Failed to delete collection {self.config.collection_name}: {str(e)}") from e
                # return False
        except Exception as e:
            logger.error(f"Error deleting collection {self.config.collection_name}: {str(e)}")
            # Raise custom exception
            raise VectorDBError(f"Failed to delete collection {self.config.collection_name}: {str(e)}") from e
            # return False
    
    def optimize_collection(self) -> bool:
        """
        Optimize the vector collection for search performance.
        
        Returns:
            True if optimization was successful
        """
        try:
            # Check if the provider supports optimization
            if self.config.provider == VectorDbProvider.QDRANT:
                # Trigger optimization
                self.client.update_collection(
                    collection_name=self.config.collection_name,
                    optimizer_config={
                        "indexing_threshold": 20000  # Adjust based on your needs
                    }
                )
                logger.info(f"Triggered optimization for collection {self.config.collection_name}")
                return True
            else:
                logger.warning(f"Optimization not supported for provider {self.config.provider}")
                return False
        except Exception as e:
            logger.error(f"Error optimizing collection: {str(e)}")
            # Raise custom exception
            raise VectorDBError(f"Failed to optimize collection: {str(e)}") from e
            # return False
    
    def backup_collection(self, backup_path: str) -> bool:
        """
        Backup the vector collection to a file.
        
        Args:
            backup_path: Path to save the backup
            
        Returns:
            True if backup was successful
        """
        try:
            # Check if the provider supports backup
            if self.config.provider == VectorDbProvider.QDRANT:
                Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Snapshot the collection (only works with persistent storage)
                if self.config.local_path:
                    # For local storage, create a snapshot
                    try:
                        self.client.create_snapshot(
                            collection_name=self.config.collection_name,
                            wait=True
                        )
                        logger.info(f"Created snapshot of collection {self.config.collection_name}")
                        return True
                    except Exception as inner_e:
                        logger.error(f"Error creating snapshot: {str(inner_e)}")
                        return False
                else:
                    # For remote storage, get all vectors and save locally
                    logger.warning("Remote backup not directly supported, consider using Qdrant's built-in backup")
                    return False
            else:
                logger.warning(f"Backup not supported for provider {self.config.provider}")
                return False
        except Exception as e:
            logger.error(f"Error backing up collection: {str(e)}")
            # Raise custom exception
            raise VectorDBError(f"Failed to backup collection: {str(e)}") from e
            # return False
    
    def restore_collection(self, backup_path: str) -> bool:
        """
        Restore the vector collection from a backup.
        
        Args:
            backup_path: Path to the backup
            
        Returns:
            True if restore was successful
        """
        try:
            # Restoring is very provider-specific and complex
            # Here we'll implement a simple version for Qdrant local
            logger.warning("Collection restore is a complex operation, consider using Qdrant's built-in restore")
            return False
        except Exception as e:
            logger.error(f"Error restoring collection: {str(e)}")
            # Raise custom exception
            raise VectorDBError(f"Failed to restore collection: {str(e)}") from e
            # return False
    
    def sync_with_duckdb(self, force_repopulate: bool = False) -> dict[str, Any]:
        """
        Synchronize the vector database with DuckDB.
 
        Ensures that embeddings in DuckDB are represented in Qdrant.
        
        Args:
            force_repopulate: If True, ignore existing vector_db_ids in DuckDB 
                              and re-insert all embeddings into Qdrant, updating 
                              the vector_db_id in DuckDB afterwards. 
                              If False (default), only insert embeddings from 
                              DuckDB that have a null vector_db_id.
        
        Returns:
            Statistics about the synchronization
        """
        if not self.config.db_config:
            logger.error("Cannot sync: No DuckDB configuration provided")
            return {"error": "No DuckDB configuration provided"}
        
        conn = get_connection(self.config.db_config)
        stats = {
            "embeddings_checked_in_duckdb": 0,
            "embeddings_to_process": 0,
            "embeddings_inserted_or_updated_in_qdrant": 0,
            "duckdb_ids_updated": 0,
            "errors": 0
        }
        points_to_insert: list[PointStruct] = []
        duckdb_qdrant_id_map: dict[str, str] = {} # {duckdb_id: new_qdrant_id}
        
        try:
            # Get all embeddings from DuckDB
            # Make sure to fetch the primary key 'id' from section_embeddings
            cursor = conn.execute(
                """
                SELECT id, law_id, section_id, segment_id, 
                       embedding_model, embedding_version, 
                       embedding, metadata, vector_db_id
                FROM section_embeddings
                """
            )
            columns = [desc[0] for desc in cursor.description]
            embeddings_in_duckdb = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            stats["embeddings_checked_in_duckdb"] = len(embeddings_in_duckdb)
            logger.info(f"Found {stats['embeddings_checked_in_duckdb']} embeddings in DuckDB.")
            if force_repopulate:
                logger.info("Force repopulate enabled: All embeddings will be re-inserted into Qdrant.")
            
            # Process each embedding from DuckDB
            for emb_data in embeddings_in_duckdb:
                try:
                    duckdb_id = emb_data["id"]
                    vector_db_id_exists = emb_data.get("vector_db_id") is not None
                    
                    # Skip if not forcing repopulation and ID already exists
                    if not force_repopulate and vector_db_id_exists:
                        continue
                    
                    stats["embeddings_to_process"] += 1
                    
                    # Generate a new Qdrant Point ID
                    new_qdrant_point_id = str(uuid.uuid4())
                    
                    # Prepare metadata payload for Qdrant
                    metadata_payload = {
                        "law_id": emb_data["law_id"],
                        "section_id": emb_data["section_id"],
                        "segment_id": emb_data["segment_id"],
                        "embedding_model": emb_data["embedding_model"],
                        "embedding_version": emb_data["embedding_version"],
                        "date_created": time.time(),
                        **(json.loads(emb_data["metadata"]) if emb_data["metadata"] else {})
                    }
                    
                    # Create the PointStruct
                    vector_list = emb_data["embedding"] # Already a list from DuckDB array
                    point = PointStruct(
                        id=new_qdrant_point_id,
                        vector=vector_list,
                        payload=metadata_payload
                    )
                    points_to_insert.append(point)
                    
                    # Store mapping to update DuckDB later
                    duckdb_qdrant_id_map[duckdb_id] = new_qdrant_point_id
                        
                except Exception as e:
                    logger.error(f"Error preparing embedding {emb_data.get('id', 'unknown')} for sync: {str(e)}")
                    stats["errors"] += 1
            
            # Insert points into Qdrant in batches
            logger.info(f"Processed {stats['embeddings_to_process']} embeddings for Qdrant insertion.")
            if points_to_insert:
                batch_size = self.config.batch_size
                total_inserted = 0
                for i in range(0, len(points_to_insert), batch_size):
                    batch = points_to_insert[i:i + batch_size]
                    try:
                        self.client.upsert(
                            collection_name=self.config.collection_name,
                            points=batch,
                            wait=True # Wait for operation to complete for accuracy
                        )
                        total_inserted += len(batch)
                        logger.debug(f"Inserted batch of {len(batch)} points into Qdrant.")
                    except Exception as e:
                        logger.error(f"Error inserting batch into Qdrant: {str(e)}")
                        stats["errors"] += 1 # Count batch insertion error
                        # Optionally, could try individual inserts here as fallback
                        # Raise custom exception for sync failure
                        raise VectorDBError(f"Failed during Qdrant upsert in sync: {str(e)}") from e
                
                stats["embeddings_inserted_or_updated_in_qdrant"] = total_inserted
                logger.info(f"Successfully inserted/updated {total_inserted} points in Qdrant.")

                # Update DuckDB with the new Qdrant IDs if forcing repopulation or if initial IDs were null
                if duckdb_qdrant_id_map: # Only update if there are mappings
                    logger.info(f"Updating {len(duckdb_qdrant_id_map)} vector_db_id references in DuckDB...")
                    updated_rows = 0
                    with conn.cursor() as cur:
                        for duckdb_id, new_qdrant_id in duckdb_qdrant_id_map.items():
                            try:
                                cur.execute(
                                    "UPDATE section_embeddings SET vector_db_id = ? WHERE id = ?",
                                    (new_qdrant_id, duckdb_id)
                                )
                                updated_rows += cur.rowcount
                            except Exception as e:
                                logger.error(f"Error updating vector_db_id for DuckDB id {duckdb_id}: {str(e)}")
                                stats["errors"] += 1
                    stats["duckdb_ids_updated"] = updated_rows
                    logger.info(f"Updated {updated_rows} rows in DuckDB.")
            else:
                 logger.info("No new embeddings needed insertion into Qdrant.")

            return stats
        except Exception as e:
            # Catch broader sync errors and raise custom exception
            logger.error(f"General error during synchronization: {str(e)}", exc_info=True)
            stats["errors"] += 1
            # Raise custom exception
            raise VectorDBError(f"Synchronization with DuckDB failed: {str(e)}") from e
            # return {"error": str(e), **stats}
    
    def close(self) -> None:
        """Close the vector database connection."""
        # For some clients, explicit closing is needed
        if hasattr(self.client, "close"):
            self.client.close()


class VectorDatabaseManager:
    """Manager for vector database operations."""
    
    def __init__(self, config: VectorDbConfig | None = None):
        """
        Initialize the vector database manager.
        
        Args:
            config: Optional configuration for the vector database
        """
        self.config = config or VectorDbConfig()
        self.db = VectorDatabase(self.config)
    
    def initialize_db(self) -> bool:
        """
        Initialize or reinitialize the vector database.
        
        Returns:
            True if initialization was successful
        """
        try:
            # Close the current connection if open
            if hasattr(self, "db"):
                self.db.close()
            
            # Create a new connection with fresh initialization
            self.db = VectorDatabase(self.config)
            return True
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            return False
    
    def process_text_embeddings(self, embeddings: list[TextEmbedding]) -> list[str]:
        """
        Process and store text embeddings.
        
        Args:
            embeddings: List of embeddings to process
            
        Returns:
            List of vector database IDs for the stored embeddings
        """
        return self.db.store_embeddings_batch(embeddings)
    
    def search_similar(self, params: SearchParameters) -> list[SearchResult]:
        """
        Search for similar texts.
        
        Args:
            params: Search parameters
            
        Returns:
            List of search results
        """
        return self.db.search(params)
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary of statistics
        """
        return self.db.get_collection_stats()
    
    def optimize(self) -> bool:
        """
        Optimize the vector database for search performance.
        
        Returns:
            True if optimization was successful
        """
        return self.db.optimize_collection()
    
    def synchronize(self, force_repopulate: bool = False) -> dict[str, Any]:
        """
        Synchronize the vector database with DuckDB.
        
        Args:
            force_repopulate: If True, ignore existing vector_db_ids in DuckDB 
                              and re-insert all embeddings into Qdrant, updating 
                              the vector_db_id in DuckDB afterwards. 
                              If False (default), only insert embeddings from 
                              DuckDB that have a null vector_db_id.

        Returns:
            Statistics about the synchronization
        """
        return self.db.sync_with_duckdb(force_repopulate=force_repopulate)
    
    def close(self) -> None:
        """Close the vector database connection."""
        if hasattr(self, "db"):
            self.db.close()