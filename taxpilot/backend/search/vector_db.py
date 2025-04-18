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
    
    @property
    def combined_score(self) -> float:
        """
        Calculate a combined score based on vector similarity and other factors.
        
        This can be extended to include other ranking signals like date freshness,
        section importance, etc.
        """
        return self.score


@dataclass
class SearchParameters:
    """Parameters for vector search."""
    
    query_text: str = ""  # Textual query to search for
    query_vector: np.ndarray | None = None  # Vector to search for (if pre-computed)
    law_id: str | None = None  # Filter by law ID
    section_id: str | None = None  # Filter by section ID
    top_k: int = 10  # Number of results to return
    min_score: float = 0.7  # Minimum similarity score to include
    embedding_model: str = EmbeddingModelType.DEFAULT.value  # Model to use for embedding
    embedding_version: str | None = None  # Specific version to filter by
    offset: int = 0  # For pagination
    include_metadata: bool = True  # Whether to include metadata in results


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
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
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
                
                # Create the collection
                distance = Distance.COSINE
                if self.config.distance_metric.lower() == "dot":
                    distance = Distance.DOT
                elif self.config.distance_metric.lower() == "euclidean":
                    distance = Distance.EUCLID
                
                create_params = {
                    "collection_name": self.config.collection_name,
                    "vectors_config": VectorParams(
                        size=self.config.embedding_dim,
                        distance=distance
                    ),
                    "on_disk_payload": True,  # Store payload on disk for large collections
                }
                
                # Add metadata_schema only if not in local mode
                if self.config.local_path is None:
                    create_params["metadata_schema"] = self.config.metadata_schema
                    
                self.client.create_collection(**create_params)
                
                # Create index for efficient filtering
                self._create_indexes()
                
                logger.info(
                    f"Collection {self.config.collection_name} created with "
                    f"{self.config.embedding_dim} dimensions and {self.config.distance_metric} metric"
                )
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise
    
    def _create_indexes(self) -> None:
        """Create indexes for efficient filtering."""
        # Key fields to create indexes for
        key_fields = ["law_id", "section_id", "embedding_model"]
        
        for field in key_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )
                logger.info(f"Created index on {field}")
            except Exception as e:
                logger.warning(f"Error creating index on {field}: {str(e)}")
    
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
            
            # Update the DuckDB record with the vector database reference
            self._update_duckdb_reference(embedding, point_id)
            
            return point_id
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            raise
    
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
                    points=batch
                )
                logger.info(f"Stored batch of {len(batch)} embeddings")
                
                # Update DuckDB references for this batch
                for j, point_id in enumerate(batch_ids):
                    self._update_duckdb_reference(embeddings[i + j], point_id)
                
            except Exception as e:
                logger.error(f"Error storing batch {i//batch_size}: {str(e)}")
                # Try to store each embedding individually
                for j, point in enumerate(batch):
                    try:
                        self.client.upsert(
                            collection_name=self.config.collection_name,
                            points=[point]
                        )
                        self._update_duckdb_reference(embeddings[i + j], batch_ids[j])
                    except Exception as inner_e:
                        logger.error(f"Error storing individual point: {str(inner_e)}")
                        point_ids[i + j] = ""  # Mark as failed
        
        return [pid for pid in point_ids if pid]
    
    def _update_duckdb_reference(self, embedding: TextEmbedding, vector_db_id: str) -> None:
        """
        Update the DuckDB record with the vector database reference.
        
        Args:
            embedding: The embedding that was stored
            vector_db_id: The ID assigned in the vector database
        """
        if not self.config.db_config:
            return
        
        conn = get_connection(self.config.db_config)
        
        try:
            # Check if there's an existing record for this segment
            result = conn.execute(
                """
                SELECT id FROM section_embeddings
                WHERE segment_id = ? AND embedding_model = ? AND embedding_version = ?
                """,
                (embedding.segment_id, embedding.embedding_model, embedding.embedding_version),
            ).fetchone()
            
            if result:
                # Update the existing record
                conn.execute(
                    """
                    UPDATE section_embeddings
                    SET vector_db_id = ?
                    WHERE id = ?
                    """,
                    (vector_db_id, result[0])
                )
            else:
                logger.warning(
                    f"No embedding record found in DuckDB for segment {embedding.segment_id}"
                )
        except Exception as e:
            logger.error(f"Error updating DuckDB reference: {str(e)}")
    
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
            
            # Construct search parameters based on whether we're in local mode
            search_kwargs = {
                "collection_name": self.config.collection_name,
                "query_vector": params.query_vector.tolist() if params.query_vector is not None else None,
                "limit": params.top_k,
                "with_payload": params.include_metadata,
                "score_threshold": params.min_score,
                "offset": params.offset,
            }
            
            # Only add filter if we're not in local mode
            if filter_dict is not None and self.config.local_path is None:
                search_kwargs["filter"] = filter_dict
            
            # Perform the search
            search_result = self.client.search(**search_kwargs)
            
            # Convert to our result format
            results = []
            for hit in search_result:
                if hit.payload:
                    result = SearchResult(
                        segment_id=hit.payload.get("segment_id", ""),
                        law_id=hit.payload.get("law_id", ""),
                        section_id=hit.payload.get("section_id", ""),
                        score=float(hit.score) if hit.score is not None else 0.0,
                        metadata={
                            k: v for k, v in hit.payload.items()
                            if k not in ["law_id", "section_id", "segment_id", "embedding_model", "embedding_version"]
                        },
                        embedding_model=hit.payload.get("embedding_model", ""),
                        embedding_version=hit.payload.get("embedding_version", "")
                    )
                    results.append(result)
            
            # Rank results by combined score
            results.sort(key=lambda x: x.combined_score, reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise
    
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
            # Prepare filter
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
            
            # Perform the search
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
                    result = SearchResult(
                        segment_id=point.payload.get("segment_id", ""),
                        law_id=point.payload.get("law_id", ""),
                        section_id=point.payload.get("section_id", ""),
                        score=1.0,  # Not a search result, so use perfect score
                        metadata={
                            k: v for k, v in point.payload.items()
                            if k not in ["law_id", "section_id", "segment_id", "embedding_model", "embedding_version"]
                        },
                        embedding_model=point.payload.get("embedding_model", ""),
                        embedding_version=point.payload.get("embedding_version", "")
                    )
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving embeddings for segment {segment_id}: {str(e)}")
            raise
    
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
            raise
    
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
            return {"error": str(e)}
    
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
            return False
    
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
            return False
    
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
            return False
    
    def sync_with_duckdb(self) -> dict[str, Any]:
        """
        Synchronize the vector database with DuckDB.
        
        This ensures that all embeddings in DuckDB are also in
        the vector database with correct references.
        
        Returns:
            Statistics about the synchronization
        """
        if not self.config.db_config:
            return {"error": "No DuckDB configuration provided"}
        
        conn = get_connection(self.config.db_config)
        stats = {
            "embeddings_checked": 0,
            "embeddings_added": 0,
            "embeddings_already_synced": 0,
            "errors": 0
        }
        
        try:
            # Get all embeddings from DuckDB
            embeddings = conn.execute(
                """
                SELECT id, law_id, section_id, segment_id, 
                       embedding_model, embedding_version, 
                       embedding, metadata, vector_db_id
                FROM section_embeddings
                """
            ).fetchall()
            
            stats["embeddings_checked"] = len(embeddings)
            
            # Process each embedding
            for emb in embeddings:
                try:
                    emb_id, law_id, section_id, segment_id, model, version, embedding_bytes, metadata_str, vector_db_id = emb
                    
                    # Skip if already has a vector_db_id
                    if vector_db_id:
                        stats["embeddings_already_synced"] += 1
                        continue
                    
                    # Create TextEmbedding object
                    vector = np.frombuffer(embedding_bytes, dtype=np.float32)
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    
                    text_embedding = TextEmbedding(
                        vector=vector,
                        segment_id=segment_id,
                        law_id=law_id,
                        section_id=section_id,
                        metadata=metadata,
                        embedding_model=model,
                        embedding_version=version
                    )
                    
                    # Store in vector database
                    new_vector_id = self.store_embedding(text_embedding)
                    stats["embeddings_added"] += 1
                    
                    # Ensure reference is updated (normally done in store_embedding)
                    conn.execute(
                        """
                        UPDATE section_embeddings
                        SET vector_db_id = ?
                        WHERE id = ?
                        """,
                        (new_vector_id, emb_id)
                    )
                except Exception as e:
                    logger.error(f"Error processing embedding {emb[0]}: {str(e)}")
                    stats["errors"] += 1
            
            return stats
        except Exception as e:
            logger.error(f"Error during synchronization: {str(e)}")
            return {"error": str(e)}
    
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
    
    def synchronize(self) -> dict[str, Any]:
        """
        Synchronize the vector database with DuckDB.
        
        Returns:
            Statistics about the synchronization
        """
        return self.db.sync_with_duckdb()
    
    def close(self) -> None:
        """Close the vector database connection."""
        if hasattr(self, "db"):
            self.db.close()