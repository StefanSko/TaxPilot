"""
Modal.com configuration for vector database operations.

This module configures Modal.com containers and functions for managing
vector embeddings and performing semantic search operations.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import modal
import numpy as np
from pydantic import BaseModel

from taxpilot.backend.search.embeddings import (
    EmbeddingModelType, 
    EmbeddingConfig, 
    TextEmbedder
)
from taxpilot.backend.search.vector_db import (
    VectorDbProvider,
    VectorDbConfig,
    SearchParameters,
    VectorDatabaseManager
)

# Define API models for request/response validation
class SearchRequest(BaseModel):
    """Model for API search request."""
    query: str
    filters: Dict[str, Any] = {}
    page: int = 1
    limit: int = 10
    highlight: bool = True


class QueryResult(BaseModel):
    """Model for individual search result."""
    id: str
    law_id: str
    section_number: str
    title: str
    content: str
    content_with_highlights: str
    relevance_score: float
    metadata: Dict[str, Any] = {}


# Create volumes for persistent storage
db_volume = modal.Volume.from_name("taxpilot-database", create_if_missing=True)
model_cache_volume = modal.Volume.from_name("model-cache", create_if_missing=True)
vector_db_volume = modal.Volume.from_name("vector-db", create_if_missing=True)

# Define image with dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "sentence-transformers>=3.4.0",
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "fastapi>=0.110.0",
    "pydantic>=2.6.0",
    "duckdb>=0.10.0",
    "qdrant-client>=1.13.0",
)

# Define additional API models
class SearchResponse(BaseModel):
    """Model for API search response."""
    results: List[QueryResult]
    total: int
    page: int
    limit: int
    query: str
    execution_time_ms: float


# Create the Modal app
app = modal.App("taxpilot-vector-search", image=image)

# Create a FastAPI web endpoint
web_app = modal.asgi_app()


@web_app.post("/api/search")
async def api_search(request: SearchRequest):
    """
    API endpoint for semantic search.
    
    Args:
        request: The search request
        
    Returns:
        SearchResponse with results
    """
    from taxpilot.backend.search.search_api import SearchService
    import time
    
    start_time = time.time()
    
    # Create the search service
    service = SearchService()
    
    try:
        # Execute search
        results = service.search(
            query=request.query,
            filters=request.filters,
            page=request.page,
            limit=request.limit,
            highlight=request.highlight
        )
        
        # Convert to response format
        return SearchResponse(
            results=results.results,
            total=results.total,
            page=results.page,
            limit=results.limit,
            query=results.query,
            execution_time_ms=results.execution_time_ms
        )
    finally:
        # Clean up resources
        service.close()


@web_app.get("/api/laws")
async def get_laws():
    """
    API endpoint to list available laws.
    
    Returns:
        List of law objects
    """
    from taxpilot.backend.data_processing.database import get_connection, close_connection
    
    try:
        # Get list of laws from database
        conn = get_connection()
        
        query = """
        SELECT 
            id, 
            full_name as name, 
            abbreviation 
        FROM laws
        ORDER BY abbreviation
        """
        result = conn.execute(query).fetchall()
        
        # Convert to list of dictionaries
        laws = []
        for row in result:
            laws.append({
                "id": row["id"],
                "name": row["name"],
                "abbreviation": row["abbreviation"]
            })
            
        return laws
    finally:
        close_connection()


@web_app.get("/health")
async def health():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    from taxpilot.backend.data_processing.database import get_connection, close_connection
    from taxpilot.backend.search.vector_db import VectorDatabaseManager, VectorDbConfig
    
    status = {
        "status": "healthy",
        "components": {}
    }
    
    # Check database connection
    try:
        conn = get_connection()
        result = conn.execute("SELECT 1").fetchone()
        status["components"]["database"] = "healthy" if result[0] == 1 else "unhealthy"
    except Exception as e:
        status["components"]["database"] = f"unhealthy: {str(e)}"
    finally:
        close_connection()
        
    # Check vector database connection
    try:
        vector_config = VectorDbConfig(
            provider=VectorDbProvider.QDRANT,
            local_path="/vector_db"
        )
        manager = VectorDatabaseManager(vector_config)
        stats = manager.get_stats()
        status["components"]["vector_db"] = "healthy"
        status["components"]["vector_db_stats"] = stats
        manager.close()
    except Exception as e:
        status["components"]["vector_db"] = f"unhealthy: {str(e)}"
        
    # Overall status
    if any(v != "healthy" for k, v in status["components"].items() if not isinstance(v, dict)):
        status["status"] = "unhealthy"
        
    return status


@app.function(
    timeout=600,  # 10 minute timeout
    cpu=2.0,
    retries=2,
    volumes={
        "/data": db_volume,
        "/model_cache": model_cache_volume,
        "/vector_db": vector_db_volume,
    },
    secrets=[
        modal.Secret.from_name("taxpilot-env"),
    ],
)
def synchronize_vector_database():
    """
    Synchronize the vector database with the embeddings in DuckDB.
    
    This ensures that all embeddings are properly indexed in the
    vector database for efficient semantic search.
    
    Returns:
        Statistics about the synchronization
    """
    from taxpilot.backend.data_processing.database import DbConfig
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("modal-vector-sync")
    
    # Configure paths
    db_path = Path("/data/taxpilot.db")
    vector_db_path = Path("/vector_db")
    
    # Configure databases
    db_config = DbConfig(db_path=str(db_path))
    vector_config = VectorDbConfig(
        provider=VectorDbProvider.QDRANT,
        local_path=vector_db_path,
        db_config=db_config
    )
    
    # Initialize vector database manager
    manager = VectorDatabaseManager(vector_config)
    
    try:
        # Perform synchronization
        logger.info("Starting vector database synchronization")
        stats = manager.synchronize()
        logger.info(f"Synchronization complete: {stats}")
        
        # Optimize if many embeddings were added
        if stats.get("embeddings_added", 0) > 1000:
            logger.info("Optimizing vector database")
            manager.optimize()
        
        return stats
    finally:
        manager.close()


@app.function(
    timeout=60,
    cpu=1.0,
    volumes={
        "/data": db_volume,
        "/model_cache": model_cache_volume,
        "/vector_db": vector_db_volume,
    },
    secrets=[
        modal.Secret.from_name("taxpilot-env"),
    ],
)
def search(
    query: str,
    law_id: str | None = None,
    section_id: str | None = None,
    top_k: int = 10,
    model_name: str = EmbeddingModelType.DEFAULT.value
):
    """
    Perform semantic search using the vector database.
    
    Args:
        query: The search query
        law_id: Optional law ID to filter results
        section_id: Optional section ID to filter results
        top_k: Number of results to return
        model_name: Name of the embedding model to use
        
    Returns:
        List of search results
    """
    from taxpilot.backend.data_processing.database import DbConfig
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("modal-vector-search")
    
    # Configure paths
    db_path = Path("/data/taxpilot.db")
    vector_db_path = Path("/vector_db")
    model_cache_path = Path("/model_cache")
    
    # Configure databases
    db_config = DbConfig(db_path=str(db_path))
    vector_config = VectorDbConfig(
        provider=VectorDbProvider.QDRANT,
        local_path=vector_db_path,
        db_config=db_config
    )
    
    # Configure embedder
    embedding_config = EmbeddingConfig(
        model_name=model_name,
        cache_dir=model_cache_path,
        use_gpu=False  # CPU is sufficient for single query embedding
    )
    
    # Initialize components
    embedder = TextEmbedder(embedding_config)
    vector_db = VectorDatabaseManager(vector_config)
    
    try:
        # Generate query embedding
        logger.info(f"Generating embedding for query: {query}")
        query_vector = embedder.embed_text(query)
        
        # Create search parameters
        params = SearchParameters(
            query_text=query,
            query_vector=query_vector,
            law_id=law_id,
            section_id=section_id,
            top_k=top_k,
            embedding_model=model_name
        )
        
        # Perform search
        logger.info(f"Performing vector search with parameters: {params}")
        results = vector_db.search_similar(params)
        logger.info(f"Found {len(results)} results")
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                "segment_id": result.segment_id,
                "law_id": result.law_id,
                "section_id": result.section_id,
                "score": float(result.score),
                "metadata": result.metadata,
                "embedding_model": result.embedding_model,
                "embedding_version": result.embedding_version
            })
        
        return serializable_results
    finally:
        vector_db.close()


@app.function(
    timeout=60,
    cpu=1.0,
    volumes={
        "/data": db_volume,
        "/vector_db": vector_db_volume,
    },
    secrets=[
        modal.Secret.from_name("taxpilot-env"),
    ],
)
def get_vector_db_stats():
    """
    Get statistics about the vector database.
    
    Returns:
        Dictionary of statistics
    """
    from taxpilot.backend.data_processing.database import DbConfig
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("modal-vector-stats")
    
    # Configure paths
    db_path = Path("/data/taxpilot.db")
    vector_db_path = Path("/vector_db")
    
    # Configure databases
    db_config = DbConfig(db_path=str(db_path))
    vector_config = VectorDbConfig(
        provider=VectorDbProvider.QDRANT,
        local_path=vector_db_path,
        db_config=db_config
    )
    
    # Initialize vector database manager
    manager = VectorDatabaseManager(vector_config)
    
    try:
        # Get statistics
        logger.info("Getting vector database statistics")
        stats = manager.get_stats()
        return stats
    finally:
        manager.close()


@app.function(
    timeout=60,
    cpu=1.0,
    volumes={
        "/data": db_volume,
        "/vector_db": vector_db_volume,
    },
    secrets=[
        modal.Secret.from_name("taxpilot-env"),
    ],
)
def optimize_vector_db():
    """
    Optimize the vector database for search performance.
    
    Returns:
        True if optimization was successful
    """
    from taxpilot.backend.data_processing.database import DbConfig
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("modal-vector-optimize")
    
    # Configure paths
    db_path = Path("/data/taxpilot.db")
    vector_db_path = Path("/vector_db")
    
    # Configure databases
    db_config = DbConfig(db_path=str(db_path))
    vector_config = VectorDbConfig(
        provider=VectorDbProvider.QDRANT,
        local_path=vector_db_path,
        db_config=db_config
    )
    
    # Initialize vector database manager
    manager = VectorDatabaseManager(vector_config)
    
    try:
        # Perform optimization
        logger.info("Optimizing vector database")
        success = manager.optimize()
        return {"success": success}
    finally:
        manager.close()


# Create a scheduled job to synchronize the vector database
@app.schedule(cron="0 1 * * *")  # Run daily at 1 AM
def daily_vector_db_sync():
    """Daily synchronization of the vector database."""
    return synchronize_vector_database.remote()


@app.schedule(cron="0 2 * * 0")  # Run weekly on Sunday at 2 AM
def weekly_vector_db_optimize():
    """Weekly optimization of the vector database."""
    return optimize_vector_db.remote()


# Mount the ASGI app
app.web_endpoint(
    function_name="web_endpoint",
    web_app=web_app,
    title="GermanLawFinder API",
    description="API for searching German tax laws",
    cpu=1.0,
    memory=512,
    volumes={
        "/data": db_volume,
        "/model_cache": model_cache_volume,
        "/vector_db": vector_db_volume,
    },
    secrets=[
        modal.Secret.from_name("taxpilot-env"),
    ],
)


# Entry point for ad-hoc runs
if __name__ == "__main__":
    with modal.run():
        # Synchronize vector database
        stats = synchronize_vector_database.remote()
        print(f"Synchronization stats: {stats}")
        
        # Get vector database statistics
        vector_stats = get_vector_db_stats.remote()
        print(f"Vector database stats: {vector_stats}")
        
        # Test search
        results = search.remote(
            query="Wann gelten Einnahmen als steuerrelevant?",
            top_k=3
        )
        print(f"Search results: {results}")