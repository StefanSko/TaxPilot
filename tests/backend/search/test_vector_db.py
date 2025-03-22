"""
Unit tests for the vector database integration.
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from taxpilot.backend.search.vector_db import (
    VectorDbProvider,
    VectorDbConfig,
    SearchParameters,
    SearchResult,
    VectorDatabase,
    VectorDatabaseManager
)
from taxpilot.backend.search.embeddings import TextEmbedding
from taxpilot.backend.data_processing.database import DbConfig


@pytest.fixture
def mock_qdrant_client():
    """Return a mock Qdrant client for testing."""
    client = MagicMock()
    
    # Mock collections
    collection = MagicMock()
    collection.name = "test_collection"
    client.get_collections.return_value = MagicMock(collections=[collection])
    
    # Mock search
    search_result = []
    for i in range(3):
        hit = MagicMock()
        hit.id = f"point{i}"
        hit.score = 0.9 - (i * 0.1)
        hit.payload = {
            "law_id": f"law{i}",
            "section_id": f"section{i}",
            "segment_id": f"segment{i}",
            "embedding_model": "test-model",
            "embedding_version": "1.0.0",
            "text_length": 100 + i,
            "paragraph_index": i
        }
        search_result.append(hit)
    client.search.return_value = search_result
    
    # Mock other methods
    client.create_collection.return_value = None
    client.create_payload_index.return_value = None
    client.upsert.return_value = None
    
    # For scroll operation
    points = []
    for i in range(5):
        point = MagicMock()
        point.id = f"point{i}"
        point.payload = {
            "law_id": f"law{i % 2}",  # Make some duplicates
            "embedding_model": f"model{i % 3}",
        }
        points.append(point)
    
    # First call returns points, second call returns empty list to end scrolling
    client.scroll.side_effect = [(points, "next_offset"), ([], None)]
    
    return client


@pytest.fixture
def sample_text_embedding():
    """Return a sample text embedding for testing."""
    return TextEmbedding(
        vector=np.random.rand(768).astype(np.float32),
        segment_id="test_segment",
        law_id="test_law",
        section_id="test_section",
        metadata={"text_length": 100, "paragraph_index": 1},
        embedding_model="test-model",
        embedding_version="1.0.0"
    )


@pytest.fixture
def sample_embeddings():
    """Return a list of sample embeddings for testing."""
    return [
        TextEmbedding(
            vector=np.random.rand(768).astype(np.float32),
            segment_id=f"segment{i}",
            law_id=f"law{i % 2}",  # Some duplicates
            section_id=f"section{i}",
            metadata={"text_length": 100 + i, "paragraph_index": i},
            embedding_model="test-model",
            embedding_version="1.0.0"
        )
        for i in range(5)
    ]


@pytest.fixture
def temp_db_dir():
    """Return a temporary directory for database storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@patch("taxpilot.backend.search.vector_db.QdrantClient")
def test_vector_db_init(mock_qdrant, mock_qdrant_client):
    """Test vector database initialization."""
    mock_qdrant.return_value = mock_qdrant_client
    collection = MagicMock()
    collection.name = "law_sections"  # Should match VectorDbConfig.collection_name default value
    mock_qdrant_client.get_collections.return_value = MagicMock(collections=[collection])
    
    # Test with default configuration
    config = VectorDbConfig(provider=VectorDbProvider.QDRANT)
    db = VectorDatabase(config)
    
    # Should have created a client
    assert mock_qdrant.call_count == 1
    
    # Should have checked if collection exists
    assert mock_qdrant_client.get_collections.call_count >= 1
    
    # Collection already exists, so shouldn't create it
    assert mock_qdrant_client.create_collection.call_count == 0
    
    # Test with non-existent collection
    # Reset the mock and make it return no collections
    mock_qdrant.reset_mock()
    mock_qdrant_client.reset_mock()
    mock_qdrant.return_value = mock_qdrant_client
    mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])
    
    # Create the database
    db = VectorDatabase(config)
    
    # Should have created the collection
    assert mock_qdrant_client.create_collection.call_count == 1
    
    # Should have created indexes
    assert mock_qdrant_client.create_payload_index.call_count > 0
    
    # Verify the collection parameters
    collection_args = mock_qdrant_client.create_collection.call_args[1]
    assert collection_args["collection_name"] == config.collection_name
    assert collection_args["vectors_config"].size == config.embedding_dim


@patch("taxpilot.backend.search.vector_db.QdrantClient")
def test_store_embedding(mock_qdrant, mock_qdrant_client, sample_text_embedding):
    """Test storing an embedding in the vector database."""
    mock_qdrant.return_value = mock_qdrant_client
    
    # Create a database with an in-memory provider for testing
    config = VectorDbConfig(provider=VectorDbProvider.MEMORY)
    db = VectorDatabase(config)
    
    # Mock the update_duckdb_reference method
    db._update_duckdb_reference = MagicMock()
    
    # Store an embedding
    point_id = db.store_embedding(sample_text_embedding)
    
    # Should have called upsert
    assert mock_qdrant_client.upsert.call_count == 1
    
    # Check that the point data was provided correctly
    upsert_args = mock_qdrant_client.upsert.call_args[1]
    assert upsert_args["collection_name"] == config.collection_name
    assert len(upsert_args["points"]) == 1
    
    # Verify that the point contains the expected data
    point = upsert_args["points"][0]
    assert point.vector == sample_text_embedding.vector.tolist()
    assert point.payload["law_id"] == sample_text_embedding.law_id
    assert point.payload["section_id"] == sample_text_embedding.section_id
    assert point.payload["segment_id"] == sample_text_embedding.segment_id
    
    # Should have called update_duckdb_reference
    assert db._update_duckdb_reference.call_count == 1
    
    # Should have returned a point ID
    assert isinstance(point_id, str)
    assert len(point_id) > 0


@patch("taxpilot.backend.search.vector_db.QdrantClient")
def test_store_embeddings_batch(mock_qdrant, mock_qdrant_client, sample_embeddings):
    """Test batch storing embeddings in the vector database."""
    mock_qdrant.return_value = mock_qdrant_client
    
    # Create a database with an in-memory provider for testing
    config = VectorDbConfig(
        provider=VectorDbProvider.MEMORY,
        use_batching=True,
        batch_size=2  # Small batch size for testing
    )
    db = VectorDatabase(config)
    
    # Mock the update_duckdb_reference method
    db._update_duckdb_reference = MagicMock()
    
    # Store the embeddings
    point_ids = db.store_embeddings_batch(sample_embeddings)
    
    # Should have multiple upsert calls due to batching
    expected_calls = len(sample_embeddings) // config.batch_size
    if len(sample_embeddings) % config.batch_size != 0:
        expected_calls += 1
    
    assert mock_qdrant_client.upsert.call_count == expected_calls
    
    # Should have multiple update_duckdb_reference calls
    assert db._update_duckdb_reference.call_count == len(sample_embeddings)
    
    # Should have returned a list of point IDs
    assert len(point_ids) == len(sample_embeddings)
    for point_id in point_ids:
        assert isinstance(point_id, str)
        assert len(point_id) > 0


@patch("taxpilot.backend.search.vector_db.QdrantClient")
def test_search(mock_qdrant, mock_qdrant_client):
    """Test searching the vector database."""
    mock_qdrant.return_value = mock_qdrant_client
    
    # Create a database with an in-memory provider for testing
    config = VectorDbConfig(provider=VectorDbProvider.MEMORY)
    db = VectorDatabase(config)
    
    # Create search parameters
    params = SearchParameters(
        query_vector=np.random.rand(768).astype(np.float32),
        law_id="test_law",
        top_k=3
    )
    
    # Perform the search
    results = db.search(params)
    
    # Should have called search
    assert mock_qdrant_client.search.call_count == 1
    
    # Check search parameters
    search_args = mock_qdrant_client.search.call_args[1]
    assert search_args["collection_name"] == config.collection_name
    assert search_args["query_vector"] == params.query_vector.tolist()
    assert search_args["limit"] == params.top_k
    
    # Should have a filter for law_id
    assert search_args["filter"] is not None
    
    # Should have returned search results
    assert len(results) == 3
    for result in results:
        assert isinstance(result, SearchResult)
        assert result.score > 0


@patch("taxpilot.backend.search.vector_db.QdrantClient")
def test_delete_by_law_id(mock_qdrant, mock_qdrant_client):
    """Test deleting embeddings for a specific law."""
    mock_qdrant.return_value = mock_qdrant_client
    
    # Set up mock for the delete method
    delete_result = MagicMock()
    delete_result.deleted = 5
    mock_qdrant_client.delete.return_value = delete_result
    
    # Create a database with an in-memory provider for testing
    config = VectorDbConfig(provider=VectorDbProvider.MEMORY)
    db = VectorDatabase(config)
    
    # Delete embeddings for a law
    deleted_count = db.delete_by_law_id("test_law")
    
    # Should have called delete
    assert mock_qdrant_client.delete.call_count == 1
    
    # Check delete parameters
    delete_args = mock_qdrant_client.delete.call_args[1]
    assert delete_args["collection_name"] == config.collection_name
    
    # Should have a filter for law_id
    assert delete_args["points_filter"] is not None
    
    # Should have returned the deleted count
    assert deleted_count == 5


@patch("taxpilot.backend.search.vector_db.QdrantClient")
def test_get_collection_stats(mock_qdrant, mock_qdrant_client):
    """Test getting statistics about the vector collection."""
    mock_qdrant.return_value = mock_qdrant_client
    
    # Create a database with an in-memory provider for testing
    config = VectorDbConfig(provider=VectorDbProvider.MEMORY)
    db = VectorDatabase(config)
    
    # Get statistics
    stats = db.get_collection_stats()
    
    # Should have called scroll
    assert mock_qdrant_client.scroll.call_count > 0
    
    # Check scroll parameters
    scroll_args = mock_qdrant_client.scroll.call_args[1]
    assert scroll_args["collection_name"] == config.collection_name
    assert scroll_args["with_payload"] is True
    
    # Should have returned statistics
    assert isinstance(stats, dict)
    assert "total_vectors" in stats
    assert "vectors_by_model" in stats
    assert "vectors_by_law" in stats


@patch("taxpilot.backend.search.vector_db.VectorDatabase")
def test_vector_database_manager(mock_vector_db_class, sample_embeddings):
    """Test the vector database manager."""
    # Mock the VectorDatabase instance
    mock_db = MagicMock()
    mock_vector_db_class.return_value = mock_db
    
    # Mock methods on the database
    mock_db.store_embeddings_batch.return_value = ["id1", "id2", "id3", "id4", "id5"]
    mock_db.get_collection_stats.return_value = {"total_vectors": 100}
    mock_db.optimize_collection.return_value = True
    mock_db.sync_with_duckdb.return_value = {"embeddings_added": 10}
    
    # Create a manager
    config = VectorDbConfig(provider=VectorDbProvider.MEMORY)
    manager = VectorDatabaseManager(config)
    
    # Test processing embeddings
    ids = manager.process_text_embeddings(sample_embeddings)
    assert mock_db.store_embeddings_batch.call_count == 1
    assert len(ids) == len(sample_embeddings)
    
    # Test getting stats
    stats = manager.get_stats()
    assert mock_db.get_collection_stats.call_count == 1
    assert stats["total_vectors"] == 100
    
    # Test optimization
    success = manager.optimize()
    assert mock_db.optimize_collection.call_count == 1
    assert success is True
    
    # Test synchronization
    sync_stats = manager.synchronize()
    assert mock_db.sync_with_duckdb.call_count == 1
    assert sync_stats["embeddings_added"] == 10
    
    # Test closing
    manager.close()
    assert mock_db.close.call_count == 1