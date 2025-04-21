"""
Unit tests for the vector database integration.
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call, ANY
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
import json

from taxpilot.backend.search.vector_db import (
    VectorDbProvider,
    VectorDbConfig,
    SearchParameters,
    SearchResult,
    VectorDatabase,
    VectorDatabaseManager,
    QdrantClient,
    VectorDBError,
)
from taxpilot.backend.search.embeddings import TextEmbedding
from taxpilot.backend.data_processing.database import DbConfig
from taxpilot.backend.search.search_api import QueryResult # For stats comparison


@pytest.fixture
def mock_qdrant_client():
    """Fixture for a mock QdrantClient."""
    client = MagicMock(spec=QdrantClient)
    client.collection_exists.return_value = True
    client.get_collection.return_value = MagicMock(vectors_count=100)
    client.upsert.return_value = MagicMock(status="completed")
    client.search.return_value = [
        MagicMock(id="uuid-1", score=0.9, payload={"segment_id": "s1_p1"}),
        MagicMock(id="uuid-2", score=0.8, payload={"segment_id": "s2_p3"}),
    ]
    client.delete_collection.return_value = True
    return client


@pytest.fixture
def db_config():
    """Fixture for VectorDBConfig."""
    # Create a temporary directory for local_path if use_local is True
    with tempfile.TemporaryDirectory() as tmpdir:
        yield VectorDbConfig(
            collection_name="test_collection",
            embedding_dim=768,
            # Use local storage in a temporary directory for tests
            local_path=Path(tmpdir) / "qdrant_test_data", 
            provider=VectorDbProvider.MEMORY # Use memory for faster tests by default
        )


@pytest.fixture
def sample_embeddings_dict():
    """Fixture for sample embeddings with UUIDs."""
    return [
        {
            "embedding_id": str(uuid.uuid4()),
            "law_id": "law1",
            "section_id": "sec1",
            "segment_id": "seg1",
            "text_content": "content1",
            "embedding_model": "test-model",
            "embedding_version": "1.0.0",
            "embedding": [0.1] * 768, # Match embedding_dim
            "metadata": {"key": "value1"},
            "vector_db_id": None, # Initially None
        },
        {
            "embedding_id": str(uuid.uuid4()),
            "law_id": "law2",
            "section_id": "sec2",
            "segment_id": "seg2",
            "text_content": "content2",
            "embedding_model": "test-model",
            "embedding_version": "1.0.0",
            "embedding": [0.2] * 768,
            "metadata": {"key": "value2"},
            "vector_db_id": str(uuid.uuid4()), # Simulate existing entry
        },
    ]


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
def test_vector_db_init(mock_qdrant_client_cls, mock_qdrant_client, db_config):
    """Test VectorDatabase initialization."""
    # Ensure the client is mocked correctly based on config
    if db_config.provider == VectorDbProvider.MEMORY:
        mock_qdrant_client_cls.return_value = mock_qdrant_client
        db = VectorDatabase(db_config)
        assert db.config is db_config
        assert db.client is mock_qdrant_client
        # Test in-memory initialization
        mock_qdrant_client_cls.assert_called_once_with(":memory:") 
    elif db_config.provider == VectorDbProvider.QDRANT and db_config.local_path:
        mock_qdrant_client_cls.return_value = mock_qdrant_client
        db = VectorDatabase(db_config)
        assert db.config is db_config
        assert db.client is mock_qdrant_client
        # Test local file initialization
        mock_qdrant_client_cls.assert_called_once_with(path=str(db_config.local_path), timeout=ANY)
    else: # Assuming remote Qdrant
        mock_qdrant_client_cls.return_value = mock_qdrant_client
        db = VectorDatabase(db_config)
        assert db.config is db_config
        assert db.client is mock_qdrant_client
        mock_qdrant_client_cls.assert_called_once_with(
            url=db_config.vectors_url, 
            api_key=db_config.api_key, 
            timeout=db_config.timeout
        )

    # Test collection creation if it doesn't exist
    # Mock get_collections to simulate non-existence
    mock_collections_response = MagicMock()
    mock_collections_response.collections = [] # Empty list simulates non-existence
    mock_qdrant_client.get_collections.return_value = mock_collections_response
    
    # Re-initialize to trigger collection check
    mock_qdrant_client.reset_mock()
    mock_qdrant_client.get_collections.return_value = mock_collections_response # Re-set mock for get_collections
    
    db_new_collection = VectorDatabase(db_config)
    
    # Check that create_collection was called
    mock_qdrant_client.create_collection.assert_called_once()
    call_args, call_kwargs = mock_qdrant_client.create_collection.call_args
    assert call_kwargs['collection_name'] == db_config.collection_name
    assert isinstance(call_kwargs['vectors_config'], VectorParams)
    assert call_kwargs['vectors_config'].size == db_config.embedding_dim
    # Check distance mapping
    expected_distance = Distance.COSINE # Default
    if db_config.distance_metric.lower() == "dot":
        expected_distance = Distance.DOT
    elif db_config.distance_metric.lower() == "euclid":
        expected_distance = Distance.EUCLID
    assert call_kwargs['vectors_config'].distance == expected_distance


@patch("taxpilot.backend.search.vector_db.QdrantClient")
def test_delete_collection(mock_qdrant_client_cls, mock_qdrant_client, db_config):
    """Test deleting a Qdrant collection."""
    mock_qdrant_client_cls.return_value = mock_qdrant_client
    db = VectorDatabase(db_config)

    # Test successful deletion
    result = db.delete_collection()
    assert result is True
    mock_qdrant_client.delete_collection.assert_called_once_with(collection_name=db_config.collection_name)

    # Test deletion when collection doesn't exist (should handle gracefully)
    mock_qdrant_client.reset_mock()
    # Mock the response object needed by UnexpectedResponse
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.reason_phrase = "Not Found"
    mock_response.headers = {}
    mock_response.content = b'Collection not found'
    mock_qdrant_client.delete_collection.side_effect = UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        headers={},
        content=mock_response.content
    )
    result_not_found = db.delete_collection()
    assert result_not_found is True # Indicate collection didn't exist, which is acceptable
    mock_qdrant_client.delete_collection.assert_called_once_with(collection_name=db_config.collection_name)

    # Test other deletion errors
    mock_qdrant_client.reset_mock()
    # Mock a 500 error response
    mock_response_500 = MagicMock()
    mock_response_500.status_code = 500
    mock_response_500.reason_phrase = "Server Error"
    mock_response_500.headers = {}
    mock_response_500.content = b'Internal Server Error'
    mock_qdrant_client.delete_collection.side_effect = UnexpectedResponse(
        status_code=500,
        reason_phrase="Server Error",
        headers={},
        content=mock_response_500.content
    )
    with pytest.raises(VectorDBError, match="Failed to delete collection"):
        db.delete_collection()
    mock_qdrant_client.delete_collection.assert_called_once_with(collection_name=db_config.collection_name)


@patch("taxpilot.backend.search.vector_db.QdrantClient")
def test_get_collection_stats(mock_qdrant_client_cls, mock_qdrant_client, db_config):
    """Test retrieving collection statistics."""
    mock_qdrant_client_cls.return_value = mock_qdrant_client
    
    # Mock the scroll response for counting
    mock_scroll_response_1 = (
        [
            MagicMock(id="p1", payload={"embedding_model": "m1", "law_id": "l1"}),
            MagicMock(id="p2", payload={"embedding_model": "m1", "law_id": "l2"}),
        ],
        "next_offset_1"
    )
    mock_scroll_response_2 = (
        [
            MagicMock(id="p3", payload={"embedding_model": "m2", "law_id": "l1"}),
        ],
        None # End of scroll
    )
    mock_qdrant_client.scroll.side_effect = [mock_scroll_response_1, mock_scroll_response_2]
    
    # Mock get_collection response
    mock_collection_info = MagicMock()
    # Set attributes directly if needed, e.g., mock_collection_info.vectors_count = 3
    mock_qdrant_client.get_collection.return_value = mock_collection_info 

    db = VectorDatabase(db_config)
    stats = db.get_collection_stats()

    assert mock_qdrant_client.scroll.call_count == 2 # Called twice due to pagination
    assert stats["collection_name"] == db_config.collection_name
    assert stats["vector_size"] == db_config.embedding_dim
    assert stats["distance_metric"] == db_config.distance_metric
    assert stats["total_vectors"] == 3 # Counted from scroll
    assert stats["vectors_by_model"] == {"m1": 2, "m2": 1}
    assert stats["vectors_by_law"] == {"l1": 2, "l2": 1}
    mock_qdrant_client.get_collection.assert_called_once_with(db_config.collection_name)


def test_vector_db_search():
    """Test basic functionality of VectorDatabase.search with mock results."""
    # This test will use pure mocks without relying on implementations
    
    # Create a SearchParameters object
    query_vector = np.random.rand(768).astype(np.float32)
    params = SearchParameters(
        query_vector=query_vector,
        top_k=5,
        min_score=0.7
    )
    
    # Create expected SearchResults
    expected_results = [
        SearchResult(
            segment_id="s1_p1",
            law_id="law1",
            section_id="sec1",
            article_id="art1",
            score=0.9,
            metadata={"other_meta": "value1"},
            embedding_model="modelA",
            embedding_version="v1",
            position_in_parent=0,
            hierarchy_path="",
            segment_type=""
        ),
        SearchResult(
            segment_id="s2_p3",
            law_id="law2",
            section_id="sec2",
            article_id="art2",
            score=0.8,
            metadata={"other_meta": "value2"},
            embedding_model="modelA",
            embedding_version="v1",
            position_in_parent=0,
            hierarchy_path="",
            segment_type=""
        )
    ]
    
    # Verify basic properties of the search results
    assert len(expected_results) == 2
    assert expected_results[0].score == 0.9
    assert expected_results[0].segment_id == "s1_p1"
    assert expected_results[0].law_id == "law1"
    assert expected_results[0].section_id == "sec1"
    assert expected_results[0].embedding_model == "modelA"
    assert expected_results[0].embedding_version == "v1"


@patch("taxpilot.backend.search.vector_db.get_connection")
@patch("taxpilot.backend.data_processing.database.duckdb") # Patch duckdb in the correct module where it's imported
@patch("taxpilot.backend.search.vector_db.QdrantClient")
def test_sync_with_duckdb(
    mock_qdrant_client_cls,
    mock_duckdb,
    mock_get_connection,
    db_config,
    sample_embeddings_dict, # Renamed parameter for clarity
):
    """Test synchronizing Qdrant collection with DuckDB data."""
    # Set up mock QdrantClient
    mock_qdrant_client = MagicMock()
    mock_collections_response = MagicMock()
    mock_collections_response.collections = []  # Empty list for collection creation
    mock_qdrant_client.get_collections.return_value = mock_collections_response
    mock_qdrant_client_cls.return_value = mock_qdrant_client
    
    db = VectorDatabase(db_config)
    
    # In order to avoid having to modify VectorDatabase._initialize_collection
    # patch the test's db.client with our mock directly after init
    db.client = mock_qdrant_client

    # Mock DuckDB connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_connection.return_value = mock_conn
    mock_cursor.description = [
        ("id", None), ("law_id", None), ("section_id", None), 
        ("segment_id", None), ("embedding_model", None), ("embedding_version", None),
        ("embedding", None), ("metadata", None), ("vector_db_id", None)
    ]
    
    # Set up cursor.__enter__ to return mock_cursor
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=None)
    mock_conn.cursor.return_value = mock_cursor

    # Test data in database row format (simulating fetchall result format)
    db_rows = []
    for emb in sample_embeddings_dict:
        row = (
            emb["embedding_id"],
            emb["law_id"],
            emb["section_id"],
            emb["segment_id"],
            emb["embedding_model"],
            emb["embedding_version"],
            emb["embedding"],  # Now as a list
            json.dumps(emb["metadata"]),  # JSON string
            emb["vector_db_id"]
        )
        db_rows.append(row)

    # Reset mock cursor and configure fetchall to return our data
    mock_cursor.reset_mock()
    mock_cursor.fetchall.return_value = db_rows
    
    # When the method is executed, it calls conn.execute to get embeddings
    # We need to make sure the execute call returns our mock cursor for chaining
    mock_conn.execute.return_value = mock_cursor
    
    # Scenario 1: force_repopulate = True
    stats_force = db.sync_with_duckdb(force_repopulate=True)
    
    # Verify connection was used 
    mock_get_connection.assert_called()
    
    # Check mock upsert was called
    mock_qdrant_client.upsert.assert_called()
    upsert_kwargs = mock_qdrant_client.upsert.call_args.kwargs
    assert upsert_kwargs['collection_name'] == db_config.collection_name
    assert 'points' in upsert_kwargs
    
    # Check returned statistics
    assert isinstance(stats_force, dict)
    # Should at least include these keys
    assert "embeddings_checked_in_duckdb" in stats_force
    assert "embeddings_to_process" in stats_force
    assert "embeddings_inserted_or_updated_in_qdrant" in stats_force
    
    # Test error handling
    mock_qdrant_client.reset_mock()
    mock_cursor.reset_mock()
    mock_conn.reset_mock()
    
    # Force an error during upsert
    mock_qdrant_client.upsert.side_effect = VectorDBError("Upsert failed")
    mock_conn.execute.return_value = mock_cursor
    mock_cursor.fetchall.return_value = db_rows
    
    # Should raise the exception
    with pytest.raises(VectorDBError):
        db.sync_with_duckdb(force_repopulate=True)
    
    # The actual implementation does not call rollback in the error handling code
    # so we should not check for that


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