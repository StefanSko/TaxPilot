"""
Unit tests for the hierarchical vector database functionality.

These tests verify the enhanced vector database that supports
hierarchical document structures for legal texts.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from taxpilot.backend.search.vector_db import (
    VectorDbConfig,
    VectorDbProvider,
    VectorDatabaseManager,
    SearchParameters
)
from taxpilot.backend.search.embeddings import TextEmbedding
from taxpilot.backend.search.segmentation import SegmentType


@pytest.fixture
def mock_hierarchical_embedding():
    """Create a test embedding with hierarchical information."""
    def _create_embedding(segment_id, article_id, hierarchy_path, segment_type):
        # Create a small random vector for testing
        vector = np.random.random(768).astype(np.float32)
        
        # Create a metadata dictionary with hierarchical information
        metadata = {
            "article_id": article_id,
            "hierarchy_path": hierarchy_path,
            "segment_type": segment_type.value if isinstance(segment_type, SegmentType) else segment_type,
            "position_in_parent": 1,
            "text_length": 100
        }
        
        # Create the embedding object
        embedding = TextEmbedding(
            vector=vector,
            law_id="estg",
            section_id="estg_13",
            segment_id=segment_id,
            embedding_model="test_model",
            embedding_version="1.0",
            metadata=metadata
        )
        
        return embedding
    
    return _create_embedding


@pytest.fixture
def test_embeddings(mock_hierarchical_embedding):
    """Create a set of test embeddings with hierarchical information."""
    return [
        mock_hierarchical_embedding(
            "estg_13_p1", 
            "estg_13", 
            "estg/§13/abs1", 
            SegmentType.PARAGRAPH
        ),
        mock_hierarchical_embedding(
            "estg_13_p2", 
            "estg_13", 
            "estg/§13/abs2", 
            SegmentType.PARAGRAPH
        ),
        mock_hierarchical_embedding(
            "estg_14_p1", 
            "estg_14", 
            "estg/§14/abs1", 
            SegmentType.PARAGRAPH
        )
    ]


@patch("taxpilot.backend.search.vector_db.VectorDatabase")
def test_vector_db_hierarchical_storage(mock_vector_db_class, test_embeddings):
    """Test storing and retrieving hierarchical data in the vector database."""
    # Set up mock vector database
    mock_db = MagicMock()
    mock_vector_db_class.return_value = mock_db
    
    # Mock the store_embeddings_batch method
    mock_db.store_embeddings_batch.return_value = ["id1", "id2", "id3"]
    
    # Create a test configuration using in-memory database for testing
    config = VectorDbConfig(
        provider=VectorDbProvider.MEMORY,
        collection_name="test_collection",
        embedding_dim=768,
        store_hierarchical_data=True
    )
    
    # Create a vector database manager
    db_manager = VectorDatabaseManager(config)
    
    # Store the embeddings
    vector_ids = db_manager.process_text_embeddings(test_embeddings)
    
    # Verify that the vector database was called with the correct embeddings
    mock_db.store_embeddings_batch.assert_called_once()
    call_args = mock_db.store_embeddings_batch.call_args[0][0]
    
    # Verify the first embedding contains the correct hierarchical data
    first_embedding = call_args[0]
    assert "article_id" in first_embedding.metadata
    assert first_embedding.metadata["article_id"] == "estg_13"
    assert first_embedding.metadata["hierarchy_path"] == "estg/§13/abs1"
    assert first_embedding.metadata["segment_type"] == SegmentType.PARAGRAPH.value
    
    # Verify the returned vector IDs
    assert len(vector_ids) == 3
    assert vector_ids == ["id1", "id2", "id3"]


@patch("taxpilot.backend.search.vector_db.VectorDatabase")
def test_article_id_filtering(mock_vector_db_class, test_embeddings):
    """Test filtering search results by article ID."""
    # Set up mock vector database
    mock_db = MagicMock()
    mock_vector_db_class.return_value = mock_db
    
    # Create mock search results
    mock_results = [
        MagicMock(article_id="estg_13"),
        MagicMock(article_id="estg_13"),
        MagicMock(article_id="estg_14")
    ]
    mock_db.search.return_value = mock_results
    
    # Create a test configuration
    config = VectorDbConfig(
        provider=VectorDbProvider.MEMORY,
        collection_name="test_collection",
        embedding_dim=768,
        store_hierarchical_data=True
    )
    
    # Create a vector database manager
    db_manager = VectorDatabaseManager(config)
    
    # Create a search parameters object with article filtering
    query_vector = np.random.random(768).astype(np.float32)
    search_params = SearchParameters(
        query_vector=query_vector,
        top_k=10,
        article_id="estg_13"
    )
    
    # Execute the search
    results = db_manager.search_similar(search_params)
    
    # Verify that the vector database was called with the correct parameters
    mock_db.search.assert_called_once()
    
    # The results should still include all mock results (filtering happens at the DB level)
    assert len(results) == 3