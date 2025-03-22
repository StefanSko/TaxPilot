"""
Unit tests for the vector embedding generation module.
"""

import tempfile
import os
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from taxpilot.backend.search.embeddings import (
    EmbeddingModelType,
    EmbeddingConfig,
    TextEmbedding,
    ModelCache,
    TextEmbedder,
    EmbeddingProcessor,
)
from taxpilot.backend.search.segmentation import TextSegment
from taxpilot.backend.data_processing.database import DbConfig


@pytest.fixture
def sample_segment():
    """Return a sample text segment for testing."""
    return TextSegment(
        text="§ 1 Persönlicher Anwendungsbereich\n\n(1) Unbeschränkt einkommensteuerpflichtig sind natürliche Personen, die im Inland einen Wohnsitz oder ihren gewöhnlichen Aufenthalt haben.",
        law_id="estg",
        section_id="s1",
        segment_id="s1_p1",
        start_idx=0,
        end_idx=150,
        metadata={"strategy": "paragraph", "paragraph_index": 0}
    )


@pytest.fixture
def sample_segments():
    """Return a list of sample text segments for testing."""
    return [
        TextSegment(
            text="§ 1 Persönlicher Anwendungsbereich",
            law_id="estg",
            section_id="s1",
            segment_id="s1_p1",
            start_idx=0,
            end_idx=30,
            metadata={"strategy": "paragraph", "paragraph_index": 0}
        ),
        TextSegment(
            text="(1) Unbeschränkt einkommensteuerpflichtig sind natürliche Personen, die im Inland einen Wohnsitz oder ihren gewöhnlichen Aufenthalt haben.",
            law_id="estg",
            section_id="s1",
            segment_id="s1_p2",
            start_idx=31,
            end_idx=150,
            metadata={"strategy": "paragraph", "paragraph_index": 1}
        ),
        TextSegment(
            text="(2) Auf Antrag werden auch natürliche Personen als unbeschränkt einkommensteuerpflichtig behandelt, die im Inland weder einen Wohnsitz noch ihren gewöhnlichen Aufenthalt haben.",
            law_id="estg",
            section_id="s1",
            segment_id="s1_p3",
            start_idx=151,
            end_idx=300,
            metadata={"strategy": "paragraph", "paragraph_index": 2}
        )
    ]


@pytest.fixture
def mock_model():
    """Return a mock embedding model for testing."""
    mock = MagicMock()
    mock.encode.return_value = np.random.rand(3, 768).astype(np.float32)
    mock.get_sentence_embedding_dimension.return_value = 768
    return mock


@pytest.fixture
def temp_db_path():
    """Return a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


def test_embedding_config():
    """Test embedding configuration."""
    # Default configuration
    config = EmbeddingConfig()
    assert config.model_name == EmbeddingModelType.DEFAULT.value
    assert config.embedding_dim == 768
    assert config.cache_dir.exists()
    
    # Configuration with large model
    config = EmbeddingConfig(model_name="deepset/gbert-large")
    assert config.embedding_dim == 1024
    
    # Configuration with custom parameters
    config = EmbeddingConfig(
        model_name=EmbeddingModelType.MULTILINGUAL_E5.value,
        embedding_dim=1024,
        batch_size=64,
        normalize_embeddings=False
    )
    assert config.model_name == EmbeddingModelType.MULTILINGUAL_E5.value
    assert config.embedding_dim == 1024
    assert config.batch_size == 64
    assert config.normalize_embeddings is False


@patch("taxpilot.backend.search.embeddings.SentenceTransformer")
def test_model_cache(mock_transformer, mock_model):
    """Test model caching."""
    mock_transformer.return_value = mock_model
    
    # Create cache and get a model
    cache = ModelCache()
    
    # Get a model (should load it)
    config = EmbeddingConfig(model_name="test-model")
    model1 = cache.get_model("test-model", config)
    
    # Should have loaded the model
    assert mock_transformer.call_count == 1
    
    # Get the same model again (should use cache)
    model2 = cache.get_model("test-model", config)
    
    # Should not have loaded again
    assert mock_transformer.call_count == 1
    assert model1 is model2
    
    # List cached models
    cached_models = cache.list_cached_models()
    assert len(cached_models) == 1
    assert cached_models[0]["name"] == "test-model"
    
    # Clear the cache
    cache.clear_cache()
    assert len(cache.list_cached_models()) == 0


@patch("taxpilot.backend.search.embeddings.ModelCache.get_model")
def test_text_embedder(mock_get_model, mock_model, sample_segment, sample_segments):
    """Test text embedding generation."""
    mock_get_model.return_value = mock_model
    
    # Create embedder
    config = EmbeddingConfig()
    embedder = TextEmbedder(config)
    
    # Test embedding a single text
    embedding = embedder.embed_text(sample_segment.text)
    assert isinstance(embedding, np.ndarray)
    assert mock_model.encode.call_count == 1
    
    # Test embedding multiple texts
    texts = [seg.text for seg in sample_segments]
    embeddings = embedder.embed_texts(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert mock_model.encode.call_count == 2
    
    # Test embedding segments
    segment_embeddings = embedder.embed_segments(sample_segments)
    assert len(segment_embeddings) == len(sample_segments)
    assert isinstance(segment_embeddings[0], TextEmbedding)
    assert segment_embeddings[0].law_id == sample_segments[0].law_id
    assert segment_embeddings[0].section_id == sample_segments[0].section_id
    assert segment_embeddings[0].segment_id == sample_segments[0].segment_id
    assert segment_embeddings[0].embedding_model == config.model_name
    assert segment_embeddings[0].embedding_version == config.model_version


@patch("taxpilot.backend.search.embeddings.get_connection")
@patch("taxpilot.backend.search.embeddings.TextEmbedder")
def test_embedding_processor(mock_embedder_cls, mock_get_connection, temp_db_path, sample_segments):
    """Test embedding processing and storage."""
    # Set up mocks
    mock_embedder = MagicMock()
    mock_embedder_cls.return_value = mock_embedder
    
    # Create mock embeddings
    mock_embeddings = []
    for i, segment in enumerate(sample_segments):
        vector = np.random.rand(768).astype(np.float32)
        embedding = TextEmbedding(
            vector=vector,
            segment_id=segment.segment_id,
            law_id=segment.law_id,
            section_id=segment.section_id,
            metadata=segment.metadata,
            embedding_model="test-model",
            embedding_version="1.0.0"
        )
        mock_embeddings.append(embedding)
    
    mock_embedder.embed_segments.return_value = mock_embeddings
    
    # Set up mock database connection
    mock_conn = MagicMock()
    mock_get_connection.return_value = mock_conn
    mock_execute = MagicMock()
    mock_conn.execute.return_value = mock_execute
    mock_execute.fetchone.return_value = None  # No existing embeddings
    
    # Create processor
    config = EmbeddingConfig(db_config=DbConfig(db_path=str(temp_db_path)))
    processor = EmbeddingProcessor(config)
    
    # Test processing segments
    embedding_ids = processor.process_segments(sample_segments)
    
    # Verify embeddings were generated and stored
    assert mock_embedder.embed_segments.call_count == 1
    assert mock_embedder.embed_segments.call_args[0][0] == sample_segments
    assert len(embedding_ids) == len(sample_segments)
    
    # Verify database operations
    assert mock_conn.execute.call_count >= 4  # 1 for init, 3 for insert
    
    # Test the processor can handle empty input
    embedding_ids = processor.process_segments([])
    assert len(embedding_ids) == 0


def test_text_embedding_dataclass():
    """Test the TextEmbedding dataclass."""
    # Create a sample embedding
    vector = np.random.rand(768).astype(np.float32)
    embedding = TextEmbedding(
        vector=vector,
        segment_id="s1_p1",
        law_id="estg",
        section_id="s1",
        metadata={"strategy": "paragraph"},
        embedding_model="test-model",
        embedding_version="1.0.0"
    )
    
    # Check fields
    assert np.array_equal(embedding.vector, vector)
    assert embedding.segment_id == "s1_p1"
    assert embedding.law_id == "estg"
    assert embedding.section_id == "s1"
    assert embedding.metadata["strategy"] == "paragraph"
    assert embedding.embedding_model == "test-model"
    assert embedding.embedding_version == "1.0.0"