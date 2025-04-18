"""
Unit tests for the vector embedding generation module.
"""

import tempfile
import os
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, ANY
import torch
import json

from taxpilot.backend.search.embeddings import (
    EmbeddingModelType,
    EmbeddingConfig,
    TextEmbedding,
    ModelCache,
    TextEmbedder,
    EmbeddingProcessor,
    EmbeddingError,
    ModelConfigError,
)
from taxpilot.backend.search.segmentation import TextSegment
from taxpilot.backend.data_processing.database import DbConfig, get_connection


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
    """Fixture for a mock transformer model."""
    model = MagicMock()
    model.return_value = MagicMock(last_hidden_state=(torch.randn(2, 4, 768),)) # Mock HF model output structure
    return model


@pytest.fixture
def mock_tokenizer():
    """Fixture for a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = { # Mock HF tokenizer output structure
        'input_ids': torch.randint(0, 1000, (2, 4)),
        'attention_mask': torch.ones(2, 4)
    }
    return tokenizer


@pytest.fixture
def temp_db_path():
    """Return a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


@pytest.fixture
def mock_hf_outputs():
    """Fixture for mock Hugging Face model and tokenizer outputs."""
    mock_model = MagicMock()
    mock_tokenizer_instance = MagicMock()
    
    # Define default behaviors (can be overridden in tests)
    mock_model_output = MagicMock(last_hidden_state=(torch.randn(1, 4, 768),)) # Default batch size 1
    # Set the return value for when the mock_model instance is called
    mock_model.return_value = mock_model_output
    # mock_model.__call__.return_value = mock_model_output # Assign mock output to __call__ - This was likely the issue
    # Make the model mock return itself when .to(device) is called
    mock_model.to.return_value = mock_model
    
    mock_tokenizer_instance.return_value = { 
        'input_ids': torch.randint(0, 1000, (1, 4)),
        'attention_mask': torch.ones(1, 4)
    }
    return mock_model, mock_tokenizer_instance


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


@patch("taxpilot.backend.search.embeddings.AutoTokenizer.from_pretrained")
@patch("taxpilot.backend.search.embeddings.AutoModel.from_pretrained")
@patch("taxpilot.backend.search.embeddings.AutoConfig.from_pretrained")
def test_model_cache(mock_auto_config, mock_auto_model, mock_auto_tokenizer, mock_hf_outputs):
    """Test ModelCache functionality."""
    mock_model, mock_tokenizer_instance = mock_hf_outputs
    mock_auto_model.return_value = mock_model
    mock_auto_tokenizer.return_value = mock_tokenizer_instance # This is the instance, not the callable
    # Mock AutoConfig return value (e.g., with a default config mock)
    mock_config_instance = MagicMock()
    mock_config_instance.hidden_size = 768 # Example attribute needed by the code
    mock_auto_config.return_value = mock_config_instance

    cache = ModelCache()
    # Clear cache before test
    cache.clear_cache()
    config = EmbeddingConfig()

    # First call - should load model and tokenizer
    model1, tokenizer1 = cache.get_model(config.model_name, config)
    mock_auto_tokenizer.assert_called_once_with(
        config.model_name,
        cache_dir=str(config.cache_dir)
    )
    # Model loading might be called multiple times if dtype changes, check first call
    mock_auto_model.assert_any_call(
        config.model_name, 
        config=ANY # Config object might be complex to mock exactly
    )
    assert model1 is mock_model
    assert tokenizer1 is mock_tokenizer_instance

    # Second call - should return cached model
    mock_auto_tokenizer.reset_mock()
    mock_auto_model.reset_mock()
    mock_auto_config.reset_mock() # Reset AutoConfig mock
    # Test different model
    config2 = EmbeddingConfig(model_name="another-model")
    mock_model2 = MagicMock()
    mock_model2.to.return_value = mock_model2 # Ensure the new mock also handles .to()
    mock_tokenizer_instance2 = MagicMock()
    mock_config_instance2 = MagicMock() # Mock config for the second model
    mock_auto_model.return_value = mock_model2
    mock_auto_tokenizer.return_value = mock_tokenizer_instance2
    mock_auto_config.return_value = mock_config_instance2

    model3, tokenizer3 = cache.get_model(config2.model_name, config2)
    mock_auto_tokenizer.assert_called_once_with(
        config2.model_name,
        cache_dir=str(config2.cache_dir)
    )
    mock_auto_model.assert_any_call(
        config2.model_name,
        config=mock_config_instance2 # Check it used the mocked config
        # config=ANY # Or keep using ANY if config obj is complex
    )
    mock_auto_config.assert_called_once_with(config2.model_name)
    assert model3 is mock_model2
    assert tokenizer3 is mock_tokenizer_instance2

    # Check cache contents (uses internal _get_model_key)
    model_key1 = cache._get_model_key(config.model_name, config)
    model_key2 = cache._get_model_key(config2.model_name, config2)
    assert model_key1 in cache._models
    assert model_key2 in cache._models
    assert len(cache.list_cached_models()) == 2
    
    # Clear cache again
    cache.clear_cache()
    assert not cache._models


@patch("taxpilot.backend.search.embeddings.get_connection") # Correct patch target
def test_embedding_processor_init(mock_get_connection):
    """Test EmbeddingProcessor initialization and DB setup."""
    mock_conn = MagicMock()
    mock_execute = MagicMock() # Mock execute directly on the connection
    mock_get_connection.return_value = mock_conn
    mock_conn.execute.return_value = mock_execute # Make execute callable

    # Mock TextEmbedder to avoid actual model loading during init test
    with patch("taxpilot.backend.search.embeddings.TextEmbedder") as MockTextEmbedder:
        mock_embedder_instance = MockTextEmbedder.return_value
        
        config = EmbeddingConfig() # Uses default DbConfig
        processor = EmbeddingProcessor(config)

        assert processor.config is config
        assert processor.embedder is mock_embedder_instance
        mock_get_connection.assert_called_once_with(config.db_config)

        # Check that initialization SQL was executed via conn.execute
        assert mock_conn.execute.call_count >= 5
        
        # Find the CREATE TABLE call
        create_table_call = None
        for call_args in mock_conn.execute.call_args_list:
            sql = call_args[0][0]
            if "CREATE TABLE IF NOT EXISTS section_embeddings" in sql:
                create_table_call = sql
                break
        assert create_table_call is not None, "CREATE TABLE statement not found in execute calls"
        
        # Check specific columns in the CREATE TABLE statement
        assert "id VARCHAR PRIMARY KEY" in create_table_call
        assert "vector_db_id VARCHAR" in create_table_call
        assert "law_id VARCHAR" in create_table_call
        assert "section_id VARCHAR" in create_table_call
        assert "segment_id VARCHAR" in create_table_call
        assert f"embedding FLOAT[{config.embedding_dim}]" in create_table_call # Check for native float array type with correct dimension
        assert "metadata JSON" in create_table_call

        # Check for CREATE INDEX calls (example)
        index_calls_found = 0
        for call_args in mock_conn.execute.call_args_list:
             sql = call_args[0][0]
             if "CREATE INDEX IF NOT EXISTS" in sql:
                 index_calls_found += 1
        assert index_calls_found >= 4 # Check that index statements were executed


@patch("taxpilot.backend.search.embeddings.get_connection") # Correct patch target
@patch("taxpilot.backend.search.embeddings.uuid.uuid4")
# Mock TextEmbedder to control its output
@patch("taxpilot.backend.search.embeddings.TextEmbedder") 
def test_embedding_processor_store(mock_text_embedder_cls, mock_uuid, mock_get_connection, sample_segments):
    """Test storing embeddings using EmbeddingProcessor."""
    mock_conn = MagicMock()
    mock_execute = MagicMock()
    mock_get_connection.return_value = mock_conn
    mock_conn.execute.return_value = mock_execute

    # Create mock embeddings corresponding to sample_segments
    mock_embeddings_data = []
    expected_sql_params = []
    test_uuids = [f"uuid-{i}" for i in range(len(sample_segments))]
    mock_uuid.side_effect = test_uuids
    
    # Mock the output of embedder.embed_segments
    mock_embedder_instance = mock_text_embedder_cls.return_value
    text_embeddings_result = []
    for i, seg in enumerate(sample_segments):
        vec = np.array([0.1 * (i + 1)] * 768) # Example vector
        emb = TextEmbedding(
            vector=vec,
            law_id=seg.law_id,
            section_id=seg.section_id,
            segment_id=seg.segment_id,
            metadata={
                **seg.metadata,
                "start_idx": seg.start_idx,
                "end_idx": seg.end_idx,
                "text_length": len(seg.text),
            },
            embedding_model="test-model", # Example model info
            embedding_version="v1.0"
        )
        text_embeddings_result.append(emb)
        
        # Prepare expected parameters for the INSERT call for this embedding
        expected_params_for_emb = (
            test_uuids[i],
            emb.law_id,
            emb.section_id,
            emb.segment_id,
            emb.embedding_model,
            emb.embedding_version,
            vec.tolist(),  # Should be list for DuckDB FLOAT[]
            json.dumps(emb.metadata)
        )
        expected_sql_params.append(expected_params_for_emb)

    mock_embedder_instance.embed_segments.return_value = text_embeddings_result

    config = EmbeddingConfig()
    processor = EmbeddingProcessor(config)

    # Call the method to test
    embedding_ids = processor.process_segments(sample_segments)
    
    # Verify embed_segments was called
    mock_embedder_instance.embed_segments.assert_called_once_with(sample_segments)
    
    # Verify DB connection was obtained
    mock_get_connection.assert_called() # Called during init and store

    # Verify INSERT statements execution
    # Execute should be called once per embedding for the INSERT
    insert_calls = [call for call in mock_conn.execute.call_args_list if "INSERT INTO section_embeddings" in call[0][0]]
    assert len(insert_calls) == len(sample_segments)
    
    # Check the parameters passed to each INSERT call
    for i, insert_call in enumerate(insert_calls):
        sql, params = insert_call[0]
        assert "VALUES (?, ?, ?, ?, ?, ?, ?, ?)" in sql # 8 columns
        assert params == expected_sql_params[i]
        
    # Check the returned IDs match the generated UUIDs
    assert embedding_ids == test_uuids


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