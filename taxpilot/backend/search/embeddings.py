"""
Vector embedding generation for German legal text.

This module provides functionality to generate vector embeddings for
German legal text segments, optimized for the legal domain and Modal.com serverless environment.
"""

import os
import json
import time
import hashlib
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from taxpilot.backend.search.segmentation import TextSegment
from taxpilot.backend.data_processing.database import get_connection, DbConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("embeddings.log"), logging.StreamHandler()],
)
logger = logging.getLogger("embeddings")


class EmbeddingModelType(Enum):
    """Types of embedding models available for use."""
    
    # Generic German language models
    GERMAN_BERT = "deepset/gbert-base"
    GERMAN_BERT_LARGE = "deepset/gbert-large"
    
    # Multilingual models with German support
    MULTILINGUAL_E5 = "intfloat/multilingual-e5-large"
    MULTILINGUAL_MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Legal domain specialized models
    LEGAL_BERT = "nlpaueb/legal-bert-base-uncased"  # English legal, but transferable
    LEGAL_GERMAN = "deepset/gelectra-large"  # German model fine-tunable for legal
    
    # Default model for our application
    DEFAULT = "deepset/gbert-base"


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding process."""
    
    model_name: str = EmbeddingModelType.DEFAULT.value
    model_version: str = "1.0.0"
    embedding_dim: int = 768  # Default dimension for base models
    batch_size: int = 32
    use_gpu: bool = torch.cuda.is_available()
    cache_dir: Path = Path("./model_cache")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_embeddings: bool = True
    max_seq_length: int = 512
    pooling_strategy: str = "mean"
    db_config: DbConfig | None = None
    
    def __post_init__(self):
        """Initialize after the instance is created."""
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set embedding dimension based on model
        if "large" in self.model_name.lower() and self.embedding_dim == 768:
            self.embedding_dim = 1024  # Large models often have higher dimensions


@dataclass
class TextEmbedding:
    """An embedding of a text segment with metadata."""
    
    vector: np.ndarray
    segment_id: str
    law_id: str
    section_id: str
    metadata: dict[str, any] = field(default_factory=dict)
    embedding_model: str = ""
    embedding_version: str = ""


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    
    def encode(self, texts: list[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        ...


class ModelCache:
    """Cache for embedding models to avoid reloading."""
    
    _instance = None
    _models: dict[str, EmbeddingModel] = {}
    _model_metadata: dict[str, dict] = {}
    
    def __new__(cls):
        """Singleton pattern to ensure only one model cache exists."""
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._models = {}
            cls._instance._model_metadata = {}
        return cls._instance
    
    def get_model(self, model_name: str, config: EmbeddingConfig) -> EmbeddingModel:
        """
        Get a model from cache or load it if not available.
        
        Args:
            model_name: Name of the model to load
            config: Embedding configuration
            
        Returns:
            The loaded model
        """
        # Generate a unique key for the model based on name and config
        model_key = self._get_model_key(model_name, config)
        
        if model_key not in self._models:
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()
            
            try:
                # Load model with appropriate device and cache settings
                model = SentenceTransformer(
                    model_name_or_path=model_name,
                    device=config.device,
                    cache_folder=str(config.cache_dir),
                )
                
                # Configure model settings
                model.max_seq_length = config.max_seq_length
                
                # Store model and its metadata
                self._models[model_key] = model
                self._model_metadata[model_key] = {
                    "name": model_name,
                    "version": config.model_version,
                    "embedding_dim": model.get_sentence_embedding_dimension(),
                    "load_time": time.time() - start_time,
                    "cached": True,
                    "device": config.device,
                }
                
                logger.info(
                    f"Model loaded in {time.time() - start_time:.2f}s. "
                    f"Dimension: {model.get_sentence_embedding_dimension()}"
                )
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                raise
        
        return self._models[model_key]
    
    def _get_model_key(self, model_name: str, config: EmbeddingConfig) -> str:
        """Generate a unique key for a model based on name and configuration."""
        key_components = f"{model_name}_{config.model_version}_{config.device}_{config.max_seq_length}"
        return hashlib.md5(key_components.encode()).hexdigest()
    
    def list_cached_models(self) -> list[dict]:
        """List all cached models with their metadata."""
        return list(self._model_metadata.values())
    
    def clear_cache(self, model_name: str | None = None):
        """
        Clear model cache.
        
        Args:
            model_name: Optional model name to clear. If None, clear all.
        """
        if model_name is None:
            logger.info("Clearing all models from cache")
            self._models.clear()
            self._model_metadata.clear()
        else:
            keys_to_remove = [
                k for k, v in self._model_metadata.items() if v["name"] == model_name
            ]
            for key in keys_to_remove:
                logger.info(f"Removing model {model_name} from cache")
                if key in self._models:
                    del self._models[key]
                if key in self._model_metadata:
                    del self._model_metadata[key]


class TextEmbedder:
    """Text embedding generator for legal text segments."""
    
    def __init__(self, config: EmbeddingConfig | None = None):
        """
        Initialize the embedder.
        
        Args:
            config: Optional embedding configuration.
        """
        self.config = config or EmbeddingConfig()
        self.model_cache = ModelCache()
        self.model = self.model_cache.get_model(self.config.model_name, self.config)
        
        logger.info(
            f"Initialized TextEmbedder with model {self.config.model_name} "
            f"running on {self.config.device}"
        )
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Encode the text and return the embedding
        embeddings = self.model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return embeddings[0]
    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embedding vectors as numpy array
        """
        if not texts:
            return np.array([])
        
        # Encode all texts in efficient batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return embeddings
    
    def embed_segments(self, segments: list[TextSegment]) -> list[TextEmbedding]:
        """
        Generate embeddings for text segments.
        
        Args:
            segments: List of text segments to embed
            
        Returns:
            List of embeddings with metadata
        """
        if not segments:
            return []
        
        # Extract text from segments
        texts = [segment.text for segment in segments]
        
        # Generate embeddings
        start_time = time.time()
        vectors = self.embed_texts(texts)
        end_time = time.time()
        
        logger.info(
            f"Generated {len(vectors)} embeddings in {end_time - start_time:.2f}s "
            f"({len(vectors) / (end_time - start_time):.2f} embeddings/s)"
        )
        
        # Create embedding objects with metadata
        embeddings = []
        for i, segment in enumerate(segments):
            embedding = TextEmbedding(
                vector=vectors[i],
                segment_id=segment.segment_id,
                law_id=segment.law_id,
                section_id=segment.section_id,
                metadata={
                    **segment.metadata,
                    "start_idx": segment.start_idx,
                    "end_idx": segment.end_idx,
                    "text_length": len(segment.text),
                },
                embedding_model=self.config.model_name,
                embedding_version=self.config.model_version,
            )
            embeddings.append(embedding)
        
        return embeddings


class EmbeddingProcessor:
    """Processor for generating and storing embeddings for legal segments."""
    
    def __init__(self, config: EmbeddingConfig | None = None):
        """
        Initialize the embedding processor.
        
        Args:
            config: Optional embedding configuration.
        """
        self.config = config or EmbeddingConfig()
        self.embedder = TextEmbedder(config)
        
        # Ensure DB config is set
        if self.config.db_config is None:
            self.config.db_config = DbConfig()
        
        # Initialize database tables
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database tables for storing embeddings."""
        conn = get_connection(self.config.db_config)
        
        try:
            # Create section_embeddings table if it doesn't exist
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS section_embeddings (
                    id VARCHAR PRIMARY KEY,
                    law_id VARCHAR,
                    section_id VARCHAR,
                    segment_id VARCHAR,
                    embedding_model VARCHAR,
                    embedding_version VARCHAR,
                    embedding BLOB,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
            # Create indexes for efficient retrieval
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_section_embeddings_law_id ON section_embeddings(law_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_section_embeddings_section_id ON section_embeddings(section_id)"
            )
            
            logger.info("Embedding database tables initialized")
        finally:
            conn.close()
    
    def process_segments(self, segments: list[TextSegment]) -> list[str]:
        """
        Process text segments, generate embeddings, and store them.
        
        Args:
            segments: List of text segments to process
            
        Returns:
            List of IDs for the processed embeddings
        """
        if not segments:
            return []
        
        # Generate embeddings
        embeddings = self.embedder.embed_segments(segments)
        
        # Store embeddings in database
        embedding_ids = self._store_embeddings(embeddings)
        
        return embedding_ids
    
    def _store_embeddings(self, embeddings: list[TextEmbedding]) -> list[str]:
        """
        Store embeddings in the database.
        
        Args:
            embeddings: List of embeddings to store
            
        Returns:
            List of embedding IDs
        """
        if not embeddings:
            return []
        
        conn = get_connection(self.config.db_config)
        embedding_ids = []
        
        try:
            for embedding in embeddings:
                # Generate a unique ID for the embedding
                embedding_id = f"{embedding.law_id}_{embedding.segment_id}_{hashlib.md5(embedding.vector.tobytes()).hexdigest()[:8]}"
                
                # Serialize the embedding vector
                vector_bytes = embedding.vector.tobytes()
                
                # Check if this segment already has an embedding with the same model
                existing = conn.execute(
                    """
                    SELECT id FROM section_embeddings 
                    WHERE segment_id = ? AND embedding_model = ? AND embedding_version = ?
                    """,
                    (embedding.segment_id, embedding.embedding_model, embedding.embedding_version),
                ).fetchone()
                
                if existing:
                    # Update existing embedding
                    conn.execute(
                        """
                        UPDATE section_embeddings SET
                        embedding = ?,
                        metadata = ?,
                        created_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (vector_bytes, json.dumps(embedding.metadata), existing[0]),
                    )
                    embedding_ids.append(existing[0])
                else:
                    # Insert new embedding
                    conn.execute(
                        """
                        INSERT INTO section_embeddings (
                            id, law_id, section_id, segment_id, 
                            embedding_model, embedding_version, 
                            embedding, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            embedding_id,
                            embedding.law_id,
                            embedding.section_id,
                            embedding.segment_id,
                            embedding.embedding_model,
                            embedding.embedding_version,
                            vector_bytes,
                            json.dumps(embedding.metadata),
                        ),
                    )
                    embedding_ids.append(embedding_id)
            
            logger.info(f"Stored {len(embeddings)} embeddings in database")
            return embedding_ids
        finally:
            conn.close()
    
    def get_embedding(self, embedding_id: str) -> TextEmbedding | None:
        """
        Retrieve an embedding from the database.
        
        Args:
            embedding_id: ID of the embedding to retrieve
            
        Returns:
            The retrieved embedding, or None if not found
        """
        conn = get_connection(self.config.db_config)
        
        try:
            result = conn.execute(
                """
                SELECT id, law_id, section_id, segment_id, 
                       embedding_model, embedding_version, 
                       embedding, metadata
                FROM section_embeddings
                WHERE id = ?
                """,
                (embedding_id,),
            ).fetchone()
            
            if not result:
                return None
            
            # Deserialize the embedding vector
            vector = np.frombuffer(result[6], dtype=np.float32)
            
            # Create and return the embedding object
            return TextEmbedding(
                vector=vector,
                segment_id=result[3],
                law_id=result[1],
                section_id=result[2],
                metadata=json.loads(result[7]),
                embedding_model=result[4],
                embedding_version=result[5],
            )
        finally:
            conn.close()
    
    def get_law_embeddings(self, law_id: str) -> list[TextEmbedding]:
        """
        Retrieve all embeddings for a specific law.
        
        Args:
            law_id: ID of the law
            
        Returns:
            List of embeddings for the law
        """
        conn = get_connection(self.config.db_config)
        
        try:
            results = conn.execute(
                """
                SELECT id, law_id, section_id, segment_id, 
                       embedding_model, embedding_version, 
                       embedding, metadata
                FROM section_embeddings
                WHERE law_id = ?
                """,
                (law_id,),
            ).fetchall()
            
            embeddings = []
            for result in results:
                # Deserialize the embedding vector
                vector = np.frombuffer(result[6], dtype=np.float32)
                
                # Create the embedding object
                embedding = TextEmbedding(
                    vector=vector,
                    segment_id=result[3],
                    law_id=result[1],
                    section_id=result[2],
                    metadata=json.loads(result[7]),
                    embedding_model=result[4],
                    embedding_version=result[5],
                )
                embeddings.append(embedding)
            
            return embeddings
        finally:
            conn.close()
    
    def delete_law_embeddings(self, law_id: str) -> int:
        """
        Delete all embeddings for a specific law.
        
        Args:
            law_id: ID of the law
            
        Returns:
            Number of embeddings deleted
        """
        conn = get_connection(self.config.db_config)
        
        try:
            cursor = conn.execute(
                "DELETE FROM section_embeddings WHERE law_id = ?", (law_id,)
            )
            deleted_count = cursor.rowcount
            
            logger.info(f"Deleted {deleted_count} embeddings for law {law_id}")
            return deleted_count
        finally:
            conn.close()


def initialize_model_for_modal():
    """
    Initialize and warm up model for Modal.com serverless environment.
    
    This function is meant to be called during the container build phase
    to download and cache the model.
    """
    config = EmbeddingConfig(
        model_name=os.environ.get("EMBEDDING_MODEL", EmbeddingModelType.DEFAULT.value),
        use_gpu=os.environ.get("USE_GPU", "false").lower() == "true",
    )
    
    # Initialize the embedder to download and cache the model
    embedder = TextEmbedder(config)
    
    # Warm up the model with a sample text
    sample_text = "Dieses ist ein Beispieltext, um das Embedding-Modell zu initialisieren."
    _ = embedder.embed_text(sample_text)
    
    logger.info(f"Model {config.model_name} initialized and ready for serverless execution")