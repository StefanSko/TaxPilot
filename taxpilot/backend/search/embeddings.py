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
from typing import Protocol, runtime_checkable, Any, Dict, List, Optional, Tuple, Union, cast
import uuid

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

from taxpilot.backend.search.segmentation import TextSegment
from taxpilot.backend.data_processing.database import get_connection, DbConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("embeddings.log"), logging.StreamHandler()],
)
logger = logging.getLogger("embeddings")


def _get_default_device() -> str:
    """Determine the default device based on availability (MPS, CUDA, CPU)."""
    if torch.backends.mps.is_available():
        logger.info("MPS backend is available. Using MPS device.")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("CUDA is available. Using CUDA device.")
        return "cuda"
    else:
        logger.info("MPS and CUDA not available. Using CPU device.")
        return "cpu"


class EmbeddingModelType(Enum):
    """Types of embedding models available for use."""
    
    # Modern and general purpose models
    MODERN_BERT = "answerdotai/ModernBERT-base"
    
    # Generic German language models
    GERMAN_BERT = "deepset/gbert-base"
    GERMAN_BERT_LARGE = "deepset/gbert-large"
    
    # Multilingual models with German support
    MULTILINGUAL_E5 = "intfloat/multilingual-e5-large"
    MULTILINGUAL_MINILM = "microsoft/Multilingual-MiniLM-L12-H384"
    
    # Legal domain specialized models
    LEGAL_BERT = "nlpaueb/legal-bert-base-uncased"  # English legal, but transferable
    LEGAL_GERMAN = "deepset/gelectra-large"  # German model fine-tunable for legal
    
    # Default model for our application
    DEFAULT = "deepset/gbert-base"  # German language model


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding process."""
    
    model_name: str = EmbeddingModelType.DEFAULT.value
    model_version: str = "1.0.0"
    embedding_dim: int = 768  # Default dimension for base models
    batch_size: int = 32
    use_accelerator: bool = True # Flag to enable MPS or CUDA if available
    cache_dir: Path = Path("./model_cache")
    device: str = field(default_factory=_get_default_device)
    normalize_embeddings: bool = True
    max_seq_length: int = 512
    pooling_strategy: str = "mean"
    db_config: DbConfig | None = None
    
    def __post_init__(self):
        """Initialize after the instance is created."""
        # Override device if use_accelerator is False
        if not self.use_accelerator:
            self.device = "cpu"
            logger.info("Accelerator usage disabled. Forcing CPU.")
            
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set embedding dimension based on model (simple check)
        if "large" in self.model_name.lower() and self.embedding_dim == 768:
            self.embedding_dim = 1024  # Large models often have higher dimensions
        elif self.model_name == EmbeddingModelType.MULTILINGUAL_MINILM.value:
             self.embedding_dim = 384 # MiniLM has smaller dimension


@dataclass
class TextEmbedding:
    """An embedding of a text segment with metadata."""
    
    vector: np.ndarray
    segment_id: str
    law_id: str
    section_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_model: str = ""
    embedding_version: str = ""


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    
    def encode(self, texts: list[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        ...


def mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling of token embeddings weighted by attention mask.
    
    Args:
        model_output: Model last_hidden_state from transformers model
        attention_mask: Attention mask from tokenizer
        
    Returns:
        Mean pooled embeddings
    """
    # First element of model_output contains token embeddings
    token_embeddings = model_output[0]
    
    # Expand attention mask to same dimension as token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum token embeddings weighted by attention mask
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Average embeddings by dividing by mask sum
    return sum_embeddings / sum_mask


class ModelCache:
    """Cache for embedding models to avoid reloading."""
    
    _instance = None
    _models: dict[str, tuple[Any, Any]] = {}  # (model, tokenizer)
    _model_metadata: dict[str, dict[str, Any]] = {}
    
    def __new__(cls):
        """Singleton pattern to ensure only one model cache exists."""
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._models = {}
            cls._instance._model_metadata = {}
        return cls._instance
    
    def get_model(self, model_name: str, config: EmbeddingConfig) -> tuple[Any, Any]:
        """
        Get a model from cache or load it if not available.
        
        Args:
            model_name: Name of the model to load
            config: Embedding configuration
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Generate a unique key for the model based on name and config
        model_key = self._get_model_key(model_name, config)
        
        if model_key not in self._models:
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()
            
            try:
                # Load model with appropriate device and cache settings
                # Note: For ModernBERT, we need transformers v4.48.0+
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(config.cache_dir)
                )
                
                # Check if flash attention is available
                use_flash_attention = False
                try:
                    import flash_attn
                    use_flash_attention = True
                    logger.info("Flash Attention 2 is available and will be used")
                except ImportError:
                    logger.info("Flash Attention not available, using standard attention")

                model_config = AutoConfig.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name, config=model_config)
                
                # Determine appropriate dtype (try fp16 on accelerators)
                dtype = torch.float32
                if config.use_accelerator and config.device != 'cpu':
                     # Check if the device supports float16 - MPS support can be tricky
                     # For now, let's assume it does, but could add more checks later
                     dtype = torch.float16
                     logger.info(f"Using {dtype} on device {config.device}")
                else:
                    logger.info(f"Using float32 on device {config.device}")

                # Reload model with potentially different dtype if necessary
                # This is a bit inefficient but ensures compatibility
                if model.dtype != dtype:
                    del model # Release memory
                    torch.cuda.empty_cache() if config.device == 'cuda' else None # Clear CUDA cache if applicable
                    model = AutoModel.from_pretrained(
                        model_name,
                        config=model_config,
                        torch_dtype=dtype # Set the determined dtype
                    )
                    logger.info(f"Reloaded model with dtype {dtype}")

                # Move model to specified device
                model = model.to(config.device)
                embedding_dim = model_config.hidden_size
                
                # Store model and its metadata
                self._models[model_key] = (model, tokenizer)
                self._model_metadata[model_key] = {
                    "name": model_name,
                    "version": config.model_version,
                    "embedding_dim": embedding_dim,
                    "load_time": time.time() - start_time,
                    "cached": True,
                    "device": config.device,
                    "using_flash_attention": use_flash_attention
                }
                
                logger.info(
                    f"Model loaded in {time.time() - start_time:.2f}s. "
                    f"Dimension: {embedding_dim}"
                )
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                raise
        
        return self._models[model_key]
    
    def _get_model_key(self, model_name: str, config: EmbeddingConfig) -> str:
        """Generate a unique key for a model based on name and configuration."""
        key_components = f"{model_name}_{config.model_version}_{config.device}_{config.max_seq_length}"
        return hashlib.md5(key_components.encode()).hexdigest()
    
    def list_cached_models(self) -> list[dict[str, Any]]:
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
        self.model, self.tokenizer = self.model_cache.get_model(self.config.model_name, self.config)
        
        logger.info(
            f"Initialized TextEmbedder with model {self.config.model_name} "
            f"running on {self.config.device}"
        )
    
    def encode(self, texts: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings using Hugging Face transformers.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            normalize: Whether to normalize vectors to unit length
            
        Returns:
            NumPy array of embeddings
        """
        # Handle empty input
        if not texts:
            return np.array([])
        
        # Use the same device as the model
        device = next(self.model.parameters()).device
        
        all_embeddings = []
        
        # Process in batches to avoid OOM errors
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and move to device
            # Note: ModernBERT does not use token_type_ids
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            ).to(device)
            
            # Remove token_type_ids if present (not used by ModernBERT)
            if 'token_type_ids' in encoded_input and 'modernbert' in self.config.model_name.lower():
                encoded_input.pop('token_type_ids')
            
            # Get model embeddings without gradient calculations
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Apply mean pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings if requested
            if normalize:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # Move to CPU and convert to NumPy
            all_embeddings.append(sentence_embeddings.cpu().numpy())
        
        # Concatenate all batches
        return np.vstack(all_embeddings)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Encode the text and return the embedding
        embeddings = self.encode(
            [text],
            batch_size=1,
            normalize=self.config.normalize_embeddings
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
        return self.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize=self.config.normalize_embeddings
        )
    
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
        """Initialize database tables for storing embeddings if they don't exist."""
        conn = get_connection(self.config.db_config)
        
        try:
            # logger.info("Dropping existing section_embeddings table (if exists)...")
            # conn.execute("DROP TABLE IF EXISTS section_embeddings") # Do not drop if we want to reuse
            
            # Create section_embeddings table IF NOT EXISTS
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS section_embeddings (
                id VARCHAR PRIMARY KEY,
                law_id VARCHAR,
                section_id VARCHAR,
                segment_id VARCHAR,
                embedding_model VARCHAR,
                embedding_version VARCHAR,
                embedding FLOAT[{self.config.embedding_dim}],  -- Use native FLOAT array
                vector_db_id VARCHAR, 
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            conn.execute(create_table_sql)
            logger.info(f"Ensured section_embeddings table exists with embedding dimension {self.config.embedding_dim}.")
            
            # Create indexes IF NOT EXISTS
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_section_embeddings_law_id ON section_embeddings(law_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_section_embeddings_section_id ON section_embeddings(section_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_section_embeddings_segment_id ON section_embeddings(segment_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_section_embeddings_model_version ON section_embeddings(embedding_model, embedding_version)"
            )

            logger.info("Embedding database tables and indexes initialized/verified")
        except Exception as e:
            logger.error(f"Error initializing embedding database: {e}")
            raise

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
                # Use a generated UUID as the primary key for uniqueness
                embedding_id = str(uuid.uuid4())
                
                # Convert NumPy vector to list for storage
                vector_list = embedding.vector.tolist()
                
                # No need to check for existing records when using UUIDs
                # Always insert new embedding with unique ID
                conn.execute(
                    """
                    INSERT INTO section_embeddings (
                        id, law_id, section_id, segment_id, 
                        embedding_model, embedding_version, 
                        embedding, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        embedding_id, # Use generated UUID
                        embedding.law_id,
                        embedding.section_id,
                        embedding.segment_id, # Store original segment ID
                        embedding.embedding_model,
                        embedding.embedding_version,
                        vector_list, # Insert the list directly
                        json.dumps(embedding.metadata),
                    ),
                )
                embedding_ids.append(embedding_id)
            
            logger.info(f"Stored {len(embeddings)} embeddings in database")
            return embedding_ids
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            raise
        # Don't close the connection here to avoid connection issues
        # The connection will be managed by the global connection pool

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
            
            # Deserialize the embedding vector from list
            vector = np.array(result[6], dtype=np.float32)
            
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
        except Exception as e:
            logger.error(f"Error retrieving embedding {embedding_id}: {str(e)}")
            return None
        # Don't close connection - managed by connection pool
    
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
                # Deserialize the embedding vector from list
                vector = np.array(result[6], dtype=np.float32)
                
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
        except Exception as e:
            logger.error(f"Error retrieving embeddings for law {law_id}: {str(e)}")
            return []
        # Don't close connection - managed by connection pool
    
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
        except Exception as e:
            logger.error(f"Error deleting embeddings for law {law_id}: {str(e)}")
            return 0
        # Don't close connection - managed by connection pool


def initialize_model_for_modal():
    """
    Initialize and warm up model for Modal.com serverless environment.
    
    This function is meant to be called during the container build phase
    to download and cache the model.
    """
    config = EmbeddingConfig(
        model_name=os.environ.get("EMBEDDING_MODEL", EmbeddingModelType.DEFAULT.value),
        use_accelerator=os.environ.get("USE_GPU", "false").lower() == "true",
    )
    
    # Initialize the embedder to download and cache the model
    embedder = TextEmbedder(config)
    
    # Warm up the model with a sample text
    sample_text = "Dieses ist ein Beispieltext, um das Embedding-Modell zu initialisieren."
    _ = embedder.embed_text(sample_text)
    
    logger.info(f"Model {config.model_name} initialized and ready for serverless execution")


# Default model changed to deepset/gbert-base
DEFAULT_MODEL_NAME = "deepset/gbert-base"
DEFAULT_MODEL_VERSION = "v1"
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CACHE_DIR = Path(os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface")))
DEFAULT_MAX_SEQ_LENGTH = 384  # gbert-base has context window of 512, but we use a bit less to be safe