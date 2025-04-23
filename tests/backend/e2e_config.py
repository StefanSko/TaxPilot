"""
End-to-end test configuration utilities.

This module provides configuration and setup utilities for end-to-end tests.
"""

import os
import tempfile
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Default paths for test data
TEST_DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "test_data"
TEST_LAW_XML = TEST_DATA_DIR / "estg_sample.xml"
TEST_DB_PATH = ":memory:"  # Use in-memory database for tests

# Vector database configuration
VECTOR_DB_IN_MEMORY = True
VECTOR_DB_COLLECTION = "test_law_sections"
VECTOR_DB_DIMENSION = 768  # Default dimension for embeddings

# Test configuration for small subset
TEST_CONFIG = {
    "laws": ["estg"],
    "law_names": {
        "estg": "Einkommensteuergesetz"
    },
    "law_abbreviations": {
        "estg": "EStG"
    },
    "sample_sections": ["13", "15"],  # Sections to include in test
    "embedding_model": "test-embedding-model"  # Model name for test
}


def get_test_db_path() -> str:
    """Get the path to the test database."""
    if TEST_DB_PATH == ":memory:":
        return TEST_DB_PATH
    
    # Create a temp directory for test database if needed
    if not isinstance(TEST_DB_PATH, str):
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")
        return db_path
    
    return TEST_DB_PATH


def get_vector_db_config() -> dict:
    """Get vector database configuration for tests."""
    from taxpilot.backend.search.vector_db import VectorDbProvider
    return {
        "provider": VectorDbProvider.MEMORY if VECTOR_DB_IN_MEMORY else VectorDbProvider.QDRANT,
        "collection_name": VECTOR_DB_COLLECTION,
        "embedding_dim": VECTOR_DB_DIMENSION
    }