"""
Tests for the GermanLawFinder database module.

This module tests the database connection, schema, and CRUD operations.
"""

import os
import tempfile
from pathlib import Path
import pytest
import duckdb

from taxpilot.backend.data_processing.database import (
    get_connection, close_connection, initialize_database,
    insert_law, get_law, get_all_laws,
    insert_section, get_section, get_sections_by_law,
    insert_section_embedding, get_section_embedding
)
from tests.backend.data_processing.test_sample_data import (
    get_sample_laws, get_sample_estg_sections, get_sample_ao_sections, get_sample_embeddings
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Set the environment variable to use the temporary path
        db_path = os.path.join(tmpdirname, "test.duckdb")
        
        # Store the original value to restore later
        original_modal_env = os.environ.get("MODAL_ENVIRONMENT")
        
        # Set environment variables for testing
        os.environ["MODAL_ENVIRONMENT"] = None
        os.environ["TEST_DB_PATH"] = db_path
        
        # Patch the get_connection function to use the test path
        original_get_connection = get_connection.__globals__["get_connection"]
        
        def mock_get_connection(config=None):
            # For testing, always use the temporary path
            return duckdb.connect(db_path)
        
        get_connection.__globals__["get_connection"] = mock_get_connection
        
        try:
            yield db_path
        finally:
            # Reset global connection
            get_connection.__globals__["_connection"] = None
            
            # Restore the original function
            get_connection.__globals__["get_connection"] = original_get_connection
            
            # Restore environment variables
            if original_modal_env is not None:
                os.environ["MODAL_ENVIRONMENT"] = original_modal_env
            else:
                os.environ.pop("MODAL_ENVIRONMENT", None)
            
            os.environ.pop("TEST_DB_PATH", None)


@pytest.fixture
def db_setup(temp_db_path):
    """Initialize the database schema for testing."""
    # Initialize the database
    initialize_database()
    
    # Return the connection for testing
    conn = get_connection()
    
    try:
        yield conn
    finally:
        close_connection()


@pytest.fixture
def sample_laws():
    """Get sample law data for testing."""
    return get_sample_laws()


@pytest.fixture
def sample_sections():
    """Get all sample section data for testing."""
    return get_sample_estg_sections() + get_sample_ao_sections()


def test_initialize_database(db_setup):
    """Test that the database initializes with the correct schema."""
    conn = db_setup
    
    # Check if laws table was created
    result = conn.execute("""
    SELECT name FROM sqlite_master WHERE type='table' AND name='laws'
    """).fetchone()
    assert result is not None
    
    # Check if sections table was created
    result = conn.execute("""
    SELECT name FROM sqlite_master WHERE type='table' AND name='sections'
    """).fetchone()
    assert result is not None
    
    # Check if section_embeddings table was created
    result = conn.execute("""
    SELECT name FROM sqlite_master WHERE type='table' AND name='section_embeddings'
    """).fetchone()
    assert result is not None


def test_insert_and_get_law(db_setup, sample_laws):
    """Test inserting and retrieving a law."""
    # Insert a sample law
    sample_law = sample_laws[0]
    insert_law(sample_law)
    
    # Retrieve the law
    retrieved_law = get_law(sample_law["id"])
    
    # Check that the retrieved law matches the sample
    assert retrieved_law is not None
    assert retrieved_law["id"] == sample_law["id"]
    assert retrieved_law["full_name"] == sample_law["full_name"]
    assert retrieved_law["abbreviation"] == sample_law["abbreviation"]
    assert retrieved_law["last_updated"] == sample_law["last_updated"]
    assert retrieved_law["issue_date"] == sample_law["issue_date"]
    assert retrieved_law["status_info"] == sample_law["status_info"]
    assert retrieved_law["metadata"] == sample_law["metadata"]


def test_get_all_laws(db_setup, sample_laws):
    """Test retrieving all laws."""
    # Insert all sample laws
    for law in sample_laws:
        insert_law(law)
    
    # Retrieve all laws
    retrieved_laws = get_all_laws()
    
    # Check that all laws were retrieved
    assert len(retrieved_laws) == len(sample_laws)
    
    # Check that the laws match by ID
    retrieved_ids = {law["id"] for law in retrieved_laws}
    sample_ids = {law["id"] for law in sample_laws}
    assert retrieved_ids == sample_ids


def test_insert_and_get_section(db_setup, sample_laws, sample_sections):
    """Test inserting and retrieving a section."""
    # Insert a sample law first (for foreign key constraint)
    insert_law(sample_laws[0])
    
    # Insert a sample section
    sample_section = next(s for s in sample_sections if s["law_id"] == sample_laws[0]["id"])
    insert_section(sample_section)
    
    # Retrieve the section
    retrieved_section = get_section(sample_section["id"])
    
    # Check that the retrieved section matches the sample
    assert retrieved_section is not None
    assert retrieved_section["id"] == sample_section["id"]
    assert retrieved_section["law_id"] == sample_section["law_id"]
    assert retrieved_section["section_number"] == sample_section["section_number"]
    assert retrieved_section["title"] == sample_section["title"]
    assert retrieved_section["content"] == sample_section["content"]
    assert retrieved_section["parent_section_id"] == sample_section["parent_section_id"]
    assert retrieved_section["hierarchy_level"] == sample_section["hierarchy_level"]
    assert retrieved_section["path"] == sample_section["path"]
    assert retrieved_section["metadata"] == sample_section["metadata"]


def test_get_sections_by_law(db_setup, sample_laws, sample_sections):
    """Test retrieving sections by law."""
    # Insert both sample laws
    for law in sample_laws:
        insert_law(law)
    
    # Insert all sample sections
    for section in sample_sections:
        insert_section(section)
    
    # Get the first law
    law_id = sample_laws[0]["id"]
    
    # Count how many sections should be associated with this law
    expected_count = sum(1 for s in sample_sections if s["law_id"] == law_id)
    
    # Retrieve sections for the law
    retrieved_sections = get_sections_by_law(law_id)
    
    # Check the count
    assert len(retrieved_sections) == expected_count
    
    # Check that all retrieved sections have the correct law_id
    for section in retrieved_sections:
        assert section["law_id"] == law_id


def test_insert_and_get_embedding(db_setup, sample_laws, sample_sections):
    """Test inserting and retrieving section embeddings."""
    # Insert a sample law
    insert_law(sample_laws[0])
    
    # Insert a sample section
    sample_section = next(s for s in sample_sections if s["law_id"] == sample_laws[0]["id"])
    insert_section(sample_section)
    
    # Generate and insert a sample embedding
    sample_embeddings = get_sample_embeddings([sample_section["id"]])
    sample_embedding = sample_embeddings[0]
    insert_section_embedding(sample_embedding)
    
    # Retrieve the embedding
    retrieved_embedding = get_section_embedding(sample_section["id"])
    
    # Check that the retrieved embedding matches the sample
    assert retrieved_embedding is not None
    assert retrieved_embedding["section_id"] == sample_embedding["section_id"]
    assert len(retrieved_embedding["embedding"]) == len(sample_embedding["embedding"])
    # Check a few values in the embedding vector
    assert retrieved_embedding["embedding"][0] == sample_embedding["embedding"][0]
    assert retrieved_embedding["embedding"][10] == sample_embedding["embedding"][10]
    assert retrieved_embedding["embedding"][-1] == sample_embedding["embedding"][-1]