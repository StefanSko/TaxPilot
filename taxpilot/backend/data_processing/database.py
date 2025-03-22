"""
Database module for GermanLawFinder.

This module provides database connection, initialization, and schema management
functions for working with DuckDB in a serverless environment.
"""

import os
from pathlib import Path
from typing import TypedDict, NotRequired, Any, cast
import duckdb
from datetime import date


# Type definitions using Python 3.12 typing features
class Law(TypedDict):
    """Type representing a law record."""
    id: str
    full_name: str
    abbreviation: str
    last_updated: date
    issue_date: date
    status_info: str
    metadata: dict[str, Any]


class Section(TypedDict):
    """Type representing a section record."""
    id: str
    law_id: str
    section_number: str
    title: str
    content: str
    parent_section_id: str | None
    hierarchy_level: int
    path: str
    metadata: dict[str, Any]


class SectionEmbedding(TypedDict):
    """Type representing a section embedding record."""
    section_id: str
    embedding: list[float]


# Optional database configuration
class DbConfig(TypedDict):
    """Database configuration options."""
    read_only: NotRequired[bool]
    memory_limit: NotRequired[str]


# Database singleton connection
_connection: duckdb.DuckDBPyConnection | None = None


def get_connection(config: DbConfig | None = None) -> duckdb.DuckDBPyConnection:
    """
    Get a connection to the DuckDB database.
    
    This function implements a singleton pattern to reuse the database connection
    across multiple function invocations in the serverless environment.
    
    Args:
        config: Optional configuration parameters for DuckDB
    
    Returns:
        A DuckDB connection object.
    """
    global _connection
    
    if _connection is None:
        # Determine the database path based on environment
        if os.getenv("MODAL_ENVIRONMENT") == "modal":
            # In Modal.com environment, use the mounted volume path
            db_path = "/data/germanlawfinder.duckdb"
        else:
            # In local development, use a local path
            db_path = "data/processed/germanlawfinder.duckdb"
            
            # Ensure the directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Apply configuration options
        config_dict = {} if config is None else config
        
        # Connect to the database
        _connection = duckdb.connect(db_path, **config_dict)
    
    return _connection


def close_connection() -> None:
    """Close the database connection if it exists."""
    global _connection
    
    if _connection is not None:
        _connection.close()
        _connection = None


def initialize_database() -> None:
    """
    Initialize the database schema.
    
    This function creates the necessary tables if they don't exist.
    """
    conn = get_connection()
    
    # Create the laws table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS laws (
        id VARCHAR PRIMARY KEY,               -- e.g., "estg"
        full_name VARCHAR,                    -- e.g., "Einkommensteuergesetz"
        abbreviation VARCHAR,                 -- e.g., "EStG"
        last_updated DATE,                    -- Last modification date
        issue_date DATE,                      -- Original issue date
        status_info VARCHAR,                  -- Current status information
        metadata JSON                         -- Additional metadata as JSON
    )
    """)
    
    # Create the sections table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS sections (
        id VARCHAR PRIMARY KEY,               -- e.g., "estg_2" for § 2
        law_id VARCHAR,                       -- Foreign key to laws table
        section_number VARCHAR,               -- e.g., "2"
        title VARCHAR,                        -- Section title
        content TEXT,                         -- Full text content
        parent_section_id VARCHAR,            -- For hierarchical structure
        hierarchy_level INTEGER,              -- Depth in document structure
        path VARCHAR,                         -- Full hierarchical path
        metadata JSON,                        -- Additional metadata
        FOREIGN KEY (law_id) REFERENCES laws(id)
    )
    """)
    
    # Create the section_embeddings table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS section_embeddings (
        section_id VARCHAR,
        embedding FLOAT[384],                 -- Vector embedding (dimension depends on model)
        PRIMARY KEY (section_id),
        FOREIGN KEY (section_id) REFERENCES sections(id)
    )
    """)


def run_migration(version: int) -> None:
    """
    Run a database migration to update the schema.
    
    Args:
        version: The migration version to run.
    """
    conn = get_connection()
    
    # Check if the migrations table exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS migrations (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Check if the migration has already been applied
    result = conn.execute(f"SELECT version FROM migrations WHERE version = {version}").fetchone()
    if result is not None:
        return
    
    # Apply the migration based on version
    if version == 1:
        # Initial schema creation
        initialize_database()
    elif version == 2:
        # Example future migration
        conn.execute("ALTER TABLE laws ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true")
    
    # Record that the migration was applied
    conn.execute(f"INSERT INTO migrations (version) VALUES ({version})")


def get_db_version() -> int:
    """
    Get the current database schema version.
    
    Returns:
        The highest migration version that has been applied.
    """
    conn = get_connection()
    
    # Check if the migrations table exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS migrations (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Get the highest migration version
    result = conn.execute("SELECT MAX(version) FROM migrations").fetchone()
    return result[0] if result[0] is not None else 0


# CRUD operations for laws
def insert_law(law: Law) -> None:
    """
    Insert a new law into the database.
    
    Args:
        law: The law data to insert.
    """
    conn = get_connection()
    
    conn.execute("""
    INSERT INTO laws (id, full_name, abbreviation, last_updated, issue_date, status_info, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        law["id"],
        law["full_name"],
        law["abbreviation"],
        law["last_updated"],
        law["issue_date"],
        law["status_info"],
        law["metadata"]
    ))


def get_law(law_id: str) -> Law | None:
    """
    Get a law by ID.
    
    Args:
        law_id: The ID of the law to retrieve.
        
    Returns:
        The law data or None if not found.
    """
    conn = get_connection()
    
    result = conn.execute("""
    SELECT id, full_name, abbreviation, last_updated, issue_date, status_info, metadata
    FROM laws
    WHERE id = ?
    """, (law_id,)).fetchone()
    
    if result is None:
        return None
    
    return cast(Law, {
        "id": result[0],
        "full_name": result[1],
        "abbreviation": result[2],
        "last_updated": result[3],
        "issue_date": result[4],
        "status_info": result[5],
        "metadata": result[6]
    })


def get_all_laws() -> list[Law]:
    """
    Get all laws from the database.
    
    Returns:
        A list of all laws.
    """
    conn = get_connection()
    
    results = conn.execute("""
    SELECT id, full_name, abbreviation, last_updated, issue_date, status_info, metadata
    FROM laws
    ORDER BY abbreviation
    """).fetchall()
    
    return [cast(Law, {
        "id": row[0],
        "full_name": row[1],
        "abbreviation": row[2],
        "last_updated": row[3],
        "issue_date": row[4],
        "status_info": row[5],
        "metadata": row[6]
    }) for row in results]


# CRUD operations for sections
def insert_section(section: Section) -> None:
    """
    Insert a new section into the database.
    
    Args:
        section: The section data to insert.
    """
    conn = get_connection()
    
    conn.execute("""
    INSERT INTO sections (
        id, law_id, section_number, title, content, 
        parent_section_id, hierarchy_level, path, metadata
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        section["id"],
        section["law_id"],
        section["section_number"],
        section["title"],
        section["content"],
        section["parent_section_id"],
        section["hierarchy_level"],
        section["path"],
        section["metadata"]
    ))


def get_section(section_id: str) -> Section | None:
    """
    Get a section by ID.
    
    Args:
        section_id: The ID of the section to retrieve.
        
    Returns:
        The section data or None if not found.
    """
    conn = get_connection()
    
    result = conn.execute("""
    SELECT id, law_id, section_number, title, content, 
           parent_section_id, hierarchy_level, path, metadata
    FROM sections
    WHERE id = ?
    """, (section_id,)).fetchone()
    
    if result is None:
        return None
    
    return cast(Section, {
        "id": result[0],
        "law_id": result[1],
        "section_number": result[2],
        "title": result[3],
        "content": result[4],
        "parent_section_id": result[5],
        "hierarchy_level": result[6],
        "path": result[7],
        "metadata": result[8]
    })


def get_sections_by_law(law_id: str) -> list[Section]:
    """
    Get all sections for a specific law.
    
    Args:
        law_id: The ID of the law to retrieve sections for.
        
    Returns:
        A list of sections for the specified law.
    """
    conn = get_connection()
    
    results = conn.execute("""
    SELECT id, law_id, section_number, title, content, 
           parent_section_id, hierarchy_level, path, metadata
    FROM sections
    WHERE law_id = ?
    ORDER BY section_number
    """, (law_id,)).fetchall()
    
    return [cast(Section, {
        "id": row[0],
        "law_id": row[1],
        "section_number": row[2],
        "title": row[3],
        "content": row[4],
        "parent_section_id": row[5],
        "hierarchy_level": row[6],
        "path": row[7],
        "metadata": row[8]
    }) for row in results]


# Operations for section embeddings
def insert_section_embedding(embedding: SectionEmbedding) -> None:
    """
    Insert a section embedding into the database.
    
    Args:
        embedding: The section embedding data to insert.
    """
    conn = get_connection()
    
    conn.execute("""
    INSERT INTO section_embeddings (section_id, embedding)
    VALUES (?, ?)
    """, (
        embedding["section_id"],
        embedding["embedding"]
    ))


def get_section_embedding(section_id: str) -> SectionEmbedding | None:
    """
    Get the embedding for a specific section.
    
    Args:
        section_id: The ID of the section to retrieve the embedding for.
        
    Returns:
        The section embedding or None if not found.
    """
    conn = get_connection()
    
    result = conn.execute("""
    SELECT section_id, embedding
    FROM section_embeddings
    WHERE section_id = ?
    """, (section_id,)).fetchone()
    
    if result is None:
        return None
    
    return cast(SectionEmbedding, {
        "section_id": result[0],
        "embedding": result[1]
    })


# Function decorator using Python 3.12 typing
def with_connection(func: callable) -> callable:
    """
    Decorator to ensure a database connection is available for a function.
    
    Args:
        func: The function to wrap.
        
    Returns:
        The wrapped function.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            # Ensure database is initialized
            if get_db_version() == 0:
                run_migration(1)
            return func(*args, **kwargs)
        finally:
            # No need to close connection in serverless environment
            # as it will be reused across function invocations
            pass
    
    return wrapper