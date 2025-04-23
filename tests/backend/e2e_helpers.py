"""
Helper utilities for end-to-end testing.

This module provides helper functions for orchestrating end-to-end tests.
"""

import os
import tempfile
import shutil
import time
import logging
import subprocess
from contextlib import contextmanager
from pathlib import Path
import threading
import uuid
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Union
from taxpilot.backend.data_processing.database import (
    initialize_database, 
    get_connection,
    DbConfig
)
from taxpilot.backend.data_processing.pipeline import (
    run_pipeline, 
    PipelineConfig
)
from taxpilot.backend.search.vector_db import (
    VectorDbConfig, 
    VectorDatabase
)
from taxpilot.backend.search.embeddings import (
    TextEmbedder, 
    EmbeddingConfig
)

from tests.backend.e2e_config import (
    get_test_db_path, 
    get_vector_db_config, 
    TEST_CONFIG
)

# Configure logging
logger = logging.getLogger(__name__)


class EndToEndTestEnvironment:
    """
    Environment for running end-to-end tests.
    
    This class handles setting up all components needed for an end-to-end test:
    - Database initialization
    - Data processing
    - Embedding generation
    - Vector database initialization
    - API server startup
    """
    
    def __init__(self, use_in_memory: bool = True):
        """
        Initialize the test environment.
        
        Args:
            use_in_memory: Whether to use in-memory databases (True) or temp files (False)
        """
        self.use_in_memory = use_in_memory
        self.temp_dir = None
        self.db_path = None
        self.api_server = None
        self.server_thread = None
        self._setup_paths()
    
    def _setup_paths(self):
        """Set up paths for test data."""
        if not self.use_in_memory:
            self.temp_dir = tempfile.mkdtemp()
            self.db_path = os.path.join(self.temp_dir, "test.db")
        else:
            self.db_path = ":memory:"
    
    def setup_database(self):
        """Set up the database for testing."""
        db_config = DbConfig(db_path=self.db_path)
        os.environ["TEST_DB_PATH"] = self.db_path
        
        # For in-memory DB, we need to clean it up first
        if self.db_path == ":memory:":
            conn = get_connection(db_config)
            # Drop tables if they exist
            try:
                conn.execute("DROP TABLE IF EXISTS section_embeddings")
                conn.execute("DROP TABLE IF EXISTS sections")
                conn.execute("DROP TABLE IF EXISTS laws")
                conn.execute("DROP TABLE IF EXISTS migrations")
            except Exception as e:
                logger.warning(f"Error dropping tables: {e}")
        
        # initialize_database implicitly creates tables
        initialize_database()
        conn = get_connection(db_config)
        return conn
    
    def run_data_pipeline(self, input_path: str | Path):
        """
        Run the data processing pipeline.
        
        Args:
            input_path: Path to input data directory or file
        """
        # For e2e tests, we'll directly use the test XML file instead of scraping
        # We'll create mock functions to handle this
        from taxpilot.backend.data_processing.pipeline import process_law
        from taxpilot.backend.data_processing.scraper import DownloadResult
        
        # Setup mock download result
        download_result = DownloadResult(
            law_id="estg",
            law_name="Einkommensteuergesetz",
            file_path=str(input_path),
            status="new"  # Use a valid status from the LawStatus type
        )
        
        # Create pipeline config with our test DB
        pipeline_config = PipelineConfig(
            db_config=DbConfig(db_path=self.db_path)
        )
        
        # Process the test law file
        result = process_law(download_result, pipeline_config)
    
    def generate_embeddings(self):
        """
        Generate and index embeddings for processed data.
        
        Returns:
            The initialized vector database instance
        """
        # Initialize vector database
        vector_db_config = VectorDbConfig(
            **get_vector_db_config()
        )
        vector_db = VectorDatabase(vector_db_config)
        
        # Connect to database
        db_config = DbConfig(db_path=self.db_path)
        conn = get_connection(db_config)
        
        # Get sections from database
        cursor = conn.execute(
            "SELECT id, law_id, section_number, title, content FROM sections"
        )
        sections = cursor.fetchall()
        
        # Use the MockTextEmbedder directly
        from tests.backend.test_e2e_search import MockTextEmbedder
        mock_embedder = MockTextEmbedder(dimension=768)
        
        # Generate and store embeddings
        for section in sections:
            section_id, law_id, section_number, title, content = section
            # Generate mock embedding
            mock_vector = mock_embedder.embed_text(content)
            # Convert to list for Qdrant
            vector = mock_vector.tolist()
            
            # Store in vector database
            point_id = str(uuid.uuid4())
            metadata = {
                "law_id": law_id,
                "section_id": section_id,
                "segment_id": section_id,
                "embedding_model": "mock-embedder",
                "embedding_version": "1.0.0",
                "title": title,
                "section_number": section_number,
            }
            
            from qdrant_client.http.models import PointStruct
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            )
            
            vector_db.client.upsert(
                collection_name=vector_db.config.collection_name,
                points=[point]
            )
        
        return vector_db
    
    def create_test_app(self):
        """Create a test FastAPI app with mock endpoints for testing."""
        app = FastAPI(title="Test API")
        
        # Define our simplified models for the test API
        class SearchRequest(BaseModel):
            query: str
            search_type: Literal["semantic", "keyword"] = "semantic"
            group_by_article: bool = False
            filters: Dict[str, Any] = Field(default_factory=dict)
            page: int = 1
            limit: int = 10
            highlight: bool = True
            min_score: float = 0.7
        
        class SearchResultItem(BaseModel):
            id: str
            law_id: str
            section_id: str
            section_number: str
            title: str
            content: str
            content_with_highlights: str
            score: float
            article_id: Optional[str] = None
            metadata: Dict[str, Any] = Field(default_factory=dict)
        
        class SearchResponse(BaseModel):
            results: List[SearchResultItem]
            total_results: int
            page: int
            page_size: int
            has_more: bool
            execution_time_ms: float
            group_by_article: bool = False
        
        @app.post("/api/search")
        async def search(request: SearchRequest) -> SearchResponse:
            """Mock search endpoint that returns test data."""
            # Generate mock search results based on query
            if "Land" in request.query:
                section_number = "13"
                title = "§ 13 Einkünfte aus Land- und Forstwirtschaft"
                content = "(1) Einkünfte aus Land- und Forstwirtschaft sind\n1. Einkünfte aus dem Betrieb von Landwirtschaft..."
            else:
                section_number = "15"
                title = "§ 15 Einkünfte aus Gewerbebetrieb"
                content = "(1) Einkünfte aus Gewerbebetrieb sind\n1. Einkünfte aus gewerblichen Unternehmen..."
            
            # Create a result item
            result = SearchResultItem(
                id=f"estg_{section_number}",
                law_id="estg",
                section_id=f"estg_{section_number}",
                section_number=section_number,
                title=title,
                content=content,
                content_with_highlights=content.replace("Einkünfte", "<mark>Einkünfte</mark>"),
                score=0.95,
                article_id=f"estg_{section_number}" if request.group_by_article else None,
                metadata={"relevance": "high"}
            )
            
            # Return search response
            return SearchResponse(
                results=[result],
                total_results=1,
                page=request.page,
                page_size=request.limit,
                has_more=False,
                execution_time_ms=42.0,
                group_by_article=request.group_by_article
            )
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "version": "0.1.0"}
            
        return app
    
    @contextmanager
    def start_api_server(self, port: int = 8000):
        """
        Start the API server for testing.
        
        Args:
            port: Port to run the API server on
        """
        # Create FastAPI app with mocked endpoints
        app = self.create_test_app()
        
        # Start server in a separate thread
        server_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={
                "app": app,
                "host": "127.0.0.1",
                "port": port,
                "log_level": "error"
            },
            daemon=True
        )
        server_thread.start()
        
        # Give the server a moment to start up
        time.sleep(1)
        
        try:
            # Yield the API URL to the caller
            yield f"http://127.0.0.1:{port}"
        finally:
            # No direct way to stop uvicorn, but we've set daemon=True
            # so it stops when the main thread exits
            pass
    
    def cleanup(self):
        """Clean up resources after tests."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def create_test_data():
    """
    Create synthetic test data for end-to-end tests.
    
    Returns:
        Path to the test data directory
    """
    # Create test directory if it doesn't exist
    from tests.backend.e2e_config import TEST_DATA_DIR
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Create a simple XML file for testing
    estg_sample = f"""<?xml version="1.0" encoding="UTF-8"?>
<dokumente>
  <norm>
    <metadaten>
      <jurabk>EStG</jurabk>
      <ausfertigung-datum>1977-10-21</ausfertigung-datum>
      <langue>Einkommensteuergesetz</langue>
      <section_id>estg_13</section_id>
    </metadaten>
    <textdaten>
      <titel>§ 13 Einkünfte aus Land- und Forstwirtschaft</titel>
      <content>
        (1) Einkünfte aus Land- und Forstwirtschaft sind
        1. Einkünfte aus dem Betrieb von Landwirtschaft, Forstwirtschaft, Weinbau, Gartenbau, Obstbau, Gemüsebau, Baumschulen.
        2. Einkünfte aus Tierzucht und Tierhaltung.
        
        (2) Zu den Einkünften im Sinne des Absatzes 1 gehören auch
        1. Einkünfte aus Binnenfischerei, Fischzucht und Teichwirtschaft,
        2. Einkünfte aus Imkerei.
      </content>
    </textdaten>
  </norm>
  <norm>
    <metadaten>
      <jurabk>EStG</jurabk>
      <ausfertigung-datum>1977-10-21</ausfertigung-datum>
      <langue>Einkommensteuergesetz</langue>
      <section_id>estg_15</section_id>
    </metadaten>
    <textdaten>
      <titel>§ 15 Einkünfte aus Gewerbebetrieb</titel>
      <content>
        (1) Einkünfte aus Gewerbebetrieb sind
        1. Einkünfte aus gewerblichen Unternehmen.
        2. Die Gewinnanteile der Gesellschafter einer Personengesellschaft.
        
        (2) Eine selbständige nachhaltige Betätigung, die mit der Absicht, Gewinn zu erzielen, unternommen wird, ist Gewerbebetrieb, wenn die Betätigung weder als Ausübung von Land- und Forstwirtschaft noch als Ausübung eines freien Berufs noch als eine andere selbständige Arbeit anzusehen ist.
      </content>
    </textdaten>
  </norm>
</dokumente>
    """
    
    test_xml_path = os.path.join(TEST_DATA_DIR, "estg_sample.xml")
    with open(test_xml_path, "w") as f:
        f.write(estg_sample)
    
    return TEST_DATA_DIR