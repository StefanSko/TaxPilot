#!/usr/bin/env python
"""
TaxPilot Local Deployment Script

This script provides a command-line interface for running the TaxPilot
pipeline locally and starting a search API server with Swagger UI.

Usage:
    python main.py run-all      # Run the entire pipeline and start the server
    python main.py scrape       # Only run the law scraper
    python main.py process      # Only process XML files
    python main.py embed        # Only generate embeddings
    python main.py index        # Only index embeddings in Qdrant
    python main.py serve        # Only start the API server
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Import pipeline components
from taxpilot.backend.data_processing.scraper import (
    ScraperConfig, run_scheduled_scraping
)
from taxpilot.backend.data_processing.pipeline import (
    PipelineConfig, run_pipeline
)
from taxpilot.backend.data_processing.database import (
    DbConfig, initialize_database, ensure_schema_current
)
from taxpilot.backend.search.embeddings import (
    TextEmbedder, EmbeddingConfig
)
from taxpilot.backend.search.vector_db import (
    VectorDbConfig, VectorDatabaseManager, VectorDbProvider
)

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("taxpilot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("taxpilot")


class Config:
    """Configuration class for TaxPilot deployment."""
    
    DEFAULT_CONFIG = {
        "data_dir": "data",
        "db_path": "data/processed/germanlawfinder.duckdb",
        "embedding_model": "deepset/gbert-base",
        "qdrant_url": "http://localhost:6333",
        "qdrant_collection": "law_sections",
        "api_host": "127.0.0.1",
        "api_port": 8000,
        "log_level": "info",
        "laws_to_scrape": [
            "estg", "kstg_1977", "ustg_1980", "ao_1977", "gewstg"
        ]
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional config file path."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self._load_from_file(config_path)
        
        # Update log level if specified
        log_level = getattr(logging, self.config["log_level"].upper())
        logging.getLogger().setLevel(log_level)
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            logger.warning("Using default configuration")
    
    def _ensure_directories(self) -> None:
        """Ensure necessary directories exist."""
        Path(self.config["data_dir"]).mkdir(exist_ok=True)
        Path(self.config["data_dir"], "raw").mkdir(exist_ok=True)
        Path(self.config["data_dir"], "processed").mkdir(exist_ok=True)
        
        # Ensure DB directory exists
        Path(self.config["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    
    def get_scraper_config(self) -> ScraperConfig:
        """Get configuration for the law scraper."""
        return ScraperConfig(
            download_dir=Path(self.config["data_dir"]),
            verify_ssl=False
        )
    
    def get_pipeline_config(self) -> PipelineConfig:
        """Get configuration for the data processing pipeline."""
        return PipelineConfig(
            db_config=DbConfig(db_path=self.config["db_path"]),
            force_update=False
        )
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """Get configuration for the embedding generation."""
        cache_dir = os.path.join(self.config["data_dir"], "model_cache")
        return EmbeddingConfig(
            model_name=self.config["embedding_model"],
            cache_dir=Path(cache_dir)
        )
    
    def get_vector_db_config(self) -> VectorDbConfig:
        """Get configuration for the vector database."""
        # Check if Qdrant server is available
        import socket
        import urllib.parse

        # Parse the URL to get host and port
        parsed_url = urllib.parse.urlparse(self.config["qdrant_url"])
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or 6333
        
        # Try to connect to check if server is running
        qdrant_available = False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((host, port)) == 0:
                    qdrant_available = True
        except Exception:
            pass
            
        if qdrant_available:
            logger.info(f"Using Qdrant server at {self.config['qdrant_url']}")
            return VectorDbConfig(
                provider=VectorDbProvider.QDRANT,
                vectors_url=self.config["qdrant_url"],
                collection_name=self.config["qdrant_collection"],
                db_config=DbConfig(db_path=self.config["db_path"])
            )
        else:
            logger.warning("Qdrant server not available. Using in-memory vector database")
            return VectorDbConfig(
                provider=VectorDbProvider.MEMORY,
                collection_name=self.config["qdrant_collection"],
                db_config=DbConfig(db_path=self.config["db_path"])
            )
    
    def __getitem__(self, key: str) -> Any:
        """Access config items dictionary-style."""
        return self.config[key]


def run_scraper(config: Config) -> None:
    """Run the law scraper to download XML files."""
    logger.info("Starting law scraping...")
    
    scraper_config = config.get_scraper_config()
    result = run_scheduled_scraping(scraper_config)
    
    # Log results
    stats = {
        "total": len(result.results),
        "new": sum(1 for r in result.results.values() if r.status == "new"),
        "updated": sum(1 for r in result.results.values() if r.status == "updated"),
        "unchanged": sum(1 for r in result.results.values() if r.status == "unchanged"),
        "error": sum(1 for r in result.results.values() if r.status == "error"),
    }
    
    logger.info(f"Scraping completed: {stats}")
    
    if stats["error"] > 0:
        errors = [f"{k}: {v.error}" for k, v in result.results.items() if v.status == "error"]
        logger.warning(f"Errors during scraping: {errors}")


def run_processor(config: Config) -> None:
    """Process XML files and store in DuckDB."""
    logger.info("Starting data processing...")
    
    # Ensure database is initialized with current schema
    initialize_database()
    ensure_schema_current()
    
    # Run the pipeline
    pipeline_config = config.get_pipeline_config()
    result = run_pipeline(pipeline_config)
    
    # Log results
    logger.info(f"Processing completed: {result.summary}")
    
    if result.summary.get("error", 0) > 0:
        errors = [f"{k}: {v.error}" for k, v in result.results.items() if v.status == "error"]
        logger.warning(f"Errors during processing: {errors}")


def run_embedder(config: Config) -> None:
    """Generate embeddings for processed law sections."""
    logger.info("Starting embedding generation...")
    
    # Create embedding model
    embedding_config = config.get_embedding_config()
    embedder = TextEmbedder(embedding_config)
    
    # Get the database connection
    db_config = DbConfig(db_path=config["db_path"])
    
    # Initialize vector DB manager for storing embeddings
    vector_config = config.get_vector_db_config()
    vector_manager = VectorDatabaseManager(vector_config)
    
    # Connect to database
    from taxpilot.backend.data_processing.database import get_connection
    conn = get_connection(db_config)
    
    # Get all sections from database
    logger.info("Fetching sections from database...")
    cursor = conn.execute("""
        SELECT id, law_id, section_number, title, content,
               parent_section_id, hierarchy_level, path
        FROM sections
    """)
    sections = cursor.fetchall()
    
    logger.info(f"Generating embeddings for {len(sections)} sections...")
    start_time = time.time()
    
    # Generate embeddings in batches
    batch_size = 10
    total_embedded = 0
    
    for i in range(0, len(sections), batch_size):
        batch = sections[i:i + batch_size]
        batch_embeddings = []
        
        for section in batch:
            section_id, law_id, section_number, title, content = section[:5]
            
            # Generate embedding
            try:
                vector = embedder.embed_text(content)
                
                # Import the TextEmbedding class to create a proper embedding object
                from taxpilot.backend.search.embeddings import TextEmbedding
                
                # Create a TextEmbedding object with the vector
                embedding = TextEmbedding(
                    vector=vector,
                    law_id=law_id,
                    section_id=section_id,
                    segment_id=section_id,
                    embedding_model=embedder.model,
                    embedding_version="1.0.0",
                    metadata={
                        "title": title,
                        "section_number": section_number
                    }
                )
                
                batch_embeddings.append(embedding)
                total_embedded += 1
                
            except Exception as e:
                logger.error(f"Error embedding section {section_id}: {e}")
        
        # Store batch in DuckDB
        try:
            from taxpilot.backend.data_processing.database import insert_section_embedding
            for emb in batch_embeddings:
                insert_section_embedding({
                    "id": f"{emb.segment_id}_{emb.embedding_model}",
                    "law_id": emb.law_id,
                    "section_id": emb.section_id,
                    "segment_id": emb.segment_id,
                    "embedding_model": emb.embedding_model,
                    "embedding_version": emb.embedding_version,
                    "embedding": emb.vector.tolist(),
                    "metadata": emb.metadata
                })
        except Exception as e:
            logger.error(f"Error storing embeddings in DuckDB: {e}")
        
        # Log progress
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(sections):
            elapsed = time.time() - start_time
            logger.info(f"Progress: {i + len(batch)}/{len(sections)} sections processed ({elapsed:.2f}s)")
    
    logger.info(f"Embedding generation completed: {total_embedded} sections embedded")


def run_indexer(config: Config) -> None:
    """Index embeddings from DuckDB into Qdrant."""
    # Initialize vector DB manager
    vector_config = config.get_vector_db_config()
    
    if vector_config.provider == VectorDbProvider.QDRANT:
        logger.info("Starting vector indexing in Qdrant...")
        vector_manager = VectorDatabaseManager(vector_config)
        
        # Sync DuckDB with Qdrant
        start_time = time.time()
        result = vector_manager.synchronize(force_repopulate=False)
        elapsed = time.time() - start_time
        
        logger.info(f"Indexing completed in {elapsed:.2f}s: {result}")
        
        # Check if there were errors
        if result.get("errors", 0) > 0:
            logger.warning(f"There were {result['errors']} errors during indexing")
        
        # Optimize the collection
        logger.info("Optimizing vector collection...")
        vector_manager.optimize()
    else:
        logger.info("Using in-memory vector database - no indexing required")
        # Still initialize the vector DB manager to create the in-memory collection
        vector_manager = VectorDatabaseManager(vector_config)


def run_server(config: Config) -> None:
    """Start the FastAPI server."""
    logger.info("Starting API server...")
    
    # Set environment variables for configuration
    os.environ["DB_PATH"] = config["db_path"]
    
    # Check if Qdrant is available and set environment variables accordingly
    vector_config = config.get_vector_db_config()
    if vector_config.provider == VectorDbProvider.QDRANT:
        logger.info(f"Using Qdrant server at {config['qdrant_url']}")
        os.environ["QDRANT_URL"] = config["qdrant_url"]
    else:
        logger.info("Qdrant server not available, using in-memory vector database")
        # Set environment variable to trigger in-memory mode
        os.environ["QDRANT_IN_MEMORY"] = "true"
    
    os.environ["QDRANT_COLLECTION"] = config["qdrant_collection"]
    
    # Import app after setting environment variables
    from taxpilot.backend.api.app import create_app
    
    app = create_app()
    
    # Start the server
    uvicorn.run(
        app,
        host=config["api_host"],
        port=config["api_port"],
        log_level=config["log_level"].lower()
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TaxPilot Local Deployment")
    
    # Main command argument
    parser.add_argument("command", choices=[
        "run-all", "scrape", "process", "embed", "index", "serve"
    ], help="Command to execute")
    
    # Optional configuration file
    parser.add_argument("--config", "-c", type=str,
                        help="Path to configuration file (JSON)")
    
    # Optional arguments to override configuration
    parser.add_argument("--data-dir", type=str,
                        help="Directory for data files")
    parser.add_argument("--db-path", type=str,
                        help="Path to DuckDB database file")
    parser.add_argument("--embedding-model", type=str,
                        help="Name of the embedding model to use")
    parser.add_argument("--qdrant-url", type=str,
                        help="URL for the Qdrant server")
    parser.add_argument("--api-port", type=int,
                        help="Port for the API server")
    parser.add_argument("--log-level", type=str, 
                        choices=["debug", "info", "warning", "error"],
                        help="Logging level")
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    for arg_name, arg_value in vars(args).items():
        if arg_name != "command" and arg_name != "config" and arg_value is not None:
            config.config[arg_name.replace("_", "-")] = arg_value
    
    # Execute the requested command
    try:
        if args.command == "run-all":
            run_scraper(config)
            run_processor(config)
            run_embedder(config)
            run_indexer(config)
            run_server(config)
        elif args.command == "scrape":
            run_scraper(config)
        elif args.command == "process":
            run_processor(config)
        elif args.command == "embed":
            run_embedder(config)
        elif args.command == "index":
            run_indexer(config)
        elif args.command == "serve":
            run_server(config)
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Error executing {args.command}: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()