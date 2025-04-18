"""
End-to-end indexing and search pipeline for TaxPilot.

This module provides a complete pipeline for indexing documents from the 
DuckDB database and exposing a simple search API.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from taxpilot.backend.data_processing.database import (
    get_connection, DbConfig, get_all_laws, get_sections_by_law, close_connection
)
from taxpilot.backend.search.segmentation import (
    SegmentationStrategy, SegmentationConfig, segment_text, TextSegment
)
from taxpilot.backend.search.embeddings import (
    EmbeddingConfig, EmbeddingProcessor, TextEmbedder, EmbeddingModelType
)
from taxpilot.backend.search.vector_db import (
    VectorDbConfig, VectorDatabaseManager, SearchParameters, SearchResult
)
from taxpilot.backend.search.search_api import (
    SearchService, QueryResult, SearchResults
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("search_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("search_pipeline")


class IndexingConfig(BaseModel):
    """Configuration for the indexing pipeline."""
    
    db_config: DbConfig = Field(default_factory=DbConfig)
    segmentation_strategy: SegmentationStrategy = Field(
        default=SegmentationStrategy.PARAGRAPH,
        description="Strategy for segmenting text"
    )
    embedding_model: str = Field(
        default=EmbeddingModelType.DEFAULT.value,
        description="Model to use for embeddings"
    )
    force_reindex: bool = Field(
        default=False,
        description="Force reindexing of all documents"
    )
    chunk_size: int = Field(
        default=512,
        description="Target size for text chunks in characters"
    )
    chunk_overlap: int = Field(
        default=128,
        description="Overlap between consecutive chunks"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    use_accelerator: bool = Field(
        default=True, # Default to using accelerator if available
        description="Use GPU (CUDA) or MPS if available for embedding generation"
    )
    laws_to_index: List[str] = Field(
        default_factory=list,
        description="List of law IDs to index, empty for all laws"
    )
    qdrant_local_path: Path | None = Field(
        default=None,
        description="Path for Qdrant local storage. If set, uses local mode instead of connecting to QDRANT_URL."
    )


class SearchPipeline:
    """
    End-to-end search pipeline for TaxPilot.
    
    This class provides methods to:
    1. Index all laws in the database
    2. Search the indexed laws
    3. Get information about the indexed laws
    """
    
    def __init__(self, config: IndexingConfig = None):
        """
        Initialize the search pipeline.
        
        Args:
            config: Configuration for the indexing pipeline
        """
        self.config = config or IndexingConfig()
        
        # Initialize components
        self.db_config = self.config.db_config
        
        # Initialize segmentation config
        self.segmentation_config = SegmentationConfig(
            strategy=self.config.segmentation_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Initialize embedding config
        self.embedding_config = EmbeddingConfig(
            model_name=self.config.embedding_model,
            batch_size=self.config.batch_size,
            use_accelerator=self.config.use_accelerator,
            db_config=self.db_config
        )
        
        # Initialize vector database config
        self.vector_db_config = VectorDbConfig(
            db_config=self.db_config,
            embedding_dim=self.embedding_config.embedding_dim,
            local_path=self.config.qdrant_local_path  # Use local path if provided
        )
        
        # Initialize processors
        self.embedding_processor = EmbeddingProcessor(self.embedding_config)
        self.vector_db = VectorDatabaseManager(self.vector_db_config)
        self.search_service = SearchService(
            vector_db=self.vector_db,
            embedder=TextEmbedder(self.embedding_config),
            db_connection=None  # Will use default
        )
        
        logger.info(f"Initialized search pipeline with model {self.config.embedding_model}")
    
    def index_all_laws(self) -> Dict[str, Any]:
        """
        Index all laws in the database.
        
        Returns:
            Statistics about the indexing process
        """
        start_time = time.time()
        stats = {
            "laws_processed": 0,
            "sections_processed": 0,
            "segments_created": 0,
            "embeddings_generated": 0,
            "errors": 0,
            "total_time_seconds": 0
        }
        
        # Get the list of laws to process
        laws = get_all_laws()
        laws_to_process = []
        
        if self.config.laws_to_index:
            # Filter laws by the provided list
            laws_to_process = [law for law in laws if law["id"] in self.config.laws_to_index]
            logger.info(f"Filtered to {len(laws_to_process)} laws from specified list")
        else:
            laws_to_process = laws
            logger.info(f"Processing all {len(laws)} laws")
        
        # Process each law
        for law in laws_to_process:
            law_id = law["id"]
            logger.info(f"Processing law {law_id} ({law['abbreviation']})")
            
            law_segments: list[TextSegment] = [] # Accumulator for segments of the current law
            law_sections_processed = 0
            
            try:
                # Get sections for this law
                sections = get_sections_by_law(law_id)
                law_sections_processed = len(sections)
                
                # Process each section to generate segments
                for section in sections:
                    section_id = section["id"]
                    section_content = section["content"]
                    
                    # Skip empty sections
                    if not section_content:
                        continue
                    
                    # Segment the section text and add to the law's list
                    segments = segment_text(
                        text=section_content,
                        law_id=law_id,
                        section_id=section_id,
                        config=self.segmentation_config
                    )
                    law_segments.extend(segments)
                
                # Now process all segments for this law in one go
                if law_segments:
                    logger.info(f"Processing {len(law_segments)} segments for law {law_id}...")
                    embedding_ids = self.embedding_processor.process_segments(law_segments)
                    stats["embeddings_generated"] += len(embedding_ids)
                    logger.info(f"Generated {len(embedding_ids)} embeddings for law {law_id}")
                else:
                    logger.warning(f"No segments generated for law {law_id}")

                stats["laws_processed"] += 1
                stats["sections_processed"] += law_sections_processed
                stats["segments_created"] += len(law_segments)
                logger.info(f"Completed law {law_id} - {law_sections_processed} sections, {len(law_segments)} segments generated.")
                
            except Exception as e:
                logger.error(f"Error processing law {law_id}: {e}", exc_info=True)
                stats["errors"] += 1
        
        # Calculate total time
        stats["total_time_seconds"] = time.time() - start_time
        
        # Synchronize with vector database
        logger.info("Synchronizing with vector database...")
        sync_stats = self.vector_db.synchronize()
        stats["sync_stats"] = sync_stats
        
        # Optimize the vector database
        logger.info("Optimizing vector database...")
        self.vector_db.optimize()
        
        logger.info(f"Indexing completed in {stats['total_time_seconds']:.2f} seconds")
        logger.info(f"Processed {stats['laws_processed']} laws, {stats['sections_processed']} sections")
        logger.info(f"Created {stats['segments_created']} segments, {stats['embeddings_generated']} embeddings")
        
        return stats
    
    def search(
        self,
        query: str,
        law_id: str | None = None,
        limit: int = 10,
        page: int = 1,
        highlight: bool = True,
        min_score: float = 0.5
    ) -> SearchResults:
        """
        Search the indexed laws.
        
        Args:
            query: Search query
            law_id: Optional law ID to limit search to
            limit: Maximum number of results to return
            page: Page number for pagination
            highlight: Whether to highlight matches in results
            min_score: Minimum relevance score for results
            
        Returns:
            Search results
        """
        filters = {}
        if law_id:
            filters["law_id"] = law_id
        
        return self.search_service.search(
            query=query,
            filters=filters,
            page=page,
            limit=limit,
            highlight=highlight,
            min_score=min_score
        )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed laws.
        
        Returns:
            Statistics about the indexed laws
        """
        stats = self.vector_db.get_stats()
        return stats
    
    def close(self):
        """Close all resources."""
        self.search_service.close()
        self.vector_db.close()
        # Explicitly close the global DB connection
        close_connection()


def run_indexing_pipeline(config: IndexingConfig) -> Dict[str, Any]:
    """
    Run the indexing pipeline.
    
    Args:
        config: Configuration for the indexing pipeline
        
    Returns:
        Statistics about the indexing process
    """
    pipeline = SearchPipeline(config)
    try:
        stats = pipeline.index_all_laws()
        return stats
    finally:
        pipeline.close()


def create_search_api(config: IndexingConfig = None) -> SearchService:
    """
    Create a search API service.
    
    Args:
        config: Optional configuration for the search API
        
    Returns:
        SearchService instance ready for searching
    """
    if config is None:
        config = IndexingConfig()
    
    # Initialize embedding config
    embedding_config = EmbeddingConfig(
        model_name=config.embedding_model,
        db_config=config.db_config
    )
    
    # Initialize vector database config
    vector_db_config = VectorDbConfig(
        db_config=config.db_config
    )
    
    # Create vector database and embedder
    vector_db = VectorDatabaseManager(vector_db_config)
    embedder = TextEmbedder(embedding_config)
    
    # Create search service
    search_service = SearchService(
        vector_db=vector_db,
        embedder=embedder
    )
    
    return search_service


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TaxPilot Search Indexing Pipeline")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/processed/germanlawfinder.duckdb",
        help="Path to the DuckDB database"
    )
    parser.add_argument(
        "--segmentation-strategy",
        type=str,
        choices=["section", "paragraph", "sentence"],
        default="paragraph",
        help="Strategy for segmenting text"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EmbeddingModelType.DEFAULT.value,
        help="Model to use for embeddings"
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindexing of all documents"
    )
    parser.add_argument(
        "--use-accelerator",
        action="store_true",
        help="Use GPU (CUDA) or MPS if available for embedding generation"
    )
    parser.add_argument(
        "--laws",
        type=str,
        nargs="*",
        help="List of law IDs to index, empty for all laws"
    )
    
    args = parser.parse_args()
    
    # Convert segmentation strategy string to enum
    strategy = SegmentationStrategy.PARAGRAPH
    if args.segmentation_strategy == "section":
        strategy = SegmentationStrategy.SECTION
    elif args.segmentation_strategy == "sentence":
        strategy = SegmentationStrategy.SENTENCE
    
    # Create configuration
    config = IndexingConfig(
        db_config=DbConfig(db_path=args.db_path),
        segmentation_strategy=strategy,
        embedding_model=args.embedding_model,
        force_reindex=args.force_reindex,
        use_accelerator=args.use_accelerator,
        laws_to_index=args.laws if args.laws else []
    )
    
    # Run the indexing pipeline
    stats = run_indexing_pipeline(config)
    
    # Print statistics
    print("Indexing completed!")
    print(f"Processed {stats['laws_processed']} laws")
    print(f"Generated {stats['embeddings_generated']} embeddings")
    print(f"Total time: {stats['total_time_seconds']:.2f} seconds") 