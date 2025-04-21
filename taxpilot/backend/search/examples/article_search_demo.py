"""
Demo script for article-based search functionality.

This script demonstrates the enhanced search with article-based grouping,
comparing it to traditional segment-based search.
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from taxpilot.backend.data_processing.database import DbConfig
from taxpilot.backend.search.indexing_pipeline import IndexingConfig, create_search_api
from taxpilot.backend.search.segmentation import SegmentationStrategy


def run_demo():
    """Run the article search demo."""
    print("\n" + "="*80)
    print(" TaxPilot Article-Based Search Demo ".center(80, "="))
    print("="*80 + "\n")
    
    # Set up the database path using absolute path from project root
    db_path = project_root / "data" / "processed" / "germanlawfinder.duckdb"
    
    if not db_path.exists():
        print(f"Database not found at {db_path}. Please run the data processing pipeline first.")
        return
        
    # Additional verification
    try:
        from taxpilot.backend.data_processing.database import get_connection
        conn = get_connection(DbConfig(db_path=str(db_path)))
        # Check if the sections table exists and has data
        cursor = conn.execute("SELECT COUNT(*) FROM sections")
        section_count = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM section_embeddings")
        embedding_count = cursor.fetchone()[0]
        print(f"Database verified: {section_count} sections, {embedding_count} embeddings")
        if embedding_count == 0:
            print("WARNING: No embeddings found in database. Search results may be limited.")
    except Exception as e:
        print(f"WARNING: Database verification failed: {e}")
        print("Continuing anyway, but example may not work as expected.")
    
    # Configure the search with in-memory Qdrant for demo
    # Note: This won't use the disk-based vector database,
    # but it will work for demonstration purposes using the DuckDB embeddings
    
    # Import explicitly for the memory provider
    from taxpilot.backend.search.vector_db import VectorDbProvider, VectorDbConfig, VectorDatabaseManager
    
    # Create an explicit memory-based vector DB config
    vector_db_config = VectorDbConfig(
        provider=VectorDbProvider.MEMORY,  # Force in-memory mode
        collection_name="law_sections",
        embedding_dim=768,
        db_config=DbConfig(db_path=str(db_path))
    )
    
    config = IndexingConfig(
        db_config=DbConfig(db_path=str(db_path)),
        segmentation_strategy=SegmentationStrategy.PARAGRAPH,
        embedding_model="deepset/gbert-base"
        # No qdrant_local_path means it will use memory mode
    )
    
    print(f"Creating search API with database at {db_path}")
    
    # Create search API with our custom vector_db_config
    from taxpilot.backend.search.vector_db import VectorDatabaseManager
    from taxpilot.backend.search.search_api import SearchService
    from taxpilot.backend.search.embeddings import TextEmbedder
    
    # Create components manually with memory-based vector DB
    vector_db = VectorDatabaseManager(vector_db_config)
    
    # Create embedding config
    from taxpilot.backend.search.embeddings import EmbeddingConfig
    embedding_config = EmbeddingConfig(
        model_name=config.embedding_model,
        use_accelerator=True
    )
    embedder = TextEmbedder(embedding_config)
    
    # Create the search API directly
    search_api = SearchService(
        vector_db=vector_db,
        embedder=embedder,
        db_connection=None  # Will use default
    )
    
    try:
        # Demo queries to compare segment vs article search
        test_queries = [
            "Einkommensteuer Landwirtschaft",
            "Gewerbliche Einkünfte",
            "Steuerliche Behandlung von Immobilien",
            "Umsatzsteuer Ausland"
        ]
        
        for query in test_queries:
            print("\n" + "-"*80)
            print(f"QUERY: {query}")
            print("-"*80)
            
            # Run segment-based search
            print("\nTraditional segment-based search:\n")
            start_time = time.time()
            segment_results = search_api.search(
                query=query,
                limit=5,
                group_by_article=False
            )
            segment_time = (time.time() - start_time) * 1000
            
            # Print segment results
            print(f"Found {segment_results.total} segments in {segment_time:.2f}ms")
            for i, result in enumerate(segment_results.results):
                print(f"\nResult {i+1}: {result.law_abbreviation} § {result.section_number} (Score: {result.relevance_score:.4f})")
                print(f"Title: {result.title}")
                content = result.content_with_highlights[:150] + "..." if len(result.content_with_highlights) > 150 else result.content_with_highlights
                content = content.replace('<mark>', '\033[1;33m').replace('</mark>', '\033[0m')
                print(f"Snippet: {content}")
            
            # Run article-based search
            print("\nArticle-based search:\n")
            start_time = time.time()
            article_results = search_api.search(
                query=query,
                limit=5,
                group_by_article=True
            )
            article_time = (time.time() - start_time) * 1000
            
            # Print article results
            print(f"Found {article_results.total} articles in {article_time:.2f}ms")
            for i, result in enumerate(article_results.results):
                print(f"\nResult {i+1}: {result.law_abbreviation} § {result.section_number} (Score: {result.relevance_score:.4f})")
                print(f"Title: {result.title}")
                print(f"Matching segments: {result.metadata.get('matching_segments', 1)}")
                content = result.content_with_highlights[:150] + "..." if len(result.content_with_highlights) > 150 else result.content_with_highlights
                content = content.replace('<mark>', '\033[1;33m').replace('</mark>', '\033[0m')
                print(f"Snippet: {content}")
            
        print("\nDemo completed successfully!")
    
    finally:
        # Clean up resources
        search_api.close()
        print("\nResources cleaned up.")


if __name__ == "__main__":
    run_demo()