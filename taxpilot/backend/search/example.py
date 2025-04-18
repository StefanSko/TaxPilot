"""
Example script demonstrating the complete TaxPilot search workflow.

This script provides an end-to-end example of:
1. Indexing laws from the DuckDB database
2. Running searches on the indexed content
3. Displaying search results
"""

import logging
import time
from pathlib import Path

from taxpilot.backend.data_processing.database import DbConfig, get_all_laws
from taxpilot.backend.search.segmentation import SegmentationStrategy
from taxpilot.backend.search.indexing_pipeline import IndexingConfig, SearchPipeline
from taxpilot.backend.search.embeddings import EmbeddingModelType


# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def run_example():
    """Run the complete search workflow example."""
    print("\n" + "="*80)
    print(" TaxPilot Search Example ".center(80, "="))
    print("="*80 + "\n")
    
    # Set up the database path - using the existing germanlawfinder.duckdb
    db_path = Path("data/processed/germanlawfinder.duckdb")
    
    print(f"Using database at: {db_path.absolute()}")
    
    if not db_path.exists():
        print(f"Database not found at {db_path}. Please run the data processing pipeline first.")
        return
    
    # --- Get available law IDs --- 
    try:
        available_laws = get_all_laws()
        available_ids = [law['id'] for law in available_laws]
        print("\nAvailable Law IDs in Database:")
        print(available_ids)
        print("-"*30)
    except Exception as e:
        print(f"Could not retrieve law IDs: {e}")
    # ------------------------------
    
    # Configure the search pipeline
    print(f"\nConfiguring pipeline to index ALL laws ({len(available_ids)} total)")
    
    config = IndexingConfig(
        db_config=DbConfig(db_path=str(db_path)),
        segmentation_strategy=SegmentationStrategy.PARAGRAPH,
        embedding_model="deepset/gbert-base",  # Explicitly use gbert-base
        chunk_size=512,
        chunk_overlap=128,
        use_accelerator=True,  # Set to True to use MPS or CUDA if available
        laws_to_index=[],  # Empty list means index ALL laws
        qdrant_local_path=Path("./qdrant_local_data") 
    )
    
    # Create the search pipeline
    pipeline = SearchPipeline(config)
    
    try:
        # Step 1: Index the laws
        print("\nStep 1: Indexing laws...")
        start_time = time.time()
        stats = pipeline.index_all_laws()
        indexing_time = time.time() - start_time
        
        print(f"Indexing completed in {indexing_time:.2f} seconds")
        print(f"Processed {stats['laws_processed']} laws")
        print(f"Created {stats['segments_created']} segments")
        print(f"Generated {stats['embeddings_generated']} embeddings")
        
        # Step 2: Run some example searches
        print("\nStep 2: Running example searches...")
        
        # Example queries to demonstrate different search capabilities
        example_queries = [
            "Einkommensteuer Grundlagen",
            "Werbungskosten",
            "Verlustverrechnung",
            "Steuerpflicht Ausland",
            "§ 32 Kinder"
        ]
        
        for i, query in enumerate(example_queries, 1):
            print(f"\nSearch {i}: '{query}'")
            
            results = pipeline.search(
                query=query,
                # law_id="estg",  # Removed incorrect filter to search all indexed laws
                limit=10,         # Increased limit
                min_score=0.5,    # Added minimum score threshold
                highlight=True
            )
            
            print(f"Found {results.total} results (showing top {len(results.results)})")
            print(f"Search took {results.execution_time_ms} ms")
            
            # Display the results
            for j, result in enumerate(results.results, 1):
                print(f"\nResult {j}:")
                print(f"Section: {result.section_number}")
                print(f"Title: {result.title}")
                print(f"Score: {result.relevance_score:.4f}")
                
                # Show a snippet of the content with highlights
                content = result.content_with_highlights
                # Replace HTML highlight tags with terminal color codes
                content = content.replace('<mark>', '\033[1;33m')
                content = content.replace('</mark>', '\033[0m')
                
                # Show a snippet of reasonable length
                max_length = 300
                snippet = content[:max_length] + "..." if len(content) > max_length else content
                print(f"Snippet: {snippet}")
        
        # Step 3: Get vector database statistics
        print("\nStep 3: Vector database statistics...")
        index_stats = pipeline.get_index_stats()
        
        print(f"Collection: {index_stats.get('collection_name', 'unknown')}")
        print(f"Total vectors: {index_stats.get('total_vectors', 0)}")
        print(f"Vector size: {index_stats.get('vector_size', 0)}")
        
        # Print vectors by model
        if "vectors_by_model" in index_stats:
            print("\nVectors by model:")
            for model, count in index_stats["vectors_by_model"].items():
                print(f"  {model}: {count}")
        
        print("\nExample completed successfully!")
    
    finally:
        # Clean up resources
        pipeline.close()
        print("\nResources cleaned up.")


if __name__ == "__main__":
    try:
        run_example()
    except KeyboardInterrupt:
        print("\nExample interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        logging.exception("An error occurred during the example") 