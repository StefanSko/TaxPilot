"""
Example script demonstrating the complete TaxPilot search workflow.

This script provides an end-to-end example of:
1. Indexing laws from the DuckDB database
2. Running searches on the indexed content
3. Displaying search results
"""

import logging
import time
import argparse
import sys
from pathlib import Path
import copy

# Add the project root to the Python path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from taxpilot.backend.data_processing.database import DbConfig, get_all_laws
from taxpilot.backend.search.segmentation import SegmentationStrategy
from taxpilot.backend.search.indexing_pipeline import IndexingConfig, SearchPipeline
from taxpilot.backend.search.embeddings import EmbeddingModelType


# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def run_example(search_only: bool = False, debug: bool = False):
    """Run the complete search workflow example."""
    print("\n" + "="*80)
    print(" TaxPilot Search Example ".center(80, "="))
    print("="*80 + "\n")
    
    # Set up the database path using absolute path from project root
    db_path = project_root / "data" / "processed" / "germanlawfinder.duckdb"
    
    print(f"Using database at: {db_path}")
    
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
        laws_to_index=[]  # Empty list means index ALL laws
        # No qdrant_local_path means it will use memory mode
    )
    
    # Create the search pipeline
    pipeline = SearchPipeline(config)
    
    try:
        # Step 1: Index the laws (conditionally)
        if not search_only:
            print("\nStep 1: Indexing laws from source (will clear Qdrant first)...")
            start_time = time.time()
            stats = pipeline.index_all_laws() # This now deletes Qdrant collection first
            indexing_time = time.time() - start_time
            
            print(f"Indexing completed in {indexing_time:.2f} seconds")
            if "error_message" in stats:
                print(f"ERROR during indexing: {stats['error_message']}")
                return # Stop if indexing failed
            print(f"Processed {stats['laws_processed']} laws")
            print(f"Created {stats['segments_created']} segments")
            print(f"Generated {stats['embeddings_generated']} embeddings")
        else:
            print("\nStep 1: Synchronizing Qdrant index from existing DuckDB data (will clear Qdrant first)...")
            start_time = time.time()
            stats = pipeline.sync_qdrant_from_duckdb()
            sync_time = time.time() - start_time
            
            print(f"Synchronization completed in {sync_time:.2f} seconds")
            if "error" in stats:
                print(f"ERROR during sync: {stats['error']}")
                return # Stop if sync failed
            print(f"Checked {stats['embeddings_checked_in_duckdb']} embeddings in DuckDB")
            print(f"{stats['embeddings_inserted_or_updated_in_qdrant']} embeddings added to Qdrant")
            print(f"{stats['duckdb_ids_updated']} DuckDB rows updated with new vector_db_id")
            print(f"{stats['errors']} errors occurred during sync.")
        
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
                if debug:
                    # Print the full object representation in debug mode, but truncate long content
                    debug_result = copy.deepcopy(result)
                    max_len = 100 # Max length for content fields in debug repr
                    if len(debug_result.content) > max_len:
                        debug_result.content = debug_result.content[:max_len] + "..."
                    if len(debug_result.content_with_highlights) > max_len:
                         debug_result.content_with_highlights = debug_result.content_with_highlights[:max_len] + "..."
                    print(repr(debug_result)) # Print the repr of the modified copy
                else:
                    # Print formatted output in normal mode
                    print(f"Law: {result.law_abbreviation} | Section: {result.section_number}")
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
    # Add argument parser
    parser = argparse.ArgumentParser(description="Run TaxPilot Search Example")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Skip the indexing step and only run searches on existing data."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full QueryResult objects instead of formatted snippets."
    )
    args = parser.parse_args()

    try:
        run_example(search_only=args.search_only, debug=args.debug)
    except KeyboardInterrupt:
        print("\nExample interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        logging.exception("An error occurred during the example") 