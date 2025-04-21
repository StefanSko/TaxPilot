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
    
    # Set up the database path - using the existing germanlawfinder.duckdb
    db_path = Path("data/processed/germanlawfinder.duckdb")
    
    if not db_path.exists():
        print(f"Database not found at {db_path}. Please run the data processing pipeline first.")
        return
    
    # Configure the search with local Qdrant
    config = IndexingConfig(
        db_config=DbConfig(db_path=str(db_path)),
        segmentation_strategy=SegmentationStrategy.PARAGRAPH,
        embedding_model="deepset/gbert-base",
        qdrant_local_path=Path("./qdrant_local_data")
    )
    
    print(f"Creating search API with database at {db_path}")
    
    # Create search API
    search_api = create_search_api(config)
    
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