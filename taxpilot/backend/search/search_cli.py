#!/usr/bin/env python
"""
Command-line interface for searching German tax laws.

This module provides a simple CLI for searching indexed tax laws
using the TaxPilot search functionality.
"""

import argparse
import json
import sys
from typing import Dict, List, Any

from taxpilot.backend.data_processing.database import DbConfig
from taxpilot.backend.search.indexing_pipeline import create_search_api, IndexingConfig


def pretty_print_result(result: Dict[str, Any], show_highlights: bool = True) -> None:
    """
    Print a search result in a readable format.
    
    Args:
        result: Search result to print
        show_highlights: Whether to show highlighted matches
    """
    print("\n" + "=" * 80)
    print(f"Law: {result['law_id']} - Section: {result['section_number']}")
    print(f"Title: {result['title']}")
    print(f"Relevance: {result['relevance_score']:.2f}")
    print("-" * 80)
    
    # Print content with or without highlights
    if show_highlights and result['content_with_highlights']:
        content = result['content_with_highlights']
        # Replace HTML mark tags with terminal color codes
        content = content.replace('<mark>', '\033[1;33m')
        content = content.replace('</mark>', '\033[0m')
        print(content)
    else:
        print(result['content'])
    
    print("=" * 80)


def run_search(
    query: str,
    db_path: str,
    law_id: str = None,
    limit: int = 5,
    json_output: bool = False,
    no_highlights: bool = False
) -> None:
    """
    Run a search query and display results.
    
    Args:
        query: Search query
        db_path: Path to DuckDB database
        law_id: Optional law ID to restrict search to
        limit: Maximum number of results to display
        json_output: Whether to output results as JSON
        no_highlights: Whether to disable highlighting
    """
    # Create search API
    config = IndexingConfig(
        db_config=DbConfig(db_path=db_path)
    )
    search_api = create_search_api(config)
    
    try:
        # Execute search
        results = search_api.search(
            query=query,
            filters={"law_id": law_id} if law_id else {},
            limit=limit,
            highlight=not no_highlights
        )
        
        # Display results
        if json_output:
            # Convert to JSON-compatible format
            output = {
                "query": results.query,
                "total_results": results.total,
                "execution_time_ms": results.execution_time_ms,
                "results": [
                    {
                        "id": r.id,
                        "law_id": r.law_id,
                        "section_number": r.section_number,
                        "title": r.title,
                        "content": r.content,
                        "relevance_score": r.relevance_score,
                        "metadata": r.metadata
                    }
                    for r in results.results
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            # Pretty print results
            print(f"Query: '{query}'")
            print(f"Found {results.total} results (showing top {len(results.results)})")
            print(f"Search took {results.execution_time_ms} ms")
            
            if results.results:
                for result in results.results:
                    pretty_print_result(result.__dict__, not no_highlights)
            else:
                print("\nNo matching results found.")
    
    finally:
        # Clean up resources
        search_api.close()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Search German tax laws")
    parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/processed/germanlawfinder.duckdb",
        help="Path to the DuckDB database"
    )
    parser.add_argument(
        "--law",
        type=str,
        help="Restrict search to a specific law (e.g., 'estg')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to display"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--no-highlights",
        action="store_true",
        help="Disable highlighting of matching terms"
    )
    
    args = parser.parse_args()
    
    run_search(
        query=args.query,
        db_path=args.db_path,
        law_id=args.law,
        limit=args.limit,
        json_output=args.json,
        no_highlights=args.no_highlights
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSearch interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1) 