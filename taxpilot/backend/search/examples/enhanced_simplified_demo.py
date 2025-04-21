"""
Enhanced simplified TaxPilot search demo comparing standard and article-based search.

This script demonstrates both standard and article-based search functionalities
using SQL queries directly on the DuckDB database without requiring a vector database.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from taxpilot.backend.data_processing.database import DbConfig, get_connection, close_connection


def run_enhanced_demo():
    """Run a demo comparing standard and article-based search using direct database queries."""
    print("\n" + "="*80)
    print(" TaxPilot Enhanced Search Comparison Demo ".center(80, "="))
    print("="*80 + "\n")
    
    # Set up the database path using absolute path from project root
    db_path = project_root / "data" / "processed" / "germanlawfinder.duckdb"
    
    if not db_path.exists():
        print(f"Database not found at {db_path}. Please run the data processing pipeline first.")
        return
    
    # Connect to the database
    try:
        conn = get_connection(DbConfig(db_path=str(db_path)))
        
        # Check database contents
        cursor = conn.execute("SELECT COUNT(*) FROM sections")
        section_count = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM section_embeddings")
        embedding_count = cursor.fetchone()[0]
        print(f"Database contains {section_count} sections and {embedding_count} embeddings")
        
        # Get available laws
        cursor = conn.execute("SELECT id, abbreviation, full_name FROM laws")
        laws = cursor.fetchall()
        print("\nAvailable laws:")
        for law in laws:
            print(f"  - {law[1]} ({law[0]}): {law[2]}")
            
        # Sample queries for comparison
        test_queries = [
            "Einkommensteuer Landwirtschaft",
            "Gewerbliche Einkünfte",
            "Steuerliche Behandlung von Immobilien",
            "Umsatzsteuer Ausland"
        ]
        
        # Search each query and compare approaches
        for query in test_queries:
            print("\n" + "-"*80)
            print(f"QUERY: {query}")
            print("-"*80)
            
            # Parse query into search terms
            search_terms = query.lower().split()
            
            # 1. Traditional segment-based search
            print("\nTraditional segment-based search:\n")
            start_time = time.time()
            
            # Construct SQL WHERE condition for each term with OR within term and AND between terms
            where_clauses = []
            for term in search_terms:
                where_clauses.append(f"(LOWER(s.content) LIKE '%{term}%' OR LOWER(s.title) LIKE '%{term}%')")
            
            where_condition = " AND ".join(where_clauses)
            
            sql_query = f"""
                SELECT 
                    s.id, 
                    s.section_number, 
                    s.title, 
                    s.content, 
                    l.abbreviation,
                    l.id as law_id
                FROM 
                    sections s
                JOIN 
                    laws l ON s.law_id = l.id 
                WHERE 
                    {where_condition}
                ORDER BY 
                    (LENGTH(s.content) - LENGTH(REPLACE(LOWER(s.content), '{search_terms[0]}', ''))) DESC
                LIMIT 5
            """
            
            cursor = conn.execute(sql_query)
            segment_results = cursor.fetchall()
            segment_time = (time.time() - start_time) * 1000
            
            print(f"Found {len(segment_results)} segments in {segment_time:.2f}ms")
            
            # Display traditional search results
            for i, (id, section_number, title, content, law_abbr, law_id) in enumerate(segment_results, 1):
                print(f"\nResult {i}: {law_abbr} § {section_number} (ID: {id})")
                print(f"Title: {title}")
                
                # Create a simplified snippet with highlighting
                content_lower = content.lower()
                snippet = content[:200] + "..." if len(content) > 200 else content
                
                # Highlight search terms
                for term in search_terms:
                    snippet = snippet.replace(term, f"\033[1;33m{term}\033[0m")
                    # Case insensitive replacement (only works for first occurrence)
                    term_lower = term.lower()
                    if term_lower in snippet.lower():
                        # Find position in lower case string
                        pos = snippet.lower().find(term_lower)
                        original_term = snippet[pos:pos+len(term)]
                        snippet = snippet.replace(original_term, f"\033[1;33m{original_term}\033[0m")
                
                print(f"Snippet: {snippet}")
            
            # 2. Article-based search (grouped by section)
            print("\nArticle-based search:\n")
            start_time = time.time()
            
            # First get all matching segments
            sql_query = f"""
                SELECT 
                    s.id, 
                    s.section_number, 
                    s.title, 
                    s.content, 
                    l.abbreviation,
                    l.id as law_id
                FROM 
                    sections s
                JOIN 
                    laws l ON s.law_id = l.id 
                WHERE 
                    {where_condition}
                ORDER BY 
                    s.section_number
            """
            
            cursor = conn.execute(sql_query)
            all_matching_segments = cursor.fetchall()
            
            # Group segments by section_number and law
            section_groups = {}
            for segment in all_matching_segments:
                id, section_number, title, content, law_abbr, law_id = segment
                key = f"{law_id}_{section_number}"
                
                if key not in section_groups:
                    section_groups[key] = {
                        "id": id,
                        "section_number": section_number,
                        "title": title,
                        "law_abbr": law_abbr,
                        "law_id": law_id,
                        "matching_segments": 1,
                        "content": content
                    }
                else:
                    section_groups[key]["matching_segments"] += 1
            
            # Sort by number of matching segments
            sorted_groups = sorted(
                section_groups.values(), 
                key=lambda x: x["matching_segments"], 
                reverse=True
            )
            
            # Take top 5
            article_results = sorted_groups[:5]
            article_time = (time.time() - start_time) * 1000
            
            print(f"Found {len(article_results)} articles in {article_time:.2f}ms")
            
            # Display article-based search results
            for i, result in enumerate(article_results, 1):
                print(f"\nResult {i}: {result['law_abbr']} § {result['section_number']} (Matches: {result['matching_segments']})")
                print(f"Title: {result['title']}")
                
                # Create a simplified snippet with highlighting
                content = result["content"]
                snippet = content[:200] + "..." if len(content) > 200 else content
                
                # Highlight search terms
                for term in search_terms:
                    # Case insensitive replacement (only works for first occurrence)
                    term_lower = term.lower()
                    if term_lower in snippet.lower():
                        # Find position in lower case string
                        pos = snippet.lower().find(term_lower)
                        original_term = snippet[pos:pos+len(term)]
                        snippet = snippet.replace(original_term, f"\033[1;33m{original_term}\033[0m")
                
                print(f"Snippet: {snippet}")
        
        print("\nDemo completed successfully!")
    
    finally:
        # Clean up
        close_connection()
        print("\nResources cleaned up.")


if __name__ == "__main__":
    run_enhanced_demo()