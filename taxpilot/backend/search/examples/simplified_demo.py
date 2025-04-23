"""
Simplified TaxPilot search demo that bypasses vector database.

This script demonstrates the article-based search functionality using
data directly from DuckDB without requiring a Qdrant vector store.
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from taxpilot.backend.data_processing.database import DbConfig, get_connection, close_connection, ensure_schema_current


def run_simplified_demo():
    """Run a simplified search demo using direct database queries."""
    print("\n" + "="*80)
    print(" TaxPilot Simplified Search Demo ".center(80, "="))
    print("="*80 + "\n")
    
    # Set up the database path using absolute path from project root
    db_path = project_root / "data" / "processed" / "germanlawfinder.duckdb"
    
    if not db_path.exists():
        print(f"Database not found at {db_path}. Please run the data processing pipeline first.")
        return
    
    # Ensure database schema is current
    db_config = DbConfig(db_path=str(db_path))
    print("Checking database schema...")
    schema_updated = ensure_schema_current()
    if not schema_updated:
        print("WARNING: Failed to update database schema. The example may not work correctly.")
    else:
        print("Database schema is current.")
    
    # Connect to the database
    try:
        conn = get_connection(db_config)
        
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
            
        # Sample queries
        test_queries = [
            "Einkommensteuer",
            "Umsatzsteuer",
            "Gewerbesteuer",
            "Körperschaftsteuer"
        ]
        
        # Search each query using simple keyword matching
        for query in test_queries:
            print("\n" + "-"*80)
            print(f"QUERY: {query}")
            print("-"*80)
            
            # Simple SQL LIKE query to find relevant sections
            cursor = conn.execute("""
                SELECT 
                    s.id, 
                    s.section_number, 
                    s.title, 
                    s.content, 
                    l.abbreviation 
                FROM 
                    sections s
                JOIN 
                    laws l ON s.law_id = l.id 
                WHERE 
                    s.content LIKE ? OR s.title LIKE ?
                LIMIT 5
            """, (f"%{query}%", f"%{query}%"))
            
            results = cursor.fetchall()
            
            print(f"Found {len(results)} matching sections")
            
            # Display results
            for i, (id, section_number, title, content, law_abbr) in enumerate(results, 1):
                print(f"\nResult {i}: {law_abbr} § {section_number}")
                print(f"Title: {title}")
                
                # Extract a snippet around the matched term
                content_lower = content.lower()
                query_lower = query.lower()
                
                if query_lower in content_lower:
                    pos = content_lower.find(query_lower)
                    start = max(0, pos - 75)
                    end = min(len(content), pos + len(query) + 75)
                    snippet = content[start:end]
                    
                    # Add ellipsis if we're not at the beginning/end
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet = snippet + "..."
                    
                    # Highlight the match with terminal colors
                    highlighted = snippet.replace(query, f"\033[1;33m{query}\033[0m")
                    print(f"Snippet: {highlighted}")
                else:
                    # If query not found in content (might be in title only)
                    print(f"Snippet: {content[:150]}...")
            
            # Simulate article-based grouping
            print("\nArticle-based results (grouped by section number):")
            sections_seen = set()
            article_count = 0
            
            for id, section_number, title, content, law_abbr in results:
                article_key = f"{law_abbr}_{section_number}"
                if article_key not in sections_seen:
                    sections_seen.add(article_key)
                    article_count += 1
                    print(f"  Article {article_count}: {law_abbr} § {section_number}")
            
        print("\nDemo completed successfully!")
    
    finally:
        # Clean up
        close_connection()
        print("\nResources cleaned up.")


if __name__ == "__main__":
    run_simplified_demo()