#!/usr/bin/env python
"""
Script to run the end-to-end test for TaxPilot.

This script sets up and runs the full end-to-end test for the TaxPilot search API.
It covers:
1. Data processing from XML to structured data
2. Embedding generation and vector database indexing
3. API server startup
4. Test queries against the API

Usage:
    poetry run python run_e2e_test.py
"""

import os
import sys
import json
import logging
import tempfile
import requests
import time
import signal
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("e2e_test")


def patch_embedder():
    """Patch the TextEmbedder to use a mock for testing."""
    from tests.backend.test_e2e_search import MockTextEmbedder
    import taxpilot.backend.search.embeddings
    
    # Store original
    original_embedder = taxpilot.backend.search.embeddings.TextEmbedder
    # Replace with mock
    taxpilot.backend.search.embeddings.TextEmbedder = MockTextEmbedder
    
    return original_embedder


def restore_embedder(original_embedder):
    """Restore the original TextEmbedder."""
    import taxpilot.backend.search.embeddings
    taxpilot.backend.search.embeddings.TextEmbedder = original_embedder


def main():
    """Run the end-to-end test."""
    from tests.backend.e2e_helpers import EndToEndTestEnvironment, create_test_data
    
    # Create test data
    logger.info("Creating test data...")
    test_data_dir = create_test_data()
    test_xml_path = os.path.join(test_data_dir, "estg_sample.xml")
    
    # Patch embedder for faster testing
    logger.info("Setting up mock embedder...")
    original_embedder = patch_embedder()
    
    try:
        # Initialize environment
        logger.info("Initializing test environment...")
        env = EndToEndTestEnvironment(use_in_memory=True)
        
        # Set up database
        logger.info("Setting up database...")
        env.setup_database()
        
        # Run data pipeline
        logger.info("Running data pipeline...")
        env.run_data_pipeline(test_xml_path)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        vector_db = env.generate_embeddings()
        
        # Start API server
        logger.info("Starting API server...")
        with env.start_api_server(port=8000) as api_url:
            logger.info(f"API server running at {api_url}")
            
            # Allow server to start
            time.sleep(2)
            
            # Run test queries
            logger.info("Running test queries...")
            
            # Basic search
            search_url = f"{api_url}/api/search"
            logger.info(f"Sending search request to {search_url}")
            
            payload = {
                "query": "Einkünfte aus Land und Forstwirtschaft",
                "search_type": "semantic",
                "group_by_article": False,
                "limit": 5
            }
            
            response = requests.post(search_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Got {len(data['results'])} results")
                if data['results']:
                    logger.info(f"First result: {data['results'][0]['title']}")
            else:
                logger.error(f"Search request failed: {response.status_code}")
                logger.error(response.text)
            
            # Article-based search
            logger.info("Testing article-based search...")
            article_payload = {
                "query": "Einkünfte aus Land und Forstwirtschaft",
                "search_type": "semantic",
                "group_by_article": True,
                "limit": 5
            }
            
            article_response = requests.post(search_url, json=article_payload)
            if article_response.status_code == 200:
                article_data = article_response.json()
                logger.info(f"Got {len(article_data['results'])} article results")
                if article_data["results"]:
                    logger.info(f"First article: {article_data['results'][0]['title']}")
            else:
                logger.error(f"Article search request failed: {article_response.status_code}")
            
            # Only wait for input if running interactively
            if os.isatty(sys.stdin.fileno()):
                input("Press Enter to end the test and shut down the server...")
            else:
                logger.info("Test completed successfully - not waiting for input in non-interactive mode")
        
        # Cleanup
        logger.info("Cleaning up test environment...")
        env.cleanup()
        
    finally:
        # Restore embedder
        restore_embedder(original_embedder)


if __name__ == "__main__":
    main()