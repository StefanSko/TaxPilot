"""
End-to-end tests for search API.

This module contains tests that verify the entire system works from data
processing to embedding generation to API serving.
"""

import os
import unittest
import pytest
import requests
import json
import tempfile
import time
import shutil
from pathlib import Path

from tests.backend.e2e_helpers import (
    EndToEndTestEnvironment,
    create_test_data
)


class MockTextEmbedder:
    """Mock text embedder for testing."""
    
    def __init__(self, dimension=768):
        """Initialize with dimension."""
        self.dimension = dimension
    
    def embed_text(self, text):
        """Create a deterministic mock embedding based on text hash."""
        import numpy as np
        import hashlib
        
        # Create a deterministic vector based on text hash
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Use the hash to seed numpy's random generator
        seed = int.from_bytes(hash_bytes[:4], byteorder='little')
        np.random.seed(seed)
        
        # Generate a random vector of the specified dimension
        vector = np.random.rand(self.dimension).astype(np.float32)
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector


class TestEndToEndSearch:
    """End-to-end tests for the search pipeline and API."""
    
    @classmethod
    def setup_class(cls):
        """Set up the test environment once for all tests."""
        # Create test data
        cls.test_data_dir = create_test_data()
        
        # Patch the TextEmbedder to use our mock
        import taxpilot.backend.search.embeddings
        cls.original_embedder = taxpilot.backend.search.embeddings.TextEmbedder
        taxpilot.backend.search.embeddings.TextEmbedder = MockTextEmbedder
        
        # Initialize environment
        cls.env = EndToEndTestEnvironment(use_in_memory=True)
    
    @classmethod
    def teardown_class(cls):
        """Clean up after all tests."""
        # Restore original embedder
        import taxpilot.backend.search.embeddings
        taxpilot.backend.search.embeddings.TextEmbedder = cls.original_embedder
        
        # Clean up test environment
        cls.env.cleanup()
    
    def test_full_pipeline(self):
        """Test the full pipeline from data processing to API serving."""
        # Create test XML file path
        test_xml_path = os.path.join(self.test_data_dir, "estg_sample.xml")
        assert os.path.exists(test_xml_path), f"Test XML file not found at {test_xml_path}"
        
        # Step 1: Set up database
        self.env.setup_database()
        
        # Step 2: Run data pipeline on test data
        self.env.run_data_pipeline(test_xml_path)
        
        # Step 3: Generate embeddings
        vector_db = self.env.generate_embeddings()
        
        # Step 4: Start API server and run tests
        with self.env.start_api_server(port=8001) as api_url:
            # Allow server to start
            time.sleep(1)
            
            # Test basic search
            search_url = f"{api_url}/api/search"
            payload = {
                "query": "Einkünfte aus Land und Forstwirtschaft",
                "search_type": "semantic",
                "group_by_article": False,
                "limit": 5
            }
            
            response = requests.post(search_url, json=payload)
            assert response.status_code == 200, f"API request failed: {response.text}"
            
            data = response.json()
            
            # Verify response structure
            assert "results" in data
            assert "total_results" in data
            assert "page" in data
            assert "execution_time_ms" in data
            
            # Verify we got some results
            assert len(data["results"]) > 0, "No search results returned"
            
            # Verify result content - should include § 13 
            section_13_found = False
            for result in data["results"]:
                if result["section_number"] == "13":
                    section_13_found = True
                    break
            
            assert section_13_found, "Expected to find § 13 in search results"
            
            # Test article-based search
            article_payload = {
                "query": "Einkünfte aus Land und Forstwirtschaft",
                "search_type": "semantic",
                "group_by_article": True,
                "limit": 5
            }
            
            article_response = requests.post(search_url, json=article_payload)
            assert article_response.status_code == 200, f"Article API request failed: {article_response.text}"
            
            article_data = article_response.json()
            
            # Verify article-based group flag
            assert article_data["group_by_article"] is True


if __name__ == "__main__":
    # This allows running the test directly
    # Create test data
    test_data_dir = create_test_data()
    
    # Set up the environment
    env = EndToEndTestEnvironment(use_in_memory=True)
    
    # Run the test
    test = TestEndToEndSearch()
    test.setup_class()
    try:
        test.test_full_pipeline()
    finally:
        test.teardown_class()