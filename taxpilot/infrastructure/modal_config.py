"""
Modal.com configuration for the TaxPilot application.

This module contains the Modal infrastructure configuration for running
the TaxPilot API backend in a serverless environment.
"""

import os
import sys
from pathlib import Path

import modal

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Define Modal image for the application
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi",
    "uvicorn",
    "lxml",
    "python-dotenv",
    "duckdb",
    "pydantic",
    "requests",
    "beautifulsoup4",
)

# Create stub for the API application
stub = modal.Stub(
    name="taxpilot-api",
    image=image,
    secrets=[
        modal.Secret.from_name("taxpilot-secrets")
    ]
)

# Volume for persistent storage
laws_volume = modal.Volume.from_name("taxpilot-laws", create_if_missing=True)
db_volume = modal.Volume.from_name("taxpilot-db", create_if_missing=True)

# FastAPI app for serving the API
web_app = modal.FastAPI(
    name="taxpilot-api",
    volumes={
        "/data": laws_volume,
        "/db": db_volume
    }
)

@web_app.router.get("/")
def root():
    """Root endpoint for the API."""
    return {"message": "Welcome to the TaxPilot API"}


@web_app.router.get("/laws")
def get_laws():
    """Get all available laws."""
    from taxpilot.backend.data_processing.database import get_all_laws, DbConfig
    
    # Set up environment
    os.environ["DB_PATH"] = "/db/laws.db"
    
    # Get laws from database
    config = DbConfig(db_path="/db/laws.db")
    laws = get_all_laws()
    
    return {"laws": laws}


@web_app.router.get("/laws/{law_id}")
def get_law(law_id: str):
    """Get a specific law by ID."""
    from taxpilot.backend.data_processing.database import get_law, get_sections_by_law, DbConfig
    
    # Set up environment
    os.environ["DB_PATH"] = "/db/laws.db"
    
    # Get law from database
    config = DbConfig(db_path="/db/laws.db")
    law = get_law(law_id)
    
    if not law:
        return {"error": f"Law {law_id} not found"}, 404
    
    # Get sections for the law
    sections = get_sections_by_law(law_id)
    
    return {
        "law": law,
        "sections": sections
    }


@web_app.router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "OK"}