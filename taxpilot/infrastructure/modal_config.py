"""
Modal.com configuration for the GermanLawFinder application.

This module defines the Modal app and resources needed for serverless deployment.
"""

import os
from pathlib import Path
from typing import Annotated

import modal
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Define the Modal app with appropriate metadata
app = modal.App(
    "germanlawfinder",
    description="German tax law search platform",
)

# Create a persistent volume for DuckDB storage
# This ensures data persists between function invocations
volume = modal.Volume.from_name("germanlawfinder-db-vol", create_if_missing=True)
VOLUME_MOUNT_PATH = "/data"

# Define the Modal image with Python 3.12 and dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi>=0.110.0",
    "uvicorn>=0.28.0",
    "lxml>=5.1.0",
    "python-dotenv>=1.0.1",
    "duckdb>=0.10.0",
    "pydantic>=2.6.3",
)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    memory=2048,  # 2GB memory
    cpu=1.0,  # 1 CPU core
    keep_warm=1,  # Keep one container warm for faster cold starts
    timeout=120,  # 2 minute timeout
)
@modal.asgi_app()
def fastapi_app():
    """
    Create and configure the FastAPI application.
    
    This function is the entry point for the Modal serverless deployment.
    It configures the FastAPI app with all required middleware and routes.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    # Create the FastAPI application
    app = FastAPI(
        title="GermanLawFinder API",
        description="API for searching German tax laws",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Configure CORS
    # In production, restrict origins to your frontend domain
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if os.getenv("ENVIRONMENT") == "development" else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routes
    @app.get("/health")
    async def health_check():
        """Health check endpoint to verify the API is running."""
        return {"status": "healthy", "version": "0.1.0"}
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Welcome to GermanLawFinder API",
            "docs": "/docs",
            "redoc": "/redoc",
        }
    
    return app


# Development function to run the app locally
@app.local_entrypoint()
def main():
    """
    Local entrypoint for development.
    
    This function is used when running the Modal app locally:
    `python -m taxpilot.infrastructure.modal_config`
    """
    # Print information about the app when run locally
    print("GermanLawFinder Modal App")
    print("-------------------------")
    print("To deploy: modal deploy taxpilot.infrastructure.modal_config")
    print("To run locally: modal serve taxpilot.infrastructure.modal_config")
    print("Ensure you've authenticated with Modal: modal token new")
    print("\nEndpoints when deployed:")
    print("- API: https://germanlawfinder--fastapi-app.modal.run")
    print("- Documentation: https://germanlawfinder--fastapi-app.modal.run/docs")