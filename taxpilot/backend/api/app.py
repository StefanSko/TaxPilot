"""
FastAPI application for the GermanLawFinder backend.

This module defines the main FastAPI application with all routes and middleware.
"""

import os
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        A configured FastAPI application instance.
    """
    # Create the FastAPI application with metadata
    app = FastAPI(
        title="GermanLawFinder API",
        description="API for searching German tax laws",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Configure CORS middleware
    # In production, restrict origins to your frontend domain
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if os.getenv("ENVIRONMENT") == "development" else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define Pydantic models for request/response validation
    class SearchRequest(BaseModel):
        query: str
        filters: Dict[str, Any] = {}
        page: int = 1
        limit: int = 10
    
    class SearchResult(BaseModel):
        id: str
        law_id: str
        section_number: str
        title: str
        content: str
        relevance_score: float
    
    class SearchResponse(BaseModel):
        results: List[SearchResult]
        total: int
        page: int
        limit: int
    
    # Health check endpoint
    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """
        Health check endpoint to verify the API is running.
        
        Returns:
            Dict with status information.
        """
        return {"status": "healthy", "version": "0.1.0"}
    
    # Root endpoint
    @app.get("/")
    async def root() -> Dict[str, str]:
        """
        Root endpoint with API information.
        
        Returns:
            Dict with welcome message and docs links.
        """
        return {
            "message": "Welcome to GermanLawFinder API",
            "docs": "/docs",
            "redoc": "/redoc",
        }
    
    # Search endpoint
    @app.post("/api/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        """
        Search endpoint for finding relevant sections in German tax laws.
        
        Args:
            request: SearchRequest object with query and filter parameters.
            
        Returns:
            SearchResponse with results.
        """
        # Placeholder implementation - will be connected to search module
        return SearchResponse(
            results=[],
            total=0,
            page=request.page,
            limit=request.limit,
        )
    
    # List laws endpoint
    @app.get("/api/laws", response_model=List[Dict[str, str]])
    async def get_laws() -> List[Dict[str, str]]:
        """
        Return a list of available laws.
        
        Returns:
            List of law objects with id, name, and abbreviation.
        """
        # Placeholder implementation - will be connected to data module
        laws = [
            {"id": "estg", "name": "Einkommensteuergesetz", "abbreviation": "EStG"},
            {"id": "kstg", "name": "Körperschaftsteuergesetz", "abbreviation": "KStG"},
            {"id": "ustg", "name": "Umsatzsteuergesetz", "abbreviation": "UStG"},
            {"id": "ao", "name": "Abgabenordnung", "abbreviation": "AO"},
            {"id": "gewstg", "name": "Gewerbesteuergesetz", "abbreviation": "GewStG"},
        ]
        return laws
    
    return app


# Application instance for ASGI servers
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Run the application with uvicorn when executed directly
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)