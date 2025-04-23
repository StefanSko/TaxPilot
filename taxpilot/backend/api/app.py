"""
FastAPI application for the GermanLawFinder backend.

This module defines the main FastAPI application with all routes and middleware.
"""

import os
import time
import logging

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from taxpilot.backend.api.models import (
    SearchRequest, 
    SearchResultItem, 
    SearchResponse
)
from taxpilot.backend.api.search_utils import highlight_text, extract_context

# Configure logging
logger = logging.getLogger(__name__)


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
    
    # Add middleware for request timing and global error handling
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time header and provide global error handling."""
        start_time = time.time()
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        except Exception as e:
            # Log the error but don't expose details to client
            logger.error(f"Unhandled error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": "An unexpected error occurred"}
            )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """
        Health check endpoint to verify the API is running.
        
        Returns:
            Dict with status information.
        """
        return {"status": "healthy", "version": "0.1.0"}
    
    # Root endpoint
    @app.get("/")
    async def root() -> dict[str, str]:
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
    @app.post("/api/search", response_model=SearchResponse, tags=["search"])
    async def search(request: SearchRequest) -> SearchResponse:
        """
        Search endpoint for finding relevant sections in German tax laws.
        
        Supports both semantic and keyword search, with options for
        article grouping, highlighting, and filtering.
        
        Args:
            request: SearchRequest object with query and search parameters
            
        Returns:
            SearchResponse with results and metadata
        """
        from taxpilot.backend.search.search_api import SearchService
        from taxpilot.backend.search.article_search import ArticleSearchService
        
        # Start timing for performance measurement
        start_time = time.time()
        
        # Choose the appropriate search service based on request
        if request.group_by_article:
            search_service = ArticleSearchService()
        else:
            search_service = SearchService()
        
        try:
            # Execute search
            search_results = search_service.search(
                query=request.query,
                filters=request.filters,
                page=request.page,
                limit=request.limit,
                highlight=request.highlight,
                cache=True,
                min_score=request.min_score,
                group_by_article=request.group_by_article
            )
            
            # Calculate execution time
            execution_time_ms = round((time.time() - start_time) * 1000, 2)
            
            # Convert QueryResult objects to SearchResultItem objects
            results = []
            for result in search_results.results:
                # Apply enhanced highlighting if enabled
                content_with_highlights = result.content_with_highlights
                if request.highlight:
                    # Apply our improved highlighting
                    content_with_highlights = highlight_text(result.content, request.query)
                
                # Extract context for long content
                if len(result.content) > 1000:
                    content = extract_context(result.content, request.query)
                else:
                    content = result.content
                
                # Create result item
                result_item = SearchResultItem(
                    id=result.id,
                    law_id=result.law_id,
                    law_abbreviation=result.law_abbreviation,
                    section_number=result.section_number,
                    title=result.title,
                    content=content,
                    content_with_highlights=content_with_highlights,
                    relevance_score=result.relevance_score,
                    is_article_result=result.metadata.get("is_article_result", False) if result.metadata else False,
                    metadata=result.metadata
                )
                
                results.append(result_item)
            
            # Create and return response
            return SearchResponse(
                results=results,
                total_results=search_results.total,
                page=request.page,
                limit=request.limit,
                query=request.query,
                search_type=request.search_type,
                group_by_article=request.group_by_article,
                execution_time_ms=execution_time_ms
            )
        
        except ValueError as e:
            # Handle validation errors
            logger.warning(f"Search validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid search request: {str(e)}"
            )
        except Exception as e:
            # Handle general errors
            logger.error(f"Search error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}"
            )
        finally:
            # Ensure resources are closed if the service has a close method
            if hasattr(search_service, 'close'):
                search_service.close()
    
    # List laws endpoint
    @app.get("/api/laws", response_model=list[dict[str, str]])
    async def get_laws() -> list[dict[str, str]]:
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