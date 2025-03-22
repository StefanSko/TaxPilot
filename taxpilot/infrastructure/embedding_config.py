"""
Modal.com configuration for vector embedding generation.

This module configures Modal.com containers and functions for generating 
vector embeddings for legal text segments.
"""

import os
from pathlib import Path

import modal

from taxpilot.backend.search.embeddings import (
    EmbeddingModelType,
    EmbeddingConfig,
    initialize_model_for_modal,
)


# Create volumes for persistent storage
db_volume = modal.Volume.from_name("taxpilot-database", create_if_missing=True)
model_cache_volume = modal.Volume.from_name("model-cache", create_if_missing=True)

# Define image with dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "sentence-transformers>=3.4.0",
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "fastapi>=0.110.0",
    "pydantic>=2.6.0",
    "duckdb>=0.10.0",
)

# Add model initialization to the image
image = image.run_function(
    initialize_model_for_modal,
    secrets=[
        modal.Secret.from_name("taxpilot-env"),
    ],
    env={
        "EMBEDDING_MODEL": EmbeddingModelType.DEFAULT.value,
        "USE_GPU": "true",  # Will be used if GPU is available
    },
)

# Create the Modal app
app = modal.App("taxpilot-embeddings", image=image)


@app.function(
    gpu="T4",  # Request a T4 GPU
    timeout=600,  # 10 minute timeout
    retries=2,
    volumes={
        "/data": db_volume,
        "/model_cache": model_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("taxpilot-env"),
    ],
)
def generate_embeddings_batch(
    law_ids: list[str],
    model_name: str = EmbeddingModelType.DEFAULT.value,
    force_update: bool = False,
):
    """
    Generate embeddings for a batch of laws.
    
    Args:
        law_ids: List of law IDs to process
        model_name: Name of the embedding model to use
        force_update: Whether to force update existing embeddings
    
    Returns:
        Statistics about the embedding generation process
    """
    from taxpilot.backend.data_processing.database import DbConfig
    from taxpilot.backend.search.segmentation import (
        SegmentationStrategy,
        SegmentationConfig,
        segment_text,
    )
    from taxpilot.backend.search.embeddings import (
        EmbeddingConfig,
        EmbeddingProcessor,
    )
    from taxpilot.backend.data_processing.database import get_connection
    import time
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("modal-embeddings")
    
    # Configure paths
    db_path = Path("/data/taxpilot.db")
    model_cache_path = Path("/model_cache")
    
    # Initialize configurations
    db_config = DbConfig(db_path=db_path)
    segmentation_config = SegmentationConfig(
        strategy=SegmentationStrategy.PARAGRAPH,
        chunk_size=512,
        chunk_overlap=128,
    )
    embedding_config = EmbeddingConfig(
        model_name=model_name,
        cache_dir=model_cache_path,
        use_gpu=True,
        db_config=db_config,
    )
    
    # Create the embedding processor
    processor = EmbeddingProcessor(embedding_config)
    
    # Track statistics
    stats = {
        "laws_processed": 0,
        "segments_processed": 0,
        "embeddings_generated": 0,
        "embeddings_updated": 0,
        "processing_time_seconds": 0,
    }
    
    start_time = time.time()
    
    # Process each law
    for law_id in law_ids:
        logger.info(f"Processing law {law_id}")
        
        # If force update, delete existing embeddings for this law
        if force_update:
            processor.delete_law_embeddings(law_id)
        
        # Get law sections from database
        conn = get_connection(db_config)
        try:
            sections = conn.execute(
                """
                SELECT id, law_id, section_number, title, content
                FROM sections 
                WHERE law_id = ?
                """,
                (law_id,),
            ).fetchall()
            
            if not sections:
                logger.warning(f"No sections found for law {law_id}")
                continue
                
            logger.info(f"Found {len(sections)} sections for law {law_id}")
            
            # Process each section
            for section in sections:
                section_id, law_id, section_number, title, content = section
                
                # Skip empty content
                if not content:
                    continue
                
                # Create a full text with title and content
                full_text = f"{section_number} {title}\n\n{content}"
                
                # Segment the text
                segments = segment_text(full_text, law_id, section_id, segmentation_config)
                
                if segments:
                    # Process segments and generate embeddings
                    embedding_ids = processor.process_segments(segments)
                    
                    # Update statistics
                    stats["segments_processed"] += len(segments)
                    stats["embeddings_generated"] += len(embedding_ids)
            
            # Mark law as processed
            stats["laws_processed"] += 1
            
        finally:
            conn.close()
    
    # Record total processing time
    stats["processing_time_seconds"] = time.time() - start_time
    
    logger.info(f"Embedding generation complete: {stats}")
    return stats


@app.function(
    timeout=60,
    volumes={
        "/data": db_volume,
    },
    secrets=[
        modal.Secret.from_name("taxpilot-env"),
    ],
)
def get_laws_needing_embeddings(
    model_name: str = EmbeddingModelType.DEFAULT.value,
) -> list[str]:
    """
    Get a list of laws that need embeddings.
    
    Args:
        model_name: Name of the embedding model to check
        
    Returns:
        List of law IDs that need embeddings
    """
    from taxpilot.backend.data_processing.database import DbConfig, get_connection
    
    # Configure database
    db_path = Path("/data/taxpilot.db")
    db_config = DbConfig(db_path=db_path)
    
    # Get connection
    conn = get_connection(db_config)
    
    try:
        # Get all laws
        all_laws = conn.execute(
            "SELECT DISTINCT id FROM laws"
        ).fetchall()
        
        all_law_ids = [law[0] for law in all_laws]
        
        # Get laws that already have embeddings for this model
        laws_with_embeddings = conn.execute(
            """
            SELECT DISTINCT law_id 
            FROM section_embeddings 
            WHERE embedding_model = ?
            """,
            (model_name,)
        ).fetchall()
        
        laws_with_embeddings_ids = [law[0] for law in laws_with_embeddings]
        
        # Find laws that need embeddings
        missing_embeddings = [
            law_id for law_id in all_law_ids if law_id not in laws_with_embeddings_ids
        ]
        
        return missing_embeddings
    finally:
        conn.close()


@app.function(
    cpu=2.0,
    timeout=60,
    volumes={
        "/data": db_volume,
    },
    secrets=[
        modal.Secret.from_name("taxpilot-env"),
    ],
)
def schedule_embedding_generation():
    """
    Check for laws that need embeddings and schedule generation jobs.
    
    This function is meant to be run on a schedule to check for new laws
    and generate embeddings for them.
    """
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("embedding-scheduler")
    
    # Get model name from environment
    model_name = os.environ.get("EMBEDDING_MODEL", EmbeddingModelType.DEFAULT.value)
    
    # Get laws that need embeddings
    laws_needing_embeddings = get_laws_needing_embeddings.remote(model_name)
    
    if not laws_needing_embeddings:
        logger.info("No laws need embeddings")
        return {"status": "no_work", "laws_count": 0}
    
    logger.info(f"Found {len(laws_needing_embeddings)} laws needing embeddings")
    
    # Process in batches of 10 laws each
    batch_size = 10
    batches = [
        laws_needing_embeddings[i:i + batch_size]
        for i in range(0, len(laws_needing_embeddings), batch_size)
    ]
    
    # Schedule batch processing
    for i, batch in enumerate(batches):
        logger.info(f"Scheduling batch {i+1}/{len(batches)} with {len(batch)} laws")
        generate_embeddings_batch.remote(batch, model_name)
    
    return {
        "status": "scheduled",
        "laws_count": len(laws_needing_embeddings),
        "batch_count": len(batches)
    }


# Create a scheduled job to check for and generate embeddings
@app.schedule(cron="0 0 * * *")  # Run daily at midnight
def daily_embedding_check():
    """Daily check for laws that need embeddings."""
    return schedule_embedding_generation.remote()


# Entry point for ad-hoc runs
if __name__ == "__main__":
    with modal.run():
        # Get laws needing embeddings
        laws = get_laws_needing_embeddings.remote()
        print(f"Found {len(laws)} laws that need embeddings")
        
        if laws:
            # Process a sample law for testing
            test_law = laws[0]
            print(f"Processing sample law: {test_law}")
            stats = generate_embeddings_batch.remote([test_law])
            print(f"Results: {stats}")