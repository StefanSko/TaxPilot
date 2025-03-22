"""
Modal.com configuration for the TaxPilot scheduled pipeline.

This module contains the Modal infrastructure configuration for running
the scheduled data processing pipeline in a serverless environment.
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
    "lxml",
    "python-dotenv",
    "duckdb",
    "pydantic",
    "requests",
    "beautifulsoup4",
    "psutil",
)

# Create stub for the scheduled pipeline
stub = modal.Stub(
    name="taxpilot-pipeline",
    image=image,
    secrets=[
        modal.Secret.from_name("taxpilot-secrets")
    ]
)

# Volume for persistent storage
laws_volume = modal.Volume.from_name("taxpilot-laws", create_if_missing=True)
db_volume = modal.Volume.from_name("taxpilot-db", create_if_missing=True)


@stub.function(
    cpu=2.0,
    memory=4096,
    timeout=3600,
    volumes={
        "/data": laws_volume,
        "/db": db_volume
    }
)
def run_pipeline():
    """
    Run the complete pipeline to scrape, parse, and index German tax laws.
    """
    # Configure environment
    os.environ["DATA_DIR"] = "/data"
    os.environ["DB_PATH"] = "/db/laws.db"
    os.environ["TRACKING_DB_PATH"] = "/db/tracking.db"
    
    # Import after environment is set up
    from taxpilot.backend.data_processing.pipeline import run_pipeline, PipelineConfig
    from taxpilot.backend.data_processing.scraper import ScraperConfig
    from taxpilot.backend.data_processing.xml_parser import ParserConfig
    from taxpilot.backend.data_processing.database import DbConfig
    from taxpilot.backend.data_processing.tracking import TrackingConfig
    
    # Configure pipeline
    config = PipelineConfig(
        scraper_config=ScraperConfig(
            download_dir=Path("/data"),
            check_interval_days=30
        ),
        parser_config=ParserConfig(),
        db_config=DbConfig(
            db_path="/db/laws.db"
        ),
        tracking_config=TrackingConfig(
            tracking_db_path=Path("/db/tracking.db"),
            log_file_path=Path("/db/pipeline.log")
        ),
        enable_transactions=True,
        send_notifications=True
    )
    
    # Run pipeline
    try:
        result = run_pipeline(config)
        print(f"Pipeline completed: {result.summary}")
        return result.summary
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


@stub.function(
    cpu=0.5,
    memory=1024,
    volumes={"/db": db_volume}
)
def get_pipeline_status():
    """
    Get the status of recent pipeline executions.
    """
    os.environ["TRACKING_DB_PATH"] = "/db/tracking.db"
    
    from taxpilot.backend.data_processing.tracking import (
        TrackingConfig, get_recent_executions, get_execution_summary
    )
    
    config = TrackingConfig(tracking_db_path=Path("/db/tracking.db"))
    
    # Get recent executions
    recent = get_recent_executions(limit=5, config=config)
    
    # Get summary statistics
    summary = get_execution_summary(days=30, config=config)
    
    return {
        "recent_executions": [r.model_dump() for r in recent],
        "summary": summary
    }


@stub.function(
    schedule=modal.Cron("0 0 1 * *")  # Run at midnight on the 1st of each month
)
def scheduled_pipeline():
    """
    Scheduled function to run the pipeline on a monthly basis.
    """
    print(f"Starting scheduled pipeline run at {modal.functions.now()}")
    result = run_pipeline.remote()
    print(f"Scheduled pipeline completed with results: {result}")
    return result