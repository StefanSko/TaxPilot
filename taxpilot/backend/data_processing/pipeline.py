"""
Processing pipeline for German legal documents.

This module orchestrates the downloading, parsing, and database storage of German tax laws.
It integrates web scraping, XML parsing, database storage, and tracking functionality.
"""

import os
import sys
import time
import logging
import traceback
from typing import Literal, TypedDict, NotRequired, cast
from datetime import datetime
from pathlib import Path
import psutil
from pydantic import BaseModel, Field

from taxpilot.backend.data_processing.scraper import (
    ScraperConfig, run_scheduled_scraping, ScraperResult, DownloadResult
)
from taxpilot.backend.data_processing.xml_parser import (
    ParserConfig, Law, process_law_file
)
from taxpilot.backend.data_processing.database import (
    DbConfig, initialize_database, insert_law, insert_section, 
    get_law, get_sections_by_law, Law as DbLaw, Section as DbSection,
    get_connection, close_connection
)

from taxpilot.backend.data_processing.tracking import (
    TrackingConfig, PipelineStatistics, ExecutionRecord,
    start_tracking, update_execution_status, log_error, complete_execution,
    create_notification_content
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("law_pipeline")


# Type definitions
ProcessingStatus = Literal["success", "error", "skipped", "unchanged"]


class ProcessingResultDict(TypedDict):
    """Dictionary representation of a processing result."""
    law_id: str
    law_name: str
    status: ProcessingStatus
    sections_processed: int
    error: NotRequired[str]
    timestamp: datetime


# Pydantic models
class PipelineConfig(BaseModel):
    """Configuration for the processing pipeline."""
    scraper_config: ScraperConfig = Field(default_factory=ScraperConfig)
    parser_config: ParserConfig = Field(default_factory=ParserConfig)
    db_config: DbConfig = Field(default_factory=DbConfig)
    tracking_config: TrackingConfig = Field(default_factory=TrackingConfig)
    force_update: bool = Field(
        default=False,
        description="Force update of laws even if they haven't changed"
    )
    enable_transactions: bool = Field(
        default=True,
        description="Enable database transactions for atomic updates"
    )
    send_notifications: bool = Field(
        default=True,
        description="Send notifications on completion or failure"
    )
    notification_threshold: int = Field(
        default=1,
        description="Number of errors to trigger notification"
    )


class ProcessingResult(BaseModel):
    """Result of processing a single law."""
    law_id: str = Field(description="ID of the law")
    law_name: str = Field(description="Name of the law")
    status: ProcessingStatus = Field(description="Status of the processing")
    sections_processed: int = Field(
        default=0, description="Number of sections processed")
    error: str | None = Field(
        default=None, description="Error message if processing failed")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp of processing")
    was_new: bool = Field(
        default=False, description="Whether this law was newly added")
    parsing_time_seconds: float = Field(
        default=0.0, description="Time spent parsing the law")
    database_time_seconds: float = Field(
        default=0.0, description="Time spent on database operations")


class PipelineResult(BaseModel):
    """Result of running the complete pipeline."""
    results: dict[str, ProcessingResult] = Field(
        default_factory=dict, description="Results for each law")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp of processing")
    summary: dict[str, int] = Field(
        default_factory=dict, description="Summary of processing results")


def process_law(download_result: DownloadResult, config: PipelineConfig) -> ProcessingResult:
    """
    Process a single law file from download to database storage.
    
    Args:
        download_result: Result of downloading the law.
        config: Pipeline configuration.
        
    Returns:
        ProcessingResult object with processing status.
    """
    result = ProcessingResult(
        law_id=download_result.law_id,
        law_name=download_result.law_name,
        status="error",
        sections_processed=0,
        was_new=False,
        parsing_time_seconds=0.0,
        database_time_seconds=0.0
    )
    
    if download_result.status == "error":
        result.error = download_result.error or "Download error"
        return result
    
    if download_result.status == "unchanged" and not config.force_update:
        result.status = "unchanged"
        return result
    
    if download_result.status == "skipped":
        result.status = "skipped"
        return result
        
    # Process only if the law was updated, new, or force_update is True
    try:
        # Check if we have a file path
        if not download_result.file_path:
            result.error = "No file path in download result"
            return result
        
        # Parse the XML file
        logger.info(f"Parsing law file for {download_result.law_id}")
        parsing_start_time = time.time()
        law = process_law_file(download_result.file_path, config.parser_config)
        result.parsing_time_seconds = time.time() - parsing_start_time
        
        # Database operations
        database_start_time = time.time()
        
        # Check if the law already exists in the database
        existing_law = get_law(law.metadata.law_id)
        
        if existing_law:
            # Update existing law - reinsert with same ID to update
            logger.info(f"Updating existing law {law.metadata.law_id}")
            law_data = cast(DbLaw, {
                "id": law.metadata.law_id,
                "full_name": law.metadata.full_title,
                "abbreviation": law.metadata.abbreviation,
                "issue_date": datetime.fromisoformat(law.metadata.issue_date).date() if "-" in law.metadata.issue_date else datetime.now().date(),
                "last_updated": datetime.now().date(),
                "status_info": law.metadata.last_changed or "",
                "metadata": {
                    "publication_info": law.metadata.publication_info,
                    "status_info": law.metadata.status_info
                }
            })
            insert_law(law_data)
            
            # Get existing sections
            existing_sections = get_sections_by_law(law.metadata.law_id)
            existing_section_ids = {s["id"] for s in existing_sections}
            
            # Update or add sections
            for section in law.sections:
                section_data = cast(DbSection, {
                    "id": section.section_id,
                    "law_id": section.law_id,
                    "section_number": section.section_number,
                    "title": section.title or "",
                    "content": section.content.text,
                    "parent_section_id": section.parent_id,
                    "hierarchy_level": section.level,
                    "path": f"{section.law_id}/{section.section_number}",
                    "metadata": {
                        "html_content": section.content.html,
                        "tables": section.content.tables,
                        "order_index": section.order_index
                    }
                })
                
                # Insert will replace if ID exists
                insert_section(section_data)
                    
            result.sections_processed = len(law.sections)
            result.was_new = False
            
        else:
            # Add new law
            logger.info(f"Adding new law {law.metadata.law_id}")
            law_data = cast(DbLaw, {
                "id": law.metadata.law_id,
                "full_name": law.metadata.full_title,
                "abbreviation": law.metadata.abbreviation,
                "issue_date": datetime.fromisoformat(law.metadata.issue_date).date() if "-" in law.metadata.issue_date else datetime.now().date(),
                "last_updated": datetime.now().date(),
                "status_info": law.metadata.last_changed or "",
                "metadata": {
                    "publication_info": law.metadata.publication_info,
                    "status_info": law.metadata.status_info
                }
            })
            insert_law(law_data)
            
            # Add sections
            for section in law.sections:
                section_data = cast(DbSection, {
                    "id": section.section_id,
                    "law_id": section.law_id,
                    "section_number": section.section_number,
                    "title": section.title or "",
                    "content": section.content.text,
                    "parent_section_id": section.parent_id,
                    "hierarchy_level": section.level,
                    "path": f"{section.law_id}/{section.section_number}",
                    "metadata": {
                        "html_content": section.content.html,
                        "tables": section.content.tables,
                        "order_index": section.order_index
                    }
                })
                insert_section(section_data)
                
            result.sections_processed = len(law.sections)
            result.was_new = True
        
        # Record database operation time
        result.database_time_seconds = time.time() - database_start_time
        
        result.status = "success"
        
    except Exception as e:
        logger.error(f"Error processing law {download_result.law_id}: {e}", exc_info=True)
        result.error = str(e)
        
    return result


def run_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """
    Run the complete data processing pipeline.
    
    Args:
        config: Optional pipeline configuration.
        
    Returns:
        PipelineResult object with processing results.
    """
    if config is None:
        config = PipelineConfig()
    
    # Start tracking this execution
    tracking_record = start_tracking(config.tracking_config)
    execution_id = tracking_record.execution_id
    
    # Initialize statistics
    statistics = PipelineStatistics()
    start_time = time.time()
    
    logger.info(f"Starting law processing pipeline (Execution ID: {execution_id})")
    
    try:
        # Initialize the database if needed
        initialize_database()
        
        # Get connection for transaction if enabled
        conn = None
        if config.enable_transactions:
            conn = get_connection(config.db_config)
            conn.execute("BEGIN TRANSACTION")
            logger.info("Started database transaction")
        
        # Scrape laws
        logger.info("Starting law scraping")
        scraper_start_time = time.time()
        try:
            scraper_result = run_scheduled_scraping(config.scraper_config)
            statistics.scraping_time_seconds = time.time() - scraper_start_time
        except Exception as e:
            log_error(
                execution_id=execution_id,
                component="scraper",
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                config=config.tracking_config
            )
            # Roll back transaction if active
            if conn:
                conn.execute("ROLLBACK")
                logger.error("Rolled back transaction due to scraper error")
            
            # Complete tracking with error
            statistics.scraping_time_seconds = time.time() - scraper_start_time
            statistics.total_time_seconds = time.time() - start_time
            complete_execution(
                execution_id=execution_id,
                statistics=statistics,
                status="failed",
                config=config.tracking_config
            )
            
            # Propagate the exception
            raise
        
        # Process each law
        results: dict[str, ProcessingResult] = {}
        parsing_time = 0.0
        database_time = 0.0
        
        for law_id, download_result in scraper_result.results.items():
            logger.info(f"Processing law {law_id}")
            
            try:
                # Process the law
                result = process_law(download_result, config)
                results[law_id] = result
                
                # Update statistics
                if hasattr(result, "parsing_time_seconds"):
                    parsing_time += result.parsing_time_seconds
                if hasattr(result, "database_time_seconds"):
                    database_time += result.database_time_seconds
                
            except Exception as e:
                logger.error(f"Unexpected error processing {law_id}: {e}", exc_info=True)
                
                # Log the error
                log_error(
                    execution_id=execution_id,
                    component="law_processor",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    law_id=law_id,
                    stack_trace=traceback.format_exc(),
                    config=config.tracking_config
                )
                
                # Add error result
                results[law_id] = ProcessingResult(
                    law_id=law_id,
                    law_name=download_result.law_name,
                    status="error",
                    error=str(e)
                )
        
        # Update statistics
        statistics.parsing_time_seconds = parsing_time
        statistics.database_time_seconds = database_time
        
        # Create a summary of the results
        summary = {
            "success": sum(1 for law in results if results[law].status == "success"),
            "error": sum(1 for law in results if results[law].status == "error"),
            "unchanged": sum(1 for law in results if results[law].status == "unchanged"),
            "skipped": sum(1 for law in results if results[law].status == "skipped"),
            "sections_processed": sum(results[law].sections_processed for law in results),
        }
        
        # Update statistics from results
        statistics.laws_processed = len(results)
        statistics.laws_added = sum(1 for law in results if results[law].status == "success" and hasattr(results[law], "was_new") and results[law].was_new)
        statistics.laws_updated = sum(1 for law in results if results[law].status == "success" and hasattr(results[law], "was_new") and not results[law].was_new)
        statistics.laws_unchanged = summary["unchanged"]
        statistics.laws_error = summary["error"]
        statistics.sections_processed = summary["sections_processed"]
        
        # Record memory usage
        process = psutil.Process(os.getpid())
        statistics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        
        # Commit transaction if active
        if conn:
            conn.execute("COMMIT")
            logger.info("Committed database transaction")
        
        # Complete tracking
        execution_status = "completed"
        if summary["error"] > 0 and summary["success"] > 0:
            execution_status = "partial_success"
        elif summary["error"] > 0 and summary["success"] == 0:
            execution_status = "failed"
        
        statistics.total_time_seconds = time.time() - start_time
        complete_execution(
            execution_id=execution_id,
            statistics=statistics,
            status=execution_status,
            config=config.tracking_config
        )
        
        # Prepare result
        pipeline_result = PipelineResult(results=results, summary=summary)
        
        logger.info(f"Pipeline completed with summary: {summary}")
        
        # Generate notification if required
        if config.send_notifications and (execution_status == "failed" or summary["error"] >= config.notification_threshold):
            notification_content = create_notification_content(execution_id, config.tracking_config)
            logger.info(f"Notification would be sent with content:\n{notification_content}")
            # In a real implementation, send the notification via email, Slack, etc.
        
        return pipeline_result
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        
        # Log the error
        log_error(
            execution_id=execution_id,
            component="pipeline",
            error_type=type(e).__name__,
            error_message=str(e),
            stack_trace=traceback.format_exc(),
            config=config.tracking_config
        )
        
        # Roll back transaction if active
        if 'conn' in locals() and conn:
            conn.execute("ROLLBACK")
            logger.error("Rolled back transaction due to pipeline error")
            close_connection()
        
        # Complete tracking with failure
        statistics.total_time_seconds = time.time() - start_time
        complete_execution(
            execution_id=execution_id,
            statistics=statistics,
            status="failed",
            config=config.tracking_config
        )
        
        # Generate notification
        if config.send_notifications:
            notification_content = create_notification_content(execution_id, config.tracking_config)
            logger.info(f"Failure notification would be sent with content:\n{notification_content}")
            # In a real implementation, send the notification via email, Slack, etc.
        
        # Propagate the exception
        raise


if __name__ == "__main__":
    # If run directly, execute the pipeline
    result = run_pipeline()
    print(f"Pipeline completed with summary: {result.summary}")