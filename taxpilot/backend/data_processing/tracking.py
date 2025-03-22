"""
Tracking module for data pipeline execution.

This module provides functionality to track pipeline executions,
including logging, statistics collection, and error reporting.
"""

import os
import json
import logging
from typing import Literal, TypedDict, NotRequired, Any, cast
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field
import duckdb

from taxpilot.backend.data_processing.database import DbConfig, get_connection


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tracking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pipeline_tracking")


# Type definitions
ExecutionStatus = Literal["running", "completed", "failed", "partial_success"]


class PipelineStatisticsDict(TypedDict):
    """Dictionary representation of pipeline execution statistics."""
    scraping_time_seconds: float
    parsing_time_seconds: float
    database_time_seconds: float
    total_time_seconds: float
    laws_processed: int
    laws_added: int
    laws_updated: int
    laws_unchanged: int
    laws_error: int
    sections_processed: int
    memory_usage_mb: float


class ErrorRecordDict(TypedDict):
    """Dictionary representation of an error record."""
    timestamp: str
    component: str
    error_type: str
    error_message: str
    law_id: NotRequired[str]
    stack_trace: NotRequired[str]


class ExecutionRecordDict(TypedDict):
    """Dictionary representation of a pipeline execution record."""
    execution_id: str
    start_time: str
    end_time: NotRequired[str]
    status: ExecutionStatus
    statistics: NotRequired[PipelineStatisticsDict]
    errors: NotRequired[list[ErrorRecordDict]]
    notification_sent: bool


# Pydantic models
class PipelineStatistics(BaseModel):
    """Statistics for a pipeline execution."""
    scraping_time_seconds: float = Field(default=0.0)
    parsing_time_seconds: float = Field(default=0.0)
    database_time_seconds: float = Field(default=0.0)
    total_time_seconds: float = Field(default=0.0)
    laws_processed: int = Field(default=0)
    laws_added: int = Field(default=0)
    laws_updated: int = Field(default=0)
    laws_unchanged: int = Field(default=0)
    laws_error: int = Field(default=0)
    sections_processed: int = Field(default=0)
    memory_usage_mb: float = Field(default=0.0)


class ErrorRecord(BaseModel):
    """Record of an error that occurred during pipeline execution."""
    timestamp: datetime = Field(default_factory=datetime.now)
    component: str = Field(description="Component where the error occurred")
    error_type: str = Field(description="Type of error")
    error_message: str = Field(description="Error message")
    law_id: str | None = Field(default=None, description="Affected law ID, if applicable")
    stack_trace: str | None = Field(default=None, description="Stack trace of the error")


class ExecutionRecord(BaseModel):
    """Record of a pipeline execution."""
    execution_id: str = Field(description="Unique ID for the execution")
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = Field(default=None)
    status: ExecutionStatus = Field(default="running")
    statistics: PipelineStatistics | None = Field(default=None)
    errors: list[ErrorRecord] = Field(default_factory=list)
    notification_sent: bool = Field(default=False)


class TrackingConfig(BaseModel):
    """Configuration for the tracking system."""
    tracking_db_path: Path = Field(
        default=Path("tracking.db"),
        description="Path to the tracking database"
    )
    log_file_path: Path = Field(
        default=Path("pipeline.log"),
        description="Path to the log file"
    )
    error_notification_threshold: int = Field(
        default=1,
        description="Number of errors to trigger notification"
    )
    expiration_days: int = Field(
        default=90,
        description="Number of days to keep tracking records"
    )


def initialize_tracking_db(config: TrackingConfig | None = None) -> None:
    """
    Initialize the tracking database with required tables.
    
    Args:
        config: Optional tracking configuration.
    """
    if config is None:
        config = TrackingConfig()
    
    # Create the parent directory if it doesn't exist
    config.tracking_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = duckdb.connect(str(config.tracking_db_path))
    
    try:
        # Executions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                execution_id VARCHAR PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status VARCHAR,
                statistics JSON,
                notification_sent BOOLEAN
            )
        """)
        
        # Create a sequence for the errors table id
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_errors_id")
        
        # Errors table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_errors_id'),
                execution_id VARCHAR,
                timestamp TIMESTAMP,
                component VARCHAR,
                error_type VARCHAR,
                error_message VARCHAR,
                law_id VARCHAR,
                stack_trace VARCHAR
            )
        """)
        
        # Create index
        conn.execute("CREATE INDEX IF NOT EXISTS idx_executions_time ON executions(start_time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_errors_execution ON errors(execution_id)")
        
        logger.info("Tracking database initialized successfully")
    finally:
        conn.close()


def start_tracking(config: TrackingConfig | None = None) -> ExecutionRecord:
    """
    Start tracking a pipeline execution.
    
    Args:
        config: Optional tracking configuration.
        
    Returns:
        An execution record for the new execution.
    """
    if config is None:
        config = TrackingConfig()
    
    # Initialize the tracking database if it doesn't exist
    if not config.tracking_db_path.exists():
        initialize_tracking_db(config)
    
    # Generate execution ID
    execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    
    # Create execution record
    record = ExecutionRecord(
        execution_id=execution_id,
        start_time=datetime.now(),
        status="running",
        notification_sent=False
    )
    
    # Store in database
    conn = duckdb.connect(str(config.tracking_db_path))
    
    try:
        conn.execute(
            """
            INSERT INTO executions (
                execution_id, start_time, status, notification_sent
            ) VALUES (?, ?, ?, ?)
            """,
            (
                record.execution_id,
                record.start_time,
                record.status,
                record.notification_sent
            )
        )
        
        logger.info(f"Started tracking execution {execution_id}")
    finally:
        conn.close()
    
    return record


def update_execution_status(
    execution_id: str, 
    status: ExecutionStatus, 
    statistics: PipelineStatistics | None = None,
    config: TrackingConfig | None = None
) -> None:
    """
    Update the status of a pipeline execution.
    
    Args:
        execution_id: The ID of the execution to update.
        status: The new status of the execution.
        statistics: Optional statistics to update.
        config: Optional tracking configuration.
    """
    if config is None:
        config = TrackingConfig()
    
    conn = duckdb.connect(str(config.tracking_db_path))
    
    try:
        # Get existing record
        result = conn.execute(
            "SELECT * FROM executions WHERE execution_id = ?",
            (execution_id,)
        ).fetchone()
        
        if not result:
            logger.error(f"Execution {execution_id} not found")
            return
        
        # Update status
        end_time = datetime.now() if status != "running" else None
        
        # Convert statistics to JSON if provided
        stats_json = json.dumps(statistics.model_dump()) if statistics else None
        
        # Update record
        if end_time:
            conn.execute(
                """
                UPDATE executions
                SET status = ?, end_time = ?, statistics = ?
                WHERE execution_id = ?
                """,
                (status, end_time, stats_json, execution_id)
            )
        else:
            conn.execute(
                """
                UPDATE executions
                SET status = ?, statistics = ?
                WHERE execution_id = ?
                """,
                (status, stats_json, execution_id)
            )
        
        logger.info(f"Updated execution {execution_id} status to {status}")
    finally:
        conn.close()


def log_error(
    execution_id: str,
    component: str,
    error_type: str,
    error_message: str,
    law_id: str | None = None,
    stack_trace: str | None = None,
    config: TrackingConfig | None = None
) -> None:
    """
    Log an error that occurred during pipeline execution.
    
    Args:
        execution_id: The ID of the execution.
        component: The component where the error occurred.
        error_type: The type of error.
        error_message: The error message.
        law_id: Optional ID of the affected law.
        stack_trace: Optional stack trace of the error.
        config: Optional tracking configuration.
    """
    if config is None:
        config = TrackingConfig()
    
    # Create error record
    error = ErrorRecord(
        timestamp=datetime.now(),
        component=component,
        error_type=error_type,
        error_message=error_message,
        law_id=law_id,
        stack_trace=stack_trace
    )
    
    # Store in database
    conn = duckdb.connect(str(config.tracking_db_path))
    
    try:
        conn.execute(
            """
            INSERT INTO errors (
                execution_id, timestamp, component, error_type, 
                error_message, law_id, stack_trace
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution_id,
                error.timestamp,
                error.component,
                error.error_type,
                error.error_message,
                error.law_id,
                error.stack_trace
            )
        )
        
        logger.error(f"Logged error in {component}: {error_message}")
        
        # Check if notification should be sent
        error_count = conn.execute(
            "SELECT COUNT(*) FROM errors WHERE execution_id = ?",
            (execution_id,)
        ).fetchone()[0]
        
        if error_count >= config.error_notification_threshold:
            # Set notification flag
            conn.execute(
                "UPDATE executions SET notification_sent = TRUE WHERE execution_id = ?",
                (execution_id,)
            )
    finally:
        conn.close()


def complete_execution(
    execution_id: str,
    statistics: PipelineStatistics,
    status: ExecutionStatus = "completed",
    config: TrackingConfig | None = None
) -> None:
    """
    Complete a pipeline execution.
    
    Args:
        execution_id: The ID of the execution to complete.
        statistics: Statistics for the execution.
        status: Final status of the execution.
        config: Optional tracking configuration.
    """
    if config is None:
        config = TrackingConfig()
    
    update_execution_status(execution_id, status, statistics, config)
    
    # Clean up old records
    cleanup_old_records(config)


def get_execution_record(execution_id: str, config: TrackingConfig | None = None) -> ExecutionRecord | None:
    """
    Get a pipeline execution record.
    
    Args:
        execution_id: The ID of the execution to retrieve.
        config: Optional tracking configuration.
        
    Returns:
        The execution record, or None if not found.
    """
    if config is None:
        config = TrackingConfig()
    
    conn = duckdb.connect(str(config.tracking_db_path))
    
    try:
        # Get execution record
        exec_result = conn.execute(
            "SELECT * FROM executions WHERE execution_id = ?",
            (execution_id,)
        ).fetchone()
        
        if not exec_result:
            return None
        
        # Convert to dictionary
        exec_dict = {
            "execution_id": exec_result[0],
            "start_time": exec_result[1],
            "end_time": exec_result[2],
            "status": exec_result[3],
            "statistics": json.loads(exec_result[4]) if exec_result[4] else None,
            "notification_sent": exec_result[5],
            "errors": []
        }
        
        # Get errors
        error_results = conn.execute(
            "SELECT * FROM errors WHERE execution_id = ?",
            (execution_id,)
        ).fetchall()
        
        for error in error_results:
            error_dict = {
                "timestamp": error[2],
                "component": error[3],
                "error_type": error[4],
                "error_message": error[5],
                "law_id": error[6],
                "stack_trace": error[7]
            }
            exec_dict["errors"].append(error_dict)
        
        # Convert to model
        return ExecutionRecord.model_validate(exec_dict)
    finally:
        conn.close()


def get_recent_executions(limit: int = 10, config: TrackingConfig | None = None) -> list[ExecutionRecord]:
    """
    Get recent pipeline executions.
    
    Args:
        limit: Maximum number of executions to retrieve.
        config: Optional tracking configuration.
        
    Returns:
        A list of execution records, ordered by start time (most recent first).
    """
    if config is None:
        config = TrackingConfig()
    
    conn = duckdb.connect(str(config.tracking_db_path))
    
    try:
        # Get execution records
        exec_results = conn.execute(
            "SELECT * FROM executions ORDER BY start_time DESC LIMIT ?",
            (limit,)
        ).fetchall()
        
        records = []
        
        for exec_result in exec_results:
            # Convert to dictionary
            exec_dict = {
                "execution_id": exec_result[0],
                "start_time": exec_result[1],
                "end_time": exec_result[2],
                "status": exec_result[3],
                "statistics": json.loads(exec_result[4]) if exec_result[4] else None,
                "notification_sent": exec_result[5],
                "errors": []
            }
            
            # Get errors
            error_results = conn.execute(
                "SELECT * FROM errors WHERE execution_id = ?",
                (exec_result[0],)
            ).fetchall()
            
            for error in error_results:
                error_dict = {
                    "timestamp": error[2],
                    "component": error[3],
                    "error_type": error[4],
                    "error_message": error[5],
                    "law_id": error[6],
                    "stack_trace": error[7]
                }
                exec_dict["errors"].append(error_dict)
            
            # Convert to model
            records.append(ExecutionRecord.model_validate(exec_dict))
        
        return records
    finally:
        conn.close()


def get_execution_summary(days: int = 30, config: TrackingConfig | None = None) -> dict[str, Any]:
    """
    Get a summary of pipeline executions for a specified period.
    
    Args:
        days: Number of days to include in the summary.
        config: Optional tracking configuration.
        
    Returns:
        A dictionary with summary statistics.
    """
    if config is None:
        config = TrackingConfig()
    
    conn = duckdb.connect(str(config.tracking_db_path))
    
    try:
        # Calculate start date
        start_date = datetime.now() - timedelta(days=days)
        
        # Get statistics
        stats = {}
        
        # Count by status
        status_counts = conn.execute(
            """
            SELECT status, COUNT(*) 
            FROM executions 
            WHERE start_time >= ? 
            GROUP BY status
            """,
            (start_date,)
        ).fetchall()
        
        status_dict = {status: count for status, count in status_counts}
        stats["executions_by_status"] = status_dict
        
        # Total executions
        stats["total_executions"] = sum(status_dict.values())
        
        # Total errors
        error_count = conn.execute(
            """
            SELECT COUNT(*) 
            FROM errors 
            WHERE timestamp >= ?
            """,
            (start_date,)
        ).fetchone()[0]
        
        stats["total_errors"] = error_count
        
        # Most common error components
        error_components = conn.execute(
            """
            SELECT component, COUNT(*) as count
            FROM errors
            WHERE timestamp >= ?
            GROUP BY component
            ORDER BY count DESC
            LIMIT 5
            """,
            (start_date,)
        ).fetchall()
        
        stats["top_error_components"] = {component: count for component, count in error_components}
        
        # Average execution time
        avg_time = conn.execute(
            """
            SELECT AVG(CAST(EXTRACT(EPOCH FROM (end_time - start_time)) AS FLOAT))
            FROM executions
            WHERE start_time >= ? AND end_time IS NOT NULL
            """,
            (start_date,)
        ).fetchone()[0]
        
        stats["average_execution_time_seconds"] = avg_time or 0
        
        # Get statistics on laws processed
        laws_processed = conn.execute(
            """
            SELECT 
                SUM(CAST(json_extract(statistics, '$.laws_processed') AS INTEGER)) as total_processed,
                SUM(CAST(json_extract(statistics, '$.laws_added') AS INTEGER)) as total_added,
                SUM(CAST(json_extract(statistics, '$.laws_updated') AS INTEGER)) as total_updated,
                SUM(CAST(json_extract(statistics, '$.laws_unchanged') AS INTEGER)) as total_unchanged,
                SUM(CAST(json_extract(statistics, '$.laws_error') AS INTEGER)) as total_error,
                SUM(CAST(json_extract(statistics, '$.sections_processed') AS INTEGER)) as total_sections
            FROM executions
            WHERE start_time >= ? AND statistics IS NOT NULL
            """,
            (start_date,)
        ).fetchone()
        
        if laws_processed and laws_processed[0] is not None:
            stats["laws_processed"] = laws_processed[0]
            stats["laws_added"] = laws_processed[1]
            stats["laws_updated"] = laws_processed[2]
            stats["laws_unchanged"] = laws_processed[3]
            stats["laws_error"] = laws_processed[4]
            stats["sections_processed"] = laws_processed[5]
        else:
            stats["laws_processed"] = 0
            stats["laws_added"] = 0
            stats["laws_updated"] = 0
            stats["laws_unchanged"] = 0
            stats["laws_error"] = 0
            stats["sections_processed"] = 0
        
        return stats
    finally:
        conn.close()


def cleanup_old_records(config: TrackingConfig | None = None) -> None:
    """
    Clean up old tracking records.
    
    Args:
        config: Optional tracking configuration.
    """
    if config is None:
        config = TrackingConfig()
    
    conn = duckdb.connect(str(config.tracking_db_path))
    
    try:
        # Calculate expiration date
        expiration_date = datetime.now() - timedelta(days=config.expiration_days)
        
        # Get old execution IDs
        old_executions = conn.execute(
            "SELECT execution_id FROM executions WHERE start_time < ?",
            (expiration_date,)
        ).fetchall()
        
        old_execution_ids = [exec_id[0] for exec_id in old_executions]
        
        if not old_execution_ids:
            return
        
        # Delete related errors
        for exec_id in old_execution_ids:
            conn.execute(
                "DELETE FROM errors WHERE execution_id = ?",
                (exec_id,)
            )
        
        # Delete old executions
        deleted_count = conn.execute(
            "DELETE FROM executions WHERE start_time < ?",
            (expiration_date,)
        ).fetchone()[0]
        
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} old execution records")
    finally:
        conn.close()


def create_notification_content(execution_id: str, config: TrackingConfig | None = None) -> str:
    """
    Create notification content for a pipeline execution.
    
    Args:
        execution_id: The ID of the execution.
        config: Optional tracking configuration.
        
    Returns:
        Notification content as a string.
    """
    if config is None:
        config = TrackingConfig()
    
    # Get execution record
    record = get_execution_record(execution_id, config)
    
    if not record:
        return f"Execution {execution_id} not found"
    
    # Build notification content
    content = [
        f"Pipeline Execution: {record.execution_id}",
        f"Status: {record.status}",
        f"Start Time: {record.start_time.isoformat()}",
        f"End Time: {record.end_time.isoformat() if record.end_time else 'N/A'}",
        ""
    ]
    
    if record.statistics:
        stats = record.statistics
        content.extend([
            "Statistics:",
            f"  Total Time: {stats.total_time_seconds:.2f} seconds",
            f"  Laws Processed: {stats.laws_processed}",
            f"  Laws Added: {stats.laws_added}",
            f"  Laws Updated: {stats.laws_updated}",
            f"  Laws Unchanged: {stats.laws_unchanged}",
            f"  Laws with Errors: {stats.laws_error}",
            f"  Sections Processed: {stats.sections_processed}",
            ""
        ])
    
    if record.errors:
        content.append(f"Errors ({len(record.errors)}):")
        for i, error in enumerate(record.errors[:5]):  # Show at most 5 errors
            content.append(f"  {i+1}. {error.component}: {error.error_message}")
        
        if len(record.errors) > 5:
            content.append(f"  ... and {len(record.errors) - 5} more errors")
    
    return "\n".join(content)