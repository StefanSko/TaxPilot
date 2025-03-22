"""
Unit tests for the pipeline tracking module.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from taxpilot.backend.data_processing.tracking import (
    TrackingConfig, PipelineStatistics, ErrorRecord, ExecutionRecord,
    initialize_tracking_db, start_tracking, update_execution_status,
    log_error, complete_execution, get_execution_record,
    get_recent_executions, get_execution_summary, cleanup_old_records,
    create_notification_content
)


@pytest.fixture
def temp_tracking_db(tmp_path):
    """Create a temporary tracking database."""
    db_path = tmp_path / "test_tracking.db"
    config = TrackingConfig(tracking_db_path=db_path)
    initialize_tracking_db(config)
    yield config


def test_initialize_tracking_db(tmp_path):
    """Test initializing the tracking database."""
    db_path = tmp_path / "test_tracking.db"
    config = TrackingConfig(tracking_db_path=db_path)
    
    # Initialize the database
    initialize_tracking_db(config)
    
    # Check that the database file was created
    assert db_path.exists()


def test_start_tracking(temp_tracking_db):
    """Test starting tracking for a pipeline execution."""
    # Start tracking
    record = start_tracking(temp_tracking_db)
    
    # Check record properties
    assert record.execution_id is not None
    assert record.start_time is not None
    assert record.status == "running"
    assert record.notification_sent is False
    
    # Check that the record was stored in the database
    stored_record = get_execution_record(record.execution_id, temp_tracking_db)
    assert stored_record is not None
    assert stored_record.execution_id == record.execution_id
    assert stored_record.status == "running"


def test_update_execution_status(temp_tracking_db):
    """Test updating the status of a pipeline execution."""
    # Start tracking
    record = start_tracking(temp_tracking_db)
    
    # Create statistics
    stats = PipelineStatistics(
        scraping_time_seconds=10.5,
        parsing_time_seconds=5.2,
        database_time_seconds=8.7,
        total_time_seconds=24.4,
        laws_processed=5,
        laws_added=2,
        laws_updated=1,
        laws_unchanged=1,
        laws_error=1,
        sections_processed=50,
        memory_usage_mb=150.5
    )
    
    # Update status
    update_execution_status(record.execution_id, "completed", stats, temp_tracking_db)
    
    # Check that the status was updated
    updated_record = get_execution_record(record.execution_id, temp_tracking_db)
    assert updated_record is not None
    assert updated_record.status == "completed"
    assert updated_record.statistics is not None
    assert updated_record.statistics.laws_processed == 5
    assert updated_record.statistics.sections_processed == 50


def test_log_error(temp_tracking_db):
    """Test logging an error during pipeline execution."""
    # Start tracking
    record = start_tracking(temp_tracking_db)
    
    # Log an error
    log_error(
        record.execution_id,
        "scraper",
        "HTTPError",
        "Failed to download law",
        "estg",
        "Traceback...",
        temp_tracking_db
    )
    
    # Check that the error was logged
    updated_record = get_execution_record(record.execution_id, temp_tracking_db)
    assert updated_record is not None
    assert len(updated_record.errors) == 1
    assert updated_record.errors[0].component == "scraper"
    assert updated_record.errors[0].error_type == "HTTPError"
    assert updated_record.errors[0].law_id == "estg"


def test_complete_execution(temp_tracking_db):
    """Test completing a pipeline execution."""
    # Start tracking
    record = start_tracking(temp_tracking_db)
    
    # Create statistics
    stats = PipelineStatistics(
        scraping_time_seconds=10.5,
        parsing_time_seconds=5.2,
        database_time_seconds=8.7,
        total_time_seconds=24.4,
        laws_processed=5,
        laws_added=2,
        laws_updated=1,
        laws_unchanged=1,
        laws_error=1,
        sections_processed=50,
        memory_usage_mb=150.5
    )
    
    # Complete execution
    complete_execution(record.execution_id, stats, "completed", temp_tracking_db)
    
    # Check that the execution was completed
    completed_record = get_execution_record(record.execution_id, temp_tracking_db)
    assert completed_record is not None
    assert completed_record.status == "completed"
    assert completed_record.end_time is not None
    assert completed_record.statistics is not None
    assert completed_record.statistics.laws_processed == 5


def test_get_recent_executions(temp_tracking_db):
    """Test getting recent pipeline executions."""
    # Insert records with different timestamps directly to avoid duplicate IDs
    import duckdb
    conn = duckdb.connect(str(temp_tracking_db.tracking_db_path))
    
    # Create test execution records with different timestamps
    exec_id1 = "test_exec_1"
    exec_id2 = "test_exec_2"
    exec_id3 = "test_exec_3"
    
    time1 = datetime.now() - timedelta(minutes=10)
    time2 = datetime.now() - timedelta(minutes=5)
    time3 = datetime.now()
    
    # Insert records directly
    conn.execute(
        """
        INSERT INTO executions (execution_id, start_time, status, notification_sent)
        VALUES (?, ?, ?, ?)
        """,
        (exec_id1, time1, "completed", False)
    )
    
    conn.execute(
        """
        INSERT INTO executions (execution_id, start_time, status, notification_sent)
        VALUES (?, ?, ?, ?)
        """,
        (exec_id2, time2, "failed", False)
    )
    
    conn.execute(
        """
        INSERT INTO executions (execution_id, start_time, status, notification_sent)
        VALUES (?, ?, ?, ?)
        """,
        (exec_id3, time3, "running", False)
    )
    
    conn.close()
    
    # Add statistics to completed executions
    stats = PipelineStatistics(laws_processed=5, sections_processed=50)
    stats_json = json.dumps(stats.model_dump())
    
    conn = duckdb.connect(str(temp_tracking_db.tracking_db_path))
    conn.execute(
        "UPDATE executions SET statistics = ? WHERE execution_id = ?",
        (stats_json, exec_id1)
    )
    conn.execute(
        "UPDATE executions SET statistics = ? WHERE execution_id = ?",
        (stats_json, exec_id2)
    )
    conn.close()
    
    # Get recent executions
    recent = get_recent_executions(limit=2, config=temp_tracking_db)
    
    # Should return the most recent 2 executions
    assert len(recent) == 2
    # Records should be sorted by start time, most recent first
    assert recent[0].execution_id == exec_id3
    assert recent[1].execution_id == exec_id2


def test_get_execution_summary(temp_tracking_db):
    """Test getting a summary of pipeline executions."""
    # Insert records with different timestamps directly to avoid duplicate IDs
    import duckdb
    conn = duckdb.connect(str(temp_tracking_db.tracking_db_path))
    
    # Create test execution records
    exec_id1 = "test_summary_1"
    exec_id2 = "test_summary_2"
    exec_id3 = "test_summary_3"
    
    # Insert records directly
    conn.execute(
        """
        INSERT INTO executions (execution_id, start_time, status, notification_sent)
        VALUES (?, ?, ?, ?)
        """,
        (exec_id1, datetime.now(), "completed", False)
    )
    
    conn.execute(
        """
        INSERT INTO executions (execution_id, start_time, status, notification_sent)
        VALUES (?, ?, ?, ?)
        """,
        (exec_id2, datetime.now(), "partial_success", False)
    )
    
    conn.execute(
        """
        INSERT INTO executions (execution_id, start_time, status, notification_sent)
        VALUES (?, ?, ?, ?)
        """,
        (exec_id3, datetime.now(), "running", False)
    )
    
    # Create statistics
    stats1 = PipelineStatistics(
        laws_processed=5, laws_added=3, laws_updated=1, 
        laws_unchanged=1, laws_error=0, sections_processed=50
    )
    stats2 = PipelineStatistics(
        laws_processed=3, laws_added=0, laws_updated=2, 
        laws_unchanged=0, laws_error=1, sections_processed=20
    )
    
    # Add statistics to executions
    conn.execute(
        "UPDATE executions SET statistics = ? WHERE execution_id = ?",
        (json.dumps(stats1.model_dump()), exec_id1)
    )
    
    conn.execute(
        "UPDATE executions SET statistics = ? WHERE execution_id = ?",
        (json.dumps(stats2.model_dump()), exec_id2)
    )
    
    # Log some errors
    conn.execute(
        """
        INSERT INTO errors (execution_id, timestamp, component, error_type, error_message, law_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (exec_id2, datetime.now(), "scraper", "HTTPError", "Download failed", "estg")
    )
    
    conn.execute(
        """
        INSERT INTO errors (execution_id, timestamp, component, error_type, error_message, law_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (exec_id2, datetime.now(), "parser", "XMLError", "Invalid XML", "kstg")
    )
    
    conn.close()
    
    # Get summary
    summary = get_execution_summary(days=30, config=temp_tracking_db)
    
    # Check summary contents
    assert summary["total_executions"] == 3
    assert summary["executions_by_status"].get("completed", 0) == 1
    assert summary["executions_by_status"].get("partial_success", 0) == 1
    assert summary["executions_by_status"].get("running", 0) == 1
    assert summary["total_errors"] == 2
    assert "scraper" in summary["top_error_components"]
    assert "parser" in summary["top_error_components"]
    assert summary["laws_processed"] == 8  # 5 + 3
    assert summary["sections_processed"] == 70  # 50 + 20


@patch("taxpilot.backend.data_processing.tracking.datetime")
def test_cleanup_old_records(mock_datetime, temp_tracking_db):
    """Test cleaning up old tracking records."""
    # Mock current time
    now = datetime(2023, 1, 15)
    mock_datetime.now.return_value = now
    
    # Start tracking with a fixed execution ID
    # Generate execution ID manually to avoid timestamp dependency
    execution_id = f"exec_old_test_{os.getpid()}"
    
    # Insert directly with an old date
    import duckdb
    conn = duckdb.connect(str(temp_tracking_db.tracking_db_path))
    old_date = (now - timedelta(days=100)).isoformat()
    conn.execute(
        """
        INSERT INTO executions (
            execution_id, start_time, status, notification_sent
        ) VALUES (?, ?, ?, ?)
        """,
        (execution_id, old_date, "completed", False)
    )
    conn.close()
    
    # Run cleanup
    cleanup_old_records(temp_tracking_db)
    
    # Check that the record was deleted by querying database directly
    conn = duckdb.connect(str(temp_tracking_db.tracking_db_path))
    result = conn.execute(
        "SELECT COUNT(*) FROM executions WHERE execution_id = ?", 
        (execution_id,)
    ).fetchone()[0]
    conn.close()
    assert result == 0


def test_create_notification_content(temp_tracking_db):
    """Test creating notification content for a pipeline execution."""
    # Insert test record directly
    import duckdb
    conn = duckdb.connect(str(temp_tracking_db.tracking_db_path))
    
    # Create test execution record
    exec_id = "test_notification"
    now = datetime.now()
    end_time = now + timedelta(minutes=5)
    
    # Insert record
    conn.execute(
        """
        INSERT INTO executions (execution_id, start_time, end_time, status, notification_sent)
        VALUES (?, ?, ?, ?, ?)
        """,
        (exec_id, now, end_time, "partial_success", False)
    )
    
    # Create statistics
    stats = PipelineStatistics(
        scraping_time_seconds=10.5,
        parsing_time_seconds=5.2,
        database_time_seconds=8.7,
        total_time_seconds=24.4,
        laws_processed=5,
        laws_added=2,
        laws_updated=1,
        laws_unchanged=1,
        laws_error=1,
        sections_processed=50,
        memory_usage_mb=150.5
    )
    
    # Add statistics
    conn.execute(
        "UPDATE executions SET statistics = ? WHERE execution_id = ?",
        (json.dumps(stats.model_dump()), exec_id)
    )
    
    # Add errors
    conn.execute(
        """
        INSERT INTO errors (execution_id, timestamp, component, error_type, error_message, law_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (exec_id, now, "scraper", "HTTPError", "Download failed", "estg")
    )
    
    conn.execute(
        """
        INSERT INTO errors (execution_id, timestamp, component, error_type, error_message, law_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (exec_id, now, "parser", "XMLError", "Invalid XML", "kstg")
    )
    
    conn.close()
    
    # Create notification content
    content = create_notification_content(exec_id, temp_tracking_db)
    
    # Check content
    assert exec_id in content
    assert "partial_success" in content
    assert "Laws Processed: 5" in content
    assert "scraper: Download failed" in content
    assert "parser: Invalid XML" in content