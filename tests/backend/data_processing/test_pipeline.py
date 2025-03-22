"""
Unit tests for the data processing pipeline.
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from taxpilot.backend.data_processing.pipeline import (
    PipelineConfig, ProcessingResult, PipelineResult,
    process_law, run_pipeline
)
from taxpilot.backend.data_processing.scraper import (
    ScraperConfig, DownloadResult
)
from taxpilot.backend.data_processing.xml_parser import (
    ParserConfig, Law, LawMetadata, Section, SectionContent
)
from taxpilot.backend.data_processing.database import (
    DbConfig, Law as DbLaw, Section as DbSection
)
from taxpilot.backend.data_processing.tracking import (
    TrackingConfig
)


@pytest.fixture
def test_config(tmp_path):
    """Return a test pipeline configuration."""
    # Create a unique tracking DB for each test run
    tracking_path = tmp_path / "tracking.db"
    
    return PipelineConfig(
        scraper_config=ScraperConfig(
            download_dir=Path("/tmp/test_pipeline")
        ),
        parser_config=ParserConfig(),
        db_config=DbConfig(
            db_path=":memory:"
        ),
        tracking_config=TrackingConfig(
            tracking_db_path=tracking_path
        )
    )


@pytest.fixture
def test_download_result():
    """Return a test download result for a law."""
    return DownloadResult(
        law_id="test_law",
        law_name="Test Law",
        file_path=Path("/tmp/test_pipeline/test_law/xml.zip"),
        source_url="https://example.com/test_law.xml",
        last_updated_website="Stand: 2023-01-01",
        status="new",
        error=None,
        timestamp=datetime.now(),
        file_hash="test_hash"
    )


@pytest.fixture
def test_law():
    """Return a test Law object."""
    return Law(
        metadata=LawMetadata(
            law_id="test_law",
            abbreviation="TL",
            full_title="Test Law",
            issue_date="2023-01-01",
            publication_info="Test Publication",
            last_changed="Last changed on 2023-03-01",
            status_info="Current"
        ),
        sections=[
            Section(
                section_id="test_law_section_1",
                law_id="test_law",
                section_number="§ 1",
                title="Test Section 1",
                content=SectionContent(
                    text="This is the content of section 1",
                    html="<p>This is the content of section 1</p>"
                ),
                parent_id=None,
                level=1,
                order_index=1
            ),
            Section(
                section_id="test_law_section_2",
                law_id="test_law",
                section_number="§ 2",
                title="Test Section 2",
                content=SectionContent(
                    text="This is the content of section 2",
                    html="<p>This is the content of section 2</p>"
                ),
                parent_id=None,
                level=1,
                order_index=2
            )
        ]
    )


@patch("taxpilot.backend.data_processing.pipeline.process_law_file")
@patch("taxpilot.backend.data_processing.pipeline.get_law")
@patch("taxpilot.backend.data_processing.pipeline.insert_law")
@patch("taxpilot.backend.data_processing.pipeline.insert_section")
def test_process_law_new(mock_insert_section, mock_insert_law, mock_get_law, 
                        mock_process_file, test_config, test_download_result, test_law):
    """Test processing a new law."""
    # Set up mocks
    mock_process_file.return_value = test_law
    mock_get_law.return_value = None  # Law doesn't exist yet
    
    # Call the function
    result = process_law(test_download_result, test_config)
    
    # Check results
    assert result.law_id == "test_law"
    assert result.law_name == "Test Law"
    assert result.status == "success"
    assert result.sections_processed == 2
    assert result.error is None
    
    # Verify law and sections were added
    mock_insert_law.assert_called_once()
    assert mock_insert_section.call_count == 2


@patch("taxpilot.backend.data_processing.pipeline.process_law_file")
@patch("taxpilot.backend.data_processing.pipeline.get_law")
@patch("taxpilot.backend.data_processing.pipeline.get_sections_by_law")
@patch("taxpilot.backend.data_processing.pipeline.insert_law")
@patch("taxpilot.backend.data_processing.pipeline.insert_section")
def test_process_law_update(mock_insert_section, mock_insert_law, mock_get_sections, 
                           mock_get_law, mock_process_file, 
                           test_config, test_download_result, test_law):
    """Test updating an existing law."""
    # Set up mocks
    mock_process_file.return_value = test_law
    mock_get_law.return_value = {"id": "test_law"}  # Law exists
    mock_get_sections.return_value = [{"id": "test_law_section_1"}]  # Only first section exists
    
    # Call the function
    result = process_law(test_download_result, test_config)
    
    # Check results
    assert result.law_id == "test_law"
    assert result.law_name == "Test Law"
    assert result.status == "success"
    assert result.sections_processed == 2
    assert result.error is None
    
    # Verify law was updated (replaced) and sections were inserted
    mock_insert_law.assert_called_once()
    assert mock_insert_section.call_count == 2


def test_process_law_error_status(test_config):
    """Test processing a law with error status."""
    # Create a download result with error status
    download_result = DownloadResult(
        law_id="test_law",
        law_name="Test Law",
        status="error",
        error="Test error"
    )
    
    # Call the function
    result = process_law(download_result, test_config)
    
    # Check results
    assert result.law_id == "test_law"
    assert result.status == "error"
    assert result.error == "Test error"
    assert result.sections_processed == 0


def test_process_law_unchanged_status(test_config):
    """Test processing a law with unchanged status."""
    # Create a download result with unchanged status
    download_result = DownloadResult(
        law_id="test_law",
        law_name="Test Law",
        status="unchanged"
    )
    
    # Call the function
    result = process_law(download_result, test_config)
    
    # Check results
    assert result.law_id == "test_law"
    assert result.status == "unchanged"
    assert result.sections_processed == 0


def test_process_law_force_update(test_config, test_download_result, test_law):
    """Test forcing an update of an unchanged law."""
    # Set up test_config with force_update=True
    test_config.force_update = True
    
    # Set up test_download_result as unchanged
    test_download_result.status = "unchanged"
    
    # Test with mocks
    with patch("taxpilot.backend.data_processing.pipeline.process_law_file") as mock_process_file, \
         patch("taxpilot.backend.data_processing.pipeline.get_law") as mock_get_law, \
         patch("taxpilot.backend.data_processing.pipeline.insert_law") as mock_insert_law, \
         patch("taxpilot.backend.data_processing.pipeline.insert_section") as mock_insert_section:
        
        # Set up mocks
        mock_process_file.return_value = test_law
        mock_get_law.return_value = None  # Law doesn't exist yet
        
        # Call the function
        result = process_law(test_download_result, test_config)
        
        # Check results
        assert result.status == "success"
        assert mock_process_file.called  # Should be called even though status is unchanged
        assert mock_insert_law.called


@patch("taxpilot.backend.data_processing.pipeline.initialize_database")
@patch("taxpilot.backend.data_processing.pipeline.run_scheduled_scraping")
@patch("taxpilot.backend.data_processing.pipeline.process_law")
def test_run_pipeline(mock_process_law, mock_scraping, mock_initialize_db, test_config):
    """Test running the complete pipeline."""
    # Set up mocks
    mock_scraping.return_value = MagicMock(
        results={
            "law1": DownloadResult(law_id="law1", law_name="Law One", status="new"),
            "law2": DownloadResult(law_id="law2", law_name="Law Two", status="unchanged"),
            "law3": DownloadResult(law_id="law3", law_name="Law Three", status="error", error="Test error")
        }
    )
    
    mock_process_law.side_effect = [
        ProcessingResult(law_id="law1", law_name="Law One", status="success", sections_processed=10),
        ProcessingResult(law_id="law2", law_name="Law Two", status="unchanged", sections_processed=0),
        ProcessingResult(law_id="law3", law_name="Law Three", status="error", error="Test error")
    ]
    
    # Call the function
    result = run_pipeline(test_config)
    
    # Check results
    assert isinstance(result, PipelineResult)
    assert len(result.results) == 3
    assert result.results["law1"].status == "success"
    assert result.results["law2"].status == "unchanged"
    assert result.results["law3"].status == "error"
    
    assert result.summary["success"] == 1
    assert result.summary["unchanged"] == 1
    assert result.summary["error"] == 1
    assert result.summary["sections_processed"] == 10
    
    # Verify pipeline steps were executed
    mock_initialize_db.assert_called_once()
    mock_scraping.assert_called_once()
    assert mock_process_law.call_count == 3


@patch("taxpilot.backend.data_processing.pipeline.initialize_database")
@patch("taxpilot.backend.data_processing.pipeline.run_scheduled_scraping")
@patch("taxpilot.backend.data_processing.pipeline.process_law")
def test_run_pipeline_with_exception(mock_process_law, mock_scraping, mock_initialize_db, test_config):
    """Test handling exceptions during pipeline execution."""
    # Set up mocks
    mock_scraping.return_value = MagicMock(
        results={
            "law1": DownloadResult(law_id="law1", law_name="Law One", status="new"),
            "law2": DownloadResult(law_id="law2", law_name="Law Two", status="new")
        }
    )
    
    # First call works, second throws exception
    mock_process_law.side_effect = [
        ProcessingResult(law_id="law1", law_name="Law One", status="success", sections_processed=10),
        Exception("Test exception")
    ]
    
    # Call the function
    result = run_pipeline(test_config)
    
    # Check results
    assert isinstance(result, PipelineResult)
    assert len(result.results) == 2
    
    assert result.results["law1"].status == "success"
    assert result.results["law2"].status == "error"
    assert "Test exception" in result.results["law2"].error
    
    assert result.summary["success"] == 1
    assert result.summary["error"] == 1
    assert result.summary["sections_processed"] == 10
    
    # Verify pipeline steps were executed
    mock_initialize_db.assert_called_once()
    mock_scraping.assert_called_once()
    assert mock_process_law.call_count == 2