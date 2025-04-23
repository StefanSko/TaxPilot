"""
Unit tests for the scraper module that downloads German tax laws.
"""

import os
from datetime import datetime, timedelta
import json
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, mock_open
import requests
from taxpilot.backend.data_processing.scraper import (
    ScraperConfig, DownloadResult, ScraperResult,
    create_session, get_law_url, scrape_law_page, download_xml,
    calculate_file_hash, check_for_changes, save_metadata,
    scrape_law, scrape_all_laws, should_run_scraper, run_scheduled_scraping
)


@pytest.fixture
def test_config():
    """Return a test scraper configuration."""
    return ScraperConfig(
        download_dir=Path("/tmp/test_scraper"),
        base_url="https://test.example.com",
        metadata_file="test_metadata.json",
        check_interval_days=1,
        timeout=1,
        retry_attempts=1
    )


@pytest.fixture
def mock_session():
    """Create a mock requests session."""
    mock = MagicMock()
    return mock


def test_create_session(test_config):
    """Test session creation with retry logic."""
    session = create_session(test_config)
    assert isinstance(session, requests.Session)
    # Verify the session has the retry adapter
    assert session.adapters.get("http://")
    assert session.adapters.get("https://")


def test_get_law_url(test_config):
    """Test generating URLs for laws."""
    url = get_law_url("estg", test_config)
    assert url == "https://test.example.com/estg/"


def test_scrape_law_page_success(mock_session, test_config):
    """Test scraping the law page to find XML download link."""
    # Set up mock response for HEAD request
    mock_response = MagicMock()
    mock_session.head.return_value = mock_response
    mock_session.head.return_value.raise_for_status.return_value = None
    
    # Call the function
    xml_url = scrape_law_page(mock_session, "estg", test_config)
    
    # Check results
    assert xml_url == "https://test.example.com/estg/xml.zip"


def test_scrape_law_page_failure(mock_session, test_config):
    """Test error handling when scraping fails."""
    # Set up mock to raise an exception
    mock_session.head.side_effect = requests.RequestException("Test error")
    
    # Call the function
    xml_url = scrape_law_page(mock_session, "estg", test_config)
    
    # Check results
    assert xml_url is None


@patch("builtins.open", new_callable=mock_open)
@patch("taxpilot.backend.data_processing.scraper.Path.mkdir")
@patch("taxpilot.backend.data_processing.scraper.Path.exists")
@patch("taxpilot.backend.data_processing.scraper.Path.unlink")
@patch("taxpilot.backend.data_processing.scraper.Path.symlink_to")
def test_download_xml_success(mock_symlink, mock_unlink, mock_exists, mock_mkdir, 
                             mock_file, mock_session, test_config):
    """Test downloading the XML file."""
    # Set up mocks
    mock_response = MagicMock()
    mock_response.content = b"test xml content"
    mock_session.get.return_value = mock_response
    mock_session.get.return_value.raise_for_status.return_value = None
    mock_exists.return_value = True
    
    # Call the function
    file_path = download_xml(mock_session, "https://test.example.com/estg.xml", "estg", test_config)
    
    # Check results
    assert file_path is not None
    mock_file.assert_called()
    mock_mkdir.assert_called_once()
    mock_symlink.assert_called_once()


def test_download_xml_failure(mock_session, test_config):
    """Test error handling when download fails."""
    # Set up mock to raise an exception
    mock_session.get.side_effect = requests.RequestException("Test error")
    
    # Call the function
    file_path = download_xml(mock_session, "https://test.example.com/estg.xml", "estg", test_config)
    
    # Check results
    assert file_path is None


@patch("builtins.open", new_callable=mock_open, read_data=b"test file content")
def test_calculate_file_hash(mock_file):
    """Test hash calculation for files."""
    # Call the function
    file_hash = calculate_file_hash(Path("/tmp/test_file.xml"))
    
    # Check results
    assert isinstance(file_hash, str)
    assert len(file_hash) == 64  # SHA-256 hex digest length
    mock_file.assert_called_once_with(Path("/tmp/test_file.xml"), 'rb')


@patch("taxpilot.backend.data_processing.scraper.calculate_file_hash")
@patch("taxpilot.backend.data_processing.scraper.Path.glob")
@patch("taxpilot.backend.data_processing.scraper.Path.stat")
@patch("taxpilot.backend.data_processing.scraper.Path.unlink")
def test_check_for_changes_unchanged(mock_unlink, mock_stat, mock_glob, mock_hash, test_config):
    """Test checking for changes when file is unchanged."""
    # Set up mocks
    mock_hash.return_value = "test_hash"
    previous_file = Path("/tmp/test_scraper/estg/xml_20220101_000000.zip")
    mock_glob.return_value = [previous_file]
    
    # Mock stat return value
    mock_stat_result = MagicMock()
    mock_stat_result.st_mtime = 1000000  # Some timestamp
    mock_stat.return_value = mock_stat_result
    
    # Call the function
    result = check_for_changes("estg", Path("/tmp/test_scraper/estg/xml_20230101_000000.zip"), test_config)
    
    # Check results
    assert result == "unchanged"
    mock_unlink.assert_called_once()  # Should delete the duplicate file


@patch("taxpilot.backend.data_processing.scraper.calculate_file_hash")
@patch("taxpilot.backend.data_processing.scraper.Path.glob")
@patch("taxpilot.backend.data_processing.scraper.Path.stat")
def test_check_for_changes_updated(mock_stat, mock_glob, mock_hash, test_config):
    """Test checking for changes when file is updated."""
    # Set up mocks
    mock_hash.side_effect = ["new_hash", "old_hash"]  # Different hashes
    previous_file = Path("/tmp/test_scraper/estg/xml_20220101_000000.zip")
    mock_glob.return_value = [previous_file]
    
    # Mock stat return value
    mock_stat_result = MagicMock()
    mock_stat_result.st_mtime = 1000000  # Some timestamp
    mock_stat.return_value = mock_stat_result
    
    # Call the function
    result = check_for_changes("estg", Path("/tmp/test_scraper/estg/xml_20230101_000000.zip"), test_config)
    
    # Check results
    assert result == "updated"


@patch("taxpilot.backend.data_processing.scraper.Path.glob")
@patch("taxpilot.backend.data_processing.scraper.calculate_file_hash")
def test_check_for_changes_new(mock_hash, mock_glob, test_config):
    """Test checking for changes when file is new."""
    # Set up mocks - no previous files
    mock_glob.return_value = []
    mock_hash.return_value = "test_hash"
    
    # Call the function
    result = check_for_changes("estg", Path("/tmp/test_scraper/estg/xml_20230101_000000.zip"), test_config)
    
    # Check results
    assert result == "new"


@patch("builtins.open", new_callable=mock_open)
@patch("taxpilot.backend.data_processing.scraper.Path.mkdir")
@patch("json.dump")
def test_save_metadata(mock_json_dump, mock_mkdir, mock_file, test_config):
    """Test saving metadata for downloaded files."""
    # Set up test data
    result = DownloadResult(
        law_id="estg",
        law_name="Einkommensteuergesetz",
        file_path=Path("/tmp/test_scraper/estg/xml_20230101_000000.zip"),
        source_url="https://test.example.com/estg.xml",
        last_updated_website="Stand: 2023-01-01",
        status="updated",
        error=None,
        timestamp=datetime.now(),
        file_hash="test_hash"
    )
    
    # Call the function
    save_metadata(result, test_config)
    
    # Check results
    mock_mkdir.assert_called_once()
    mock_file.assert_called_once()
    mock_json_dump.assert_called_once()
    # Check that the Path was converted to string in the JSON
    args, kwargs = mock_json_dump.call_args
    assert isinstance(args[0]["file_path"], str)


@patch("taxpilot.backend.data_processing.scraper.create_session")
@patch("taxpilot.backend.data_processing.scraper.scrape_law_page")
@patch("taxpilot.backend.data_processing.scraper.download_xml")
@patch("taxpilot.backend.data_processing.scraper.calculate_file_hash")
@patch("taxpilot.backend.data_processing.scraper.check_for_changes")
@patch("taxpilot.backend.data_processing.scraper.save_metadata")
def test_scrape_law_success(mock_save, mock_check, mock_hash, mock_download, 
                            mock_scrape, mock_session, test_config):
    """Test successful law scraping process."""
    # Set up mocks
    mock_session.return_value = MagicMock()
    mock_scrape.return_value = ("https://test.example.com/estg.xml", "Stand: 2023-01-01")
    mock_download.return_value = Path("/tmp/test_scraper/estg/xml_20230101_000000.zip")
    mock_hash.return_value = "test_hash"
    mock_check.return_value = "updated"
    
    # Call the function
    result = scrape_law("estg", test_config)
    
    # Check results
    assert result.law_id == "estg"
    assert result.status == "updated"
    assert result.error is None
    mock_save.assert_called_once()


@patch("taxpilot.backend.data_processing.scraper.create_session")
@patch("taxpilot.backend.data_processing.scraper.scrape_law_page")
def test_scrape_law_error_no_url(mock_scrape, mock_session, test_config):
    """Test error handling when no XML URL is found."""
    # Set up mocks
    mock_session.return_value = MagicMock()
    mock_scrape.return_value = None  # No URL found
    
    # Call the function
    result = scrape_law("estg", test_config)
    
    # Check results
    assert result.law_id == "estg"
    assert result.status == "error"
    assert result.error == "No XML URL found"


@patch("taxpilot.backend.data_processing.scraper.scrape_law")
@patch("time.sleep")  # To avoid actual sleep
def test_scrape_all_laws(mock_sleep, mock_scrape, test_config):
    """Test scraping all laws."""
    # Set up mock
    mock_scrape.return_value = DownloadResult(
        law_id="estg",
        law_name="Einkommensteuergesetz",
        status="updated"
    )
    
    # Call the function
    result = scrape_all_laws(test_config)
    
    # Check results
    assert isinstance(result, ScraperResult)
    assert "estg" in result.results
    assert "summary" in result.model_dump()
    assert mock_scrape.call_count == len(test_config.laws_to_scrape)


@patch("taxpilot.backend.data_processing.scraper.Path.exists")
def test_should_run_scraper_first_time(mock_exists, test_config):
    """Test should_run_scraper for first run."""
    # Set up mock - directory doesn't exist
    mock_exists.return_value = False
    
    # Call the function
    result = should_run_scraper(test_config)
    
    # Check results
    assert result is True


@patch("taxpilot.backend.data_processing.scraper.Path.exists")
@patch("builtins.open", new_callable=mock_open, read_data='{"timestamp": "2023-01-01T00:00:00"}')
def test_should_run_scraper_interval_passed(mock_file, mock_exists, test_config):
    """Test should_run_scraper when interval has passed."""
    # Set up mocks
    mock_exists.return_value = True
    
    # Call the function with test date
    with patch("taxpilot.backend.data_processing.scraper.datetime") as mock_datetime:
        # Set current time to be after the check interval
        mock_datetime.now.return_value = datetime(2023, 1, 3)  # 2 days later, interval is 1 day
        mock_datetime.fromisoformat.return_value = datetime(2023, 1, 1)
        
        result = should_run_scraper(test_config)
        
        # Check results
        assert result is True


@patch("taxpilot.backend.data_processing.scraper.should_run_scraper")
@patch("taxpilot.backend.data_processing.scraper.scrape_all_laws")
def test_run_scheduled_scraping_skip(mock_scrape_all, mock_should_run, test_config):
    """Test skipping scheduled scraping when it was run recently."""
    # Set up mock
    mock_should_run.return_value = False
    
    # Call the function
    result = run_scheduled_scraping(test_config)
    
    # Check results
    assert result.summary.get("skipped") == len(test_config.laws_to_scrape)
    mock_scrape_all.assert_not_called()


@patch("taxpilot.backend.data_processing.scraper.should_run_scraper")
@patch("taxpilot.backend.data_processing.scraper.scrape_all_laws")
def test_run_scheduled_scraping_run(mock_scrape_all, mock_should_run, test_config):
    """Test running scheduled scraping."""
    # Set up mocks
    mock_should_run.return_value = True
    mock_scrape_all.return_value = ScraperResult(
        results={"estg": DownloadResult(law_id="estg", law_name="Test", status="updated")},
        summary={"updated": 1}
    )
    
    # Call the function
    result = run_scheduled_scraping(test_config)
    
    # Check results
    assert result.summary.get("updated") == 1
    mock_scrape_all.assert_called_once()