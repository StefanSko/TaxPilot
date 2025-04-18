"""
Web scraper module for extracting German tax laws from government websites.

This module provides functions to scrape and download tax laws from 
gesetze-im-internet.de, focusing on the 5 key tax laws (EStG, KStG, UStG, AO, GewStG).
"""

import os
import time
import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal, Self, TypedDict, NotRequired
from pydantic import BaseModel, Field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("law_scraper")


# Constants
TAX_LAWS = {
    "estg": "Einkommensteuergesetz",
    "kstg_1977": "Körperschaftsteuergesetz",
    "ustg_1980": "Umsatzsteuergesetz",
    "ao_1977": "Abgabenordnung",
    "gewstg": "Gewerbesteuergesetz",
}

BASE_URL = "https://www.gesetze-im-internet.de"


# Type definitions
LawStatus = Literal["unchanged", "updated", "new", "error", "skipped"]


# Pydantic models for configuration
class ScraperConfig(BaseModel):
    """Configuration for the law scraper."""
    download_dir: Path = Field(
        default=Path("data"),
        description="Directory where downloaded files will be stored"
    )
    base_url: str = Field(
        default="https://www.gesetze-im-internet.de",
        description="Base URL for the law website"
    )
    metadata_file: str = Field(
        default="metadata.json",
        description="Filename for storing metadata about downloads"
    )
    check_interval_days: int = Field(
        default=30,
        description="Interval in days between checks for updates"
    )
    timeout: int = Field(
        default=30,
        description="Timeout for HTTP requests in seconds"
    )
    retry_attempts: int = Field(
        default=5,
        description="Number of retry attempts for HTTP requests"
    )
    verify_ssl: bool = Field(default=False, description="Verify SSL certificates when scraping")
    
    @property
    def laws_to_scrape(self) -> dict[str, str]:
        """Get the list of laws to scrape."""
        return TAX_LAWS


class DownloadResult(BaseModel):
    """Result of a law download operation."""
    law_id: str
    law_name: str
    file_path: Path | None = None
    source_url: str | None = None
    last_updated_website: str | None = None
    status: LawStatus
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    file_hash: str | None = None


class ScraperResult(BaseModel):
    """Result of a scraping operation for all laws."""
    results: dict[str, DownloadResult]
    timestamp: datetime = Field(default_factory=datetime.now)
    summary: dict[str, int] = Field(default_factory=dict)


def create_session(config: ScraperConfig) -> requests.Session:
    """
    Create a requests session with retry logic for robust network requests.
    
    Args:
        config: The scraper configuration.
    
    Returns:
        A configured requests session.
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=config.retry_attempts,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def get_law_url(law_id: str, config: ScraperConfig) -> str:
    """
    Get the URL for a specific law.
    
    Args:
        law_id: The ID of the law (e.g., "estg").
        config: The scraper configuration.
        
    Returns:
        The URL to the law's page.
    """
    return f"{config.base_url}/{law_id}/"


def scrape_law_page(session: requests.Session, law_id: str, config: ScraperConfig) -> str | None:
    """
    Get the direct XML download link for a law.
    
    Args:
        session: The requests session.
        law_id: The ID of the law (e.g., "estg").
        config: The scraper configuration.
        
    Returns:
        A tuple containing (xml_url, last_updated_date) or (None, None) if not found.
        The last_updated_date will always be None with this direct approach.
    """
    url = get_law_url(law_id, config)
    xml_url = f"{url}xml.zip"
    
    logger.info(f"Trying direct XML URL: {xml_url}")
    
    try:
        # Try to access the XML zip file directly
        response = session.head(xml_url, timeout=config.timeout, verify=config.verify_ssl)
        response.raise_for_status()
        
        # If we get here, the URL exists
        logger.info(f"Found direct XML URL: {xml_url}")
        
        # Return the URL without attempting to get a date
        return xml_url
    except requests.RequestException as e:
        logger.error(f"Error accessing {xml_url}: {e}")
        return None


def download_xml(session: requests.Session, url: str, law_id: str, config: ScraperConfig) -> Path | None:
    """
    Download the XML file for a law.
    
    Args:
        session: The requests session.
        url: The URL to the XML file.
        law_id: The ID of the law.
        config: The scraper configuration.
        
    Returns:
        The path to the downloaded file or None if download failed.
    """
    logger.info(f"Downloading XML from {url}")
    
    try:
        response = session.get(url, timeout=config.timeout * 2, verify=config.verify_ssl)  # Longer timeout for downloads
        response.raise_for_status()
        
        # Create directory for this law if it doesn't exist
        law_dir = config.download_dir / law_id
        law_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the XML file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = law_dir / f"xml_{timestamp}.zip"
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
            
        logger.info(f"Downloaded XML file to {file_path}")
        
        # Create a simpler symlink for the latest version
        latest_path = law_dir / "xml.zip"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(file_path.name)
        
        return file_path
        
    except requests.RequestException as e:
        logger.error(f"Error downloading from {url}: {e}")
        return None
    except OSError as e:
        logger.error(f"Error saving file: {e}")
        return None


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate the SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        The hexadecimal digest of the hash.
    """
    hash_obj = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
            
    return hash_obj.hexdigest()


def check_for_changes(law_id: str, new_file_path: Path, config: ScraperConfig) -> LawStatus:
    """
    Check if a law has changed by comparing the hash of the new file with the previous version.
    
    Args:
        law_id: The ID of the law.
        new_file_path: Path to the newly downloaded file.
        config: The scraper configuration.
        
    Returns:
        The status of the law: "new", "updated", "unchanged", or "error".
    """
    new_hash = calculate_file_hash(new_file_path)
    
    # Look for previous versions
    law_dir = config.download_dir / law_id
    previous_files = sorted([f for f in law_dir.glob("xml_*.zip") if f != new_file_path], 
                           key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not previous_files:
        logger.info(f"No previous version found for {law_id}. This is a new law.")
        return "new"
        
    previous_file = previous_files[0]
    previous_hash = calculate_file_hash(previous_file)
    
    if new_hash == previous_hash:
        logger.info(f"No changes detected for {law_id}. Files are identical.")
        # Remove the new file since it's a duplicate
        new_file_path.unlink()
        return "unchanged"
    else:
        logger.info(f"Changes detected for {law_id}.")
        return "updated"


def save_metadata(result: DownloadResult, config: ScraperConfig) -> None:
    """
    Save metadata about the downloaded law.
    
    Args:
        result: The download result.
        config: The scraper configuration.
    """
    law_dir = config.download_dir / result.law_id
    metadata_path = law_dir / config.metadata_file
    
    # Make sure the directory exists
    law_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable dict
    metadata = result.model_dump()
    
    # Convert Path objects to strings for JSON serialization
    if metadata["file_path"] is not None:
        metadata["file_path"] = str(metadata["file_path"])
    
    metadata["timestamp"] = metadata["timestamp"].isoformat()
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Saved metadata to {metadata_path}")


def scrape_law(law_id: str, config: ScraperConfig) -> DownloadResult:
    """
    Scrape a specific law from gesetze-im-internet.de.
    
    Args:
        law_id: The ID of the law (e.g., "estg").
        config: The scraper configuration.
        
    Returns:
        A DownloadResult with information about the operation.
    """
    result = DownloadResult(
        law_id=law_id,
        law_name=config.laws_to_scrape.get(law_id, "Unknown"),
        status="error"
    )
    
    session = create_session(config)
    
    try:
        # Get the XML download URL
        xml_url = scrape_law_page(session, law_id, config)
        result.source_url = xml_url
        
        if not xml_url:
            logger.error(f"Failed to find XML URL for {law_id}")
            result.error = "No XML URL found"
            return result
            
        # Download the XML file
        file_path = download_xml(session, xml_url, law_id, config)
        
        if not file_path:
            logger.error(f"Failed to download XML for {law_id}")
            result.error = "Download failed"
            return result
            
        # Set file path and calculate hash
        result.file_path = file_path
        result.file_hash = calculate_file_hash(file_path)
        
        # Check if the law has changed
        result.status = check_for_changes(law_id, file_path, config)
        
    except Exception as e:
        logger.error(f"Unexpected error scraping {law_id}: {e}", exc_info=True)
        result.error = str(e)
        
    # Save metadata
    save_metadata(result, config)
    
    return result


def scrape_all_laws(config: ScraperConfig) -> ScraperResult:
    """
    Scrape all tax laws defined in the configuration.
    
    Args:
        config: The scraper configuration.
        
    Returns:
        A ScraperResult with results for each law.
    """
    results: dict[str, DownloadResult] = {}
    
    for law_id, law_name in config.laws_to_scrape.items():
        logger.info(f"Starting to scrape {law_id} ({law_name})")
        
        try:
            result = scrape_law(law_id, config)
            results[law_id] = result
            
            # Be nice to the server
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Unexpected error scraping {law_id}: {e}", exc_info=True)
            results[law_id] = DownloadResult(
                law_id=law_id,
                law_name=law_name,
                status="error",
                error=str(e)
            )
    
    # Create a summary of the results
    summary = {
        "new": sum(1 for law in results if results[law].status == "new"),
        "updated": sum(1 for law in results if results[law].status == "updated"),
        "unchanged": sum(1 for law in results if results[law].status == "unchanged"),
        "error": sum(1 for law in results if results[law].status == "error"),
    }
    
    return ScraperResult(results=results, summary=summary)


def should_run_scraper(config: ScraperConfig) -> bool:
    """
    Check if the scraper should run based on the last run time.
    
    Args:
        config: The scraper configuration.
        
    Returns:
        True if the scraper should run, False otherwise.
    """
    # Always run if download directory doesn't exist
    if not config.download_dir.exists():
        return True
    
    # Check each law directory for a metadata file
    for law_id in config.laws_to_scrape:
        metadata_path = config.download_dir / law_id / config.metadata_file
        
        if not metadata_path.exists():
            # No metadata file, so we should run
            return True
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # Check if the last run was too long ago
            last_run = datetime.fromisoformat(metadata.get('timestamp', '2000-01-01T00:00:00'))
            days_since_last_run = (datetime.now() - last_run).days
            
            if days_since_last_run >= config.check_interval_days:
                return True
                
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.error(f"Error reading metadata file for {law_id}: {e}")
            return True
    
    return False


def run_scheduled_scraping(config: ScraperConfig | None = None) -> ScraperResult:
    """
    Run a scheduled scraping job for all tax laws.
    
    This function should be called by a scheduler (e.g., cron) monthly.
    
    Args:
        config: Optional configuration for the scraper.
        
    Returns:
        A ScraperResult with results for each law.
    """
    if config is None:
        config = ScraperConfig()
    
    logger.info("Starting scheduled scraping of tax laws")
    
    # Check if we should run the scraper
    if not should_run_scraper(config):
        logger.info("Skipping scraping as it was run recently")
        return ScraperResult(
            results={},
            summary={"skipped": len(config.laws_to_scrape)}
        )
    
    start_time = time.time()
    result = scrape_all_laws(config)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Scraping completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results summary: {result.summary}")
    
    return result


if __name__ == "__main__":
    # If run directly, execute a scraping job
    run_scheduled_scraping()