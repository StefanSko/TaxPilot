# TaxPilot Data Processing

A robust pipeline for scraping, parsing, and storing German tax laws in a structured database.

## Overview

The TaxPilot data processing package automatically downloads, parses, and indexes German tax laws from official government sources. It focuses on the 5 key tax laws (EStG, KStG, UStG, AO, GewStG) and provides a complete ETL pipeline from web scraping to database storage.

## Features

- **Automated Scraping**: Downloads tax law XML files from gesetze-im-internet.de
- **Intelligent XML Parsing**: Processes complex legal documents into structured data
- **Persistent Storage**: Stores laws in a DuckDB database for efficient querying
- **Change Detection**: Only processes laws that have been updated
- **Execution Tracking**: Records statistics and errors for monitoring
- **Serverless Ready**: Designed to run in serverless environments like Modal.com

## Package Structure

```
taxpilot/backend/data_processing/
├── scraper.py       # Downloads law files from government websites
├── xml_parser.py    # Parses XML files into structured data
├── database.py      # Manages database connections and queries
├── pipeline.py      # Orchestrates the entire workflow
├── tracking.py      # Records execution statistics
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/taxpilot.git

# Install dependencies using Poetry
cd taxpilot
poetry install
```

## Usage

### Running the Pipeline Locally

```python
from taxpilot.backend.data_processing.pipeline import run_pipeline

# Run with default configuration
result = run_pipeline()

# Print results summary
print(f"Pipeline completed: {result.summary}")
```

### Custom Configuration

```python
from taxpilot.backend.data_processing.pipeline import run_pipeline, PipelineConfig
from taxpilot.backend.data_processing.database import DbConfig

# Create custom configuration
config = PipelineConfig(
    force_update=True,  # Force update even if laws haven't changed
    db_config=DbConfig(
        db_path="custom/path/to/laws.db",
        read_only=False
    )
)

# Run with custom configuration
result = run_pipeline(config)
```

### Serverless Deployment

The package includes Modal.com integration for serverless execution:

```bash
# Deploy the scheduled pipeline
poetry run python -m taxpilot.infrastructure.pipeline_scheduler
```

## Database Schema

The database stores the following information:

### Laws Table

Stores metadata about each law:

- `id`: Unique identifier (e.g., "estg")
- `full_name`: Complete name of the law
- `abbreviation`: Official abbreviation (e.g., "EStG")
- `last_updated`: Date when the law was last updated
- `issue_date`: Original issue date
- `status_info`: Current status information
- `metadata`: Additional JSON metadata

### Sections Table

Stores individual sections of laws:

- `id`: Unique identifier for the section
- `law_id`: Foreign key to the laws table
- `section_number`: Section number (e.g., "2")
- `title`: Section title
- `content`: Full text content
- `parent_section_id`: For hierarchical structure
- `hierarchy_level`: Depth in document structure
- `path`: Full hierarchical path
- `metadata`: Additional JSON metadata

### Section Embeddings Table

Stores vector embeddings for semantic search:

- `section_id`: Foreign key to the sections table
- `embedding`: Vector representation of the section content

## Configuration Options

All configuration is handled through Pydantic models:

### Pipeline Configuration

```python
from taxpilot.backend.data_processing.pipeline import PipelineConfig

config = PipelineConfig(
    force_update=False,              # Force update of laws even if unchanged
    enable_transactions=True,        # Enable database transactions
    send_notifications=True,         # Send notifications on completion/failure
    notification_threshold=1         # Number of errors to trigger notification
)
```

### Database Configuration

```python
from taxpilot.backend.data_processing.database import DbConfig

db_config = DbConfig(
    db_path="data/processed/laws.duckdb",  # Path to database file
    read_only=False,                       # Open in read-only mode
    memory_limit="4GB"                     # Memory limit for DuckDB
)
```

## Development

### Running Tests

```bash
poetry run pytest tests/backend/data_processing
```

### Type Checking

```bash
poetry run mypy taxpilot/backend/data_processing
```

## Requirements

- Python 3.12+
- Poetry for dependency management
- Dependencies:
  - lxml
  - python-dotenv
  - duckdb
  - pydantic
  - requests
  - beautifulsoup4
  - psutil

## License

[MIT License](LICENSE)
