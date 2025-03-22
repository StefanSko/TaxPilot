# GermanLawFinder - Local Development & Usage Guide

This guide provides instructions for setting up and running the GermanLawFinder (TaxPilot) project locally.

## Prerequisites

- Python 3.12 or higher
- Poetry (dependency management)
- Git
- At least 2GB of free disk space
- Recommended: 8GB RAM for embedding generation

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/TaxPilot.git
cd TaxPilot
```

2. **Install dependencies with Poetry**

```bash
poetry install
```

3. **Activate the virtual environment**

```bash
poetry shell
```

## Running the Project Locally

### 1. Data Pipeline

The data pipeline consists of several steps:

#### a. Extract and Parse XML Data

```bash
python -m taxpilot.backend.data_processing.pipeline run --laws estg,kstg,ustg,ao,gewstg
```

This command will:
- Extract XML files from the data directory
- Parse the German law XML structure
- Store laws and sections in the DuckDB database

#### b. Generate Embeddings

```bash
python -m taxpilot.backend.search.embeddings run
```

This will:
- Process text from the database
- Generate vector embeddings using the German BERT model
- Store embeddings in the DuckDB database

#### c. Initialize Vector Database

```bash
python -m taxpilot.backend.search.vector_db sync
```

This will:
- Initialize the Qdrant vector database
- Synchronize embeddings from DuckDB to Qdrant
- Create appropriate indices for efficient search

### 2. Running the Search API

You can run the search API locally using Uvicorn:

```bash
uvicorn taxpilot.backend.api.app:app --reload
```

The API will be available at http://127.0.0.1:8000 with the following endpoints:
- `/api/search` - POST endpoint for semantic search
- `/api/laws` - GET endpoint to list available laws
- `/health` - GET endpoint for health check
- `/docs` - Interactive API documentation

## Running on Modal.com (Serverless)

For deploying to Modal.com, you'll need:

1. **Install Modal CLI**

```bash
pip install modal
```

2. **Configure Modal**

```bash
modal token new
```

3. **Deploy the API**

```bash
python -m taxpilot.infrastructure.vector_db_config
```

This will deploy the serverless application to Modal.com with:
- Vector database integration
- Search API endpoints
- Scheduled synchronization and optimization

## Running Tests

To run all tests:

```bash
poetry run pytest
```

To run specific test modules:

```bash
poetry run pytest tests/backend/search/test_vector_db.py
```

## Project Structure Overview

- `taxpilot/backend/data_processing/` - Data extraction and processing modules
  - `pipeline.py` - Main ETL pipeline
  - `xml_parser.py` - German law XML parser
  - `database.py` - DuckDB database integration
  - `tracking.py` - Execution tracking system

- `taxpilot/backend/search/` - Search functionality
  - `embeddings.py` - Vector embedding generation
  - `segmentation.py` - Text segmentation for embedding
  - `vector_db.py` - Qdrant vector database integration
  - `search_api.py` - Search service implementation

- `taxpilot/backend/api/` - API interfaces
  - `app.py` - FastAPI application with endpoints

- `taxpilot/infrastructure/` - Deployment configurations
  - `vector_db_config.py` - Modal.com configuration for vector search
  - `embedding_config.py` - Modal.com configuration for embedding generation
  - `pipeline_scheduler.py` - Modal.com scheduled pipeline

## Configuration

The project uses environment variables for configuration:

```bash
# Create a .env file
touch .env

# Add configuration
echo "QDRANT_URL=http://localhost:6333" >> .env
echo "USE_GPU=false" >> .env
```

Key environment variables:
- `QDRANT_URL` - URL for Qdrant vector database (default: http://localhost:6333)
- `QDRANT_API_KEY` - API key for Qdrant Cloud (if used)
- `USE_GPU` - Whether to use GPU for embedding generation (default: false)
- `MODEL_NAME` - Embedding model to use (default: deepset/gbert-base)

## Troubleshooting

### Common Issues

1. **Database connection errors**
   - Check if database files have appropriate permissions
   - Ensure DuckDB version compatibility

2. **Vector database synchronization issues**
   - Verify Qdrant is running locally (`docker run -p 6333:6333 qdrant/qdrant`)
   - Check for sufficient disk space

3. **Embedding generation errors**
   - Ensure enough memory for model loading
   - Try with smaller batch sizes (`EMBEDDING_BATCH_SIZE=16`)

4. **API errors**
   - Check API endpoint URLs and payloads
   - Verify vector database connection

## Examples

### Sample Search Query

```python
import requests

# Search for tax-related information
response = requests.post(
    "http://localhost:8000/api/search",
    json={
        "query": "Homeoffice Steuerabzug",
        "filters": {"law_id": "estg"},
        "page": 1,
        "limit": 5
    }
)

# Print results
for result in response.json()["results"]:
    print(f"{result['section_number']} - {result['title']}")
    print(f"Score: {result['relevance_score']}")
    print(result['content_with_highlights'])
    print("-" * 50)
```

### Modal.com API Usage

```python
from modal import Stub, web_endpoint
from taxpilot.infrastructure.vector_db_config import app

# Example usage with Modal client
@web_endpoint
def search_laws(query: str, law_id: str = None):
    results = app.app.search.remote(query, law_id=law_id, top_k=5)
    return {"results": results}
```