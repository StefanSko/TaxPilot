# Local Deployment Plan for TaxPilot

This document outlines the plan for creating a local deployment of TaxPilot that:
1. Scrapes real German tax laws
2. Processes them into a DuckDB database
3. Generates real embeddings
4. Stores them in a local Qdrant instance
5. Serves an API with Swagger documentation

## Components

### 1. Main Script (`main.py`)

A command-line script that orchestrates the entire pipeline with the following capabilities:
- Run specific parts of the pipeline or the entire process
- Configure paths, models, and other settings
- Deploy the API server after processing
- Support incremental updates

### 2. Pipeline Components

The main script will integrate these existing components:
- **Scraper**: Download laws from gesetze-im-internet.de
- **Parser**: Process XML into structured data
- **Database**: Store structured data in DuckDB
- **Embedding**: Generate text embeddings
- **Vector DB**: Store embeddings in Qdrant
- **API Server**: Serve the search API with FastAPI

## Implementation Plan

### Step 1: Command-Line Interface

Create a CLI with the following commands:
- `scrape`: Run the law scraper
- `process`: Process downloaded XML files
- `embed`: Generate embeddings
- `index`: Index embeddings in Qdrant
- `serve`: Start the API server
- `run-all`: Run the entire pipeline

### Step 2: Configuration Management

Implement configuration handling:
- File paths for input/output data
- Database connection settings
- Embedding model selection
- Qdrant connection settings
- API server settings

### Step 3: Pipeline Integration

Connect all pipeline components:
1. **Scraping Stage**: 
   - Download laws from government websites
   - Store XML files locally
   
2. **Processing Stage**:
   - Parse XML files into structured data
   - Store structured data in DuckDB
   
3. **Embedding Stage**:
   - Load processed data
   - Generate embeddings
   - Store embeddings in DuckDB
   
4. **Indexing Stage**:
   - Connect to Qdrant
   - Index embeddings from DuckDB
   - Optimize vector search
   
5. **Serving Stage**:
   - Start FastAPI server
   - Connect to databases
   - Enable Swagger UI

### Step 4: Error Handling and Logging

Implement comprehensive error handling:
- Clear error messages for each pipeline step
- Logging to file and console
- Recovery mechanisms from partial failures
- Validation checks between stages

### Step 5: API Server Configuration

Configure the API server for local use:
- Enable CORS for local development
- Set up Swagger UI with examples
- Expose health check endpoints
- Include performance metrics

## Usage Examples

### Basic Usage

```bash
# Run the entire pipeline and start server
python main.py run-all

# Run individual components
python main.py scrape
python main.py process
python main.py embed
python main.py index
python main.py serve
```

### Advanced Usage

```bash
# Use custom configuration file
python main.py run-all --config config.json

# Specify custom paths
python main.py process --input-dir ./data/raw --output-dir ./data/processed

# Use a specific embedding model
python main.py embed --model-name deepset/gbert-base

# Run with increased logging
python main.py run-all --log-level debug
```

## Implementation Details

### `main.py` Structure

```python
# High-level structure
def main():
    args = parse_arguments()
    config = load_configuration(args.config)
    
    if args.command == 'run-all':
        run_scraper(config)
        run_processor(config)
        run_embedder(config)
        run_indexer(config)
        run_server(config)
    elif args.command == 'scrape':
        run_scraper(config)
    # ... other commands
    
def run_scraper(config):
    # Initialize and run the scraper
    
def run_processor(config):
    # Initialize and run the processor
    
# ... other component functions

def run_server(config):
    # Initialize and start the API server
```

### Prerequisites

- Local Qdrant server running
  - Docker: `docker run -p 6333:6333 qdrant/qdrant`
  - Or standalone installation
- Python environment set up with Poetry
- Access to embedding model (internet connection or local model)

## Dependencies

- taxpilot.backend.data_processing.scraper
- taxpilot.backend.data_processing.pipeline
- taxpilot.backend.search.embeddings
- taxpilot.backend.search.vector_db
- taxpilot.backend.api.app

## Next Steps

1. Create `main.py` implementing the command interface
2. Create default configuration file
3. Test each pipeline component individually
4. Test the entire pipeline end-to-end
5. Document usage with examples