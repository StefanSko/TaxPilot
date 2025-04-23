# GermanLawFinder (TaxPilot)

A modern legal research platform that transforms how tax attorneys access and search German tax laws using vector search technology and a clean, user-friendly interface.

## Project Overview

GermanLawFinder provides efficient access to key German tax laws:
- Einkommensteuergesetz (EStG) - Income Tax Act
- Körperschaftsteuergesetz (KStG) - Corporate Income Tax Act
- Umsatzsteuergesetz (UStG) - Value Added Tax Act
- Abgabenordnung (AO) - Fiscal Code
- Gewerbesteuergesetz (GewStG) - Trade Tax Act

## Technology Stack

- **Backend**: Python 3.12, FastAPI, Modal.com serverless
- **Database**: DuckDB and vector database for embeddings
- **Frontend**: Vue.js 3 with Composition API
- **XML Processing**: lxml for German legal document parsing

## Setup Instructions

### Prerequisites
- Python 3.12
- Poetry package manager
- Node.js and npm (for frontend development)
- Docker and Docker Compose (optional, for containerized deployment)

### Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd taxpilot
   ```

2. Install backend dependencies:
   ```
   poetry install
   ```

3. Install frontend dependencies:
   ```
   cd frontend
   npm install
   ```

4. Create a `.env` file based on `.env.example`

### Development

#### Running the Local Server

Use the `main.py` script to run different parts of the pipeline:

```bash
# Run the entire pipeline and start server
poetry run python main.py run-all

# Run individual components
poetry run python main.py scrape    # Download laws
poetry run python main.py process   # Process XML files
poetry run python main.py embed     # Generate embeddings
poetry run python main.py index     # Index in Qdrant
poetry run python main.py serve     # Start API server
```

#### Docker Deployment

For a complete deployment with Qdrant vector database, use Docker Compose:

```bash
# Build and start the containers
docker-compose up -d

# Run the complete pipeline within the container
docker-compose exec taxpilot python main.py run-all

# Optionally, run just the API server
docker-compose exec taxpilot python main.py serve
```

After running the server, access the Swagger UI at:
```
http://localhost:8000/docs
```

#### Configuration

You can customize the deployment using a configuration file:

```bash
poetry run python main.py run-all --config custom-config.json
```

Or override specific settings:

```bash
poetry run python main.py serve --api-port 9000 --qdrant-url http://localhost:7000
```

#### Frontend Development

Start the frontend development server:
```
cd frontend
npm run dev
```

## Development Guidelines

- Use strong typing with Python 3.12's type annotations
- Format code with Black and lint with Flake8
- Type-check with MyPy
- Write tests for all new features
- Follow the project's code style guidelines

## License

[License information]