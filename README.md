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

1. Start the backend server:
   ```
   poetry run uvicorn taxpilot.backend.api.main:app --reload
   ```

2. Start the frontend development server:
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