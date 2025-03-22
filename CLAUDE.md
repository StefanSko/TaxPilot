# CLAUDE.md for TaxPilot

## Development Commands
- Install dependencies: `poetry install`
- Run backend: *TBD - add command when implemented*
- Tests: `poetry run pytest`
- Single test: `poetry run pytest path/to/test_file.py::test_function`
- Linting: `poetry run flake8` and `poetry run mypy --python-version 3.12 .`

## Code Style Guidelines
- Python 3.12 with strict typing utilizing latest features:
  - Use `|` for union types (PEP 604) instead of `Union[]`
  - Use builtin collection types for type annotations (PEP 585) 
  - Use TypedDict for dictionary typing
  - Leverage Python 3.12's enhanced type narrowing
- FastAPI type annotations for all API endpoints
- XML processing with lxml library
- Format imports: stdlib, third-party, local (alphabetized within groups)
- Use dataclasses or Pydantic models for data structures
- Exception handling: specific exceptions, proper error messages
- Vue 3 with Composition API for frontend components

## Naming Conventions
- Classes: PascalCase
- Functions/methods: snake_case
- Variables: snake_case
- Constants: UPPER_SNAKE_CASE
- Descriptive names reflecting purpose and content

## Project Structure
- Backend: Python 3.12, FastAPI, Modal.com serverless
- Frontend: Vue.js 3, Vuetify/PrimeVue, Pinia, Vite
- Database: DuckDB (relational) and vector DB (embeddings)