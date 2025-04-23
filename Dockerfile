FROM python:3.12-slim

WORKDIR /app

# Install poetry and system dependencies
RUN pip install --no-cache-dir poetry==1.8.2 \
    && apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry configuration files
COPY pyproject.toml poetry.lock ./

# Configure poetry for container environment 
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/processed data/raw

# Expose API port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["python", "main.py", "serve"]