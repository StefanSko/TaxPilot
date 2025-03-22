# GermanLawFinder Modal.com Infrastructure

This directory contains the Modal.com configuration and deployment scripts for the GermanLawFinder application.

## Files

- `modal_config.py`: Defines the Modal app, volumes, and serverless endpoints
- `deploy.py`: Script to deploy the application to Modal.com
- `README.md`: This file

## Deployment Instructions

### Prerequisites

1. Install the Modal CLI:
   ```
   pip install modal
   ```

2. Authenticate with Modal:
   ```
   modal token new
   ```

### Deployment

Deploy to the development environment:
```
python -m taxpilot.infrastructure.deploy
```

Deploy to staging or production:
```
python -m taxpilot.infrastructure.deploy staging
python -m taxpilot.infrastructure.deploy production
```

## Environment Configuration

The deployment script sets the `ENVIRONMENT` environment variable, which is used to configure environment-specific settings:

- `development`: CORS allows all origins, more verbose logging
- `staging`: Restricted CORS, moderate logging
- `production`: Strict CORS, minimal logging

## Resources

- Modal Volume: `germanlawfinder-db-vol` - Used to store the DuckDB database
- Modal App: `germanlawfinder` - Contains the serverless API endpoint

## Customizing Resources

To customize the compute resources:

1. Edit `modal_config.py` and modify the `@app.function` decorator:
   ```python
   @app.function(
       image=image,
       volumes={VOLUME_MOUNT_PATH: volume},
       memory=2048,  # Adjust memory (in MB)
       cpu=1.0,      # Adjust CPU cores
       gpu="any",    # Uncomment for GPU access
       keep_warm=1,  # Number of warm containers
       timeout=120,  # Timeout in seconds
   )
   ```

2. Redeploy the application with the updated configuration.