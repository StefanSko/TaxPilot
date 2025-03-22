#!/usr/bin/env python
"""
Deployment script for GermanLawFinder to Modal.com.

This script deploys the application to Modal.com and outputs the deployment URL.
Run this script with: python -m taxpilot.infrastructure.deploy
"""

import subprocess
import sys
import time
from typing import Optional
import os


def deploy(environment: str = "development") -> None:
    """
    Deploy the application to Modal.com.
    
    Args:
        environment: The deployment environment (development, staging, production).
    """
    print(f"Deploying GermanLawFinder to Modal.com ({environment} environment)...")
    
    # Set environment variable for the deployment
    os.environ["ENVIRONMENT"] = environment
    
    # Deploy the app using Modal CLI
    try:
        result = subprocess.run(
            ["modal", "deploy", "taxpilot.infrastructure.modal_config"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        
        # Extract the deployment URL from the output
        for line in result.stdout.splitlines():
            if "https://" in line and ".modal.run" in line:
                print("\n✅ Deployment successful!")
                print(f"API URL: {line.strip()}")
                print(f"API Docs: {line.strip()}/docs")
                break
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        print(e.stderr)
        sys.exit(1)


def ensure_modal_authenticated() -> bool:
    """
    Ensure that the user is authenticated with Modal.
    
    Returns:
        True if authenticated, False otherwise.
    """
    try:
        result = subprocess.run(
            ["modal", "token", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Check if there's at least one token
        return "No tokens found" not in result.stdout
        
    except subprocess.CalledProcessError:
        return False


def create_resources() -> None:
    """Create necessary Modal resources before deployment."""
    # Create the volume if it doesn't exist
    try:
        subprocess.run(
            ["modal", "volume", "create", "germanlawfinder-db-vol"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("✅ Volume created: germanlawfinder-db-vol")
    except subprocess.CalledProcessError as e:
        if "already exists" in e.stderr:
            print("ℹ️ Volume already exists: germanlawfinder-db-vol")
        else:
            print(f"❌ Failed to create volume: {e}")
            print(e.stderr)


if __name__ == "__main__":
    # Check authentication
    if not ensure_modal_authenticated():
        print("❌ Not authenticated with Modal. Please run 'modal token new' first.")
        sys.exit(1)
    
    # Create resources
    create_resources()
    
    # Get environment from command line args or default to development
    env = "development"
    if len(sys.argv) > 1:
        env = sys.argv[1]
    
    # Deploy the app
    deploy(environment=env)