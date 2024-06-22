#!/bin/bash

# Create virtual environment (if not already present)
python3 -m venv .venv  # Use python3 to ensure Python 3

# Activate virtual environment
source .venv/bin/activate  # Use source for activation

# Install requirements (assuming requirements.txt is in the same directory)
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found. Skipping installation."
fi