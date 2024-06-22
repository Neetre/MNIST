#!/bin/bash

# Create virtual environment (if not already present)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install requirements (assuming requirements.txt is in the same directory)
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found. Skipping installation."
fi