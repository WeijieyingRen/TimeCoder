#!/bin/bash
# Run Script

echo "Starting experiments..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run main program
python core/engine.py "$@"

echo "Experiments completed."
