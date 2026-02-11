#!/bin/bash
# Setup script for smart buildings worker
# Run after cloning: ./setup_worker.sh

set -e

echo "Setting up smart buildings worker environment..."

# Create virtual environment
python3 -m venv venv
echo "Created virtual environment"

# Activate and install
source venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install flask  # for central server

echo ""
echo "Setup complete!"
echo ""
echo "To activate: source venv/bin/activate"
echo ""
echo "To run central server:"
echo "  python central_server.py"
echo ""
echo "To run worker:"
echo "  SERVER=http://server-ip:5000 python worker_server.py 8"
