#!/bin/bash
# Setup script for smart buildings worker
# Run after cloning: ./setup_worker.sh

set -e

echo "Setting up smart buildings worker environment..."

# Create conda environment with Python 3.11 (required for tensorflow 2.15)
conda create -n sbsim python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate sbsim

# Install from frozen requirements
pip install --upgrade pip
pip install -r requirements-frozen.txt
pip install -e .
pip install flask  # for central server

echo ""
echo "Setup complete!"
echo ""
echo "To activate: conda activate sbsim"
echo ""
echo "To run central server:"
echo "  python central_server.py"
echo ""
echo "To run worker:"
echo "  SERVER=http://server-ip:5000 python worker_server.py 8"
