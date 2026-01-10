#! /usr/bin/env bash

# Install unixODBC (Required for the dlv.dlv.x86-64-linux-elf-unixodbc version)
sudo apt-get update
sudo apt-get install -y unixodbc unixodbc-dev

# Install uv and Python dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv sync --dev

# Download and Install DLV
# Replace the URL with the official download link for the unixodbc version
DLV_URL="https://www.dlvsystem.it/files/dlv.x86-64-linux-elf-unixodbc"
curl -Lo dlv $DLV_URL
chmod +x dlv
sudo mv dlv /usr/local/bin/dlv

# Install pre-commit hooks
uv run pre-commit install --install-hooks
