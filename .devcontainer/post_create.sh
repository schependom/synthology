#! /usr/bin/env bash

# Install unixODBC (Required for the dlv.dlv.x86-64-linux-elf-unixodbc version)
sudo apt-get update
sudo apt-get install -y unixodbc unixodbc-dev

# Install uv and Python dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv sync --dev

# Download and Install DLV
./install-dlv-linux.sh

# Install pre-commit hooks
uv run pre-commit install --install-hooks
