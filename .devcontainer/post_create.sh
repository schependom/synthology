#! /usr/bin/env bash

# Install unixODBC (Required for the dlv.dlv.x86-64-linux-elf-unixodbc version)
sudo apt-get update
sudo apt-get install -y unixodbc unixodbc-dev

# Install uv and Python dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv sync --dev

# Download and Install DLV
chmod +x install-dlv-linux.sh
./install-dlv-linux.sh

# Move DLV to global path and update config
sudo mv dlv /usr/local/bin/
sed -i 's@dlv: .*@dlv: /usr/local/bin/dlv@' configs/asp_generator/config.yaml

# Install pre-commit hooks
uv run pre-commit install --install-hooks
