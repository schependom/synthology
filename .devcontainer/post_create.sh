#! /usr/bin/env bash

# Install uv and Python dependencies
# uv is installed in Dockerfile, just sync
uv sync --dev

# Update DLV path in config (DLV is installed in Dockerfile)
CONFIG_FILE="configs/asp_generator/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    sed -i 's@dlv: .*@dlv: /usr/local/bin/dlv@' "$CONFIG_FILE"
fi

# Install pre-commit hooks
uv run pre-commit install --install-hooks
