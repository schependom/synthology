#! /usr/bin/env bash

# Install uv and Python dependencies
# uv is installed in Dockerfile, just sync
uv sync --dev

./install-dlv-linux.sh

# Install pre-commit hooks
uv run pre-commit install --install-hooks
