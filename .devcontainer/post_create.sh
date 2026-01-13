#! /usr/bin/env bash

# Install uv and Python dependencies
# uv is installed in Dockerfile, just sync
uv sync --dev
echo "Installed uv and Python dependencies."

# Source virtual environment
if [ -f ".venv/bin/activate" ]; then
    # Activate
    source .venv/bin/activate
    echo "Activated virtual environment."
else
    # error
    echo "Virtual environment activation script not found!"
    exit 1
fi

# Update DLV path in config (DLV is installed in Dockerfile)
CONFIG_FILE="configs/asp_generator/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    sed -i 's@dlv: .*@dlv: /usr/local/bin/dlv@' "$CONFIG_FILE"
    echo "Updated DLV path in $CONFIG_FILE to '/usr/local/bin/dlv'."
else
    # error
    echo "Config file $CONFIG_FILE not found!"
    exit 1
fi


# Install pre-commit hooks
uv run pre-commit install --install-hooks
echo "Pre-commit hooks installed."

# Set unbuffered output for Python
export PYTHONUNBUFFERED=1
echo "Set PYTHONUNBUFFERED=1 for unbuffered Python output."

echo "##########################################"
echo "Post-create script completed successfully."
echo "##########################################"
echo "You can now start using the Synthology development environment."
