#!/bin/bash

# Setup Variables
CONFIG_FILE="configs/asp_generator/config.yaml"
DLV_URL="https://www.dlvsystem.it/files/dlv.x86-64-linux-elf-unixodbc"

# OS Check
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Error: This script is intended for Linux systems only."
    echo "Please follow the instructions in the README.md for MacOS."
    exit 1
fi

# Check for Config File
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    echo "Please ensure you are running this script from the root of the repository."
    exit 1
fi

# Check for existing DLV or Download
if command -v dlv &> /dev/null; then
    DLV_PATH=$(command -v dlv)
    echo "====================================================================="
    echo "DLV is already installed at $DLV_PATH. Skipping download."
    echo "====================================================================="
else
    echo "============================================"
    echo "DLV not found. Downloading..."
    echo "============================================"

    # Download with -f (fail silently on server error) to detect broken links
    if curl -fLo dlv "$DLV_URL"; then
        chmod +x dlv
        DLV_PATH="$PWD/dlv"
        echo "Download successful."
    else
        echo "Error: Failed to download DLV. Please check your internet connection or the URL."
        exit 1
    fi
fi

# Sanity Check: Does the binary work?
echo "Verifying DLV executable..."
if ! "$DLV_PATH" --version &> /dev/null && ! "$DLV_PATH" -help &> /dev/null; then
     echo "WARNING: The DLV binary at $DLV_PATH appears to be broken."
     echo "You may be missing dependencies. Try running: sudo apt-get install unixodbc"
     # We don't exit here, just warn, as --version might not be supported by all DLV versions
fi

echo "Updating DLV path in $CONFIG_FILE to $DLV_PATH"

# Replace DLV path in config file using sed
sed -i.bak "s@^([[:space:]]*)dlv:.*@\1dlv: $DLV_PATH@" "$CONFIG_FILE"

# Clean up backup only if it exists
[ -f "$CONFIG_FILE.bak" ] && rm "$CONFIG_FILE.bak"

echo "DLV path updated successfully."
echo "-------------------------------------"
grep "dlv:" "$CONFIG_FILE"
echo "-------------------------------------"
