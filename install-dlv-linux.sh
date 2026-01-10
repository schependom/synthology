# Install dependency if not present (Debian/Ubuntu)
# sudo apt-get update && sudo apt-get install -y unixodbc

# If on MacOS, exit with message
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "This script is intended for Linux systems only. Please follow the instructions in the README.md for MacOS."
    exit 1
fi

CONFIG_FILE="configs/asp_generator/config.yaml"

# Check if already installed on linux
if command -v dlv &> /dev/null
then
    echo "====================================================================="
    echo "DLV is already installed at $(command -v dlv). Skipping installation."
    echo "====================================================================="
else
    echo "============================================"
    echo "DLV not found. Proceeding with installation."
    echo "============================================"
    echo ""

    DLV_URL="https://www.dlvsystem.it/files/dlv.x86-64-linux-elf-unixodbc"
    curl -Lo dlv $DLV_URL
    chmod +x dlv
fi

DLV_PATH="$(command -v dlv || echo "$PWD/dlv")"
echo "Updating DLV path in $CONFIG_FILE to $DLV_PATH"

# Using '@' as a delimiter to avoid issues with '/' or '|' in paths
sed -i.bak "s@^dlv: .*@dlv: $DLV_PATH@" "$CONFIG_FILE"
rm "$CONFIG_FILE.bak"

echo "DLV path updated to: $DLV_PATH"

echo "Here's the config file (please verify):"
cat "$CONFIG_FILE"
