# Install dependency if not present (Debian/Ubuntu)
# sudo apt-get update && sudo apt-get install -y unixodbc

# Check if already installed
if command -v dlv &> /dev/null
then
    echo "DLV is already installed at $(command -v dlv). Skipping installation."
    echo "Here's the config file:"
    cat "$CONFIG_FILE"
    exit 0
fi

echo "Downloading and installing DLV..."

DLV_URL="https://www.dlvsystem.it/files/dlv.x86-64-linux-elf-unixodbc"
curl -Lo dlv $DLV_URL
chmod +x dlv

CONFIG_FILE="configs/asp_generator/config.yaml"
DLV_PATH="$(pwd)/dlv"

# Using '@' as a delimiter to avoid issues with '/' or '|' in paths
sed -i.bak "s@^dlv: .*@dlv: $DLV_PATH@" "$CONFIG_FILE"
rm "$CONFIG_FILE.bak"

echo "DLV path updated to: $DLV_PATH"

echo "Here's the config file:"
cat "$CONFIG_FILE"
