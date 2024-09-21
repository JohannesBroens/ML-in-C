#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Determine the project root directory
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Set the data directory path
DATA_DIR="$PROJECT_ROOT/data"

# Create the data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Change to the data directory
cd "$DATA_DIR"

echo "Downloading datasets to $DATA_DIR..."

# Function to download a file if it doesn't exist
download_if_not_exists() {
    local url=$1
    local filename=$2
    wget -nc "$url" -O "$filename"
    #if [ ! -f "$filename" ]; then
    #    echo "Downloading $filename..."
    #    wget "$url" -O "$filename"
    #else
    #    echo "$filename already exists. Skipping download."
    #fi
}

# Download Iris Dataset
download_if_not_exists "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" "iris.data"

# Download Wine Quality Dataset (Red)
download_if_not_exists "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" "winequality-red.csv"

# Download Wine Quality Dataset (White)
download_if_not_exists "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv" "winequality-white.csv"

# Download Breast Cancer Wisconsin Dataset
download_if_not_exists "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" "wdbc.data"

echo "Datasets are ready in $DATA_DIR."

# Return to the script directory (optional)
cd "$SCRIPT_DIR"
