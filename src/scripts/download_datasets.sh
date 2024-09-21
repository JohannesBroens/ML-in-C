#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Determine the project root directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set the data directory path
DATA_DIR="$PROJECT_ROOT/data"

# Create the data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Change to the data directory
cd "$DATA_DIR"

# Download datasets
echo "Downloading datasets to $DATA_DIR..."
wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

echo "Datasets downloaded to $DATA_DIR."

# Return to the original directory (optional)
cd "$SCRIPT_DIR"
