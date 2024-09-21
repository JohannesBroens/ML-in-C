#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Determine the project root directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Step 1: Download datasets
echo "Downloading datasets..."
chmod +x "$SCRIPT_DIR/download_datasets.sh"
"$SCRIPT_DIR/download_datasets.sh"

# Step 2: Preprocess datasets
echo "Preprocessing datasets..."
python3 "$SCRIPT_DIR/preprocess_iris.py"

# Step 3: Build the project
echo "Building the project..."
mkdir -p "$PROJECT_ROOT/c/build"
cd "$PROJECT_ROOT/c/build"
cmake .. -DUSE_CUDA=ON
make

# Step 4: Run the program
echo "Running the program..."
./main
cd "$PROJECT_ROOT"
echo "Pipeline completed."
