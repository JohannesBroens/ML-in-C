#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Determine the project root directory
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Starting the pipeline..."

# Step 1: Download datasets
echo "Checking for datasets..."
chmod +x "$SCRIPT_DIR/download_datasets.sh"
"$SCRIPT_DIR/download_datasets.sh"

# Step 2: Preprocess datasets
echo "Checking for preprocessed data..."
python3 "$SCRIPT_DIR/preprocess_iris.py"

# Step 3: Build the project
echo "Building the project..."
mkdir -p "$PROJECT_ROOT/src/c/build"
cd "$PROJECT_ROOT/src/c/build"
cmake .. -DUSE_CUDA=ON
make

# Step 4: Run the program
# Step 4: Run the program
echo "Running the program..."

cd "$PROJECT_ROOT"

if [ $# -eq 1 ]; then
    DATASET=$1
    echo "Running on dataset: $DATASET"
    ./src/c/build/main --dataset "$DATASET"
else
    DATASETS=("generated" "iris" "wine-red" "wine-white" "breast-cancer")
    for DATASET in "${DATASETS[@]}"; do
        echo "--------------------------------------------"
        echo "Running on dataset: $DATASET"
        ./src/c/build/main --dataset "$DATASET"
    done
fi

echo "Pipeline completed."
