import os
import numpy as np

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Determine the project root directory
project_root = os.path.dirname(script_dir)

# Set the data directory path
data_dir = os.path.join(project_root, 'data')

# Input and output file paths
input_file = os.path.join(data_dir, 'iris.data')
output_file = os.path.join(data_dir, 'iris_processed.txt')

# Check if the input file exists
if not os.path.isfile(input_file):
    print(f"Input file {input_file} not found. Please ensure the dataset has been downloaded.")
    exit(1)

# Load the dataset
data = []
with open(input_file, 'r') as f:
    for line in f:
        if line.strip() == '':
            continue
        tokens = line.strip().split(',')
        features = list(map(float, tokens[:4]))
        label = tokens[4]
        data.append(features + [label])

# Convert to numpy array
data = np.array(data)

# Encode labels
label_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
labels = np.array([label_mapping[label] for label in data[:, -1]], dtype=np.int32)

# Extract features and convert to float
features = data[:, :-1].astype(np.float32)

# Normalize features
features_mean = np.mean(features, axis=0)
features_std = np.std(features, axis=0)
features_normalized = (features - features_mean) / features_std

# Combine features and labels
processed_data = np.hstack((features_normalized, labels.reshape(-1, 1)))

# Save to a text file in the data directory
np.savetxt(output_file, processed_data, fmt='%.6f', delimiter=',')

print(f'Preprocessing completed. Processed data saved to {output_file}')
