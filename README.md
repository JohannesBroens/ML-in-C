# ML-in-C
ML-in-C is a project aimed at enhancing machine learning capabilities using the C programming language. This repository contains implementations of machine learning models, such as Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs), both in C and Python. The project focuses on performance optimization, leveraging CPU and GPU computations using OpenMP, CUDA, and providing benchmarking scripts for performance comparison.

## Table of Contents
- [ML-in-C](#ml-in-c)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Folder Structure](#folder-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Building the Project](#building-the-project)
    - [Running the Pipeline](#running-the-pipeline)
  - [Usage](#usage)
    - [Running on Specific Datasets](#running-on-specific-datasets)
  - [Benchmarking](#benchmarking)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
  - [License](#license)

## Features
- **Machine Learning Models in C**: Implementations of MLP and CNN models optimized for performance.
- **CUDA Support**: GPU acceleration using CUDA for intensive computations.
- **Python Implementations**: Equivalent models in Python for ease of use and comparison.
- **Data Loading and Preprocessing**: Scripts to download and preprocess popular datasets.
- **Benchmarking Scripts**: Tools to compare performance between CPU and GPU implementations.
- **Unit Testing**: Testing framework for ensuring code correctness and reliability.

## Folder Structure
```scss
├── data
│   ├── iris.data
│   ├── iris_processed.txt
│   ├── wdbc.data
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── docs
├── examples
│   ├── c
│   └── python
├── figs
├── LICENSE
├── README.md
└── src
    ├── c
    │   ├── CMakeLists.txt
    │   ├── data_loader.c
    │   ├── data_loader.h
    │   ├── main.c
    │   └── models
    │       ├── cnn
    │       └── mlp
    │           ├── CMakeLists.txt
    │           ├── mlp_cpu.c
    │           ├── mlp.cu
    │           └── mlp.h
    ├── python
    │   ├── models
    │   │   ├── cnn
    │   │   └── mlp
    │   │       └── mlp.py
    │   ├── setup.py
    │   └── utils
    ├── scripts
    │   ├── build_run_c_CPU.sh
    │   ├── download_datasets.sh
    │   ├── preprocess_iris.py
    │   └── run_pipeline.sh
    └── tests
        ├── c
        └── python
```
- **data/**: Contains datasets and preprocessed data files.
- **docs/**: Documentation files.
- **examples/**: Example programs in C and Python.
- **figs/**: Figures and images for documentation or results.
- **src/**: Source code for the project.
  - **c/**: C implementations.
    - **models/**: Machine learning models in C.
      - **mlp/**: Multi-Layer Perceptron implementations.
      - **cnn/**: Convolutional Neural Network implementations.
  - **python/**: Python implementations.
    - **models/**: Machine learning models in Python.
  - **scripts/**: Helper scripts for building, running, and benchmarking.
  - **tests/**: Unit tests for C and Python code.


## Getting Started
### Prerequisites
- **C Compiler**: GCC or any C99-compatible compiler.
- **CUDA Toolkit**: Required for GPU acceleration (if building with CUDA support).
- **CMake**: Version 3.10 or higher.
- **Python 3**: For running preprocessing scripts and Python implementations.
- **Python Packages**: e.g. pytorch etc. (can be installed via requirements.txt). However, I have not made Python versions yet.

### Building the Project
1. **Clone the Repositiory**: 
```bash
git clone https://github.com/JohannesBroens/ML-in-C.git
cd ML-in-C
```
2. **Install Dependencies**: 
    - **For C code**:
      - Ensure that you have a C compiler and CUDA Toolkit installed. 
    - **For Python code**:
      - Install required Python packages: 
        ```bash
        pip install -r requirements.txt
        ```
3. **Build the C Project**:
```bash
cd src/c
mkdir build
cd build
cmake .. -DUSE_CUDA=ON
make
```
- Set `-DUSE_CUDA=OFF` if you want to build without CUDA support.
### Running the Pipeline
A pipeline script is provided to automate the process of downloading datasets, preprocessing, building the project, and running the program.
```bash
cd src/scripts
./run_pipeline.sh
```
- **Note**: Ensure the script has execute permissions: 
```bash
chmod +x run_pipeline.sh
```
## Usage
### Running on Specific Datasets
You can run the program on specific datasets by providing the dataset name as an argument. 
```bash
./run_pipeline.sh iris
```
Supported datasets:
- `generated`
- `iris`
- `wine-red`
- `wine-white`
- `breast-cancer`
## Benchmarking
Not yet implemented. 

## Testing
Not yet implemented. 

## Documentation
Not fully documented yet. However, it is the plan to do the following: 
- **Generating Code Documentation**: 
  - **C Code**: Documentation generated using Doxygen. 
    ```bash
    doxygen Doxyfile
    ```
  - **Python Code**: Documentation generated using Sphinx. 
    ```bash
    cd docs
    make html
    ```
- **README.md**:
  - Contains instructions on how to build and run the code, including dependencies and prerequisites.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/YourFeature`)
3. Commit your Changes (`git commit -m 'Add YourFeature'`)
4. Push to the Branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License
This project is licensed under the Apache License (Version 2.0) - see the [LICENSE](LICENSE) file for details. 
  
  
  
  
Feel free to explore the repository, run the models, and contribute to the project. If you encounter any issues or have suggestions, please open an issue on GitHub.
<!---

# DELETE BELOW
I wish to improve my  "ML in C" abilities. Hence making a repo without any other pupose than that - for now. 

## Current next steps
**Test the Updated Code**: Compile and run the program to ensure that it works correctly and efficiently.

**Benchmarking**: Measure the performance of both the CPU and GPU implementations under similar conditions to compare their execution times and resource usage.

**Further Optimization**: Explore additional optimization techniques, such as vectorization or using optimized math libraries.

## Folder structure
```scss
├── LICENSE
├── README.md
├── .gitignore
├── docs/
│   └── ... (Documentation files)
├── examples/
│   ├── c/
│   │   └── ... (Example C programs)
│   └── python/
│       └── ... (Example Python scripts)
├── src/
│   ├── c/
│   │   ├── CMakeLists.txt
│   │   ├── models/
│   │   │   ├── mlp/
│   │   │   │   ├── mlp.cu
│   │   │   │   ├── mlp.h
│   │   │   │   └── ... (Other MLP-related files)
│   │   │   ├── cnn/
│   │   │   │   └── ... (CNN implementation in C)
│   │   │   └── ... (Other models)
│   │   └── utils/
│   │       └── ... (Utility functions)
│   ├── python/
│   │   ├── setup.py
│   │   ├── models/
│   │   │   ├── mlp.py
│   │   │   ├── cnn.py
│   │   │   └── ... (Other models)
│   │   └── utils/
│   │       └── ... (Utility modules)
|   └── data/
|       └── ... (Datasets or data processing scripts)
├── tests/
│   ├── c/
│   │   └── ... (Unit tests for C code)
│   └── python/
│       └── ... (Unit tests for Python code)
├── scripts/
│   └── ... (Helper scripts, e.g., for building or running benchmarks)
└── requirements.txt
```
update to (add comments and remove some stuff to give idea):
```scss
├── data
│   ├── iris.data
│   ├── iris_processed.txt
│   ├── wdbc.data
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── docs
├── examples
│   ├── c
│   └── python
├── figs
├── LICENSE
├── README.md
└── src
    ├── c
    │   ├── CMakeLists.txt
    │   ├── data_loader.c
    │   ├── data_loader.h
    │   ├── main.c
    │   └── models
    │       ├── cnn
    │       └── mlp
    │           ├── CMakeLists.txt
    │           ├── mlp_cpu.c
    │           ├── mlp.cu
    │           └── mlp.h
    ├── python
    │   ├── models
    │   │   ├── cnn
    │   │   └── mlp
    │   │       └── mlp.py
    │   ├── setup.py
    │   └── utils
    ├── scripts
    │   ├── build_run_c_CPU.sh
    │   ├── download_datasets.sh
    │   ├── preprocess_iris.py
    │   └── run_pipeline.sh
    └── tests
        ├── c
        └── python
```

## Checklist
**Benchmark Scripts**: Under `scripts/` create scripts that:
- Train models with the same data.
- Measure training time and performance metrics.
- Generate comparison reports or plots.

**Data Formats**: Standardize data loading and saving formats to ensure consistency.

### Documentation
- **README.md**: Update with instructions on how to build and run my code, including dependencies and prerequisites.
- **Docs Generation**: Use Doxygen for C code and Sphinx for Python code to generate documentation from comments.

### Testing
**Unit Tests**: Under `tests/`, write unit tests for both C and Python code.

**C Tests**: Use a testing framework like CUnit or Unity.
**Python Tests**: Use unittest, pytest, or similar frameworks.

-->