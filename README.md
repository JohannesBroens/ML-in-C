# ML-in-C
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

## Potential Enhancements
### Mini-Batch Gradient Decent
Implement mini-batch gradient descent to reduce the overhead of updating weights and biases within critical sections.
- Accumulate weight updates over a batch of samples.
- Update weights and biases after processing the batch.

### Atomic Operations
Use atomic operations for weight updates if the hardware and compiler support them:
```c
#pragma omp atomic
mlp->output_weights[i] += LEARNING_RATE * d_output * hidden[i];
```
### Thread-Private Variables
Declare variables that should be private to each thread to prevent unintended sharing.
