#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Common function to initialize random number generator
static void seed_random() {
    srand((unsigned int)time(NULL));
}

// ========================================================
// CUDA Implementations
// ========================================================

// Kernel for initializing weights and biases
__global__ void init_weights_biases(float *weights, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = (curand_uniform(&state) - 0.5f) * 0.1f;
    }
}

// Function to initialize the MLP
void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size) {
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;

    size_t input_w_size = input_size * hidden_size * sizeof(float);
    size_t output_w_size = hidden_size * output_size * sizeof(float);
    size_t hidden_b_size = hidden_size * sizeof(float);
    size_t output_b_size = output_size * sizeof(float);

    // Allocate managed memory
    cudaMallocManaged(&mlp->input_weights, input_w_size);
    cudaMallocManaged(&mlp->output_weights, output_w_size);
    cudaMallocManaged(&mlp->hidden_biases, hidden_b_size);
    cudaMallocManaged(&mlp->output_biases, output_b_size);

    // Initialize weights and biases using CUDA kernels
    int threads_per_block = 256;
    int blocks_per_grid_input = (input_size * hidden_size + threads_per_block - 1) / threads_per_block;
    int blocks_per_grid_output = (hidden_size * output_size + threads_per_block - 1) / threads_per_block;

    unsigned long seed = (unsigned long)time(NULL);

    init_weights_biases<<<blocks_per_grid_input, threads_per_block>>>(mlp->input_weights, input_size * hidden_size, seed);
    init_weights_biases<<<blocks_per_grid_output, threads_per_block>>>(mlp->output_weights, hidden_size * output_size, seed + 1);

    // Set biases to zero
    cudaMemset(mlp->hidden_biases, 0, hidden_b_size);
    cudaMemset(mlp->output_biases, 0, output_b_size);

    cudaDeviceSynchronize();
}

// Function to free the MLP resources
void mlp_free(MLP *mlp) {
    cudaFree(mlp->input_weights);
    cudaFree(mlp->output_weights);
    cudaFree(mlp->hidden_biases);
    cudaFree(mlp->output_biases);
}

// Kernel for training
__global__ void train_kernel(MLP mlp,
                             float *inputs, int *targets, int num_samples,
                             float learning_rate, int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_samples)
        return;

    // Forward pass
    extern __shared__ float shared[];
    float *hidden = shared;  // Shared memory for hidden layer activations
    float *output = &shared[hidden_size];  // Output layer

    // Compute hidden layer activations
    for (int i = 0; i < hidden_size; ++i) {
        float sum = mlp.hidden_biases[i];
        for (int j = 0; j < input_size; ++j) {
            sum += inputs[idx * input_size + j] * mlp.input_weights[j * hidden_size + i];
        }
        // Activation function (ReLU)
        hidden[i] = fmaxf(0.0f, sum);
    }

    // Compute output layer activations
    for (int i = 0; i < output_size; ++i) {
        float sum = mlp.output_biases[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden[j] * mlp.output_weights[j * output_size + i];
        }
        // Activation function (Softmax or Sigmoid)
        // For simplicity, using Sigmoid for binary classification
        output[i] = 1.0f / (1.0f + expf(-sum));
    }

    // Compute error
    float *errors = &shared[hidden_size + output_size];  // For output errors
    for (int i = 0; i < output_size; ++i) {
        float target = (targets[idx] == i) ? 1.0f : 0.0f;  // One-hot encoding
        errors[i] = target - output[i];
    }

    // Backward pass (simple gradient descent)
    // Output layer gradients
    float *d_output = &shared[hidden_size + 2 * output_size];
    for (int i = 0; i < output_size; ++i) {
        d_output[i] = errors[i] * output[i] * (1.0f - output[i]);  // Derivative of Sigmoid
    }

    // Hidden layer gradients
    float *d_hidden = &shared[hidden_size + 3 * output_size];
    for (int i = 0; i < hidden_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < output_size; ++j) {
            sum += d_output[j] * mlp.output_weights[i * output_size + j];
        }
        d_hidden[i] = sum * ((hidden[i] > 0) ? 1.0f : 0.0f);  // Derivative of ReLU
    }

    // Update output weights and biases
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            atomicAdd(&mlp.output_weights[i * output_size + j], learning_rate * d_output[j] * hidden[i]);
        }
    }
    for (int i = 0; i < output_size; ++i) {
        atomicAdd(&mlp.output_biases[i], learning_rate * d_output[i]);
    }

    // Update input weights and biases
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            atomicAdd(&mlp.input_weights[i * hidden_size + j], learning_rate * d_hidden[j] * inputs[idx * input_size + i]);
        }
    }
    for (int i = 0; i < hidden_size; ++i) {
        atomicAdd(&mlp.hidden_biases[i], learning_rate * d_hidden[i]);
    }
}

// Function to train the MLP
void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples) {
    // Copy inputs and targets to device memory
    float *d_inputs;
    int *d_targets;
    size_t input_size_bytes = num_samples * mlp->input_size * sizeof(float);
    size_t target_size_bytes = num_samples * sizeof(int);
    cudaMalloc(&d_inputs, input_size_bytes);
    cudaMalloc(&d_targets, target_size_bytes);

    cudaMemcpy(d_inputs, inputs, input_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, target_size_bytes, cudaMemcpyHostToDevice);

    // Training parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_samples + threads_per_block - 1) / threads_per_block;

    size_t shared_memory_size = (mlp->hidden_size + mlp->output_size * 4) * sizeof(float);  // Adjust as needed

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        train_kernel<<<blocks_per_grid, threads_per_block, shared_memory_size>>>(*mlp,
                                                                                 d_inputs, d_targets, num_samples,
                                                                                 LEARNING_RATE,
                                                                                 mlp->input_size, mlp->hidden_size, mlp->output_size);
        cudaDeviceSynchronize();

        if (epoch % 100 == 0) {
            printf("Epoch %d completed.\n", epoch);
        }
    }

    // Free device memory
    cudaFree(d_inputs);
    cudaFree(d_targets);
}

// Kernel for evaluation
__global__ void evaluate_kernel(MLP mlp,
                                float *inputs, int *targets, int num_samples,
                                float *loss, int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_samples)
        return;

    // Forward pass
    extern __shared__ float shared[];
    float *hidden = shared;  // Shared memory for hidden layer activations
    float *output = &shared[hidden_size];  // Output layer

    // Compute hidden layer activations
    for (int i = 0; i < hidden_size; ++i) {
        float sum = mlp.hidden_biases[i];
        for (int j = 0; j < input_size; ++j) {
            sum += inputs[idx * input_size + j] * mlp.input_weights[j * hidden_size + i];
        }
        // Activation function (ReLU)
        hidden[i] = fmaxf(0.0f, sum);
    }

    // Compute output layer activations
    for (int i = 0; i < output_size; ++i) {
        float sum = mlp.output_biases[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden[j] * mlp.output_weights[j * output_size + i];
        }
        // Activation function (Sigmoid)
        output[i] = 1.0f / (1.0f + expf(-sum));
    }

    // Compute error (Cross-entropy loss)
    float target = (targets[idx] == 1) ? 1.0f : 0.0f;  // Adjust for multi-class
    float sample_loss = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        float t = (targets[idx] == i) ? 1.0f : 0.0f;
        sample_loss -= t * logf(output[i] + 1e-7f);  // Add epsilon to prevent log(0)
    }

    // Accumulate loss
    atomicAdd(loss, sample_loss);
}

// Function to evaluate the MLP
void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples, float *loss) {
    // Copy inputs and targets to device memory
    float *d_inputs;
    int *d_targets;
    float *d_loss;
    size_t input_size_bytes = num_samples * mlp->input_size * sizeof(float);
    size_t target_size_bytes = num_samples * sizeof(int);
    cudaMalloc(&d_inputs, input_size_bytes);
    cudaMalloc(&d_targets, target_size_bytes);
    cudaMallocManaged(&d_loss, sizeof(float));

    cudaMemcpy(d_inputs, inputs, input_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, target_size_bytes, cudaMemcpyHostToDevice);
    *d_loss = 0.0f;

    // Evaluation parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_samples + threads_per_block - 1) / threads_per_block;
    size_t shared_memory_size = (mlp->hidden_size + mlp->output_size) * sizeof(float);  // Adjust as needed

    evaluate_kernel<<<blocks_per_grid, threads_per_block, shared_memory_size>>>(*mlp,
                                                                                d_inputs, d_targets, num_samples,
                                                                                d_loss,
                                                                                mlp->input_size, mlp->hidden_size, mlp->output_size);
    cudaDeviceSynchronize();

    *loss = *d_loss / num_samples;

    // Free device memory
    cudaFree(d_inputs);
    cudaFree(d_targets);
    cudaFree(d_loss);
}

// ========================================================
// Common Functions (Optional, if needed)
// ========================================================

// You can remove or update the mlp_generate_data function as per your requirements.
// It's no longer necessary if you're loading real datasets.

// Function to generate data points (if needed)
void mlp_generate_data(float *inputs, int *targets, int num_samples) {
    seed_random();
    for (int i = 0; i < num_samples; ++i) {
        float x = ((float)rand() / RAND_MAX) * 2 - 1;  // Random value between -1 and 1
        float y = ((float)rand() / RAND_MAX) * 2 - 1;
        inputs[i * 2] = x;
        inputs[i * 2 + 1] = y;
        // Simple function: target = 1 if inside circle of radius 0.5
        targets[i] = (x * x + y * y < 0.25f) ? 1 : 0;
    }
}
