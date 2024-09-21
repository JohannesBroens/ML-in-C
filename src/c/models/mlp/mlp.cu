#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

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

// Kernel for training with Mini-Batch Gradient Descent
__global__ void train_kernel(MLP mlp,
                             float *inputs, int *targets, int num_samples,
                             float learning_rate, int input_size, int hidden_size, int output_size) {
    // Calculate the sample index within the batch
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= num_samples)
        return;

    // Thread-private variables stored in registers/local memory
    // Using fixed-size arrays might not be ideal for large networks; adjust sizes accordingly
    const int MAX_HIDDEN_SIZE = 1024;
    const int MAX_OUTPUT_SIZE = 10;

    float hidden_t[MAX_HIDDEN_SIZE];
    float output_t[MAX_OUTPUT_SIZE];
    float d_output_t[MAX_OUTPUT_SIZE];
    float d_hidden_t[MAX_HIDDEN_SIZE];

    // Load input and target
    float *input_sample = &inputs[sample_idx * input_size];
    int target_label = targets[sample_idx];

    // Forward pass
    for (int i = 0; i < hidden_size; ++i) {
        float sum = mlp.hidden_biases[i];
        for (int j = 0; j < input_size; ++j) {
            sum += input_sample[j] * mlp.input_weights[j * hidden_size + i];
        }
        // Activation function (ReLU)
        hidden_t[i] = fmaxf(0.0f, sum);
    }

    // Output layer
    for (int i = 0; i < output_size; ++i) {
        float sum = mlp.output_biases[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden_t[j] * mlp.output_weights[j * output_size + i];
        }
        // Activation function (Sigmoid)
        output_t[i] = 1.0f / (1.0f + expf(-sum));
    }

    // Compute error and gradients
    for (int i = 0; i < output_size; ++i) {
        float target = (target_label == i) ? 1.0f : 0.0f;  // One-hot encoding
        float error = target - output_t[i];
        d_output_t[i] = error * output_t[i] * (1.0f - output_t[i]);  // Derivative of Sigmoid
    }

    // Hidden layer gradients
    for (int i = 0; i < hidden_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < output_size; ++j) {
            sum += d_output_t[j] * mlp.output_weights[i * output_size + j];
        }
        d_hidden_t[i] = sum * ((hidden_t[i] > 0.0f) ? 1.0f : 0.0f);  // Derivative of ReLU
    }

    // Update weights and biases using atomic operations
    // Hidden to Output weights and biases
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float gradient = d_output_t[j] * hidden_t[i];
            atomicAdd(&mlp.output_weights[i * output_size + j], learning_rate * gradient);
        }
    }

    for (int i = 0; i < output_size; ++i) {
        float gradient = d_output_t[i];
        atomicAdd(&mlp.output_biases[i], learning_rate * gradient);
    }

    // Input to Hidden weights and biases
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            float gradient = d_hidden_t[j] * input_sample[i];
            atomicAdd(&mlp.input_weights[i * hidden_size + j], learning_rate * gradient);
        }
    }

    for (int i = 0; i < hidden_size; ++i) {
        float gradient = d_hidden_t[i];
        atomicAdd(&mlp.hidden_biases[i], learning_rate * gradient);
    }
}

// Function to train the MLP using Mini-Batch Gradient Descent
void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples, int batch_size) {
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
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    int threads_per_block = 256; // Adjust based on your GPU's capabilities
    int max_threads_per_block = threads_per_block;

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        for (int batch = 0; batch < num_batches; ++batch) {
            int batch_start = batch * batch_size;
            int current_batch_size = min(batch_size, num_samples - batch_start);

            int blocks_per_grid = (current_batch_size + threads_per_block - 1) / threads_per_block;

            // Launch kernel for the current batch
            train_kernel<<<blocks_per_grid, threads_per_block>>>(
                *mlp,
                d_inputs + batch_start * mlp->input_size,
                d_targets + batch_start,
                current_batch_size,
                LEARNING_RATE,
                mlp->input_size, mlp->hidden_size, mlp->output_size
            );

            cudaDeviceSynchronize();
        }

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
                                float *loss, int *correct_count,
                                int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_samples)
        return;

    // Thread-private variables
    const int MAX_HIDDEN_SIZE = 1024;
    const int MAX_OUTPUT_SIZE = 10;

    float hidden_t[MAX_HIDDEN_SIZE];
    float output_t[MAX_OUTPUT_SIZE];

    // Load input and target
    float *input_sample = &inputs[idx * input_size];
    int target_label = targets[idx];

    // Forward pass
    for (int i = 0; i < hidden_size; ++i) {
        float sum = mlp.hidden_biases[i];
        for (int j = 0; j < input_size; ++j) {
            sum += input_sample[j] * mlp.input_weights[j * hidden_size + i];
        }
        // Activation function (ReLU)
        hidden_t[i] = fmaxf(0.0f, sum);
    }

    // Output layer
    for (int i = 0; i < output_size; ++i) {
        float sum = mlp.output_biases[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden_t[j] * mlp.output_weights[j * output_size + i];
        }
        // Activation function (Sigmoid)
        output_t[i] = 1.0f / (1.0f + expf(-sum));
    }

    // Compute loss (Cross-Entropy) and accuracy
    float sample_loss = 0.0f;
    float target_vector = (float)(target_label);
    for (int i = 0; i < output_size; ++i) {
        float t = (target_label == i) ? 1.0f : 0.0f;
        sample_loss -= t * logf(output_t[i] + 1e-7f);  // Add epsilon to prevent log(0)
    }

    // Accumulate loss
    atomicAdd(loss, sample_loss);

    // Determine if the prediction is correct
    int predicted_label = 0;
    float max_output = output_t[0];
    for (int i = 1; i < output_size; ++i) {
        if (output_t[i] > max_output) {
            max_output = output_t[i];
            predicted_label = i;
        }
    }
    if (predicted_label == target_label) {
        atomicAdd(correct_count, 1);
    }
}

// Function to evaluate the MLP
void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples, float *loss, float *accuracy) {
    // Copy inputs and targets to device memory
    float *d_inputs;
    int *d_targets;
    float *d_loss;
    int *d_correct_count;
    size_t input_size_bytes = num_samples * mlp->input_size * sizeof(float);
    size_t target_size_bytes = num_samples * sizeof(int);
    cudaMalloc(&d_inputs, input_size_bytes);
    cudaMalloc(&d_targets, target_size_bytes);
    cudaMallocManaged(&d_loss, sizeof(float));
    cudaMallocManaged(&d_correct_count, sizeof(int));

    cudaMemcpy(d_inputs, inputs, input_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, target_size_bytes, cudaMemcpyHostToDevice);
    *d_loss = 0.0f;
    *d_correct_count = 0;

    // Evaluation parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_samples + threads_per_block - 1) / threads_per_block;

    evaluate_kernel<<<blocks_per_grid, threads_per_block>>>(
        *mlp,
        d_inputs, d_targets, num_samples,
        d_loss, d_correct_count,
        mlp->input_size, mlp->hidden_size, mlp->output_size
    );
    cudaDeviceSynchronize();

    *loss = *d_loss / num_samples;
    *accuracy = (float)(*d_correct_count) / num_samples * 100.0f;

    // Free device memory
    cudaFree(d_inputs);
    cudaFree(d_targets);
    cudaFree(d_loss);
    cudaFree(d_correct_count);
}

// ========================================================
// Common Functions (Optional)
// ========================================================

// Function to generate data points (if needed)
void mlp_generate_data(float *inputs, int *targets, int num_samples) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < num_samples; ++i) {
        float x = ((float)rand() / RAND_MAX) * 2 - 1;  // Random value between -1 and 1
        float y = ((float)rand() / RAND_MAX) * 2 - 1;
        inputs[i * 2] = x;
        inputs[i * 2 + 1] = y;
        // Simple function: target = 1 if inside circle of radius 0.5
        targets[i] = (x * x + y * y < 0.25f) ? 1 : 0;
    }
}
