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
__global__ void init_weights_biases(float *weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        weights[idx] = (curand_uniform(&state) - 0.5f) * 0.1f;
    }
}

// Function to initialize the MLP
void mlp_initialize(MLP *mlp) {
    size_t input_w_size = INPUT_SIZE * HIDDEN_SIZE * sizeof(float);
    size_t output_w_size = HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float);
    size_t hidden_b_size = HIDDEN_SIZE * sizeof(float);
    size_t output_b_size = OUTPUT_SIZE * sizeof(float);

    // Allocate managed memory
    cudaMallocManaged(&mlp->input_weights, input_w_size);
    cudaMallocManaged(&mlp->output_weights, output_w_size);
    cudaMallocManaged(&mlp->hidden_biases, hidden_b_size);
    cudaMallocManaged(&mlp->output_biases, output_b_size);

    // Initialize weights and biases using CUDA kernels
    int threads_per_block = 256;
    int blocks_per_grid_input = (INPUT_SIZE * HIDDEN_SIZE + threads_per_block - 1) / threads_per_block;
    // Commented out for now
    // int blocks_per_grid_hidden = (HIDDEN_SIZE + threads_per_block - 1) / threads_per_block;

    init_weights_biases<<<blocks_per_grid_input, threads_per_block>>>(mlp->input_weights, INPUT_SIZE * HIDDEN_SIZE);
    init_weights_biases<<<blocks_per_grid_input, threads_per_block>>>(mlp->output_weights, HIDDEN_SIZE * OUTPUT_SIZE);

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
__global__ void train_kernel(float *input_weights, float *output_weights,
                             float *hidden_biases, float *output_biases,
                             float *inputs, float *targets, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_samples)
        return;

    // Forward pass
    float hidden[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hidden[i] = hidden_biases[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            hidden[i] += inputs[idx * INPUT_SIZE + j] * input_weights[j * HIDDEN_SIZE + i];
        }
        // Activation function (ReLU)
        hidden[i] = fmaxf(0.0f, hidden[i]);
    }

    float output = output_biases[0];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        output += hidden[i] * output_weights[i];
    }
    // Activation function (Sigmoid)
    output = 1.0f / (1.0f + expf(-output));

    // Compute error
    float error = targets[idx] - output;

    // Backward pass (simple gradient descent)
    float d_output = error * output * (1.0f - output);  // Derivative of sigmoid

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        float d_hidden = d_output * output_weights[i];
        d_hidden = d_hidden * ((hidden[i] > 0) ? 1.0f : 0.0f);  // Derivative of ReLU

        // Update weights and biases using atomic operations
        atomicAdd(&output_weights[i], LEARNING_RATE * d_output * hidden[i]);
        atomicAdd(&input_weights[i + INPUT_SIZE * 0], LEARNING_RATE * d_hidden * inputs[idx * INPUT_SIZE + 0]);
        atomicAdd(&input_weights[i + INPUT_SIZE * 1], LEARNING_RATE * d_hidden * inputs[idx * INPUT_SIZE + 1]);

        // Update biases (only once per thread block)
        if (threadIdx.x == 0) {
            atomicAdd(&hidden_biases[i], LEARNING_RATE * d_hidden);
        }
    }
    if (threadIdx.x == 0) {
        atomicAdd(&output_biases[0], LEARNING_RATE * d_output);
    }
}

// Function to train the MLP
void mlp_train(MLP *mlp, float *inputs, float *targets, int num_samples) {
    // Copy inputs and targets to device memory
    float *d_inputs, *d_targets;
    size_t input_size = num_samples * INPUT_SIZE * sizeof(float);
    size_t target_size = num_samples * sizeof(float);
    cudaMalloc(&d_inputs, input_size);
    cudaMalloc(&d_targets, target_size);

    cudaMemcpy(d_inputs, inputs, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, target_size, cudaMemcpyHostToDevice);

    // Training parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_samples + threads_per_block - 1) / threads_per_block;

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        train_kernel<<<blocks_per_grid, threads_per_block>>>(mlp->input_weights, mlp->output_weights,
                                                             mlp->hidden_biases, mlp->output_biases,
                                                             d_inputs, d_targets, num_samples);
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
__global__ void evaluate_kernel(float *input_weights, float *output_weights,
                                float *hidden_biases, float *output_biases,
                                float *inputs, float *targets, int num_samples,
                                float *loss) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_samples)
        return;

    // Forward pass
    float hidden[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hidden[i] = hidden_biases[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            hidden[i] += inputs[idx * INPUT_SIZE + j] * input_weights[j * HIDDEN_SIZE + i];
        }
        // Activation function (ReLU)
        hidden[i] = fmaxf(0.0f, hidden[i]);
    }

    float output = output_biases[0];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        output += hidden[i] * output_weights[i];
    }
    // Activation function (Sigmoid)
    output = 1.0f / (1.0f + expf(-output));

    // Compute error
    float error = targets[idx] - output;

    // Accumulate loss
    atomicAdd(loss, error * error);
}

// Function to evaluate the MLP
void mlp_evaluate(MLP *mlp, float *inputs, float *targets, int num_samples, float *loss) {
    // Copy inputs and targets to device memory
    float *d_inputs, *d_targets, *d_loss;
    size_t input_size = num_samples * INPUT_SIZE * sizeof(float);
    size_t target_size = num_samples * sizeof(float);
    cudaMalloc(&d_inputs, input_size);
    cudaMalloc(&d_targets, target_size);
    cudaMallocManaged(&d_loss, sizeof(float));

    cudaMemcpy(d_inputs, inputs, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, target_size, cudaMemcpyHostToDevice);
    *d_loss = 0.0f;

    // Evaluation parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_samples + threads_per_block - 1) / threads_per_block;

    evaluate_kernel<<<blocks_per_grid, threads_per_block>>>(mlp->input_weights, mlp->output_weights,
                                                            mlp->hidden_biases, mlp->output_biases,
                                                            d_inputs, d_targets, num_samples, d_loss);
    cudaDeviceSynchronize();

    *loss = *d_loss / num_samples;

    // Free device memory
    cudaFree(d_inputs);
    cudaFree(d_targets);
    cudaFree(d_loss);
}

// ========================================================
// Common Functions
// ========================================================

// Function to generate data points
void mlp_generate_data(float *inputs, float *targets, int num_samples) {
    seed_random();
    for (int i = 0; i < num_samples; ++i) {
        float x = ((float)rand() / RAND_MAX) * 2 - 1;  // Random value between -1 and 1
        float y = ((float)rand() / RAND_MAX) * 2 - 1;
        inputs[i * INPUT_SIZE] = x;
        inputs[i * INPUT_SIZE + 1] = y;
        // Simple function: target = 1 if inside circle of radius 0.5
        targets[i] = (x * x + y * y < 0.25f) ? 1.0f : 0.0f;
    }
}
