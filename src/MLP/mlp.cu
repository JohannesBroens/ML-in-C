#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include "mlp.h"

// CUDA kernel for training
__global__ void train_kernel(float *input_weights, float *output_weights,
                             float *hidden_biases, float *output_biases,
                             float *inputs, float *targets, float *losses, int num_samples) {
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
    // Store squared error for loss computation
    losses[idx] = error * error;

    // Backward pass (simple gradient descent)
    float d_output = error * output * (1.0f - output);  // Derivative of sigmoid

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        float d_hidden = d_output * output_weights[i];
        d_hidden = d_hidden * ((hidden[i] > 0) ? 1.0f : 0.0f);  // Derivative of ReLU

        // Update weights and biases
        atomicAdd(&output_weights[i], LEARNING_RATE * d_output * hidden[i]);
        for (int j = 0; j < INPUT_SIZE; ++j) {
            atomicAdd(&input_weights[j * HIDDEN_SIZE + i], LEARNING_RATE * d_hidden * inputs[idx * INPUT_SIZE + j]);
        }

        if (threadIdx.x == 0) {  // Update biases once per block
            atomicAdd(&hidden_biases[i], LEARNING_RATE * d_hidden);
        }
    }
    if (threadIdx.x == 0) {
        atomicAdd(&output_biases[0], LEARNING_RATE * d_output);
    }
}

// CUDA kernel for evaluation
__global__ void evaluate_kernel(float *input_weights, float *output_weights,
                                float *hidden_biases, float *output_biases,
                                float *inputs, float *targets, float *losses, int num_samples) {
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
    // Store squared error for loss computation
    losses[idx] = error * error;
}

void initialize_mlp(MLP *mlp) {
    // Allocate memory for weights and biases
    size_t input_w_size = INPUT_SIZE * HIDDEN_SIZE * sizeof(float);
    size_t output_w_size = HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float);
    size_t hidden_b_size = HIDDEN_SIZE * sizeof(float);
    size_t output_b_size = OUTPUT_SIZE * sizeof(float);

    cudaMallocManaged(&mlp->input_weights, input_w_size);
    cudaMallocManaged(&mlp->output_weights, output_w_size);
    cudaMallocManaged(&mlp->hidden_biases, hidden_b_size);
    cudaMallocManaged(&mlp->output_biases, output_b_size);

    // Initialize weights and biases with small random values
    srand(time(NULL));
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; ++i)
        mlp->input_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        mlp->output_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        mlp->hidden_biases[i] = 0.0f;
    mlp->output_biases[0] = 0.0f;
}

void free_mlp(MLP *mlp) {
    cudaFree(mlp->input_weights);
    cudaFree(mlp->output_weights);
    cudaFree(mlp->hidden_biases);
    cudaFree(mlp->output_biases);
}

void generate_data(float *inputs, float *targets, int num_samples) {
    // Generate data points from a distribution
    for (int i = 0; i < num_samples; ++i) {
        float x = ((float)rand() / RAND_MAX) * 2 - 1;  // Random value between -1 and 1
        float y = ((float)rand() / RAND_MAX) * 2 - 1;
        inputs[i * INPUT_SIZE] = x;
        inputs[i * INPUT_SIZE + 1] = y;
        // Simple function: target = 1 if inside circle of radius 0.5
        targets[i] = (x * x + y * y < 0.25f) ? 1.0f : 0.0f;
    }
}

void train_mlp(MLP *mlp, float *train_inputs, float *train_targets, int train_size,
               float *val_inputs, float *val_targets, int val_size,
               float *train_losses, float *val_losses) {
    // Allocate device memory for inputs, targets, and losses
    float *d_inputs, *d_targets, *d_losses;
    size_t input_size = train_size * INPUT_SIZE * sizeof(float);
    size_t target_size = train_size * sizeof(float);
    size_t loss_size = train_size * sizeof(float);
    cudaMalloc(&d_inputs, input_size);
    cudaMalloc(&d_targets, target_size);
    cudaMalloc(&d_losses, loss_size);

    // For validation data
    float *d_val_inputs, *d_val_targets, *d_val_losses;
    size_t val_input_size = val_size * INPUT_SIZE * sizeof(float);
    size_t val_target_size = val_size * sizeof(float);
    size_t val_loss_size = val_size * sizeof(float);
    cudaMalloc(&d_val_inputs, val_input_size);
    cudaMalloc(&d_val_targets, val_target_size);
    cudaMalloc(&d_val_losses, val_loss_size);

    cudaMemcpy(d_inputs, train_inputs, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, train_targets, target_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_inputs, val_inputs, val_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_targets, val_targets, val_target_size, cudaMemcpyHostToDevice);

    // Training loop
    int threads_per_block = 256;
    int blocks_per_grid = (train_size + threads_per_block - 1) / threads_per_block;
    int val_blocks_per_grid = (val_size + threads_per_block - 1) / threads_per_block;

    // Remove existing loss file
    remove("losses.txt");

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        // Zero the losses
        cudaMemset(d_losses, 0, loss_size);

        // Training step
        train_kernel<<<blocks_per_grid, threads_per_block>>>(mlp->input_weights, mlp->output_weights,
                                                             mlp->hidden_biases, mlp->output_biases,
                                                             d_inputs, d_targets, d_losses, train_size);
        cudaDeviceSynchronize();

        // Compute training loss
        cudaMemcpy(train_losses, d_losses, loss_size, cudaMemcpyDeviceToHost);
        float train_loss = 0.0f;
        for (int i = 0; i < train_size; ++i) {
            train_loss += train_losses[i];
        }
        train_loss /= train_size;

        // Validation step
        cudaMemset(d_val_losses, 0, val_loss_size);
        evaluate_kernel<<<val_blocks_per_grid, threads_per_block>>>(mlp->input_weights, mlp->output_weights,
                                                                    mlp->hidden_biases, mlp->output_biases,
                                                                    d_val_inputs, d_val_targets, d_val_losses, val_size);
        cudaDeviceSynchronize();

        // Compute validation loss
        cudaMemcpy(val_losses, d_val_losses, val_loss_size, cudaMemcpyDeviceToHost);
        float val_loss = 0.0f;
        for (int i = 0; i < val_size; ++i) {
            val_loss += val_losses[i];
        }
        val_loss /= val_size;

        // Print losses
        if (epoch % 10 == 0) {
            printf("Epoch %d, Training Loss: %f, Validation Loss: %f\n", epoch, train_loss, val_loss);
        }

        // Save losses to a file for plotting
        FILE *loss_file = fopen("losses.txt", "a");
        if (loss_file) {
            fprintf(loss_file, "%d %f %f\n", epoch, train_loss, val_loss);
            fclose(loss_file);
        }
    }

    cudaFree(d_inputs);
    cudaFree(d_targets);
    cudaFree(d_losses);
    cudaFree(d_val_inputs);
    cudaFree(d_val_targets);
    cudaFree(d_val_losses);
}

float evaluate_mlp(MLP *mlp, float *inputs, float *targets, int num_samples) {
    // Allocate device memory for inputs, targets, and losses
    float *d_inputs, *d_targets, *d_losses;
    size_t input_size = num_samples * INPUT_SIZE * sizeof(float);
    size_t target_size = num_samples * sizeof(float);
    size_t loss_size = num_samples * sizeof(float);
    cudaMalloc(&d_inputs, input_size);
    cudaMalloc(&d_targets, target_size);
    cudaMalloc(&d_losses, loss_size);

    cudaMemcpy(d_inputs, inputs, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, target_size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (num_samples + threads_per_block - 1) / threads_per_block;

    evaluate_kernel<<<blocks_per_grid, threads_per_block>>>(mlp->input_weights, mlp->output_weights,
                                                            mlp->hidden_biases, mlp->output_biases,
                                                            d_inputs, d_targets, d_losses, num_samples);
    cudaDeviceSynchronize();

    // Compute loss
    float *losses = (float *)malloc(loss_size);
    cudaMemcpy(losses, d_losses, loss_size, cudaMemcpyDeviceToHost);
    float total_loss = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
        total_loss += losses[i];
    }
    total_loss /= num_samples;

    // Clean up
    cudaFree(d_inputs);
    cudaFree(d_targets);
    cudaFree(d_losses);
    free(losses);

    return total_loss;
}

int main() {
    MLP mlp;
    initialize_mlp(&mlp);

    // Total number of samples
    int num_samples = NUM_SAMPLES;

    // Calculate sizes for splits
    int train_size = (int)(num_samples * TRAIN_RATIO);
    int val_size = (int)(num_samples * VAL_RATIO);
    int test_size = num_samples - train_size - val_size;

    // Allocate memory for data
    float *inputs = (float *)malloc(num_samples * INPUT_SIZE * sizeof(float));
    float *targets = (float *)malloc(num_samples * sizeof(float));
    generate_data(inputs, targets, num_samples);

    // Shuffle data
    // Create an array of indices and shuffle them
    int *indices = (int *)malloc(num_samples * sizeof(int));
    for (int i = 0; i < num_samples; ++i) {
        indices[i] = i;
    }
    // Simple Fisher-Yates shuffle
    for (int i = num_samples - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        // Swap indices[i] and indices[j]
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    // Create shuffled inputs and targets
    float *shuffled_inputs = (float *)malloc(num_samples * INPUT_SIZE * sizeof(float));
    float *shuffled_targets = (float *)malloc(num_samples * sizeof(float));
    for (int i = 0; i < num_samples; ++i) {
        shuffled_targets[i] = targets[indices[i]];
        shuffled_inputs[i * INPUT_SIZE] = inputs[indices[i] * INPUT_SIZE];
        shuffled_inputs[i * INPUT_SIZE + 1] = inputs[indices[i] * INPUT_SIZE + 1];
    }
    free(indices);
    free(inputs);
    free(targets);
    inputs = shuffled_inputs;
    targets = shuffled_targets;

    // Split data
    float *train_inputs = inputs;
    float *train_targets = targets;

    float *val_inputs = inputs + train_size * INPUT_SIZE;
    float *val_targets = targets + train_size;

    float *test_inputs = inputs + (train_size + val_size) * INPUT_SIZE;
    float *test_targets = targets + train_size + val_size;

    // Arrays to hold per-sample losses
    float *train_losses = (float *)malloc(train_size * sizeof(float));
    float *val_losses = (float *)malloc(val_size * sizeof(float));

    // Train the MLP
    train_mlp(&mlp, train_inputs, train_targets, train_size,
              val_inputs, val_targets, val_size,
              train_losses, val_losses);

    // Evaluate on test set
    float test_loss = evaluate_mlp(&mlp, test_inputs, test_targets, test_size);
    printf("Test Loss: %f\n", test_loss);

    // Optionally, write test loss to a file
    FILE *test_loss_file = fopen("test_loss.txt", "w");
    if (test_loss_file) {
        fprintf(test_loss_file, "%f\n", test_loss);
        fclose(test_loss_file);
    }

    // Clean up
    free_mlp(&mlp);
    free(inputs);
    free(targets);
    free(train_losses);
    free(val_losses);

    return 0;
}
