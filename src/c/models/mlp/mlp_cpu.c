#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h> // Include OpenMP header

// Common function to initialize random number generator
static void seed_random() {
    srand((unsigned int)time(NULL));
}

// ========================================================
// CPU Implementations with OpenMP
// ========================================================

// Function to initialize the MLP
void mlp_initialize(MLP *mlp) {
    size_t input_w_size = INPUT_SIZE * HIDDEN_SIZE * sizeof(float);
    size_t output_w_size = HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float);
    size_t hidden_b_size = HIDDEN_SIZE * sizeof(float);
    size_t output_b_size = OUTPUT_SIZE * sizeof(float);

    // Allocate memory using malloc
    mlp->input_weights = (float *)malloc(input_w_size);
    mlp->output_weights = (float *)malloc(output_w_size);
    mlp->hidden_biases = (float *)malloc(hidden_b_size);
    mlp->output_biases = (float *)malloc(output_b_size);

    // Initialize weights and biases
    seed_random();

    // Initialize weights using parallel for loops
    #pragma omp parallel for
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; ++i)
        mlp->input_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; ++i)
        mlp->output_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // Initialize biases to zero
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        mlp->hidden_biases[i] = 0.0f;

    mlp->output_biases[0] = 0.0f;
}

// Function to free the MLP resources
void mlp_free(MLP *mlp) {
    free(mlp->input_weights);
    free(mlp->output_weights);
    free(mlp->hidden_biases);
    free(mlp->output_biases);
}

// Function to train the MLP
void mlp_train(MLP *mlp, float *inputs, float *targets, int num_samples) {
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        // Parallelize over samples
        #pragma omp parallel for
        for (int idx = 0; idx < num_samples; ++idx) {
            // Each thread needs its own copies of variables to avoid race conditions
            float hidden[HIDDEN_SIZE];
            float output;
            float error;
            float d_output;
            float d_hidden[HIDDEN_SIZE];

            // Forward pass
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                hidden[i] = mlp->hidden_biases[i];
                for (int j = 0; j < INPUT_SIZE; ++j) {
                    hidden[i] += inputs[idx * INPUT_SIZE + j] * mlp->input_weights[j * HIDDEN_SIZE + i];
                }
                // Activation function (ReLU)
                hidden[i] = fmaxf(0.0f, hidden[i]);
            }

            output = mlp->output_biases[0];
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                output += hidden[i] * mlp->output_weights[i];
            }
            // Activation function (Sigmoid)
            output = 1.0f / (1.0f + expf(-output));

            // Compute error
            error = targets[idx] - output;

            // Backward pass (simple gradient descent)
            d_output = error * output * (1.0f - output);  // Derivative of sigmoid

            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                d_hidden[i] = d_output * mlp->output_weights[i];
                d_hidden[i] *= (hidden[i] > 0) ? 1.0f : 0.0f;  // Derivative of ReLU
            }

            // Update weights and biases
            // Use critical sections or atomic operations to prevent race conditions
            #pragma omp critical
            {
                // Update output weights and biases
                for (int i = 0; i < HIDDEN_SIZE; ++i) {
                    mlp->output_weights[i] += LEARNING_RATE * d_output * hidden[i];
                }
                mlp->output_biases[0] += LEARNING_RATE * d_output;
            }

            #pragma omp critical
            {
                // Update input weights and hidden biases
                for (int i = 0; i < HIDDEN_SIZE; ++i) {
                    for (int j = 0; j < INPUT_SIZE; ++j) {
                        mlp->input_weights[j * HIDDEN_SIZE + i] += LEARNING_RATE * d_hidden[i] * inputs[idx * INPUT_SIZE + j];
                    }
                    mlp->hidden_biases[i] += LEARNING_RATE * d_hidden[i];
                }
            }
        }

        if (epoch % 100 == 0) {
            printf("Epoch %d completed.\n", epoch);
        }
    }
}

// Function to evaluate the MLP
void mlp_evaluate(MLP *mlp, float *inputs, float *targets, int num_samples, float *loss) {
    float total_loss = 0.0f;

    // Parallelize over samples
    #pragma omp parallel for reduction(+:total_loss)
    for (int idx = 0; idx < num_samples; ++idx) {
        float hidden[HIDDEN_SIZE];
        float output;
        float error;

        // Forward pass
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            hidden[i] = mlp->hidden_biases[i];
            for (int j = 0; j < INPUT_SIZE; ++j) {
                hidden[i] += inputs[idx * INPUT_SIZE + j] * mlp->input_weights[j * HIDDEN_SIZE + i];
            }
            // Activation function (ReLU)
            hidden[i] = fmaxf(0.0f, hidden[i]);
        }

        output = mlp->output_biases[0];
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            output += hidden[i] * mlp->output_weights[i];
        }
        // Activation function (Sigmoid)
        output = 1.0f / (1.0f + expf(-output));

        // Compute error
        error = targets[idx] - output;
        total_loss += error * error;
    }

    *loss = total_loss / num_samples;
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
