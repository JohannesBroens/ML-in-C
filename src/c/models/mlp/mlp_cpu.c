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
void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size) {
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;

    size_t input_w_size = input_size * hidden_size * sizeof(float);
    size_t output_w_size = hidden_size * output_size * sizeof(float);
    size_t hidden_b_size = hidden_size * sizeof(float);
    size_t output_b_size = output_size * sizeof(float);

    // Allocate memory using malloc
    mlp->input_weights = (float *)malloc(input_w_size);
    mlp->output_weights = (float *)malloc(output_w_size);
    mlp->hidden_biases = (float *)malloc(hidden_b_size);
    mlp->output_biases = (float *)malloc(output_b_size);

    // Check for allocation failure
    if (!mlp->input_weights || !mlp->output_weights || !mlp->hidden_biases || !mlp->output_biases) {
        fprintf(stderr, "Failed to allocate memory for MLP parameters.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize weights and biases
    seed_random();

    // Initialize weights using parallel for loops
    #pragma omp parallel for
    for (int i = 0; i < input_size * hidden_size; ++i)
        mlp->input_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    #pragma omp parallel for
    for (int i = 0; i < hidden_size * output_size; ++i)
        mlp->output_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // Initialize biases to zero
    #pragma omp parallel for
    for (int i = 0; i < hidden_size; ++i)
        mlp->hidden_biases[i] = 0.0f;

    #pragma omp parallel for
    for (int i = 0; i < output_size; ++i)
        mlp->output_biases[i] = 0.0f;
}

// Function to free the MLP resources
void mlp_free(MLP *mlp) {
    free(mlp->input_weights);
    free(mlp->output_weights);
    free(mlp->hidden_biases);
    free(mlp->output_biases);
}

// Function to train the MLP
void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples) {
    int input_size = mlp->input_size;
    int hidden_size = mlp->hidden_size;
    int output_size = mlp->output_size;

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        // Arrays to accumulate weight updates
        float *delta_input_weights = calloc(input_size * hidden_size, sizeof(float));
        float *delta_output_weights = calloc(hidden_size * output_size, sizeof(float));
        float *delta_hidden_biases = calloc(hidden_size, sizeof(float));
        float *delta_output_biases = calloc(output_size, sizeof(float));

        // Check for allocation failure
        if (!delta_input_weights || !delta_output_weights || !delta_hidden_biases || !delta_output_biases) {
            fprintf(stderr, "Failed to allocate memory for weight updates.\n");
            exit(EXIT_FAILURE);
        }

        // Parallelize over samples
        #pragma omp parallel for
        for (int idx = 0; idx < num_samples; ++idx) {
            // Each thread needs its own copies of variables to avoid race conditions
            float *hidden = (float *)malloc(hidden_size * sizeof(float));
            float *output = (float *)malloc(output_size * sizeof(float));
            float *d_output = (float *)malloc(output_size * sizeof(float));
            float *d_hidden = (float *)malloc(hidden_size * sizeof(float));

            if (!hidden || !output || !d_output || !d_hidden) {
                fprintf(stderr, "Failed to allocate memory for activations.\n");
                exit(EXIT_FAILURE);
            }

            // Forward pass
            for (int i = 0; i < hidden_size; ++i) {
                float sum = mlp->hidden_biases[i];
                for (int j = 0; j < input_size; ++j) {
                    sum += inputs[idx * input_size + j] * mlp->input_weights[j * hidden_size + i];
                }
                // Activation function (ReLU)
                hidden[i] = fmaxf(0.0f, sum);
            }

            // Output layer
            for (int i = 0; i < output_size; ++i) {
                float sum = mlp->output_biases[i];
                for (int j = 0; j < hidden_size; ++j) {
                    sum += hidden[j] * mlp->output_weights[j * output_size + i];
                }
                // Activation function (Sigmoid)
                output[i] = 1.0f / (1.0f + expf(-sum));
            }

            // Compute error using one-hot encoding for multi-class classification
            float *target_vector = (float *)calloc(output_size, sizeof(float));
            if (!target_vector) {
                fprintf(stderr, "Failed to allocate memory for target vector.\n");
                exit(EXIT_FAILURE);
            }
            target_vector[targets[idx]] = 1.0f;

            // Backward pass
            // Output layer gradients
            for (int i = 0; i < output_size; ++i) {
                float error = target_vector[i] - output[i];
                d_output[i] = error * output[i] * (1.0f - output[i]); // Derivative of Sigmoid
            }

            // Hidden layer gradients
            for (int i = 0; i < hidden_size; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < output_size; ++j) {
                    sum += d_output[j] * mlp->output_weights[i * output_size + j];
                }
                d_hidden[i] = sum * ((hidden[i] > 0) ? 1.0f : 0.0f); // Derivative of ReLU
            }

            // Accumulate weight updates using atomic operations
            // Input to Hidden weights and biases
            for (int i = 0; i < input_size; ++i) {
                for (int j = 0; j < hidden_size; ++j) {
                    float delta = LEARNING_RATE * d_hidden[j] * inputs[idx * input_size + i];
                    #pragma omp atomic
                    delta_input_weights[i * hidden_size + j] += delta;
                }
            }
            for (int i = 0; i < hidden_size; ++i) {
                float delta = LEARNING_RATE * d_hidden[i];
                #pragma omp atomic
                delta_hidden_biases[i] += delta;
            }

            // Hidden to Output weights and biases
            for (int i = 0; i < hidden_size; ++i) {
                for (int j = 0; j < output_size; ++j) {
                    float delta = LEARNING_RATE * d_output[j] * hidden[i];
                    #pragma omp atomic
                    delta_output_weights[i * output_size + j] += delta;
                }
            }
            for (int i = 0; i < output_size; ++i) {
                float delta = LEARNING_RATE * d_output[i];
                #pragma omp atomic
                delta_output_biases[i] += delta;
            }

            // Free temporary memory
            free(hidden);
            free(output);
            free(d_output);
            free(d_hidden);
            free(target_vector);
        }

        // Update weights and biases
        for (int i = 0; i < input_size * hidden_size; ++i) {
            mlp->input_weights[i] += delta_input_weights[i];
        }
        for (int i = 0; i < hidden_size * output_size; ++i) {
            mlp->output_weights[i] += delta_output_weights[i];
        }
        for (int i = 0; i < hidden_size; ++i) {
            mlp->hidden_biases[i] += delta_hidden_biases[i];
        }
        for (int i = 0; i < output_size; ++i) {
            mlp->output_biases[i] += delta_output_biases[i];
        }

        // Free accumulated updates
        free(delta_input_weights);
        free(delta_output_weights);
        free(delta_hidden_biases);
        free(delta_output_biases);

        if (epoch % 100 == 0) {
            printf("Epoch %d completed.\n", epoch);
        }
    }
}

// Function to evaluate the MLP
void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples, float *loss) {
    int input_size = mlp->input_size;
    int hidden_size = mlp->hidden_size;
    int output_size = mlp->output_size;

    float total_loss = 0.0f;
    int correct_predictions = 0;

    // Parallelize over samples
    #pragma omp parallel for reduction(+:total_loss, correct_predictions)
    for (int idx = 0; idx < num_samples; ++idx) {
        float *hidden = (float *)malloc(hidden_size * sizeof(float));
        float *output = (float *)malloc(output_size * sizeof(float));

        if (!hidden || !output) {
            fprintf(stderr, "Failed to allocate memory for activations.\n");
            exit(EXIT_FAILURE);
        }

        // Forward pass
        for (int i = 0; i < hidden_size; ++i) {
            float sum = mlp->hidden_biases[i];
            for (int j = 0; j < input_size; ++j) {
                sum += inputs[idx * input_size + j] * mlp->input_weights[j * hidden_size + i];
            }
            // Activation function (ReLU)
            hidden[i] = fmaxf(0.0f, sum);
        }

        // Output layer
        for (int i = 0; i < output_size; ++i) {
            float sum = mlp->output_biases[i];
            for (int j = 0; j < hidden_size; ++j) {
                sum += hidden[j] * mlp->output_weights[j * output_size + i];
            }
            // Activation function (Sigmoid)
            output[i] = 1.0f / (1.0f + expf(-sum));
        }

        // Compute loss (Cross-entropy)
        float *target_vector = (float *)calloc(output_size, sizeof(float));
        if (!target_vector) {
            fprintf(stderr, "Failed to allocate memory for target vector.\n");
            exit(EXIT_FAILURE);
        }
        target_vector[targets[idx]] = 1.0f;

        float sample_loss = 0.0f;
        for (int i = 0; i < output_size; ++i) {
            sample_loss -= target_vector[i] * logf(output[i] + 1e-7f); // Add epsilon to prevent log(0)
        }
        total_loss += sample_loss;

        // Check if prediction is correct
        int predicted_class = 0;
        float max_output = output[0];
        for (int i = 1; i < output_size; ++i) {
            if (output[i] > max_output) {
                max_output = output[i];
                predicted_class = i;
            }
        }
        if (predicted_class == targets[idx]) {
            correct_predictions++;
        }

        // Free temporary memory
        free(hidden);
        free(output);
        free(target_vector);
    }

    *loss = total_loss / num_samples;
    float accuracy = (float)correct_predictions / num_samples * 100.0f;
    printf("Evaluation Loss: %.6f, Accuracy: %.2f%%\n", *loss, accuracy);
}

// ========================================================
// Common Functions (Optional)
// ========================================================

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
