#ifndef MLP_H
#define MLP_H

#ifdef __cplusplus
extern "C" {
#endif

// Removed hardcoded sizes to allow dynamic configuration
// #define INPUT_SIZE    2
// #define HIDDEN_SIZE   16
// #define OUTPUT_SIZE   1

#define NUM_EPOCHS    1000   // Number of training epochs
#define LEARNING_RATE 0.01f  // Learning rate for training

typedef struct {
    int input_size;     // Added to store the input size
    int hidden_size;    // Added to store the hidden layer size
    int output_size;    // Added to store the output size

    float *input_weights;    // Size: input_size * hidden_size
    float *output_weights;   // Size: hidden_size * output_size
    float *hidden_biases;    // Size: hidden_size
    float *output_biases;    // Size: output_size
} MLP;

// Function declarations with updated parameters
void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size);
void mlp_free(MLP *mlp);

// Removed mlp_generate_data function as data generation is handled separately

void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples);
void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples, float *loss);

#ifdef __cplusplus
}
#endif

#endif // MLP_H
