#ifndef MLP_H
#define MLP_H

#define INPUT_SIZE    2     // Number of input neurons
#define HIDDEN_SIZE   16    // Number of hidden neurons
#define OUTPUT_SIZE   1     // Number of output neurons
#define NUM_SAMPLES   4096  // Total number of samples

#define TRAIN_RATIO   0.6  // 60% for training
#define VAL_RATIO     0.2  // 20% for validation
#define TEST_RATIO    0.2  // 20% for testing

#define NUM_EPOCHS    1000  // Number of training epochs
#define LEARNING_RATE 0.01f

// Structure to hold the MLP parameters
typedef struct {
    float *input_weights;   // Weights between input and hidden layer
    float *output_weights;  // Weights between hidden and output layer
    float *hidden_biases;   // Biases of hidden layer
    float *output_biases;   // Biases of output layer
} MLP;

// Function declarations
void initialize_mlp(MLP *mlp);
void free_mlp(MLP *mlp);

void generate_data(float *inputs, float *targets, int num_samples);

void train_mlp(MLP *mlp, float *train_inputs, float *train_targets, int train_size,
               float *val_inputs, float *val_targets, int val_size,
               float *train_losses, float *val_losses);

float evaluate_mlp(MLP *mlp, float *inputs, float *targets, int num_samples);

#endif // MLP_H
