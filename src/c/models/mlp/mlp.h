#ifndef MLP_H
#define MLP_H

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_EPOCHS    1000   // Number of training epochs
#define LEARNING_RATE 0.01f  // Learning rate for training

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;

    float *input_weights;    // Size: input_size * hidden_size
    float *output_weights;   // Size: hidden_size * output_size
    float *hidden_biases;    // Size: hidden_size
    float *output_biases;    // Size: output_size
} MLP;

// Function declarations
void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size);
void mlp_free(MLP *mlp);

// Updated function signatures with additional parameters
void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples, int batch_size);
void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples, float *loss, float *accuracy);

#ifdef __cplusplus
}
#endif

#endif // MLP_H
