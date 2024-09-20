#ifndef MLP_H
#define MLP_H

#ifdef __cplusplus
extern "C" {
#endif

#define INPUT_SIZE    2
#define HIDDEN_SIZE   16
#define OUTPUT_SIZE   1
#define NUM_EPOCHS    1000   // Number of training epochs
#define LEARNING_RATE 0.01f  // Learning rate for training

typedef struct {
    float *input_weights;
    float *output_weights;
    float *hidden_biases;
    float *output_biases;
} MLP;

// Function declarations
void mlp_initialize(MLP *mlp);
void mlp_free(MLP *mlp);

void mlp_generate_data(float *inputs, float *targets, int num_samples);

void mlp_train(MLP *mlp, float *inputs, float *targets, int num_samples);
void mlp_evaluate(MLP *mlp, float *inputs, float *targets, int num_samples, float *loss);

#ifdef __cplusplus
}
#endif

#endif // MLP_H
