#include <stdio.h>
#include <stdlib.h>
#include "models/mlp/mlp.h"

int main() {
    MLP mlp;
    mlp_initialize(&mlp);

    int num_samples = 1000;
    float *inputs = (float *)malloc(num_samples * INPUT_SIZE * sizeof(float));
    float *targets = (float *)malloc(num_samples * sizeof(float));

    mlp_generate_data(inputs, targets, num_samples);

    mlp_train(&mlp, inputs, targets, num_samples);

    float loss;
    mlp_evaluate(&mlp, inputs, targets, num_samples, &loss);

    printf("Loss: %f\n", loss);

    mlp_free(&mlp);
    free(inputs);
    free(targets);

    return 0;
}
