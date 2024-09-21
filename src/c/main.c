// main.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mlp.h"
#include "data_loader.h"  // Include the data loader header

// Function prototypes
void print_usage(const char *program_name);

int main(int argc, char *argv[]) {
    char *dataset_name = NULL;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--dataset") == 0 || strcmp(argv[i], "-d") == 0) && i + 1 < argc) {
            dataset_name = argv[++i];
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (dataset_name == NULL) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Variables for data and labels
    float *inputs = NULL;
    int *labels = NULL;
    int num_samples = 0;
    int input_size = 0;
    int output_size = 0; // Number of classes
    int hidden_size = 16; // You can adjust this as needed

    // Load the selected dataset
    if (strcmp(dataset_name, "generated") == 0) {
        if (load_generated_data(&inputs, &labels, &num_samples, &input_size, &output_size) != 0) {
            fprintf(stderr, "Failed to load generated data.\n");
            return EXIT_FAILURE;
        }
    } else if (strcmp(dataset_name, "iris") == 0) {
        if (load_iris_data(&inputs, &labels, &num_samples, &input_size, &output_size) != 0) {
            fprintf(stderr, "Failed to load Iris dataset.\n");
            return EXIT_FAILURE;
        }
    } else if (strcmp(dataset_name, "wine-red") == 0) {
        if (load_wine_quality_data(&inputs, &labels, &num_samples, &input_size, &output_size, "data/winequality-red.csv") != 0) {
            fprintf(stderr, "Failed to load Wine Quality (Red) dataset.\n");
            return EXIT_FAILURE;
        }
    } else if (strcmp(dataset_name, "wine-white") == 0) {
        if (load_wine_quality_data(&inputs, &labels, &num_samples, &input_size, &output_size, "data/winequality-white.csv") != 0) {
            fprintf(stderr, "Failed to load Wine Quality (White) dataset.\n");
            return EXIT_FAILURE;
        }
    } else if (strcmp(dataset_name, "breast-cancer") == 0) {
        if (load_breast_cancer_data(&inputs, &labels, &num_samples, &input_size, &output_size) != 0) {
            fprintf(stderr, "Failed to load Breast Cancer dataset.\n");
            return EXIT_FAILURE;
        }
    } else {
        fprintf(stderr, "Unknown dataset: %s\n", dataset_name);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Initialize the MLP model with appropriate input size and output size
    MLP mlp;
    mlp_initialize(&mlp, input_size, hidden_size, output_size);

    // Train the MLP
    int batch_size = 64; // Adjust the batch size as needed
    mlp_train(&mlp, inputs, labels, num_samples, batch_size);

    // Evaluate the MLP
    float loss, accuracy;
    mlp_evaluate(&mlp, inputs, labels, num_samples, &loss, &accuracy);
    printf("Final loss on %s dataset: %.6f\n", dataset_name, loss);
    printf("Accuracy on %s dataset: %.2f%%\n", dataset_name, accuracy);

    // Free resources
    mlp_free(&mlp);
    free(inputs);
    free(labels);

    return EXIT_SUCCESS;
}

void print_usage(const char *program_name) {
    printf("Usage: %s --dataset [generated|iris|wine-red|wine-white|breast-cancer]\n", program_name);
}
