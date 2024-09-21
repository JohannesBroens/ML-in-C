// data_loader.c

#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024

// Function to load generated data
int load_generated_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size) {
    *input_size = 2;    // For generated data with x and y
    *output_size = 2;   // Assuming binary classification
    *num_samples = 1000;  // Set the number of samples

    *inputs = (float *)malloc((*num_samples) * (*input_size) * sizeof(float));
    *labels = (int *)malloc((*num_samples) * sizeof(int));

    if (!(*inputs) || !(*labels)) {
        fprintf(stderr, "Memory allocation failed for inputs or labels.\n");
        return -1;
    }

    // Generate data
    srand((unsigned int)time(NULL));
    for (int i = 0; i < *num_samples; ++i) {
        float x = ((float)rand() / RAND_MAX) * 2 - 1;
        float y = ((float)rand() / RAND_MAX) * 2 - 1;
        (*inputs)[i * (*input_size)] = x;
        (*inputs)[i * (*input_size) + 1] = y;
        (*labels)[i] = (x * x + y * y < 0.25f) ? 1 : 0;
    }

    return 0;
}

// Function to load the Iris dataset
int load_iris_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size) {
    const char *filename = "data/iris_processed.txt";
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open Iris dataset file");
        return -1;
    }

    *input_size = 4;
    *output_size = 3; // Iris dataset has 3 classes

    // Count the number of samples
    *num_samples = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        (*num_samples)++;
    }
    rewind(file);

    *inputs = (float *)malloc((*num_samples) * (*input_size) * sizeof(float));
    *labels = (int *)malloc((*num_samples) * sizeof(int));

    if (!(*inputs) || !(*labels)) {
        fprintf(stderr, "Memory allocation failed for inputs or labels.\n");
        fclose(file);
        return -1;
    }

    // Read the data
    for (int i = 0; i < *num_samples; ++i) {
        for (int j = 0; j < *input_size; ++j) {
            fscanf(file, "%f,", &(*inputs)[i * (*input_size) + j]);
        }
        fscanf(file, "%d", &(*labels)[i]);
    }

    fclose(file);
    return 0;
}

// Function to load the Wine Quality dataset
int load_wine_quality_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open Wine Quality dataset file");
        return -1;
    }

    *input_size = 11; // Number of features in Wine Quality dataset
    *output_size = 11; // Quality scores range from 0 to 10

    // Skip header line
    char line[MAX_LINE_LENGTH];
    fgets(line, sizeof(line), file);

    // Count the number of samples
    *num_samples = 0;
    while (fgets(line, sizeof(line), file)) {
        (*num_samples)++;
    }
    rewind(file);
    fgets(line, sizeof(line), file); // Skip header again

    *inputs = (float *)malloc((*num_samples) * (*input_size) * sizeof(float));
    *labels = (int *)malloc((*num_samples) * sizeof(int));

    if (!(*inputs) || !(*labels)) {
        fprintf(stderr, "Memory allocation failed for inputs or labels.\n");
        fclose(file);
        return -1;
    }

    // Read the data
    for (int i = 0; i < *num_samples; ++i) {
        for (int j = 0; j < *input_size; ++j) {
            fscanf(file, "%f;", &(*inputs)[i * (*input_size) + j]);
        }
        fscanf(file, "%d", &(*labels)[i]); // Quality score as label
    }

    fclose(file);
    return 0;
}

// Function to load the Breast Cancer Wisconsin dataset
int load_breast_cancer_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size) {
    const char *filename = "data/wdbc.data";
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open Breast Cancer dataset file");
        return -1;
    }

    *input_size = 30; // Number of features in the dataset
    *output_size = 2; // Malignant or Benign

    // Count the number of samples
    *num_samples = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        (*num_samples)++;
    }
    rewind(file);

    *inputs = (float *)malloc((*num_samples) * (*input_size) * sizeof(float));
    *labels = (int *)malloc((*num_samples) * sizeof(int));

    if (!(*inputs) || !(*labels)) {
        fprintf(stderr, "Memory allocation failed for inputs or labels.\n");
        fclose(file);
        return -1;
    }

    // Read the data
    for (int i = 0; i < *num_samples; ++i) {
        int id;
        char diagnosis;
        fscanf(file, "%d,%c,", &id, &diagnosis);
        (*labels)[i] = (diagnosis == 'M') ? 1 : 0; // Malignant:1, Benign:0

        for (int j = 0; j < *input_size; ++j) {
            fscanf(file, "%f,", &(*inputs)[i * (*input_size) + j]);
        }
    }

    fclose(file);
    return 0;
}
