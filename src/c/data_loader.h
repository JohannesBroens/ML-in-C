// data_loader.h

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

int load_generated_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size);
int load_iris_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size);
int load_wine_quality_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size, const char *filename);
int load_breast_cancer_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size);

#ifdef __cplusplus
}
#endif

#endif // DATA_LOADER_H
