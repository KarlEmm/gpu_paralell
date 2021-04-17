#include <iostream>

#define CUDAMALLOC_ERROR(_err)          \
do {                                \
    if (_err != cudaSuccess) {      \
        printf("%s in %s at line %d\n", cudaGetErrorString(_err),__FILE__,__LINE__); \
        exit(EXIT_FAILURE);         \
        }                           \
    }while(0)

void init_array(float *array, int size, float value) {
    for (size_t i = 0; i < size; ++i) {
        array[i] = value;
    }
}

void vecAdd(float *h_A, float *h_B, float *h_C, int n) {
    float *d_A, *d_B, *d_C;

    cudaError_t err = cudaMalloc((void **) &d_A, n);
    CUDAMALLOC_ERROR(err);
    err = cudaMalloc((void **) &d_B, n);
    CUDAMALLOC_ERROR(err);
    err = cudaMalloc((void **) &d_C, n);
    CUDAMALLOC_ERROR(err);

    cudaMemcpy(d_A, h_A, n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, n, cudaMemcpyHostToDevice);

    // KERNEL INVOCATION
    // -----------------

    cudaMemcpy(h_C, d_C, n, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int nbElements = 10e6;
    int size = nbElements * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *) malloc(size);
    h_B = (float *) malloc(size);
    h_C = (float *) malloc(size);

    init_array(h_A, nbElements, 1);
    init_array(h_B, nbElements, 2);
    init_array(h_C, nbElements, 0);

    vecAdd(h_A, h_B, h_C, size);

    free(h_A); free(h_B); free(h_C);


    return 0;
}
