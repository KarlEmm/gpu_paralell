#include <iostream>

#define THREAD_PER_BLOCK 1024.0

#define CUDAMALLOC_ERROR(_err)      \
do {                                \
    if (_err != cudaSuccess) {      \
        printf("%s in %s at line %d\n", cudaGetErrorString(_err),__FILE__,__LINE__); \
        exit(EXIT_FAILURE);         \
        }                           \
    }while(0)

void init_array(float *array, long size, float value) {
    for (size_t i = 0; i < size; ++i) {
        array[i] = value;
    }
}

__global__
void vecAddKernel(float *A, float *B, float *C, long n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float *h_A, float *h_B, float *h_C, long n) {
    float *d_A, *d_B, *d_C;

    long size = n * sizeof (float);

    cudaError_t err = cudaMalloc((void **) &d_A, size);
    CUDAMALLOC_ERROR(err);
    err = cudaMalloc((void **) &d_B, size);
    CUDAMALLOC_ERROR(err);
    err = cudaMalloc((void **) &d_C, size);
    CUDAMALLOC_ERROR(err);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    dim3 dimGrid{(uint) ceil(n/THREAD_PER_BLOCK),1,1};
    dim3 dimBlock = {(uint) THREAD_PER_BLOCK,1,1};

    // KERNEL LAUNCH
    vecAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    clock_t begin = clock();
    long nbElements = 10e7;
    long size = nbElements * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *) malloc(size);
    h_B = (float *) malloc(size);
    h_C = (float *) malloc(size);

    init_array(h_A, nbElements, 1);
    init_array(h_B, nbElements, 2);
    init_array(h_C, nbElements, 0);

    vecAdd(h_A, h_B, h_C, nbElements);

    for (int i = 10e6; i < 10e7; ++i) {
        std::cout << h_C[i] << std::endl;
    }

    free(h_A); free(h_B); free(h_C);

    clock_t end = clock();

    double elapsed_time = (double)(end - begin) / CLOCKS_PER_SEC;
    std::cout << elapsed_time << "s" << std::endl;


    return 0;
}
