#include <iostream>

#define MATRIX_SIZE 4

#define CUDAMALLOC_ERROR(_err) \
do {                           \
    if (_err != cudaSuccess) { \
        printf("%s failed in file %s at line #%d\n", cudaGetErrorString(_err),__FILE__,__LINE__); \
        exit(EXIT_FAILURE);   \
        }                      \
    } while(0)

void fillMatrix(float *matrix, float value)
{
    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; ++i)
    {
        matrix[i] = value;
    }
}

__global__
void matrixAddKernel(float *A, float *B, float *C, size_t n)
{
    unsigned int Col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Row < n && Col < n)
    {
        // 1D coordinates
        int coord = Row * n + Col;
        C[coord] = A[coord] + B[coord];
    }
}

void addMatrices()
{
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);

    // For this machine, sqrt(1024) = 32. Therefore, we have a 32-square matrix per block.
    dim3 dimBlock(sqrt(dev_prop.maxThreadsPerBlock),sqrt(dev_prop.maxThreadsPerBlock),1);
    dim3 dimGrid(1,1,1);

    int matrixSize = MATRIX_SIZE * MATRIX_SIZE;
    int matrixByteSize = matrixSize * sizeof (float);

    float *h_A;
    float *h_B;
    float *h_C;

    h_A = (float *) malloc (matrixByteSize);
    h_B = (float *) malloc (matrixByteSize);
    h_C = (float *) malloc(matrixByteSize);

    fillMatrix(h_A, 1);
    fillMatrix(h_B, 2);
    fillMatrix(h_C, 0);

    float *d_A;
    cudaError_t err = cudaMalloc((void**)&d_A, matrixByteSize);
    CUDAMALLOC_ERROR(err);
    float *d_B;
    err = cudaMalloc((void**)&d_B, matrixByteSize);
    CUDAMALLOC_ERROR(err);
    float *d_C;
    err = cudaMalloc((void**)&d_C, matrixByteSize);
    CUDAMALLOC_ERROR(err);

    cudaMemcpy(d_A, h_A, matrixByteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixByteSize, cudaMemcpyHostToDevice);

    matrixAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, MATRIX_SIZE);

    cudaMemcpy(h_C, d_C, matrixByteSize, cudaMemcpyDeviceToHost);

    for (int i = 0 ; i < 16; ++i) {
        h_A[i] = 0;
    }

    cudaMemcpy(h_A, d_A, matrixByteSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 15; ++i) {
        std::cout << h_C[i] << std::endl;
    }


    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}


int main() {
    addMatrices();
    return 0;
}
