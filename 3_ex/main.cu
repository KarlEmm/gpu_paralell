#include <iostream>

#define MATRIX_SIZE 4

void fillMatrix(float matrix[MATRIX_SIZE*MATRIX_SIZE], float value)
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
    //dim3 dimBlock(sqrt(dev_prop.maxthreadsperblock),sqrt(dev_prop.maxthreadsperblock),5);
    dim3 dimBlock(4,4,4);
    dim3 dimGrid(1,1,1);

    float h_A[MATRIX_SIZE*MATRIX_SIZE];
    float h_B[MATRIX_SIZE*MATRIX_SIZE];
    float h_C[MATRIX_SIZE*MATRIX_SIZE];

    fillMatrix(h_A, 1);
    fillMatrix(h_B, 2);
    fillMatrix(h_C, 0);

    float *d_A;
    cudaMalloc((void**)&d_A, sizeof (float) * MATRIX_SIZE * MATRIX_SIZE);
    float *d_B;
    cudaMalloc((void**)&d_B, sizeof (float) * MATRIX_SIZE * MATRIX_SIZE);
    float *d_C;
    cudaMalloc((void**)&d_C, sizeof (float) * MATRIX_SIZE * MATRIX_SIZE);

    cudaMemcpy(d_A, h_A, MATRIX_SIZE*MATRIX_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_SIZE*MATRIX_SIZE, cudaMemcpyHostToDevice);

    matrixAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, MATRIX_SIZE);

    cudaMemcpy(h_C, d_C, MATRIX_SIZE*MATRIX_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}


int main() {
    addMatrices();
    return 0;
}
