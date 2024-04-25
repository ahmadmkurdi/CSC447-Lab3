%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel for basic matrix multiplication
__global__ void matrixMultiplyBasic(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Function to initialize a matrix with random values
void initializeMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX; // Random values between 0 and 1
    }
}

// Multiplication of matrices A (MxK) and B (KxN) to produce matrix C (MxN)
int main() {
    int M = 256; // Number of rows in A and C
    int K = 256; // Number of columns in A and rows in B
    int N = 256; // Number of columns in B and C

    // Allocate host memory
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Configure kernel launch parameters
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMultiplyBasic<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

    // Record stop event and synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Wait for the kernel to complete

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Output the elapsed time
    printf("CUDA Matrix multiplication time: %.3f ms\n", elapsedTime);

    // Clean up device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Clean up host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}