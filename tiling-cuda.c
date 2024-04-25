%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Tile size for tiling
#define TILE_SIZE 16

// Kernel for matrix multiplication with tiling
__global__ void matrixMultiplyTiled(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (row < M && (tile * TILE_SIZE + threadIdx.x) < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (tile * TILE_SIZE + threadIdx.y) < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to initialize a matrix with random values
void initializeMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX; // Random values between 0 and 1
    }
}

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
    dim3 blockSize(TILE_SIZE, TILE_SIZE); // 16x16 threads per block
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel with tiling
    matrixMultiplyTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

    // Record stop event and synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Wait for the kernel to complete

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Output the elapsed time
    printf("CUDA Matrix multiplication with tiling time: %.3f ms\n", elapsedTime);

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