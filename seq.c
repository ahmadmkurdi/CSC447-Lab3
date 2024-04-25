#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to initialize a matrix with random values
void initializeMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX; // Random values between 0 and 1
    }
}

// Function for sequential matrix multiplication
void matrixMultiplySequential(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    int M = 256; // Number of rows in A and C
    int K = 256; // Number of columns in A and rows in B
    int N = 256; // Number of columns in B and C

    // Allocate memory for matrices
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices
    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);

    // Start the timer
    clock_t start = clock();

    // Perform sequential matrix multiplication
    matrixMultiplySequential(A, B, C, M, K, N);

    // Stop the timer
    clock_t stop = clock();

    // Calculate elapsed time in seconds
    double elapsedTime = ((double)(stop - start)) / CLOCKS_PER_SEC;

    // Convert to milliseconds
    double elapsedTimeMs = elapsedTime * 1000;

    // Output the elapsed time in milliseconds
    printf("Sequential matrix multiplication time: %.3f milliseconds\n", elapsedTimeMs);

    // Clean up memory
    free(A);
    free(B);
    free(C);

    return 0;
}
