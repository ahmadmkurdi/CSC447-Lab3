#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to initialize a matrix with random values
void initializeMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX; // Random values between 0 and 1
    }
}

void matrixMultiplyBasic(const float* A, const float* B, float* C, int M, int K, int N) {
    #pragma acc parallel loop collapse(2) copyin(A[0:M*K], B[0:K*N]) copyout(C[0:M*N])
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    int M = 256; // Rows of A and C
    int K = 256; // Columns of A and rows of B
    int N = 256; // Columns of B and C

    // Allocate memory for matrices
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices
    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);

    // Start the timer
    clock_t start = clock();

    // Perform basic matrix multiplication
    matrixMultiplyBasic(A, B, C, M, K, N);

    // Stop the timer
    clock_t stop = clock();

    // Calculate elapsed time in milliseconds
    double elapsedTime = ((double)(stop - start)) / (CLOCKS_PER_SEC / 1000);

    // Output the elapsed time
    printf("Basic matrix multiplication with OpenACC time: %.3f milliseconds\n", elapsedTime);

    // Clean up memory
    free(A);
    free(B);
    free(C);

    return 0;
}
