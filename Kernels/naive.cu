#include <cuda_runtime.h>
#include <math.h>
#include<stdio.h>
#include<stdlib.h>   



__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      // Compute dot product
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

// Function to calculate the ceiling division
int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }

int main() {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  float alpha = 1.0;
  float beta = 0.0;

  printf("Allocating memory to Host: CPU \n");  // Allocate memory on the host
  float *A, *B, *C;
  cudaMallocHost((void **)&A, M * K * sizeof(float));
  cudaMallocHost((void **)&B, K * N * sizeof(float));
  cudaMallocHost((void **)&C, M * N * sizeof(float));

  // Initialize A and B with random values
  for (int i = 0; i < M * K; ++i) {
    A[i] = (float)rand() / RAND_MAX;
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = (float)rand() / RAND_MAX;
  }

  // Create as many blocks as necessary to map all of C
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  // 32 * 32 = 1024 thread per block
  dim3 blockDim(32, 32, 1);

  // Allocate memory on the device
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, M * K * sizeof(float));
  cudaMalloc((void **)&d_B, K * N * sizeof(float));
  cudaMalloc((void **)&d_C, M * N * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

  printf("Copy data from host to device done.\n");

  // launch the asynchronous execution of the kernel on the device
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  cudaDeviceSynchronize();

  // Check for any errors
  printf("Launch kernel done.\n");

  // Copy data from device to host
  cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // // Print result on the host
  // printf("Copy data from device to host done.\n");
  // for (int i = 0; i < M * N; ++i) {
  //   printf("Result[C[%d]]: %f\n", i, C[i]);
  // }

  // Deallocate memory on the device
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Deallocate memory on the host
  cudaFreeHost(A);
  cudaFreeHost(B);
  cudaFreeHost(C);

}
