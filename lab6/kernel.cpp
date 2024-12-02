#include "kernel.h"
#include "helper_maca.h"
#include <math.h>


#define BLOCK_SIZE 16 // Block size
#define WARP_SIZE 64  // Warp size on MACA platform

__global__ void sgemm_kernel(int M, int N, int K, float alpha, float beta,
                             const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ C) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Compute the row and column indices of the element
  int Row = by * BLOCK_SIZE + ty;
  int Col = bx * BLOCK_SIZE + tx;

  // Shared memory for As and Bs
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  float Cvalue = 0.0f;

  // Loop over the tiles of K
  for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
    // Load elements of A into shared memory
    if (Row < M && (t * BLOCK_SIZE + tx) < K) {
      As[ty][tx] = A[Row * K + t * BLOCK_SIZE + tx];
    } else {
      As[ty][tx] = 0.0f;
    }

    // Load elements of B into shared memory
    if ((t * BLOCK_SIZE + ty) < K && Col < N) {
      Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + Col];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Multiply the two tiles together
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Cvalue += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  // Write the result to C
  if (Row < M && Col < N) {
    C[Row * N + Col] = alpha * Cvalue + beta * C[Row * N + Col];
  }
}

void call_sgemm_kernel(int M, int N, int K, float alpha, float beta, float *A, float *B, float *C) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  sgemm_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, beta, A, B, C);

  // Synchronize and check for errors
  checkMacaErrors(mcDeviceSynchronize());
}