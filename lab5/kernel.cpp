#include "kernel.h"
#include "helper_maca.h"

#define WARP_SIZE 64
#define WARPS_PER_BLOCK 8 // Adjusted for higher occupancy

__device__ __forceinline__ float waveReduceSum(float sum) {
  // Perform warp-level reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffffffffff, sum, offset);
  }
  return sum;
}

__global__ void sgemv_kernel(const float *__restrict__ A,
                             const float *__restrict__ x, float *__restrict__ y,
                             const int M, const int N) {
  // Thread index
  int tx = threadIdx.x;
  int warpId = tx / WARP_SIZE; // Warp index within the block
  int laneId = tx % WARP_SIZE; // Thread index within the warp

  // Each warp processes one row
  int row = blockIdx.x * WARPS_PER_BLOCK + warpId;

  if (row < M) {
    float sum = 0.0f;

    // Vectorized memory access using float4
    int vectorSize = 4;
    int totalThreads = WARP_SIZE * vectorSize;

    // Ensure alignment
    const float *A_row = A + row * N;
    const float *x_ptr = x;

    // Loop over columns with vectorization
    for (int col = laneId * vectorSize; col <= N - vectorSize;
         col += totalThreads) {
      float4 a_vec = reinterpret_cast<const float4 *>(A_row + col)[0];
      float4 x_vec = reinterpret_cast<const float4 *>(x_ptr + col)[0];

      sum += a_vec.x * x_vec.x + a_vec.y * x_vec.y + a_vec.z * x_vec.z + a_vec.w * x_vec.w;
    }

    // Handle remaining elements
    for (int col = (N / vectorSize) * vectorSize + laneId; col < N; col += WARP_SIZE) {
      float a_val = A_row[col];
      float x_val = __ldg(&x_ptr[col]);
      sum += a_val * x_val;
    }

    // Perform warp-level reduction
    sum = waveReduceSum(sum);

    // Write the result from the first lane of each warp
    if (laneId == 0) {
      y[row] = sum;
    }
  }
}

void call_sgemv_kernel(float *__restrict__ d_A, float *__restrict__ d_x,
                       float *__restrict__ d_y, const int M, const int N) {
  const int warps_per_block = WARPS_PER_BLOCK;
  int total_threads = WARP_SIZE * warps_per_block;

  dim3 dimBlock(total_threads);
  dim3 dimGrid((M + warps_per_block - 1) / warps_per_block);

  sgemv_kernel<<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, N);

  // Synchronize and check for errors
  checkMacaErrors(mcDeviceSynchronize());
}