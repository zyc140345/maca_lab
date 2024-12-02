#include "kernel.h"
#include "helper_maca.h"

#define BLOCK_SIZE 16 // Adjust based on hardware capabilities

void call_image_filtering_kernel(float *out, float const *in, int nx, int ny) {
  // Define block and grid dimensions
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((nx + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (ny + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Launch the optimized kernel
  image_mean_filtering_kernel<<<gridDim, blockDim>>>(nx, ny, in, out);

  // Synchronize and check for errors
  checkMacaErrors(mcDeviceSynchronize());
}

__global__ void image_mean_filtering_kernel(int width, int height, float const *in, float *out) {
  // Calculate global indices
  int i = blockIdx.x * blockDim.x + threadIdx.x; // Column index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index

  int pos = j * width + i; // Position in the 1D array

  // Define shared memory with halo regions
  __shared__ double shared_mem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

  // Calculate shared memory indices with halo offset
  int s_i = threadIdx.x + 1;
  int s_j = threadIdx.y + 1;

  // Initialize shared memory to zero
  shared_mem[s_j][s_i] = 0.0;

  // Load the main data into shared memory
  if (i < width && j < height) {
    shared_mem[s_j][s_i] = static_cast<double>(in[pos]);
  }

  // Load halo regions
  // Left and right halos
  if (threadIdx.x == 0 && i > 0) {
    shared_mem[s_j][s_i - 1] = static_cast<double>(in[pos - 1]);
  }
  if (threadIdx.x == blockDim.x - 1 && i < width - 1) {
    shared_mem[s_j][s_i + 1] = static_cast<double>(in[pos + 1]);
  }
  // Top and bottom halos
  if (threadIdx.y == 0 && j > 0) {
    shared_mem[s_j - 1][s_i] = static_cast<double>(in[pos - width]);
  }
  if (threadIdx.y == blockDim.y - 1 && j < height - 1) {
    shared_mem[s_j + 1][s_i] = static_cast<double>(in[pos + width]);
  }
  // Corner halos
  if (threadIdx.x == 0 && threadIdx.y == 0 && i > 0 && j > 0) {
    shared_mem[s_j - 1][s_i - 1] = static_cast<double>(in[pos - width - 1]);
  }
  if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && i < width - 1 && j > 0) {
    shared_mem[s_j - 1][s_i + 1] = static_cast<double>(in[pos - width + 1]);
  }
  if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && i > 0 && j < height - 1) {
    shared_mem[s_j + 1][s_i - 1] = static_cast<double>(in[pos + width - 1]);
  }
  if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && i < width - 1 && j < height - 1) {
    shared_mem[s_j + 1][s_i + 1] = static_cast<double>(in[pos + width + 1]);
  }

  // Synchronize threads to ensure all data is loaded
  __syncthreads();

  if (i < width && j < height) {
    if (i > 0 && i < width - 1 && j > 0 && j < height - 1) {
      // Compute the mean of the 3x3 neighborhood
      double temp = 0.0;
      temp += shared_mem[s_j][s_i];         // Center
      temp += shared_mem[s_j][s_i + 1];     // Right
      temp += shared_mem[s_j][s_i - 1];     // Left
      temp += shared_mem[s_j - 1][s_i - 1]; // Top-left
      temp += shared_mem[s_j - 1][s_i];     // Top
      temp += shared_mem[s_j - 1][s_i + 1]; // Top-right
      temp += shared_mem[s_j + 1][s_i - 1]; // Bottom-left
      temp += shared_mem[s_j + 1][s_i];     // Bottom
      temp += shared_mem[s_j + 1][s_i + 1]; // Bottom-right

      temp = temp / 9.0;

      // Write the result to the output array
      out[pos] = static_cast<float>(temp);
    } else {
      // Edge pixels remain unchanged
      out[pos] = in[pos];
    }
  }
}