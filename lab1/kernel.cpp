#include "kernel.h"

__global__ void parallel_sum(int *sum, int const *arr, int n) {
  // 每个线程块处理 1024 个元素
  __shared__ int sdata[256]; // 根据线程块大小调整共享内存大小

  int tid = threadIdx.x;
  int i = blockIdx.x;

  // 计算全局索引
  int idx = i * 1024 + tid * 4;

  // 每个线程处理 4 个元素
  int local_sum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (idx + j < n) {
      local_sum += arr[idx + j];
    }
  }

  // 将局部和写入共享内存
  sdata[tid] = local_sum;
  __syncthreads();

  // 在共享内存中进行规约
  for (unsigned int stride = blockDim.x / 2; stride > 64; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  // 对最后一个 warp 进行展开
  if (tid < 64) {
    volatile int *vsmem = sdata;
    vsmem[tid] += vsmem[tid + 64];
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // 线程 0 写入结果
  if (tid == 0) {
    sum[i] = sdata[0];
  }
}

void call_parallel_sum_kernel(int *sum, int const *arr, int n) {
  size_t threadsPerBlock = 256; // 根据硬件优化
  size_t numberOfBlocks = (n + 1024 - 1) / 1024;
  parallel_sum<<<numberOfBlocks, threadsPerBlock>>>(sum, arr, n);
}
