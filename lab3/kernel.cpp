#include "kernel.h"
#include <mc_runtime.h>

// 定义瓦片大小和指令级并行度
#define TILE_DIM 64
#define ILP 8 // 指令级并行度

// 优化后的矩阵转置内核，利用波级同步和 ILP

__global__ void macaTransposeKernel(const float *input, float *output, int n) {
  // 计算线程的全局索引
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y0 = blockIdx.y * TILE_DIM;

  // 定义共享内存，带填充以避免银行冲突
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  // 加载数据到共享内存
  for (int i = 0; i < TILE_DIM; i += ILP) {
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      int y = y0 + i + j;
      if (x < n && y < n) {
        tile[i + j][threadIdx.x] = input[y * n + x];
      }
    }
  }

  // 使用波级同步
  __syncwave();

  // 计算转置后的索引
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y0 = blockIdx.x * TILE_DIM;

  // 将转置后的数据写回全局内存
  for (int i = 0; i < TILE_DIM; i += ILP) {
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      int y = y0 + i + j;
      if (x < n && y < n) {
        output[y * n + x] = tile[threadIdx.x][i + j];
      }
    }
  }
}

void macaTranspose(const float *d_input, float *d_output, int n) {
  // 定义线程块和网格大小
  dim3 blockSize(TILE_DIM, ILP); // (64, 8)
  dim3 gridSize((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);

  // 启动优化后的转置内核
  macaTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);

  // 确保内核执行完毕
  mcDeviceSynchronize();
}