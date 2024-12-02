#include "kernel.h"
#include "helper_maca.h"
#include <mc_runtime.h>

#define BLOCK_SIZE 64 // 线程块大小，与 wave size 一致
#define TILE_SIZE 64  // Tile 大小，与 BLOCK_SIZE 一致
#define ILP 8         // 指令级并行度

/*
 * 优化后的 body_force_kernel，使用波级同步和增加 ILP
 */
__global__ void body_force_kernel(Body *p, float dt, int n) {
  // 计算线程索引
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  // 定义共享内存，每个 wave 使用自己的共享内存区域
  extern __shared__ Body shared_bodies[]; // 大小为 BLOCK_SIZE * TILE_SIZE
  Body *shared_bodies_wave = &shared_bodies[0]; // 每个 wave 的起始位置

  // 预取 Body i 的位置到寄存器
  float pos_ix = p[i].x;
  float pos_iy = p[i].y;
  float pos_iz = p[i].z;

  // 初始化力的累加器，使用 ILP 个累加器
  float Fx[ILP] = {0.0f};
  float Fy[ILP] = {0.0f};
  float Fz[ILP] = {0.0f};

  // 计算需要处理的 tile 数量
  int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

  for (int tile = 0; tile < numTiles; tile++) {
    int j = tile * TILE_SIZE + threadIdx.x;

    // 加载当前 tile 的 Bodies 到共享内存
    if (j < n) {
      shared_bodies_wave[threadIdx.x] = p[j];
    } else {
      // 边界条件处理，避免越界
      shared_bodies_wave[threadIdx.x].x = 0.0f;
      shared_bodies_wave[threadIdx.x].y = 0.0f;
      shared_bodies_wave[threadIdx.x].z = 0.0f;
    }

    // 波级同步
    __syncwave();

    // 遍历共享内存中的 Bodies，使用 ILP
    for (int k = 0; k < TILE_SIZE; k += ILP) {
#pragma unroll
      for (int l = 0; l < ILP; l++) {
        int idx = k + l;
        int j_index = tile * TILE_SIZE + idx;
        if (j_index >= n || j_index == i)
          continue;

        float dx = shared_bodies_wave[idx].x - pos_ix;
        float dy = shared_bodies_wave[idx].y - pos_iy;
        float dz = shared_bodies_wave[idx].z - pos_iz;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        // 累加力
        Fx[l] += dx * invDist3;
        Fy[l] += dy * invDist3;
        Fz[l] += dz * invDist3;
      }
    }

    // 波级同步
    __syncwave();
  }

  // 汇总累加器的值
  float sumFx = 0.0f;
  float sumFy = 0.0f;
  float sumFz = 0.0f;
#pragma unroll
  for (int l = 0; l < ILP; l++) {
    sumFx += Fx[l];
    sumFy += Fy[l];
    sumFz += Fz[l];
  }

  // 更新速度
  p[i].vx += dt * sumFx;
  p[i].vy += dt * sumFy;
  p[i].vz += dt * sumFz;
}

/*
 * 更新 Body 位置
 * 该操作相对简单，无需特殊优化
 */
__global__ void integrate_position_kernel(Body *p, float dt, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  p[i].x += p[i].vx * dt;
  p[i].y += p[i].vy * dt;
  p[i].z += p[i].vz * dt;
}

/*
 * 调用计算力的内核，并管理内存拷贝
 */
void call_body_force_kernel(Body *p, float dt, int n) {
  size_t threadsPerBlock = BLOCK_SIZE;
  size_t numberOfBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  size_t sharedMemSize = TILE_SIZE * sizeof(Body);

  // 调用优化后的 body_force_kernel
  body_force_kernel<<<numberOfBlocks, threadsPerBlock, sharedMemSize>>>(p, dt, n);

  // 错误检查和同步
  checkMacaErrors(mcDeviceSynchronize());
}

/*
 * 调用更新位置的内核
 */
void call_integrate_position_kernel(Body *p, float dt, int n) {
  size_t threadsPerBlock = BLOCK_SIZE;
  size_t numberOfBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  // 调用 integrate_position_kernel
  integrate_position_kernel<<<numberOfBlocks, threadsPerBlock>>>(p, dt, n);

  // 错误检查和同步
  checkMacaErrors(mcDeviceSynchronize());
}