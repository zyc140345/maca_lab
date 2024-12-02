/*
 * SGEMV Kernel Header
 * Optimized Based on Initial Code and Teacher's Suggestions
 * Dezheng Yan
 */
#ifndef KERNEL_H
#define KERNEL_H

#include "common.h"
#include <mc_runtime.h>


__device__ __forceinline__ float waveReduceSum(float sum);

__global__ void sgemv_kernel(const float *__restrict__ A,
                             const float *__restrict__ x, float *__restrict__ y,
                             const int M, const int N);

void call_sgemv_kernel(float *__restrict__ d_A, float *__restrict__ d_x,
                       float *__restrict__ d_y, const int M, const int N);

#endif // KERNEL_H