/* 
 * MACA Kernels
 * Dezheng Yan
 */

#include <vector>
#include <mc_runtime.h>
#include "common.h"

__global__ void sgemm_kernel(int M, int N, int K, float alpha, float beta, const float *A, const float *B, float *C);

void call_sgemm_kernel(int M, int N, int K, float alpha, float beta, float *A, float *B, float *C) ;

