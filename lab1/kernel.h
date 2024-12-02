#pragma once

#include <mc_runtime.h>
void call_parallel_sum_kernel(int *sum, int const *arr, int n);
__global__ void parallel_sum(int *sum, int const *arr, int n);