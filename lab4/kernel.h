/* 
 * MACA image filtering
 * Dezheng Yan
 */

#ifndef BLUR_DEVICE_CUH
#define BLUR_DEVICE_CUH

#include "common.h"

void call_image_filtering_kernel(float *out, float const *in, int nx, int ny);

__global__ void image_mean_filtering_kernel(int width, int height, float const *in, float *out);

#endif
