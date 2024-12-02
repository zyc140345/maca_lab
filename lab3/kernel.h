/* 
 * MACA Kernels
 * Dezheng Yan, 2024 
 */

#ifndef MACA_TRANSPOSE_MCH
#define MACA_TRANSPOSE_MCH

#include "common.h"

void macaTranspose(
    const float *d_input,
    float *d_output,
    int n);

#endif
