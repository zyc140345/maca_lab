/* 
 * MXMACA Common Header Files
 * Dezheng Yan, 2024
 */

/*
 * 
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <cassert>

#include "mc_runtime.h"

#ifndef MACA_HEADER_MCH_
#define MACA_HEADER_MCH_

#ifdef __MACA_ARCH__
#define MACA_CALLABLE __host__ __device__
#else
// Host function attributes
#define MACA_CALLABLE
#endif // __MACA_ARCH__

#endif // MACA_HEADER_MCH_

/*
 * NOTE: You can use this macro to easily check mxmaca error codes 
 * and get more information. 
 */
#define gpu_errchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(mcError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != mcSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            mcGetErrorString(code), file, line);
        exit(code);
    }
}

