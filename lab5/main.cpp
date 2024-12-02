#include <mc_runtime.h>
#include <mcblas.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "check.h"
#include "helper/input.h"
#include "helper/output.h"
#include "kernel.h"
#include "timer.h"

/* #define MAKE_TEST_CASE_DATA */

// cal offset from row col and ld , in row-major matrix, ld is the width of the
// matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkmcErrors(func)                                               \
  {                                                                       \
    mcError_t e = (func);                                                 \
    if (e != mcSuccess)                                                   \
      printf("%s %dMACA: %s\n", __FILE__, __LINE__, mcGetErrorString(e)); \
  }

void randomize_matrix(float* mat, int N) {
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

int run(const int argc, const char** argv,
        const char* input_file_name = nullptr,
        const char* output_file_name = nullptr) {
  const bool oj_mode = argc > 1;  // read input data from std::cin, write
                                  // output data to file

  size_t M = 8192;
  size_t N = 256;
  size_t bytes_A = sizeof(float) * M * N;
  size_t bytes_x = sizeof(float) * N;
  size_t bytes_y = sizeof(float) * M;
  float* h_A = (float*)malloc(bytes_A);
  float* h_x = (float*)malloc(bytes_x);
  float* h_y = (float*)malloc(bytes_y);
  float* h_y1 = (float*)malloc(bytes_y);
  float* d_A;
  float* d_x;
  float* d_y;
  checkmcErrors(mcMalloc(&d_A, bytes_A));
  checkmcErrors(mcMalloc(&d_x, bytes_x));
  checkmcErrors(mcMalloc(&d_y, bytes_y));
  if (oj_mode) {
    ReadInputData(M, N, h_A, h_x, h_y);
  } else {
    randomize_matrix(h_A, M * N);
    randomize_matrix(h_x, N);
    randomize_matrix(h_y, M);

#ifdef MAKE_TEST_CASE_DATA
    WriteInputDataToFile(input_file_name, M, N, h_A, h_x, h_y);
#endif
  }
  memcpy(h_y1, h_y, M * sizeof(float));

#ifdef MAKE_TEST_CASE_DATA
  const int nIters = 1;
#else
  const int nIters = 1000;
#endif

  checkmcErrors(mcMemcpy(d_A, h_A, bytes_A, mcMemcpyHostToDevice));
  checkmcErrors(mcMemcpy(d_x, h_x, bytes_x, mcMemcpyHostToDevice));
  checkmcErrors(mcMemcpy(d_y, h_y, bytes_y, mcMemcpyHostToDevice));

  mcEvent_t start;
  mcEvent_t stop;

#define START_TIMER()                  \
  {                                    \
    gpu_errchk(mcEventCreate(&start)); \
    gpu_errchk(mcEventCreate(&stop));  \
    gpu_errchk(mcEventRecord(start));  \
  }

#define STOP_RECORD_TIMER(name)                         \
  {                                                     \
    gpu_errchk(mcEventRecord(stop));                    \
    gpu_errchk(mcEventSynchronize(stop));               \
    gpu_errchk(mcEventElapsedTime(&name, start, stop)); \
    gpu_errchk(mcEventDestroy(start));                  \
    gpu_errchk(mcEventDestroy(stop));                   \
  }

  // Initialize timers
  float sgemv_ms = -1;
  float mcblas_ms = -1;
  double totalTime = 0.0;
  double minTime = std::numeric_limits<double>::max();

  for (int run = 0; run < nIters; run++) {
    START_TIMER();
    call_sgemv_kernel(d_A, d_x, d_y, M, N);
    STOP_RECORD_TIMER(sgemv_ms);
    totalTime += sgemv_ms;
    if (minTime > sgemv_ms) {
      minTime = sgemv_ms;
    }
    // printf("sgemv_kernel iter=%d: totalTime is %f; minTime is %f\n", run,
    // totalTime, minTime);
  }
  float avgTime = totalTime / nIters;
  printf("sgemv_kernel nIters = %d: avgTime is %f; minTime is %f\n", nIters,
         avgTime, minTime);
  checkmcErrors(mcMemcpy(h_y, d_y, bytes_y, mcMemcpyDeviceToHost));

  if (oj_mode) {
    WriteOutputDataToFile(argv[1], h_y, M);
  }

  // mcblas
  mcblasHandle_t blas_handle;
  mcblasCreate(&blas_handle);
  float alpha = 1.0;
  float beta = 0;
  // checkmcErrors(mcMemcpy( d_A, h_A, bytes_A, mcMemcpyHostToDevice));
  // checkmcErrors(mcMemcpy( d_x, h_x, bytes_x, mcMemcpyHostToDevice));
  checkmcErrors(mcMemcpy(d_y, h_y1, bytes_y, mcMemcpyHostToDevice));
  totalTime = 0.0;
  minTime = std::numeric_limits<double>::max();
  for (int run = 0; run < nIters; run++) {
    START_TIMER();
    mcblasSgemv(blas_handle, MCBLAS_OP_T, N, M, &alpha, d_A, N, d_x, 1, &beta,
                d_y, 1);
    STOP_RECORD_TIMER(mcblas_ms);
    totalTime += mcblas_ms;
    if (minTime > mcblas_ms) {
      minTime = mcblas_ms;
    }
    // printf("mcblas iter=%d: totalTime is %f; minTime is %f\n", run,
    // totalTime, minTime);
  }
  avgTime = totalTime / nIters;
  printf("mcblas nIters = %d: avgTime is %f; minTime is %f\n", nIters, avgTime,
         minTime);
  checkmcErrors(mcMemcpy(h_y1, d_y, bytes_y, mcMemcpyDeviceToHost));
  mcblasDestroy(blas_handle);

  bool correct = MetaXOJ::VerifyLab5Result(h_y, h_y1, M);
  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

#ifdef MAKE_TEST_CASE_DATA
  WriteOutputDataToFile(output_file_name, h_y1, M);
#endif

  // Free Memory
  mcFree(d_A);
  mcFree(d_x);
  mcFree(d_y);

  free(h_A);
  free(h_x);
  free(h_y);
  free(h_y1);

  return 0;
}

int main(const int argc, const char** argv) {
  std::srand(std::time(nullptr));  // seed once

#ifdef MAKE_TEST_CASE_DATA
  for (int i = 1; i <= 10; ++i) {
    std::string intput_file_name;
    intput_file_name += std::to_string(i);
    intput_file_name += "-input.dat";

    std::string output_file_name;
    output_file_name += std::to_string(i);
    output_file_name += "-output.dat";
    run(0, nullptr, intput_file_name.c_str(), output_file_name.c_str());
  }

  return 0;

#else
  return run(argc, argv);
#endif
}
