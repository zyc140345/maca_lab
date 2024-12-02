#include <mc_runtime.h>
#include <mcblas.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "check.h"
#include "helper/input.h"
#include "helper/output.h"
#include "kernel.h"
#include "timer.h"

// cal offset from row col and ld , in row-major matrix, ld is the width of the
// matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

void randomize_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed;  // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i];  // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N) fs << ";\n";
    }
  }
  fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

void runMcblasFP32(mcblasHandle_t handle, int M, int N, int K, float alpha,
                   float beta, float *A, float *B, float *C) {
  // mcBLAS uses column-major order. So we change the order of our row-major A &
  // B, since (B^T*A^T)^T = (A*B)
  // This runs mcBLAS in full fp32 mode
  mcblasGemmEx(handle, MCBLAS_OP_N, MCBLAS_OP_N, N, M, K, &alpha, B, MACA_R_32F,
               N, A, MACA_R_32F, K, &beta, C, MACA_R_32F, N, MCBLAS_COMPUTE_32F,
               MCBLAS_GEMM_DEFAULT_TENSOR_OP);
}

const std::string errLogFile = "matrixValidationFailure.txt";

int run(const int argc, const char **argv,
        const char *input_file_name = nullptr,
        const char *output_file_name = nullptr) {
  const bool oj_mode = argc > 1;  // read input data from std::cin, write
                                  // output data to file

  int m, n, k, matrix_size;
  /* matrix_size = 2048; */
  matrix_size = 1024;
  m = n = k = matrix_size;

  float alpha = 0.5, beta = 3.0;  // GEMM input parameters, C=α*AB+β*C

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr;  // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr;  // device matrices

  A = (float *)malloc(sizeof(float) * matrix_size * matrix_size);
  B = (float *)malloc(sizeof(float) * matrix_size * matrix_size);
  C = (float *)malloc(sizeof(float) * matrix_size * matrix_size);
  C_ref = (float *)malloc(sizeof(float) * matrix_size * matrix_size);

  if (oj_mode) {
    ReadInputData(matrix_size * matrix_size, A, B, C);
  } else {
    randomize_matrix(A, matrix_size * matrix_size);
    randomize_matrix(B, matrix_size * matrix_size);
    randomize_matrix(C, matrix_size * matrix_size);

#ifdef MAKE_TEST_CASE_DATA
    WriteInputDataToFile(input_file_name, matrix_size * matrix_size, A, B, C);
#endif
  }
  printf("Inputs: matrix_size(m=n=k) is %d; alpha is %f; beta is %f.\n",
         matrix_size, alpha, beta);

  gpu_errchk(mcMalloc((void **)&dA, sizeof(float) * matrix_size * matrix_size));
  gpu_errchk(mcMalloc((void **)&dB, sizeof(float) * matrix_size * matrix_size));
  gpu_errchk(mcMalloc((void **)&dC, sizeof(float) * matrix_size * matrix_size));
  gpu_errchk(
      mcMalloc((void **)&dC_ref, sizeof(float) * matrix_size * matrix_size));

  gpu_errchk(mcMemcpy(dA, A, sizeof(float) * matrix_size * matrix_size,
                      mcMemcpyHostToDevice));
  gpu_errchk(mcMemcpy(dB, B, sizeof(float) * matrix_size * matrix_size,
                      mcMemcpyHostToDevice));
  gpu_errchk(mcMemcpy(dC, C, sizeof(float) * matrix_size * matrix_size,
                      mcMemcpyHostToDevice));
  gpu_errchk(mcMemcpy(dC_ref, C, sizeof(float) * matrix_size * matrix_size,
                      mcMemcpyHostToDevice));

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

  // use mcblas to get dC_ref
  mcblasHandle_t blas_handle;
  mcblasCreate(&blas_handle);
  runMcblasFP32(blas_handle, m, n, k, alpha, beta, dA, dB, dC_ref);
  gpu_errchk(
      mcMemcpy(C_ref, dC_ref, sizeof(float) * m * n, mcMemcpyDeviceToHost));

#ifdef MAKE_TEST_CASE_DATA
  WriteOutputDataToFile(output_file_name, C_ref, m * n);

#else

  // Initialize timers
  float sgemm_ms = -1;
  float mcblas_ms = -1;
  double totalTime = 0.0;
  double minTime = std::numeric_limits<double>::max();

  const int nIters = 10;
  for (int run = 0; run < nIters; run++) {
    START_TIMER();
    call_sgemm_kernel(m, n, k, alpha, beta, dA, dB, dC);
    gpu_errchk(mcDeviceSynchronize());
    STOP_RECORD_TIMER(sgemm_ms);

    gpu_errchk(mcGetLastError());  // Check for async errors during kernel run
    if (run == 0) {
      // verify_matrix with the first run results, where dC is initialzed the
      // same as dC_ref.
      gpu_errchk(mcMemcpy(C, dC, sizeof(float) * m * n, mcMemcpyDeviceToHost));
      if (!verify_matrix(C_ref, C, m * n)) {
        printf("Failed to pass the correctness verification against mcBLAS\n");
        if (m <= 4096) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m, n, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(-1);
      }
      if (oj_mode) {
        WriteOutputDataToFile(argv[1], C, m * n);
      }
    }

    totalTime += sgemm_ms;
    if (minTime > sgemm_ms) {
      minTime = sgemm_ms;
    }
    printf("sgemm_kernel iter=%d: totalTime is %f; minTime is %f\n", run,
           totalTime, minTime);
  }
  float avgTime = totalTime / nIters;
  printf("sgemm_kernel nIters = %d: avgTime is %f; minTime is %f\n", nIters,
         avgTime, minTime);
#endif

  mcblasDestroy(blas_handle);

  // Free Memory
  free(A);
  free(B);
  free(C);
  free(C_ref);
  mcFree(dA);
  mcFree(dB);
  mcFree(dC);
  mcFree(dC_ref);

  return 0;
}

int main(const int argc, const char **argv) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);  // seed once

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
