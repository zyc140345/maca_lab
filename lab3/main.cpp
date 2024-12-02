#include <mc_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "check.h"
#include "helper/input.h"
#include "helper/output.h"
#include "kernel.h"
#include "timer.h"

// #define DEBUG_WRITE_TO_FILE

/*
 * Fills fill with random numbers is [0, 1]. Size is number of elements to
 * assign.
 */
void randomFill(std::vector<std::vector<float>> &matrix, int size) {
  std::srand(std::time(nullptr));

  matrix.resize(size);
  for (int row = 0; row < size; ++row) {
    matrix[row].resize(size);
    for (int col = 0; col < size; ++col) {
      matrix[row][col] =
          static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      //   matrix[row][col] = row * size + col;  // debug
    }
  }
}

/* CPU transpose, takes an n x n matrix in input and writes to output. */
void cpuTranspose(const float *input, float *output, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      output[j + n * i] = input[i + n * j];
    }
  }
}

int main(const int argc, const char **argv) {
  const bool oj_mode = argc > 1;  // read input data from args, write output
                                  // data to file

  std::string kernel = "gpu";
  int n = 512;  // 512, 1024, 2048, 4096

  std::vector<std::vector<float>> matrix;
  if (oj_mode) {
    ReadMatrix(matrix);
    n = matrix[0].size();
  } else {
    randomFill(matrix, n);
  }

    if (!(n == 512 || n == 1024 || n == 2048 || n == 4096)) {
    fprintf(stderr,
            "Program only designed to run sizes 512, 1024, 2048, 4096\n");
  }
  assert(n % 64 == 0);

  assert(kernel == "all" || kernel == "cpu" || kernel == "gpu");

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
  float cpu_ms = -1;
  float gpu_ms = -1;

  // Allocate host memory
  float *input = new float[n * n];
  float *output = new float[n * n];
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      input[r * n + c] = matrix[r][c];
    }
  }
#ifdef DEBUG_WRITE_TO_FILE
  if (!oj_mode) WriteFile("1-input.txt", input, n, n);
#endif

  // Allocate device memory
  float *d_input;
  float *d_output;
  gpu_errchk(mcMalloc(&d_input, n * n * sizeof(float)));
  gpu_errchk(mcMalloc(&d_output, n * n * sizeof(float)));

  // Copy input to GPU
  gpu_errchk(
      mcMemcpy(d_input, input, n * n * sizeof(float), mcMemcpyHostToDevice));

  // CPU implementation
  if (kernel == "cpu" || kernel == "all") {
    START_TIMER();
    cpuTranspose(input, output, n);
    STOP_RECORD_TIMER(cpu_ms);

    checkTransposed(input, output, n);
    memset(output, 0, n * n * sizeof(float));

    printf("Size %d CPU: %f ms\n", n, cpu_ms);
  }

  // GPU implementation
  const int nIters = 10;  // repeat iterations
  float totalTime = 0.0;
  float minTime = std::numeric_limits<double>::max();
  if (kernel == "gpu" || kernel == "all") {
    for (int iter = 0; iter < nIters; iter++) {
      START_TIMER();
      macaTranspose(d_input, d_output, n);
      STOP_RECORD_TIMER(gpu_ms);
      totalTime += gpu_ms;
      if (minTime > gpu_ms) {
        minTime = gpu_ms;
      }
      printf("totalTime: %f; minTime: %f\n", totalTime, minTime);
    }
    gpu_errchk(mcMemcpy(output, d_output, n * n * sizeof(float),
                        mcMemcpyDeviceToHost));
    if (oj_mode) {
      WriteFile(argv[1], output, n, n);
    } else {
      checkTransposed(input, output, n);
#ifdef DEBUG_WRITE_TO_FILE
      WriteFile("1-output.txt", output, n, n);
#endif
    }

    memset(output, 0, n * n * sizeof(float));
    gpu_errchk(mcMemset(d_output, 0, n * n * sizeof(float)));

    float avgTime = totalTime / nIters;
    printf("avgTime: %f; minTime: %f\n", avgTime, minTime);
    printf("Size %d GPU: %f ms\n", n, gpu_ms);
  }

  // Free host memory
  delete[] input;
  delete[] output;

  // Free device memory
  gpu_errchk(mcFree(d_input));
  gpu_errchk(mcFree(d_output));

  printf("\n");
}
