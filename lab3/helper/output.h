#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

inline void WriteFile(const char *file_name, float *matrix, int rows,
                      int cols) {
  // write result to `file_name`
  auto *OFD = fopen(file_name, "w");
  if (OFD == nullptr) {
    fprintf(stderr, "Error open result.txt by %s\n", strerror(errno));
    std::abort();
  }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      fprintf(OFD, "%f ", matrix[r * cols + c]);
    }
    fprintf(OFD, "\n");
  }
  fprintf(OFD, "\n");
}

#define OutputResult WriteFile
