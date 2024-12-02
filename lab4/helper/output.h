#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

inline void WriteFile(const char *file_name, const float *p, int size) {
  // write result to `file_name`
  auto *OFD = fopen(file_name, "w");
  if (OFD == nullptr) {
    fprintf(stderr, "Error open result.txt by %s\n", strerror(errno));
    std::abort();
  }
  for (int i = 0; i < size; ++i) {
    fprintf(OFD, "%f ", p[i]);
  }
  fprintf(OFD, "\n");
}

#define OutputResult WriteFile
