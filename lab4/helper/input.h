#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.hpp"

inline void ReadInputData(float* out_data, int size) {
  std::vector<float> data;
  MetaXOJ::DataLoader::loadData(data);

  for (int i = 0; i < size; ++i) {
    if (i < data.size()) {
      out_data[i] = data[i];
    } else {
      out_data[i] = 0;
    }
  }
}
