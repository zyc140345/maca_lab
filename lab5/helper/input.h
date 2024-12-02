#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.hpp"
#include "external/json.hpp"

struct InputData {
  int m;
  int n;
  std::vector<float> matrix_a;
  std::vector<float> x;
  std::vector<float> y;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(InputData, m, n, matrix_a, x, y)
};

void ReadInputData(const int m, const int n, float* matrix, float* x,
                   float* y) {
  assert(!matrix && !x && !y);

  std::vector<unsigned char> buffer;
  MetaXOJ::DataLoader::loadData(buffer);

  nlohmann::json j = nlohmann::json::from_cbor(buffer);
  InputData d = j;
  assert(d.m == m);
  assert(d.n == n);

  for (int i = 0; i < m * n; ++i) {
    matrix[i] = d.matrix_a[i];
  }
  for (int i = 0; i < n; ++i) {
    x[i] = d.x[i];
  }
  for (int i = 0; i < m; ++i) {
    y[i] = d.y[i];
  }
}

void WriteInputDataToFile(const char* input_file_name, int m, int n,
                          const float* matrix, const float* x, const float* y) {
  InputData d;
  d.m = m;
  d.n = n;
  for (int i = 0; i < m * n; ++i) {
    d.matrix_a.push_back(matrix[i]);
  }
  for (int i = 0; i < n; ++i) {
    d.x.push_back(x[i]);
  }
  for (int i = 0; i < m; ++i) {
    d.y.push_back(y[i]);
  }

  nlohmann::json j = d;
  std::vector<std::uint8_t> v = nlohmann::json::to_cbor(j);
  std::ofstream ofs(input_file_name);
  assert(ofs);
  ofs.write(reinterpret_cast<char*>(v.data()), v.size());
}
