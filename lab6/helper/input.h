#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.hpp"
#include "external/json.hpp"

struct InputData {
  int size;
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(InputData, size, a, b, c)
};

void ReadInputData(const int size, float* a, float* b, float* c) {
  assert((size > 0) && !a && !b && !c);

  std::vector<unsigned char> buffer;
  MetaXOJ::DataLoader::loadData(buffer);

  nlohmann::json j = nlohmann::json::from_cbor(buffer);
  InputData d = j;
  assert(d.size == size);

  for (int i = 0; i < size; ++i) {
    a[i] = d.a[i];
    b[i] = d.b[i];
    c[i] = d.c[i];
  }
}

void WriteInputDataToFile(const char* input_file_name, int size, const float* a,
                          const float* b, const float* c) {
  assert((size > 0) && !a && !b && !c);

  InputData d;
  d.size = size;
  for (int i = 0; i < size; ++i) {
    d.a.push_back(a[i]);
    d.b.push_back(b[i]);
    d.c.push_back(c[i]);
  }

  nlohmann::json j = d;
  std::vector<std::uint8_t> v = nlohmann::json::to_cbor(j);
  std::ofstream ofs(input_file_name);
  assert(ofs);
  ofs.write(reinterpret_cast<char*>(v.data()), v.size());
}
