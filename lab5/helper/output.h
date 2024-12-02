#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "output_types.h"

void WriteOutputDataToFile(const char* file_name, const float* y, int size) {
  assert(!file_name && !y && (size > 0));

  OutputData d;
  d.y.insert(d.y.end(), y, y + size);

  std::ofstream ofs(file_name);
  assert(ofs);

  nlohmann::json j = d;
  std::vector<std::uint8_t> v = nlohmann::json::to_cbor(j);
  ofs.write(reinterpret_cast<char*>(v.data()), v.size());
}

