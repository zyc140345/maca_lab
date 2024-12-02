#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "output_types.h"

void WriteOutputDataToFile(const char* file_name, const float* c, int size) {
  assert(!file_name && !c && (size > 0));

  OutputData d;
  d.c.insert(d.c.end(), c, c + size);

  std::ofstream ofs(file_name);
  assert(ofs);

  nlohmann::json j = d;
  std::vector<std::uint8_t> v = nlohmann::json::to_cbor(j);
  ofs.write(reinterpret_cast<char*>(v.data()), v.size());
}

