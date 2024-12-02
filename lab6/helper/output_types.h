#pragma once

#include <vector>

#include "external/json.hpp"

struct OutputData {
  std::vector<float> c;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(OutputData, c)
};

