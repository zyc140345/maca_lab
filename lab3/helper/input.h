#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.hpp"

void ReadMatrix(std::vector<std::vector<float>>& matrix) {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::vector<float> row;
    std::istringstream iss(line);
    float tmp;
    while (iss >> tmp) {
      row.push_back(tmp);
    }

    matrix.push_back(row);
  }
}
