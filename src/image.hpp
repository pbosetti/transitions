#pragma once
#include <vector>

namespace phasecorr {

struct Image {
  int rows;
  int cols;
  std::vector<float> data;

  Image(int r, int c) : rows(r), cols(c), data(r * c) {}

  float &operator()(int r, int c) {
    return data[r * cols + c];
  }

  const float &operator()(int r, int c) const {
    return data[r * cols + c];
  }
};

}  // namespace phasecorr
