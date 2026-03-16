#include <iostream>
#include "../phase_corr.hpp"

int main() {
  const int height = 512;
  const int width = 512;

  phasecorr::Image a(height, width);
  phasecorr::Image b(height, width);

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      a(r, c) = std::sin(0.01 * r) + std::cos(0.01 * c);
    }
  }

  const int dx = 20;
  const int dy = -15;

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      const int rr = (r + dy + height) % height;
      const int cc = (c + dx + width) % width;
      b(rr, cc) = a(r, c);
    }
  }

  const auto shift = phasecorr::phaseCorrelation(a, b);

  std::cout << "Estimated shift: " << shift.transpose() << std::endl;
}
