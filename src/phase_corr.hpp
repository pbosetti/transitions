#pragma once

#include <Eigen/Dense>
#include "image.hpp"

namespace phasecorr {

struct PhaseCorrelationResult {
  Eigen::Vector2d shift = Eigen::Vector2d::Zero();
  double peak = 0.0;
};

PhaseCorrelationResult phaseCorrelation(const Image &a, const Image &b);

}  // namespace phasecorr
