#pragma once

#include <Eigen/Dense>
#include "image.hpp"

namespace phasecorr {

Eigen::Vector2d phaseCorrelation(const Image &a, const Image &b);

}  // namespace phasecorr
