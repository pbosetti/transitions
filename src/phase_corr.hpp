#pragma once

#include <fftw3.h>
#include <Eigen/Dense>
#include "image.hpp"

namespace phasecorr {

struct PhaseCorrelationResult {
  Eigen::Vector2d shift = Eigen::Vector2d::Zero();
  double peak = 0.0;
};

// Reusable FFTW context for a fixed image size.
// Holds pre-allocated buffers and pre-created plans so that repeated
// phaseCorrelation calls on the same image size avoid repeated malloc/free
// and plan creation/destruction.  Each thread must own its own FftwContext.
struct FftwContext {
  int height;
  int width;
  float *input_a;
  float *input_b;
  fftwf_complex *freq_a;
  fftwf_complex *freq_b;
  fftwf_plan plan_a;
  fftwf_plan plan_b;
  fftwf_plan plan_inverse;

  FftwContext(int h, int w);
  ~FftwContext();
  FftwContext(const FftwContext &) = delete;
  FftwContext &operator=(const FftwContext &) = delete;
};

PhaseCorrelationResult phaseCorrelation(const Image &a, const Image &b);
PhaseCorrelationResult phaseCorrelation(FftwContext &ctx, const Image &a, const Image &b);

}  // namespace phasecorr
