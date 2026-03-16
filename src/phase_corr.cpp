#include "phase_corr.hpp"
#include <fftw3.h>
#include <cmath>
#include <cstring>
#include <mutex>

namespace phasecorr {

namespace {

std::mutex PlannerMutex;

}  // namespace

Eigen::Vector2d phaseCorrelation(const Image &a, const Image &b) {
  const int height = a.rows;
  const int width = a.cols;
  const int complex_size = height * (width / 2 + 1);

  float *a_buffer = static_cast<float *>(fftwf_malloc(sizeof(float) * height * width));
  float *b_buffer = static_cast<float *>(fftwf_malloc(sizeof(float) * height * width));
  fftwf_complex *fa =
      static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * complex_size));
  fftwf_complex *fb =
      static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * complex_size));

  std::memcpy(a_buffer, a.data.data(), sizeof(float) * height * width);
  std::memcpy(b_buffer, b.data.data(), sizeof(float) * height * width);

  fftwf_plan plan_a = nullptr;
  fftwf_plan plan_b = nullptr;
  fftwf_plan plan_inverse = nullptr;

  {
    std::lock_guard<std::mutex> lock(PlannerMutex);
    plan_a = fftwf_plan_dft_r2c_2d(height, width, a_buffer, fa, FFTW_ESTIMATE);
    plan_b = fftwf_plan_dft_r2c_2d(height, width, b_buffer, fb, FFTW_ESTIMATE);
  }

  fftwf_execute(plan_a);
  fftwf_execute(plan_b);

  for (int i = 0; i < complex_size; ++i) {
    const float ar = fa[i][0];
    const float ai = fa[i][1];
    const float br = fb[i][0];
    const float bi = -fb[i][1];

    const float rr = ar * br - ai * bi;
    const float ri = ar * bi + ai * br;
    const float magnitude = std::sqrt(rr * rr + ri * ri) + 1e-12f;

    fa[i][0] = rr / magnitude;
    fa[i][1] = ri / magnitude;
  }

  {
    std::lock_guard<std::mutex> lock(PlannerMutex);
    plan_inverse = fftwf_plan_dft_c2r_2d(height, width, fa, a_buffer, FFTW_ESTIMATE);
  }
  fftwf_execute(plan_inverse);

  int peak = 0;
  for (int i = 1; i < height * width; ++i) {
    if (a_buffer[i] > a_buffer[peak]) {
      peak = i;
    }
  }

  int peak_y = peak / width;
  int peak_x = peak % width;
  if (peak_x > width / 2) {
    peak_x -= width;
  }
  if (peak_y > height / 2) {
    peak_y -= height;
  }

  {
    std::lock_guard<std::mutex> lock(PlannerMutex);
    fftwf_destroy_plan(plan_a);
    fftwf_destroy_plan(plan_b);
    fftwf_destroy_plan(plan_inverse);
  }
  fftwf_free(a_buffer);
  fftwf_free(b_buffer);
  fftwf_free(fa);
  fftwf_free(fb);

  return Eigen::Vector2d(peak_x, peak_y);
}

}  // namespace phasecorr
