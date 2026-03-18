#include "phase_corr.hpp"
#include <fftw3.h>
#include <cmath>
#include <cstring>
#include <mutex>

namespace phasecorr {

namespace {

std::mutex &planner_mutex() {
  // FFTW plan creation/destruction is serialized.
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::mutex &execution_mutex() {
  // Release-mode crashes suggest FFTW use is not safe in our current threaded setup.
  static auto *mutex = new std::mutex();
  return *mutex;
}

}  // namespace

PhaseCorrelationResult phaseCorrelation(const Image &a, const Image &b) {
  const int height = a.rows;
  const int width = a.cols;
  const int complex_size = height * (width / 2 + 1);

  float *input_a = static_cast<float *>(fftwf_malloc(sizeof(float) * height * width));
  float *input_b = static_cast<float *>(fftwf_malloc(sizeof(float) * height * width));
  fftwf_complex *freq_a =
      static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * complex_size));
  fftwf_complex *freq_b =
      static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * complex_size));
  if (!input_a || !input_b || !freq_a || !freq_b) {
    if (input_a) {
      fftwf_free(input_a);
    }
    if (input_b) {
      fftwf_free(input_b);
    }
    if (freq_a) {
      fftwf_free(freq_a);
    }
    if (freq_b) {
      fftwf_free(freq_b);
    }
    throw std::bad_alloc();
  }

  std::memcpy(input_a, a.data.data(), sizeof(float) * height * width);
  std::memcpy(input_b, b.data.data(), sizeof(float) * height * width);

  fftwf_plan plan_a = nullptr;
  fftwf_plan plan_b = nullptr;
  fftwf_plan plan_inverse = nullptr;
  {
    std::lock_guard<std::mutex> lock(planner_mutex());
    plan_a = fftwf_plan_dft_r2c_2d(height, width, input_a, freq_a, FFTW_ESTIMATE);
    plan_b = fftwf_plan_dft_r2c_2d(height, width, input_b, freq_b, FFTW_ESTIMATE);
    plan_inverse = fftwf_plan_dft_c2r_2d(height, width, freq_a, input_a, FFTW_ESTIMATE);
  }
  if (!plan_a || !plan_b || !plan_inverse) {
    if (plan_a) {
      fftwf_destroy_plan(plan_a);
    }
    if (plan_b) {
      fftwf_destroy_plan(plan_b);
    }
    if (plan_inverse) {
      fftwf_destroy_plan(plan_inverse);
    }
    fftwf_free(input_a);
    fftwf_free(input_b);
    fftwf_free(freq_a);
    fftwf_free(freq_b);
    throw std::runtime_error("Failed to create FFTW plans.");
  }

  {
    std::lock_guard<std::mutex> lock(execution_mutex());
    fftwf_execute(plan_a);
    fftwf_execute(plan_b);

    for (int i = 0; i < complex_size; ++i) {
      const float ar = freq_a[i][0];
      const float ai = freq_a[i][1];
      const float br = freq_b[i][0];
      const float bi = -freq_b[i][1];

      const float rr = ar * br - ai * bi;
      const float ri = ar * bi + ai * br;
      const float magnitude = std::sqrt(rr * rr + ri * ri) + 1e-12f;

      freq_a[i][0] = rr / magnitude;
      freq_a[i][1] = ri / magnitude;
    }

    fftwf_execute(plan_inverse);
  }

  int peak = 0;
  for (int i = 1; i < height * width; ++i) {
    if (input_a[i] > input_a[peak]) {
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

  PhaseCorrelationResult result{};
  result.shift = Eigen::Vector2d(peak_x, peak_y);
  result.peak = static_cast<double>(input_a[peak]);

  {
    std::lock_guard<std::mutex> lock(planner_mutex());
    fftwf_destroy_plan(plan_a);
    fftwf_destroy_plan(plan_b);
    fftwf_destroy_plan(plan_inverse);
  }
  fftwf_free(input_a);
  fftwf_free(input_b);
  fftwf_free(freq_a);
  fftwf_free(freq_b);
  return result;
}

}  // namespace phasecorr
