#include "phase_corr.hpp"
#include <fftw3.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace phasecorr {

namespace {

std::mutex PlannerMutex;

struct PlanKey {
  int width = 0;
  int height = 0;

  bool operator==(const PlanKey &other) const {
    return width == other.width && height == other.height;
  }
};

struct PlanKeyHash {
  std::size_t operator()(const PlanKey &key) const {
    return (static_cast<std::size_t>(static_cast<unsigned int>(key.width)) << 32U) ^
           static_cast<std::size_t>(static_cast<unsigned int>(key.height));
  }
};

struct PlanCacheEntry {
  int width = 0;
  int height = 0;
  int complex_size = 0;
  float *plan_input_a = nullptr;
  float *plan_input_b = nullptr;
  fftwf_complex *plan_freq_a = nullptr;
  fftwf_complex *plan_freq_b = nullptr;
  fftwf_plan plan_a = nullptr;
  fftwf_plan plan_b = nullptr;
  fftwf_plan plan_inverse = nullptr;

  PlanCacheEntry(int image_width, int image_height) : width(image_width), height(image_height) {
    complex_size = height * (width / 2 + 1);
    plan_input_a = static_cast<float *>(fftwf_malloc(sizeof(float) * height * width));
    plan_input_b = static_cast<float *>(fftwf_malloc(sizeof(float) * height * width));
    plan_freq_a =
        static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * complex_size));
    plan_freq_b =
        static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * complex_size));
    if (!plan_input_a || !plan_input_b || !plan_freq_a || !plan_freq_b) {
      throw std::bad_alloc();
    }

    std::lock_guard<std::mutex> lock(PlannerMutex);
    plan_a = fftwf_plan_dft_r2c_2d(height, width, plan_input_a, plan_freq_a, FFTW_ESTIMATE);
    plan_b = fftwf_plan_dft_r2c_2d(height, width, plan_input_b, plan_freq_b, FFTW_ESTIMATE);
    plan_inverse =
        fftwf_plan_dft_c2r_2d(height, width, plan_freq_a, plan_input_a, FFTW_ESTIMATE);
    if (!plan_a || !plan_b || !plan_inverse) {
      throw std::runtime_error("Failed to create FFTW plans.");
    }
  }

  ~PlanCacheEntry() {
    std::lock_guard<std::mutex> lock(PlannerMutex);
    if (plan_a) {
      fftwf_destroy_plan(plan_a);
    }
    if (plan_b) {
      fftwf_destroy_plan(plan_b);
    }
    if (plan_inverse) {
      fftwf_destroy_plan(plan_inverse);
    }
    if (plan_input_a) {
      fftwf_free(plan_input_a);
    }
    if (plan_input_b) {
      fftwf_free(plan_input_b);
    }
    if (plan_freq_a) {
      fftwf_free(plan_freq_a);
    }
    if (plan_freq_b) {
      fftwf_free(plan_freq_b);
    }
  }

  PlanCacheEntry(const PlanCacheEntry &) = delete;
  PlanCacheEntry &operator=(const PlanCacheEntry &) = delete;
};

PlanCacheEntry &get_plan_cache_entry(int width, int height) {
  thread_local std::unordered_map<PlanKey, std::unique_ptr<PlanCacheEntry>, PlanKeyHash> cache;

  const PlanKey key{width, height};
  const auto it = cache.find(key);
  if (it != cache.end()) {
    return *it->second;
  }

  auto entry = std::make_unique<PlanCacheEntry>(width, height);
  PlanCacheEntry &entry_ref = *entry;
  cache.emplace(key, std::move(entry));
  return entry_ref;
}

}  // namespace

Eigen::Vector2d phaseCorrelation(const Image &a, const Image &b) {
  const int height = a.rows;
  const int width = a.cols;
  PlanCacheEntry &plans = get_plan_cache_entry(width, height);
  const int complex_size = plans.complex_size;

  float *a_buffer = static_cast<float *>(fftwf_malloc(sizeof(float) * height * width));
  float *b_buffer = static_cast<float *>(fftwf_malloc(sizeof(float) * height * width));
  fftwf_complex *fa =
      static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * complex_size));
  fftwf_complex *fb =
      static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * complex_size));

  std::memcpy(a_buffer, a.data.data(), sizeof(float) * height * width);
  std::memcpy(b_buffer, b.data.data(), sizeof(float) * height * width);
  fftwf_execute_dft_r2c(plans.plan_a, a_buffer, fa);
  fftwf_execute_dft_r2c(plans.plan_b, b_buffer, fb);

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

  fftwf_execute_dft_c2r(plans.plan_inverse, fa, a_buffer);

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

  fftwf_free(a_buffer);
  fftwf_free(b_buffer);
  fftwf_free(fa);
  fftwf_free(fb);

  return Eigen::Vector2d(peak_x, peak_y);
}

}  // namespace phasecorr
