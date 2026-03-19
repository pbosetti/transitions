#include "phase_corr.hpp"
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace phasecorr {

namespace {

std::mutex &planner_mutex() {
  // FFTW plan creation/destruction is not thread-safe and must be serialized.
  static auto *mutex = new std::mutex();
  return *mutex;
}

}  // namespace

FftwContext::FftwContext(int h, int w)
    : height(h), width(w),
      input_a(nullptr), input_b(nullptr),
      freq_a(nullptr), freq_b(nullptr),
      plan_a(nullptr), plan_b(nullptr), plan_inverse(nullptr) {
  const int complex_size = h * (w / 2 + 1);

  input_a = static_cast<float *>(fftwf_malloc(sizeof(float) * h * w));
  input_b = static_cast<float *>(fftwf_malloc(sizeof(float) * h * w));
  freq_a = static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * complex_size));
  freq_b = static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * complex_size));

  if (!input_a || !input_b || !freq_a || !freq_b) {
    if (input_a) fftwf_free(input_a);
    if (input_b) fftwf_free(input_b);
    if (freq_a) fftwf_free(freq_a);
    if (freq_b) fftwf_free(freq_b);
    throw std::bad_alloc();
  }

  {
    std::lock_guard<std::mutex> lock(planner_mutex());
    plan_a = fftwf_plan_dft_r2c_2d(h, w, input_a, freq_a, FFTW_ESTIMATE);
    plan_b = fftwf_plan_dft_r2c_2d(h, w, input_b, freq_b, FFTW_ESTIMATE);
    plan_inverse = fftwf_plan_dft_c2r_2d(h, w, freq_a, input_a, FFTW_ESTIMATE);
  }

  if (!plan_a || !plan_b || !plan_inverse) {
    std::lock_guard<std::mutex> lock(planner_mutex());
    if (plan_a) { fftwf_destroy_plan(plan_a); plan_a = nullptr; }
    if (plan_b) { fftwf_destroy_plan(plan_b); plan_b = nullptr; }
    if (plan_inverse) { fftwf_destroy_plan(plan_inverse); plan_inverse = nullptr; }
    fftwf_free(input_a); input_a = nullptr;
    fftwf_free(input_b); input_b = nullptr;
    fftwf_free(freq_a); freq_a = nullptr;
    fftwf_free(freq_b); freq_b = nullptr;
    throw std::runtime_error("Failed to create FFTW plans.");
  }
}

FftwContext::~FftwContext() {
  {
    std::lock_guard<std::mutex> lock(planner_mutex());
    if (plan_a) fftwf_destroy_plan(plan_a);
    if (plan_b) fftwf_destroy_plan(plan_b);
    if (plan_inverse) fftwf_destroy_plan(plan_inverse);
  }
  if (input_a) fftwf_free(input_a);
  if (input_b) fftwf_free(input_b);
  if (freq_a) fftwf_free(freq_a);
  if (freq_b) fftwf_free(freq_b);
}

PhaseCorrelationResult phaseCorrelation(FftwContext &ctx, const Image &a, const Image &b) {
  const int height = a.rows;
  const int width = a.cols;
  const int complex_size = height * (width / 2 + 1);

  std::memcpy(ctx.input_a, a.data.data(), sizeof(float) * height * width);
  std::memcpy(ctx.input_b, b.data.data(), sizeof(float) * height * width);

  fftwf_execute(ctx.plan_a);
  fftwf_execute(ctx.plan_b);

  for (int i = 0; i < complex_size; ++i) {
    const float ar = ctx.freq_a[i][0];
    const float ai = ctx.freq_a[i][1];
    const float br = ctx.freq_b[i][0];
    const float bi = -ctx.freq_b[i][1];

    const float rr = ar * br - ai * bi;
    const float ri = ar * bi + ai * br;
    const float magnitude = std::sqrt(rr * rr + ri * ri) + 1e-12f;

    ctx.freq_a[i][0] = rr / magnitude;
    ctx.freq_a[i][1] = ri / magnitude;
  }

  fftwf_execute(ctx.plan_inverse);

  int peak = 0;
  for (int i = 1; i < height * width; ++i) {
    if (ctx.input_a[i] > ctx.input_a[peak]) {
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
  result.peak = static_cast<double>(ctx.input_a[peak]);
  return result;
}

PhaseCorrelationResult phaseCorrelation(const Image &a, const Image &b) {
  FftwContext ctx(a.rows, a.cols);
  return phaseCorrelation(ctx, a, b);
}

}  // namespace phasecorr
