#include <cstddef>
#include <cstdio>

#include <jpeglib.h>
#include <libraw/libraw.h>
#include <tiffio.h>

#include "phase_corr.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <csetjmp>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numbers>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace Fs = std::filesystem;

namespace {

std::mutex &jpeg_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::mutex &libraw_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::mutex &tiff_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

enum class SortMode {
  kName,
  kCaptureTime,
};

enum class SliceDirection {
  kVertical,
  kHorizontal,
};

struct Options {
  SortMode sort_mode = SortMode::kCaptureTime;
  SliceDirection direction = SliceDirection::kVertical;
  bool reverse = false;
  bool auto_align = false;
  bool global_auto_align = false;
  bool dump_intermediate = false;
  int jpeg_quality = 95;
  std::size_t parallelism = 1;
  std::size_t preview_count = 0;
  Fs::path input_dir;
  Fs::path output_path;
};

struct Timestamp {
  int year = 0;
  int month = 0;
  int day = 0;
  int hour = 0;
  int minute = 0;
  int second = 0;

  auto operator<=>(const Timestamp &) const = default;
};

struct Image {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;
};

struct InputEntry {
  Fs::path path;
  std::string filename;
  std::optional<Timestamp> capture_time;
  int width = 0;
  int height = 0;
};

struct ImageInfo {
  InputEntry entry;
  Image image;
};

struct Offset {
  int dx = 0;
  int dy = 0;
};

struct Transform {
  double dx = 0.0;
  double dy = 0.0;
  double rotation_degrees = 0.0;
};

struct CropRect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

struct AlignmentData {
  std::vector<Transform> relative_transforms;
  std::vector<Transform> total_transforms;
};

struct JpegErrorManager {
  jpeg_error_mgr base;
  std::jmp_buf jump_buffer;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void print_progress(std::string_view operation, std::size_t current, std::size_t total) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::cout << operation << ": [" << current << "/" << total << "]" << std::endl;
}

void print_alignment_progress(std::string_view operation, std::size_t current, std::size_t total,
                              const Transform &transform) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::cout << operation << ": [" << current << "/" << total << "] "
            << "x=" << std::lround(transform.dx) << "px, y=" << std::lround(transform.dy)
            << "px, r=" << std::fixed << std::setprecision(3) << transform.rotation_degrees
            << "deg" << std::defaultfloat << std::endl;
}

void print_usage(const char *argv0) {
  std::cout
      << "Usage: " << argv0 << " [options] <input_dir> <output.{jpg,tif}>\n"
      << "Options:\n"
      << "  --sort <name|time>       Sort input files by filename or capture time (default: time).\n"
      << "  --reverse                Reverse the selected ordering.\n"
      << "  --horizontal             Build the output from horizontal slices.\n"
      << "  --auto-align             Align each image to the previous one with black padding.\n"
      << "  --global-auto-align      Align all images, crop to common content, disable --auto-align.\n"
      << "  --dump-intermediate      Save slices and alignment data in a temp folder before merging.\n"
      << "  --preview <n>            Sample n images uniformly, always including first and last.\n"
      << "  --parallel <n>           Load input images using n threads (default: 1).\n"
      << "  --quality <1-100>        JPEG quality for JPEG output only (default: 95).\n"
      << "  --help                   Show this message.\n";
}

std::string lowercase(std::string_view value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return result;
}

bool is_supported_input(const Fs::path &path) {
  if (!path.has_extension()) {
    return false;
  }

  const std::string extension = lowercase(path.extension().string());
  return extension == ".jpg" || extension == ".jpeg" || extension == ".dng";
}

std::size_t parse_positive_count(std::string_view value, std::string_view option_name) {
  std::size_t position = 0;
  const unsigned long long parsed = std::stoull(std::string(value), &position);
  if (position != value.size() || parsed == 0) {
    fail(std::string(option_name) + " must be a positive integer.");
  }
  return static_cast<std::size_t>(parsed);
}

Options parse_args(int argc, char **argv) {
  Options options;
  std::vector<std::string> positional_args;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--sort") {
      if (i + 1 >= argc) {
        fail("Missing value for --sort.");
      }
      const std::string value = lowercase(argv[++i]);
      if (value == "name") {
        options.sort_mode = SortMode::kName;
      } else if (value == "time") {
        options.sort_mode = SortMode::kCaptureTime;
      } else {
        fail("Invalid value for --sort: " + value);
      }
      continue;
    }
    if (arg == "--reverse") {
      options.reverse = true;
      continue;
    }
    if (arg == "--horizontal") {
      options.direction = SliceDirection::kHorizontal;
      continue;
    }
    if (arg == "--auto-align") {
      options.auto_align = true;
      continue;
    }
    if (arg == "--global-auto-align") {
      options.global_auto_align = true;
      options.auto_align = false;
      continue;
    }
    if (arg == "--dump-intermediate") {
      options.dump_intermediate = true;
      continue;
    }
    if (arg == "--preview") {
      if (i + 1 >= argc) {
        fail("Missing value for --preview.");
      }
      options.preview_count = parse_positive_count(argv[++i], "--preview");
      continue;
    }
    if (arg.rfind("--preview=", 0) == 0) {
      options.preview_count = parse_positive_count(std::string_view(arg).substr(10), "--preview");
      continue;
    }
    if (arg == "--parallel") {
      if (i + 1 >= argc) {
        fail("Missing value for --parallel.");
      }
      options.parallelism = parse_positive_count(argv[++i], "--parallel");
      continue;
    }
    if (arg.rfind("--parallel=", 0) == 0) {
      options.parallelism = parse_positive_count(std::string_view(arg).substr(11), "--parallel");
      continue;
    }
    if (arg == "--quality") {
      if (i + 1 >= argc) {
        fail("Missing value for --quality.");
      }
      options.jpeg_quality = std::stoi(argv[++i]);
      if (options.jpeg_quality < 1 || options.jpeg_quality > 100) {
        fail("JPEG quality must be between 1 and 100.");
      }
      continue;
    }
    if (arg.rfind("--", 0) == 0) {
      fail("Unknown option: " + arg);
    }
    positional_args.push_back(arg);
  }

  if (positional_args.size() != 2) {
    print_usage(argv[0]);
    std::exit(1);
  }

  options.input_dir = positional_args[0];
  options.output_path = positional_args[1];
  return options;
}

int compare_natural(std::string_view lhs, std::string_view rhs) {
  std::size_t i = 0;
  std::size_t j = 0;

  while (i < lhs.size() && j < rhs.size()) {
    const unsigned char left = static_cast<unsigned char>(lhs[i]);
    const unsigned char right = static_cast<unsigned char>(rhs[j]);
    const bool left_is_digit = std::isdigit(left) != 0;
    const bool right_is_digit = std::isdigit(right) != 0;

    if (left_is_digit && right_is_digit) {
      std::size_t i_end = i;
      std::size_t j_end = j;
      while (i_end < lhs.size() &&
             std::isdigit(static_cast<unsigned char>(lhs[i_end])) != 0) {
        ++i_end;
      }
      while (j_end < rhs.size() &&
             std::isdigit(static_cast<unsigned char>(rhs[j_end])) != 0) {
        ++j_end;
      }

      std::size_t i_trim = i;
      std::size_t j_trim = j;
      while (i_trim < i_end && lhs[i_trim] == '0') {
        ++i_trim;
      }
      while (j_trim < j_end && rhs[j_trim] == '0') {
        ++j_trim;
      }

      const std::size_t left_digits = i_end - i_trim;
      const std::size_t right_digits = j_end - j_trim;
      if (left_digits != right_digits) {
        return left_digits < right_digits ? -1 : 1;
      }

      for (std::size_t k = 0; k < left_digits; ++k) {
        if (lhs[i_trim + k] != rhs[j_trim + k]) {
          return lhs[i_trim + k] < rhs[j_trim + k] ? -1 : 1;
        }
      }

      const std::size_t left_run = i_end - i;
      const std::size_t right_run = j_end - j;
      if (left_run != right_run) {
        return left_run < right_run ? -1 : 1;
      }

      i = i_end;
      j = j_end;
      continue;
    }

    const char left_lower = static_cast<char>(std::tolower(left));
    const char right_lower = static_cast<char>(std::tolower(right));
    if (left_lower != right_lower) {
      return left_lower < right_lower ? -1 : 1;
    }
    if (lhs[i] != rhs[j]) {
      return lhs[i] < rhs[j] ? -1 : 1;
    }
    ++i;
    ++j;
  }

  if (i == lhs.size() && j == rhs.size()) {
    return 0;
  }
  return i == lhs.size() ? -1 : 1;
}

std::optional<Timestamp> parse_timestamp_string(std::string_view value) {
  if (value.size() < 19) {
    return std::nullopt;
  }

  Timestamp timestamp{};
  const auto parse_field = [&](std::size_t start, std::size_t length) -> std::optional<int> {
    if (start + length > value.size()) {
      return std::nullopt;
    }
    int result = 0;
    for (std::size_t i = 0; i < length; ++i) {
      const unsigned char ch = static_cast<unsigned char>(value[start + i]);
      if (std::isdigit(ch) == 0) {
        return std::nullopt;
      }
      result = result * 10 + (ch - '0');
    }
    return result;
  };

  const auto year = parse_field(0, 4);
  const auto month = parse_field(5, 2);
  const auto day = parse_field(8, 2);
  const auto hour = parse_field(11, 2);
  const auto minute = parse_field(14, 2);
  const auto second = parse_field(17, 2);

  if (!year || !month || !day || !hour || !minute || !second) {
    return std::nullopt;
  }

  if (value[4] != ':' || value[7] != ':' || value[10] != ' ' ||
      value[13] != ':' || value[16] != ':') {
    return std::nullopt;
  }

  timestamp.year = *year;
  timestamp.month = *month;
  timestamp.day = *day;
  timestamp.hour = *hour;
  timestamp.minute = *minute;
  timestamp.second = *second;
  return timestamp;
}

std::optional<Timestamp> timestamp_from_epoch(std::time_t epoch) {
  if (epoch <= 0) {
    return std::nullopt;
  }

  std::tm local_time{};
#if defined(_WIN32)
  if (localtime_s(&local_time, &epoch) != 0) {
    return std::nullopt;
  }
#else
  if (localtime_r(&epoch, &local_time) == nullptr) {
    return std::nullopt;
  }
#endif

  Timestamp timestamp{};
  timestamp.year = local_time.tm_year + 1900;
  timestamp.month = local_time.tm_mon + 1;
  timestamp.day = local_time.tm_mday;
  timestamp.hour = local_time.tm_hour;
  timestamp.minute = local_time.tm_min;
  timestamp.second = local_time.tm_sec;
  return timestamp;
}

std::uint16_t read_u16(std::span<const std::uint8_t> data, std::size_t offset, bool little_endian) {
  if (offset + 2 > data.size()) {
    fail("Invalid EXIF structure.");
  }
  if (little_endian) {
    return static_cast<std::uint16_t>(data[offset]) |
           (static_cast<std::uint16_t>(data[offset + 1]) << 8U);
  }
  return (static_cast<std::uint16_t>(data[offset]) << 8U) |
         static_cast<std::uint16_t>(data[offset + 1]);
}

std::uint32_t read_u32(std::span<const std::uint8_t> data, std::size_t offset, bool little_endian) {
  if (offset + 4 > data.size()) {
    fail("Invalid EXIF structure.");
  }
  if (little_endian) {
    return static_cast<std::uint32_t>(data[offset]) |
           (static_cast<std::uint32_t>(data[offset + 1]) << 8U) |
           (static_cast<std::uint32_t>(data[offset + 2]) << 16U) |
           (static_cast<std::uint32_t>(data[offset + 3]) << 24U);
  }
  return (static_cast<std::uint32_t>(data[offset]) << 24U) |
         (static_cast<std::uint32_t>(data[offset + 1]) << 16U) |
         (static_cast<std::uint32_t>(data[offset + 2]) << 8U) |
         static_cast<std::uint32_t>(data[offset + 3]);
}

std::optional<Timestamp> extract_exif_datetime_from_tiff(std::span<const std::uint8_t> data,
                                                         std::uint32_t ifd_offset,
                                                         bool little_endian,
                                                         std::size_t tiff_start) {
  if (tiff_start + ifd_offset + 2 > data.size()) {
    return std::nullopt;
  }

  const std::size_t dir_offset = tiff_start + ifd_offset;
  const std::uint16_t entry_count = read_u16(data, dir_offset, little_endian);
  for (std::uint16_t index = 0; index < entry_count; ++index) {
    const std::size_t entry_offset = dir_offset + 2U + static_cast<std::size_t>(index) * 12U;
    if (entry_offset + 12 > data.size()) {
      return std::nullopt;
    }

    const std::uint16_t tag = read_u16(data, entry_offset, little_endian);
    const std::uint16_t type = read_u16(data, entry_offset + 2, little_endian);
    const std::uint32_t count = read_u32(data, entry_offset + 4, little_endian);
    const std::uint32_t value_or_offset = read_u32(data, entry_offset + 8, little_endian);

    if (tag == 0x9003 && type == 2 && count >= 19) {
      const std::size_t value_offset = tiff_start + value_or_offset;
      if (value_offset + count > data.size()) {
        return std::nullopt;
      }
      const std::string_view value(reinterpret_cast<const char *>(data.data() + value_offset),
                                   count > 0 ? count - 1 : 0);
      return parse_timestamp_string(value);
    }
  }

  return std::nullopt;
}

std::optional<Timestamp> extract_exif_datetime(std::span<const std::uint8_t> exif_block) {
  if (exif_block.size() < 14) {
    return std::nullopt;
  }
  if (std::string_view(reinterpret_cast<const char *>(exif_block.data()), 6) != "Exif\0\0") {
    return std::nullopt;
  }

  const std::size_t tiff_start = 6;
  const bool little_endian =
      exif_block[tiff_start] == 'I' && exif_block[tiff_start + 1] == 'I';
  const bool big_endian =
      exif_block[tiff_start] == 'M' && exif_block[tiff_start + 1] == 'M';
  if (!little_endian && !big_endian) {
    return std::nullopt;
  }

  const std::uint16_t marker = read_u16(exif_block, tiff_start + 2, little_endian);
  if (marker != 42) {
    return std::nullopt;
  }

  const std::uint32_t ifd0_offset = read_u32(exif_block, tiff_start + 4, little_endian);
  if (tiff_start + ifd0_offset + 2 > exif_block.size()) {
    return std::nullopt;
  }

  const std::size_t ifd0 = tiff_start + ifd0_offset;
  const std::uint16_t entry_count = read_u16(exif_block, ifd0, little_endian);
  for (std::uint16_t index = 0; index < entry_count; ++index) {
    const std::size_t entry_offset = ifd0 + 2U + static_cast<std::size_t>(index) * 12U;
    if (entry_offset + 12 > exif_block.size()) {
      return std::nullopt;
    }

    const std::uint16_t tag = read_u16(exif_block, entry_offset, little_endian);
    if (tag != 0x8769) {
      continue;
    }

    const std::uint32_t exif_ifd_offset = read_u32(exif_block, entry_offset + 8, little_endian);
    return extract_exif_datetime_from_tiff(exif_block, exif_ifd_offset, little_endian, tiff_start);
  }

  return std::nullopt;
}

void jpeg_error_exit(j_common_ptr cinfo) {
  auto *error = reinterpret_cast<JpegErrorManager *>(cinfo->err);
  char buffer[JMSG_LENGTH_MAX];
  (*cinfo->err->format_message)(cinfo, buffer);
  std::cerr << "JPEG error: " << buffer << '\n';
  std::longjmp(error->jump_buffer, 1);
}

InputEntry read_jpeg_entry(const Fs::path &path) {
  std::lock_guard<std::mutex> lock(jpeg_mutex());
  FILE *file = std::fopen(path.string().c_str(), "rb");
  if (file == nullptr) {
    fail("Unable to open JPEG file: " + path.string());
  }

  jpeg_decompress_struct cinfo{};
  JpegErrorManager jerr{};
  cinfo.err = jpeg_std_error(&jerr.base);
  jerr.base.error_exit = jpeg_error_exit;

  if (setjmp(jerr.jump_buffer) != 0) {
    jpeg_destroy_decompress(&cinfo);
    std::fclose(file);
    fail("Failed to inspect JPEG file: " + path.string());
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, file);
  jpeg_save_markers(&cinfo, JPEG_APP0 + 1, 0xFFFF);
  jpeg_read_header(&cinfo, TRUE);

  InputEntry entry{};
  entry.path = path;
  entry.filename = path.filename().string();
  entry.width = static_cast<int>(cinfo.image_width);
  entry.height = static_cast<int>(cinfo.image_height);

  for (jpeg_saved_marker_ptr marker = cinfo.marker_list; marker != nullptr; marker = marker->next) {
    if (marker->marker != JPEG_APP0 + 1) {
      continue;
    }
    const std::span<const std::uint8_t> exif(marker->data, marker->data_length);
    entry.capture_time = extract_exif_datetime(exif);
    if (entry.capture_time) {
      break;
    }
  }

  jpeg_destroy_decompress(&cinfo);
  std::fclose(file);
  return entry;
}

InputEntry read_dng_entry(const Fs::path &path) {
  std::lock_guard<std::mutex> lock(libraw_mutex());
  LibRaw raw_processor;
  int result = raw_processor.open_file(path.string().c_str());
  if (result != LIBRAW_SUCCESS) {
    fail("Unable to open DNG file: " + path.string());
  }

  result = raw_processor.unpack();
  if (result != LIBRAW_SUCCESS) {
    fail("Unable to inspect DNG file: " + path.string());
  }

  InputEntry entry{};
  entry.path = path;
  entry.filename = path.filename().string();
  entry.capture_time = timestamp_from_epoch(raw_processor.imgdata.other.timestamp);
  entry.width = static_cast<int>(raw_processor.imgdata.sizes.iwidth);
  entry.height = static_cast<int>(raw_processor.imgdata.sizes.iheight);
  if (entry.width <= 0 || entry.height <= 0) {
    fail("Invalid DNG dimensions: " + path.string());
  }
  return entry;
}

InputEntry read_input_entry(const Fs::path &path) {
  const std::string extension = lowercase(path.extension().string());
  if (extension == ".jpg" || extension == ".jpeg") {
    return read_jpeg_entry(path);
  }
  if (extension == ".dng") {
    return read_dng_entry(path);
  }
  fail("Unsupported file type: " + path.string());
}

Image load_jpeg_image(const Fs::path &path) {
  std::lock_guard<std::mutex> lock(jpeg_mutex());
  FILE *file = std::fopen(path.string().c_str(), "rb");
  if (file == nullptr) {
    fail("Unable to open JPEG file: " + path.string());
  }

  jpeg_decompress_struct cinfo{};
  JpegErrorManager jerr{};
  cinfo.err = jpeg_std_error(&jerr.base);
  jerr.base.error_exit = jpeg_error_exit;

  if (setjmp(jerr.jump_buffer) != 0) {
    jpeg_destroy_decompress(&cinfo);
    std::fclose(file);
    fail("Failed to decode JPEG file: " + path.string());
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, file);
  jpeg_read_header(&cinfo, TRUE);
  cinfo.out_color_space = JCS_RGB;
  jpeg_start_decompress(&cinfo);

  Image image{};
  image.width = static_cast<int>(cinfo.output_width);
  image.height = static_cast<int>(cinfo.output_height);
  image.pixels.resize(static_cast<std::size_t>(image.width) *
                      static_cast<std::size_t>(image.height) * 3U);

  const std::size_t row_stride = static_cast<std::size_t>(image.width) * 3U;
  while (cinfo.output_scanline < cinfo.output_height) {
    auto *row = image.pixels.data() + row_stride * cinfo.output_scanline;
    JSAMPROW rows[] = {row};
    jpeg_read_scanlines(&cinfo, rows, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  std::fclose(file);
  return image;
}

Image load_dng_image(const Fs::path &path) {
  std::lock_guard<std::mutex> lock(libraw_mutex());
  LibRaw raw_processor;

  int result = raw_processor.open_file(path.string().c_str());
  if (result != LIBRAW_SUCCESS) {
    fail("Unable to open DNG file: " + path.string());
  }

  raw_processor.imgdata.params.use_camera_wb = 1;
  raw_processor.imgdata.params.output_bps = 8;
  raw_processor.imgdata.params.no_auto_bright = 1;
  raw_processor.imgdata.params.use_auto_wb = 0;
  raw_processor.imgdata.params.output_color = 1;

  result = raw_processor.unpack();
  if (result != LIBRAW_SUCCESS) {
    fail("Unable to unpack DNG file: " + path.string());
  }

  result = raw_processor.dcraw_process();
  if (result != LIBRAW_SUCCESS) {
    fail("Unable to process DNG file: " + path.string());
  }

  int error_code = LIBRAW_SUCCESS;
  libraw_processed_image_t *processed = raw_processor.dcraw_make_mem_image(&error_code);
  if (processed == nullptr || error_code != LIBRAW_SUCCESS) {
    fail("Unable to convert DNG file: " + path.string());
  }

  if (processed->type != LIBRAW_IMAGE_BITMAP || processed->colors != 3 || processed->bits != 8) {
    LibRaw::dcraw_clear_mem(processed);
    fail("Unsupported DNG conversion result for file: " + path.string());
  }

  Image image{};
  image.width = processed->width;
  image.height = processed->height;
  image.pixels.assign(processed->data, processed->data + processed->data_size);
  LibRaw::dcraw_clear_mem(processed);
  return image;
}

Image load_tiff_image(const Fs::path &path) {
  std::lock_guard<std::mutex> lock(tiff_mutex());
  TIFF *tiff = TIFFOpen(path.string().c_str(), "r");
  if (tiff == nullptr) {
    fail("Unable to open TIFF file: " + path.string());
  }

  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint16_t samples_per_pixel = 0;
  std::uint16_t bits_per_sample = 0;
  std::uint16_t planar_config = 0;
  std::uint16_t photometric = 0;

  if (TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) != 1 ||
      TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) != 1 ||
      TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel) != 1 ||
      TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bits_per_sample) != 1 ||
      TIFFGetField(tiff, TIFFTAG_PLANARCONFIG, &planar_config) != 1 ||
      TIFFGetField(tiff, TIFFTAG_PHOTOMETRIC, &photometric) != 1) {
    TIFFClose(tiff);
    fail("Unable to read TIFF metadata: " + path.string());
  }

  if (samples_per_pixel != 3 || bits_per_sample != 8 ||
      planar_config != PLANARCONFIG_CONTIG || photometric != PHOTOMETRIC_RGB) {
    TIFFClose(tiff);
    fail("Unsupported TIFF layout: " + path.string());
  }

  Image image{};
  image.width = static_cast<int>(width);
  image.height = static_cast<int>(height);
  image.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);

  for (std::uint32_t row = 0; row < height; ++row) {
    auto *scanline = image.pixels.data() +
                     static_cast<std::size_t>(row) * static_cast<std::size_t>(width) * 3U;
    if (TIFFReadScanline(tiff, scanline, row, 0) < 0) {
      TIFFClose(tiff);
      fail("Unable to read TIFF scanline: " + path.string());
    }
  }

  TIFFClose(tiff);
  return image;
}

Image load_image(const Fs::path &path) {
  const std::string extension = lowercase(path.extension().string());
  if (extension == ".jpg" || extension == ".jpeg") {
    return load_jpeg_image(path);
  }
  if (extension == ".dng") {
    return load_dng_image(path);
  }
  if (extension == ".tif" || extension == ".tiff") {
    return load_tiff_image(path);
  }
  fail("Unsupported file type: " + path.string());
}

void write_jpeg(const Fs::path &path, const Image &image, int quality) {
  std::lock_guard<std::mutex> lock(jpeg_mutex());
  FILE *file = std::fopen(path.string().c_str(), "wb");
  if (file == nullptr) {
    fail("Unable to open output file: " + path.string());
  }

  jpeg_compress_struct cinfo{};
  JpegErrorManager jerr{};
  cinfo.err = jpeg_std_error(&jerr.base);
  jerr.base.error_exit = jpeg_error_exit;

  if (setjmp(jerr.jump_buffer) != 0) {
    jpeg_destroy_compress(&cinfo);
    std::fclose(file);
    fail("Failed to write JPEG file: " + path.string());
  }

  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, file);

  cinfo.image_width = static_cast<JDIMENSION>(image.width);
  cinfo.image_height = static_cast<JDIMENSION>(image.height);
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);
  jpeg_start_compress(&cinfo, TRUE);

  const std::size_t row_stride = static_cast<std::size_t>(image.width) * 3U;
  while (cinfo.next_scanline < cinfo.image_height) {
    auto *row = const_cast<JSAMPLE *>(image.pixels.data() + row_stride * cinfo.next_scanline);
    JSAMPROW rows[] = {row};
    jpeg_write_scanlines(&cinfo, rows, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  std::fclose(file);
}

void write_tiff(const Fs::path &path, const Image &image) {
  std::lock_guard<std::mutex> lock(tiff_mutex());
  TIFF *tiff = TIFFOpen(path.string().c_str(), "w");
  if (tiff == nullptr) {
    fail("Unable to open output TIFF file: " + path.string());
  }

  if (TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, static_cast<std::uint32_t>(image.width)) != 1 ||
      TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, static_cast<std::uint32_t>(image.height)) != 1 ||
      TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 3) != 1 ||
      TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 8) != 1 ||
      TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT) != 1 ||
      TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) != 1 ||
      TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB) != 1 ||
      TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_LZW) != 1 ||
      TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP,
                   TIFFDefaultStripSize(tiff, static_cast<std::uint32_t>(image.width) * 3U)) != 1) {
    TIFFClose(tiff);
    fail("Failed to configure TIFF output: " + path.string());
  }

  for (int row = 0; row < image.height; ++row) {
    auto *scanline = const_cast<std::uint8_t *>(
        image.pixels.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(image.width) * 3U);
    if (TIFFWriteScanline(tiff, scanline, static_cast<std::uint32_t>(row), 0) < 0) {
      TIFFClose(tiff);
      fail("Failed to write TIFF scanline: " + path.string());
    }
  }

  TIFFClose(tiff);
}

void write_output(const Fs::path &path, const Image &image, int jpeg_quality) {
  const std::string extension = lowercase(path.extension().string());
  if (extension == ".jpg" || extension == ".jpeg") {
    write_jpeg(path, image, jpeg_quality);
    return;
  }
  if (extension == ".tif" || extension == ".tiff") {
    write_tiff(path, image);
    return;
  }
  fail("Unsupported output format. Use .jpg, .jpeg, .tif, or .tiff.");
}

std::vector<Fs::path> list_input_files(const Fs::path &input_dir) {
  if (!Fs::exists(input_dir)) {
    fail("Input directory does not exist: " + input_dir.string());
  }
  if (!Fs::is_directory(input_dir)) {
    fail("Input path is not a directory: " + input_dir.string());
  }

  std::vector<Fs::path> paths;
  for (const auto &entry : Fs::directory_iterator(input_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (!is_supported_input(entry.path())) {
      continue;
    }
    paths.push_back(entry.path());
  }

  if (paths.empty()) {
    fail("No supported input files found in directory: " + input_dir.string());
  }
  return paths;
}

template <typename T, typename Loader>
std::vector<T> parallel_load(std::vector<Fs::path> paths, std::size_t parallelism,
                             std::string_view progress_name, Loader loader) {
  std::vector<T> results(paths.size());
  (void)parallelism;
  for (std::size_t index = 0; index < paths.size(); ++index) {
    results[index] = loader(paths[index]);
    print_progress(progress_name, index + 1, paths.size());
  }
  return results;
}

template <typename Worker>
void parallel_for(std::size_t task_count,
                  std::size_t parallelism,
                  std::string_view progress_name,
                  Worker worker_fn) {
  if (task_count == 0) {
    return;
  }

  const std::size_t worker_count = std::max<std::size_t>(1, std::min(parallelism, task_count));
  if (worker_count == 1) {
    for (std::size_t index = 0; index < task_count; ++index) {
      worker_fn(index);
      print_progress(progress_name, index + 1, task_count);
    }
    return;
  }

  std::atomic<std::size_t> next_index{0};
  std::atomic<std::size_t> completed{0};
  std::mutex error_mutex;
  std::exception_ptr first_error;

  auto worker = [&]() {
    while (true) {
      const std::size_t index = next_index.fetch_add(1);
      if (index >= task_count) {
        break;
      }

      {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error) {
          break;
        }
      }

      try {
        worker_fn(index);
        const std::size_t current = completed.fetch_add(1) + 1;
        print_progress(progress_name, current, task_count);
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!first_error) {
          first_error = std::current_exception();
        }
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::size_t i = 0; i < worker_count; ++i) {
    workers.emplace_back(worker);
  }
  for (auto &worker_thread : workers) {
    worker_thread.join();
  }

  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

void validate_dimensions(const std::vector<InputEntry> &entries) {
  if (entries.empty()) {
    fail("No input images available after filtering.");
  }
  const int width = entries.front().width;
  const int height = entries.front().height;
  for (const auto &entry : entries) {
    if (entry.width != width || entry.height != height) {
      fail("All input images must have the same dimensions.");
    }
  }
}

std::vector<InputEntry> select_preview(std::vector<InputEntry> entries, std::size_t preview_count) {
  if (preview_count == 0 || preview_count >= entries.size()) {
    return entries;
  }
  if (preview_count == 1) {
    return {entries.front()};
  }

  std::vector<InputEntry> selected;
  selected.reserve(preview_count);
  std::size_t last_index = std::numeric_limits<std::size_t>::max();
  for (std::size_t i = 0; i < preview_count; ++i) {
    const double position = static_cast<double>(i) *
                            static_cast<double>(entries.size() - 1) /
                            static_cast<double>(preview_count - 1);
    std::size_t index = static_cast<std::size_t>(std::llround(position));
    if (index >= entries.size()) {
      index = entries.size() - 1;
    }
    if (index == last_index) {
      continue;
    }
    selected.push_back(entries[index]);
    last_index = index;
  }

  if (selected.back().path != entries.back().path) {
    selected.back() = entries.back();
  }
  return selected;
}

std::vector<InputEntry> gather_inputs(const Options &options) {
  std::vector<Fs::path> paths = list_input_files(options.input_dir);
  std::vector<InputEntry> entries =
      parallel_load<InputEntry>(std::move(paths), options.parallelism, "scanning inputs", read_input_entry);

  validate_dimensions(entries);

  auto by_name = [](const InputEntry &lhs, const InputEntry &rhs) {
    const int cmp = compare_natural(lhs.filename, rhs.filename);
    if (cmp != 0) {
      return cmp < 0;
    }
    return lhs.path.string() < rhs.path.string();
  };

  std::sort(entries.begin(), entries.end(), [&](const InputEntry &lhs, const InputEntry &rhs) {
    if (options.sort_mode == SortMode::kName) {
      return by_name(lhs, rhs);
    }
    if (lhs.capture_time && rhs.capture_time && lhs.capture_time != rhs.capture_time) {
      return *lhs.capture_time < *rhs.capture_time;
    }
    if (lhs.capture_time.has_value() != rhs.capture_time.has_value()) {
      return lhs.capture_time.has_value();
    }
    return by_name(lhs, rhs);
  });

  if (options.reverse) {
    std::reverse(entries.begin(), entries.end());
  }

  entries = select_preview(std::move(entries), options.preview_count);
  validate_dimensions(entries);
  return entries;
}

std::vector<ImageInfo> load_selected_images(const std::vector<InputEntry> &entries, std::size_t parallelism) {
  std::vector<Fs::path> paths;
  paths.reserve(entries.size());
  for (const auto &entry : entries) {
    paths.push_back(entry.path);
  }

  std::vector<Image> images =
      parallel_load<Image>(std::move(paths), parallelism, "loading images", load_image);

  std::vector<ImageInfo> result(entries.size());
  for (std::size_t i = 0; i < entries.size(); ++i) {
    result[i].entry = entries[i];
    result[i].image = std::move(images[i]);
  }
  return result;
}

phasecorr::Image to_phasecorr_image(const Image &image) {
  phasecorr::Image gray(image.height, image.width);
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      const std::size_t index =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) +
           static_cast<std::size_t>(x)) * 3U;
      const float r = static_cast<float>(image.pixels[index]);
      const float g = static_cast<float>(image.pixels[index + 1]);
      const float b = static_cast<float>(image.pixels[index + 2]);
      gray(y, x) = 0.299f * r + 0.587f * g + 0.114f * b;
    }
  }
  return gray;
}

double degrees_to_radians(double degrees) {
  return degrees * std::numbers::pi_v<double> / 180.0;
}

Eigen::Vector2d rotate_vector(const Eigen::Vector2d &value, double degrees) {
  const double radians = degrees_to_radians(degrees);
  const double cosine = std::cos(radians);
  const double sine = std::sin(radians);
  return Eigen::Vector2d(cosine * value.x() - sine * value.y(),
                         sine * value.x() + cosine * value.y());
}

Transform compose_transform(const Transform &lhs, const Transform &rhs) {
  const Eigen::Vector2d rhs_translation(rhs.dx, rhs.dy);
  const Eigen::Vector2d rotated = rotate_vector(rhs_translation, lhs.rotation_degrees);

  Transform result{};
  result.dx = lhs.dx + rotated.x();
  result.dy = lhs.dy + rotated.y();
  result.rotation_degrees = lhs.rotation_degrees + rhs.rotation_degrees;
  return result;
}

std::vector<Transform> accumulate_total_transforms(const std::vector<Transform> &relative_transforms) {
  std::vector<Transform> total_transforms(relative_transforms.size(), Transform{});
  for (std::size_t i = 1; i < relative_transforms.size(); ++i) {
    total_transforms[i] = compose_transform(total_transforms[i - 1], relative_transforms[i]);
  }
  return total_transforms;
}

float sample_phasecorr_image(const phasecorr::Image &image, double x, double y) {
  if (x < 0.0 || y < 0.0 || x > static_cast<double>(image.cols - 1) ||
      y > static_cast<double>(image.rows - 1)) {
    return 0.0f;
  }

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, image.cols - 1);
  const int y1 = std::min(y0 + 1, image.rows - 1);
  const double tx = x - static_cast<double>(x0);
  const double ty = y - static_cast<double>(y0);

  const double top = static_cast<double>(image(y0, x0)) * (1.0 - tx) +
                     static_cast<double>(image(y0, x1)) * tx;
  const double bottom = static_cast<double>(image(y1, x0)) * (1.0 - tx) +
                        static_cast<double>(image(y1, x1)) * tx;
  return static_cast<float>(top * (1.0 - ty) + bottom * ty);
}

phasecorr::Image rotate_phasecorr_image(const phasecorr::Image &image, double rotation_degrees) {
  phasecorr::Image rotated(image.rows, image.cols);
  const double cx = 0.5 * static_cast<double>(image.cols - 1);
  const double cy = 0.5 * static_cast<double>(image.rows - 1);
  const double radians = degrees_to_radians(rotation_degrees);
  const double cosine = std::cos(radians);
  const double sine = std::sin(radians);

  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      const double dx = static_cast<double>(x) - cx;
      const double dy = static_cast<double>(y) - cy;
      const double source_x = cosine * dx + sine * dy + cx;
      const double source_y = -sine * dx + cosine * dy + cy;
      rotated(y, x) = sample_phasecorr_image(image, source_x, source_y);
    }
  }

  return rotated;
}

Transform estimate_transform_phase_corr(const Image &reference, const Image &candidate) {
  const phasecorr::Image ref_gray = to_phasecorr_image(reference);
  const phasecorr::Image cand_gray = to_phasecorr_image(candidate);
  static constexpr std::array<double, 11> coarse_angles = {
      -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0};

  Transform best{};
  double best_peak = -std::numeric_limits<double>::infinity();
  double best_angle = 0.0;

  for (double angle : coarse_angles) {
    const phasecorr::Image rotated = rotate_phasecorr_image(cand_gray, angle);
    const phasecorr::PhaseCorrelationResult result = phasecorr::phaseCorrelation(ref_gray, rotated);
    if (result.peak > best_peak) {
      best_peak = result.peak;
      best_angle = angle;
      best.dx = result.shift.x();
      best.dy = result.shift.y();
      best.rotation_degrees = angle;
    }
  }

  for (int step = -5; step <= 5; ++step) {
    const double angle = best_angle + 0.2 * static_cast<double>(step);
    const phasecorr::Image rotated = rotate_phasecorr_image(cand_gray, angle);
    const phasecorr::PhaseCorrelationResult result = phasecorr::phaseCorrelation(ref_gray, rotated);
    if (result.peak > best_peak) {
      best_peak = result.peak;
      best.dx = result.shift.x();
      best.dy = result.shift.y();
      best.rotation_degrees = angle;
    }
  }

  return best;
}

AlignmentData compute_alignment_from_loaded_images(const std::vector<ImageInfo> &inputs,
                                                   std::size_t parallelism,
                                                   std::string_view operation) {
  AlignmentData alignment{};
  alignment.relative_transforms.assign(inputs.size(), Transform{});
  if (inputs.size() <= 1) {
    alignment.total_transforms.assign(inputs.size(), Transform{});
    return alignment;
  }

  const std::size_t task_count = inputs.size() - 1;
  const std::size_t worker_count = std::max<std::size_t>(1, std::min(parallelism, task_count));
  if (worker_count == 1) {
    for (std::size_t index = 1; index < inputs.size(); ++index) {
      const Transform delta = estimate_transform_phase_corr(inputs[index - 1].image, inputs[index].image);
      alignment.relative_transforms[index] = delta;
      print_alignment_progress(operation, index, task_count, delta);
    }
    alignment.total_transforms = accumulate_total_transforms(alignment.relative_transforms);
    return alignment;
  }

  std::atomic<std::size_t> next_index{1};
  std::mutex error_mutex;
  std::exception_ptr first_error;

  auto worker = [&]() {
    while (true) {
      const std::size_t index = next_index.fetch_add(1);
      if (index >= inputs.size()) {
        break;
      }

      {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error) {
          break;
        }
      }

      try {
        const Transform delta = estimate_transform_phase_corr(inputs[index - 1].image, inputs[index].image);
        alignment.relative_transforms[index] = delta;
        print_alignment_progress(operation, index, task_count, delta);
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!first_error) {
          first_error = std::current_exception();
        }
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::size_t i = 0; i < worker_count; ++i) {
    workers.emplace_back(worker);
  }
  for (auto &worker_thread : workers) {
    worker_thread.join();
  }

  if (first_error) {
    std::rethrow_exception(first_error);
  }

  alignment.total_transforms = accumulate_total_transforms(alignment.relative_transforms);
  return alignment;
}

std::array<Eigen::Vector2d, 4> transformed_corners(int width, int height, const Transform &transform) {
  const double cx = 0.5 * static_cast<double>(width - 1);
  const double cy = 0.5 * static_cast<double>(height - 1);
  const std::array<Eigen::Vector2d, 4> corners = {
      Eigen::Vector2d(0.0, 0.0),
      Eigen::Vector2d(static_cast<double>(width - 1), 0.0),
      Eigen::Vector2d(0.0, static_cast<double>(height - 1)),
      Eigen::Vector2d(static_cast<double>(width - 1), static_cast<double>(height - 1)),
  };

  std::array<Eigen::Vector2d, 4> transformed{};
  const double radians = degrees_to_radians(transform.rotation_degrees);
  const double cosine = std::cos(radians);
  const double sine = std::sin(radians);

  for (std::size_t i = 0; i < corners.size(); ++i) {
    const double dx = corners[i].x() - cx;
    const double dy = corners[i].y() - cy;
    transformed[i].x() = cosine * dx - sine * dy + cx + transform.dx;
    transformed[i].y() = sine * dx + cosine * dy + cy + transform.dy;
  }
  return transformed;
}

CropRect compute_crop_rect(int width, int height, const std::vector<Transform> &transforms) {
  double common_left = 0.0;
  double common_top = 0.0;
  double common_right = static_cast<double>(width - 1);
  double common_bottom = static_cast<double>(height - 1);

  for (const Transform &transform : transforms) {
    const auto corners = transformed_corners(width, height, transform);
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const Eigen::Vector2d &corner : corners) {
      min_x = std::min(min_x, corner.x());
      min_y = std::min(min_y, corner.y());
      max_x = std::max(max_x, corner.x());
      max_y = std::max(max_y, corner.y());
    }
    common_left = std::max(common_left, min_x);
    common_top = std::max(common_top, min_y);
    common_right = std::min(common_right, max_x);
    common_bottom = std::min(common_bottom, max_y);
  }

  CropRect crop{};
  crop.x = std::max(0, static_cast<int>(std::ceil(common_left)));
  crop.y = std::max(0, static_cast<int>(std::ceil(common_top)));
  crop.width = std::min(width, static_cast<int>(std::floor(common_right)) + 1) - crop.x;
  crop.height = std::min(height, static_cast<int>(std::floor(common_bottom)) + 1) - crop.y;
  if (crop.width <= 0 || crop.height <= 0) {
    fail("Global auto alignment found no common content.");
  }
  return crop;
}

std::array<double, 3> sample_rgb(const Image &image, double x, double y) {
  if (x < 0.0 || y < 0.0 || x > static_cast<double>(image.width - 1) ||
      y > static_cast<double>(image.height - 1)) {
    return {0.0, 0.0, 0.0};
  }

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, image.width - 1);
  const int y1 = std::min(y0 + 1, image.height - 1);
  const double tx = x - static_cast<double>(x0);
  const double ty = y - static_cast<double>(y0);

  std::array<double, 3> result{};
  for (int channel = 0; channel < 3; ++channel) {
    const std::size_t idx00 = (static_cast<std::size_t>(y0) * static_cast<std::size_t>(image.width) +
                               static_cast<std::size_t>(x0)) * 3U + static_cast<std::size_t>(channel);
    const std::size_t idx10 = (static_cast<std::size_t>(y0) * static_cast<std::size_t>(image.width) +
                               static_cast<std::size_t>(x1)) * 3U + static_cast<std::size_t>(channel);
    const std::size_t idx01 = (static_cast<std::size_t>(y1) * static_cast<std::size_t>(image.width) +
                               static_cast<std::size_t>(x0)) * 3U + static_cast<std::size_t>(channel);
    const std::size_t idx11 = (static_cast<std::size_t>(y1) * static_cast<std::size_t>(image.width) +
                               static_cast<std::size_t>(x1)) * 3U + static_cast<std::size_t>(channel);

    const double top = static_cast<double>(image.pixels[idx00]) * (1.0 - tx) +
                       static_cast<double>(image.pixels[idx10]) * tx;
    const double bottom = static_cast<double>(image.pixels[idx01]) * (1.0 - tx) +
                          static_cast<double>(image.pixels[idx11]) * tx;
    result[static_cast<std::size_t>(channel)] = top * (1.0 - ty) + bottom * ty;
  }
  return result;
}

Image apply_transform_with_padding(const Image &image, const Transform &transform) {
  Image result{};
  result.width = image.width;
  result.height = image.height;
  result.pixels.assign(static_cast<std::size_t>(image.width) *
                           static_cast<std::size_t>(image.height) * 3U,
                       0);
  const double cx = 0.5 * static_cast<double>(image.width - 1);
  const double cy = 0.5 * static_cast<double>(image.height - 1);
  const double radians = degrees_to_radians(transform.rotation_degrees);
  const double cosine = std::cos(radians);
  const double sine = std::sin(radians);

  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      const double dest_dx = static_cast<double>(x) - cx - transform.dx;
      const double dest_dy = static_cast<double>(y) - cy - transform.dy;
      const double source_x = cosine * dest_dx + sine * dest_dy + cx;
      const double source_y = -sine * dest_dx + cosine * dest_dy + cy;
      const auto rgb = sample_rgb(image, source_x, source_y);
      const std::size_t destination_index =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) +
           static_cast<std::size_t>(x)) * 3U;
      result.pixels[destination_index] = static_cast<std::uint8_t>(std::clamp(std::lround(rgb[0]), 0L, 255L));
      result.pixels[destination_index + 1] = static_cast<std::uint8_t>(std::clamp(std::lround(rgb[1]), 0L, 255L));
      result.pixels[destination_index + 2] = static_cast<std::uint8_t>(std::clamp(std::lround(rgb[2]), 0L, 255L));
    }
  }

  return result;
}

Image crop_image(const Image &image, int x0, int y0, int width, int height) {
  Image result{};
  result.width = width;
  result.height = height;
  result.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);

  for (int y = 0; y < height; ++y) {
    const std::size_t source_index =
        (static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(image.width) +
         static_cast<std::size_t>(x0)) * 3U;
    const std::size_t destination_index =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 3U;
    std::copy_n(image.pixels.data() + source_index, static_cast<std::size_t>(width) * 3U,
                result.pixels.data() + destination_index);
  }

  return result;
}

Image transform_image(const Image &image, const Transform &transform, bool global_crop, CropRect crop_rect) {
  Image transformed = apply_transform_with_padding(image, transform);
  if (!global_crop) {
    return transformed;
  }
  return crop_image(transformed, crop_rect.x, crop_rect.y, crop_rect.width, crop_rect.height);
}

Image extract_slice(const Image &image, std::size_t index, std::size_t count, SliceDirection direction) {
  if (direction == SliceDirection::kVertical) {
    const int x0 = static_cast<int>((index * static_cast<std::size_t>(image.width)) / count);
    const int x1 = static_cast<int>(((index + 1) * static_cast<std::size_t>(image.width)) / count);
    const int slice_width = x1 - x0;
    Image slice{};
    slice.width = slice_width;
    slice.height = image.height;
    slice.pixels.resize(static_cast<std::size_t>(slice_width) * static_cast<std::size_t>(image.height) * 3U);

    for (int y = 0; y < image.height; ++y) {
      const std::size_t source_index =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) +
           static_cast<std::size_t>(x0)) * 3U;
      const std::size_t destination_index =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(slice_width) * 3U;
      std::copy_n(image.pixels.data() + source_index, static_cast<std::size_t>(slice_width) * 3U,
                  slice.pixels.data() + destination_index);
    }
    return slice;
  }

  const int y0 = static_cast<int>((index * static_cast<std::size_t>(image.height)) / count);
  const int y1 = static_cast<int>(((index + 1) * static_cast<std::size_t>(image.height)) / count);
  const int slice_height = y1 - y0;
  Image slice{};
  slice.width = image.width;
  slice.height = slice_height;
  slice.pixels.resize(static_cast<std::size_t>(image.width) * static_cast<std::size_t>(slice_height) * 3U);

  for (int y = 0; y < slice_height; ++y) {
    const std::size_t source_index =
        (static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(image.width)) * 3U;
    const std::size_t destination_index =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) * 3U;
    std::copy_n(image.pixels.data() + source_index, static_cast<std::size_t>(image.width) * 3U,
                slice.pixels.data() + destination_index);
  }
  return slice;
}

Image compose_slices(const std::vector<Image> &images, SliceDirection direction) {
  const std::size_t count = images.size();
  const int width = images.front().width;
  const int height = images.front().height;

  Image result{};
  result.width = width;
  result.height = height;
  result.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);

  if (direction == SliceDirection::kVertical) {
    for (std::size_t index = 0; index < count; ++index) {
      const int x0 = static_cast<int>((index * static_cast<std::size_t>(width)) / count);
      const int x1 = static_cast<int>(((index + 1) * static_cast<std::size_t>(width)) / count);
      const int slice_width = x1 - x0;
      if (slice_width <= 0) {
        continue;
      }

      for (int y = 0; y < height; ++y) {
        const std::size_t source_index =
            (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x0)) * 3U;
        const std::size_t destination_index = source_index;
        std::copy_n(images[index].pixels.data() + source_index,
                    static_cast<std::size_t>(slice_width) * 3U,
                    result.pixels.data() + destination_index);
      }
    }
  } else {
    for (std::size_t index = 0; index < count; ++index) {
      const int y0 = static_cast<int>((index * static_cast<std::size_t>(height)) / count);
      const int y1 = static_cast<int>(((index + 1) * static_cast<std::size_t>(height)) / count);
      const int slice_height = y1 - y0;
      if (slice_height <= 0) {
        continue;
      }

      const std::size_t bytes = static_cast<std::size_t>(slice_height) *
                                static_cast<std::size_t>(width) * 3U;
      const std::size_t offset = static_cast<std::size_t>(y0) *
                                 static_cast<std::size_t>(width) * 3U;
      std::copy_n(images[index].pixels.data() + offset, bytes,
                  result.pixels.data() + offset);
    }
  }

  return result;
}

std::string escape_json(std::string_view input) {
  std::string result;
  for (const unsigned char ch : input) {
    switch (ch) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\b':
        result += "\\b";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        result += static_cast<char>(ch);
        break;
    }
  }
  return result;
}

void write_alignment_json(const Fs::path &path,
                          const std::vector<InputEntry> &entries,
                          const AlignmentData &alignment) {
  std::ofstream stream(path);
  if (!stream) {
    fail("Unable to write alignment JSON: " + path.string());
  }

  stream << "{\n";
  for (std::size_t i = 0; i < entries.size(); ++i) {
    stream << "  \"" << escape_json(entries[i].filename) << "\": {"
           << "\"relative\": {\"x\": " << alignment.relative_transforms[i].dx
           << ", \"y\": " << alignment.relative_transforms[i].dy
           << ", \"rotation_degrees\": " << alignment.relative_transforms[i].rotation_degrees
           << "}, "
           << "\"total\": {\"x\": " << alignment.total_transforms[i].dx
           << ", \"y\": " << alignment.total_transforms[i].dy
           << ", \"rotation_degrees\": " << alignment.total_transforms[i].rotation_degrees
           << "}"
           << "}";
    if (i + 1 != entries.size()) {
      stream << ",";
    }
    stream << "\n";
  }
  stream << "}\n";
}

Fs::path output_directory(const Fs::path &output_path) {
  if (output_path.has_parent_path()) {
    return output_path.parent_path();
  }
  return Fs::current_path();
}

Fs::path prepare_temp_directory(const Fs::path &output_path) {
  const Fs::path temp_dir = output_directory(output_path) / "temp";
  std::error_code ec;
  Fs::remove_all(temp_dir, ec);
  if (ec) {
    fail("Unable to clear temp directory: " + temp_dir.string());
  }
  Fs::create_directories(temp_dir, ec);
  if (ec) {
    fail("Unable to create temp directory: " + temp_dir.string());
  }
  return temp_dir;
}

Fs::path slice_path_for(const Fs::path &temp_dir, std::size_t index) {
  std::ostringstream name;
  name << "slice_" << std::setw(6) << std::setfill('0') << index << ".tiff";
  return temp_dir / name.str();
}

void dump_slices_no_alignment(const std::vector<InputEntry> &entries,
                              SliceDirection direction,
                              const Fs::path &temp_dir,
                              std::size_t parallelism) {
  (void)parallelism;
  parallel_for(entries.size(), 1, "dumping slices", [&](std::size_t i) {
    const Image image = load_image(entries[i].path);
    const Image slice = extract_slice(image, i, entries.size(), direction);
    write_tiff(slice_path_for(temp_dir, i), slice);
  });
}

void dump_transformed_slices(const std::vector<InputEntry> &entries,
                             const std::vector<Transform> &transforms,
                             SliceDirection direction,
                             bool global_crop,
                             CropRect crop_rect,
                             const Fs::path &temp_dir,
                             std::size_t parallelism) {
  (void)parallelism;
  parallel_for(entries.size(), 1, "dumping slices", [&](std::size_t i) {
    const Image image = load_image(entries[i].path);
    const Image transformed = transform_image(image, transforms[i], global_crop, crop_rect);
    const Image slice = extract_slice(transformed, i, entries.size(), direction);
    write_tiff(slice_path_for(temp_dir, i), slice);
  });
}

Image merge_slices_from_temp(const std::vector<InputEntry> &entries,
                             SliceDirection direction,
                             const Fs::path &temp_dir,
                             int output_width,
                             int output_height) {
  Image result{};
  result.width = output_width;
  result.height = output_height;
  result.pixels.assign(static_cast<std::size_t>(output_width) *
                           static_cast<std::size_t>(output_height) * 3U,
                       0);

  for (std::size_t i = 0; i < entries.size(); ++i) {
    print_progress("merging slices", i + 1, entries.size());
    const Image slice = load_tiff_image(slice_path_for(temp_dir, i));
    if (direction == SliceDirection::kVertical) {
      const int x0 = static_cast<int>((i * static_cast<std::size_t>(output_width)) / entries.size());
      for (int y = 0; y < output_height; ++y) {
        const std::size_t source_index =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(slice.width) * 3U;
        const std::size_t destination_index =
            (static_cast<std::size_t>(y) * static_cast<std::size_t>(output_width) +
             static_cast<std::size_t>(x0)) * 3U;
        std::copy_n(slice.pixels.data() + source_index, static_cast<std::size_t>(slice.width) * 3U,
                    result.pixels.data() + destination_index);
      }
    } else {
      const int y0 = static_cast<int>((i * static_cast<std::size_t>(output_height)) / entries.size());
      const std::size_t bytes = static_cast<std::size_t>(slice.height) *
                                static_cast<std::size_t>(slice.width) * 3U;
      const std::size_t destination_index =
          static_cast<std::size_t>(y0) * static_cast<std::size_t>(output_width) * 3U;
      std::copy_n(slice.pixels.data(), bytes, result.pixels.data() + destination_index);
    }
  }

  return result;
}

Image run_in_memory_pipeline(const Options &options, const std::vector<InputEntry> &entries) {
  std::vector<ImageInfo> inputs = load_selected_images(entries, options.parallelism);

  if (options.global_auto_align) {
    const AlignmentData alignment =
        compute_alignment_from_loaded_images(inputs, options.parallelism, "global alignment");
    const CropRect crop = compute_crop_rect(inputs.front().image.width, inputs.front().image.height,
                                            alignment.total_transforms);
    std::vector<Image> images;
    images.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      print_progress("applying global alignment", i + 1, inputs.size());
      images.push_back(transform_image(inputs[i].image, alignment.total_transforms[i], true, crop));
    }
    return compose_slices(images, options.direction);
  }

  if (options.auto_align) {
    const AlignmentData alignment =
        compute_alignment_from_loaded_images(inputs, options.parallelism, "auto alignment");
    std::vector<Image> images;
    images.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      print_progress("applying auto alignment", i + 1, inputs.size());
      images.push_back(transform_image(inputs[i].image, alignment.total_transforms[i], false, CropRect{}));
    }
    return compose_slices(images, options.direction);
  }

  std::vector<Image> images;
  images.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    print_progress("preparing images", i + 1, inputs.size());
    images.push_back(std::move(inputs[i].image));
  }
  return compose_slices(images, options.direction);
}

Image run_dump_pipeline(const Options &options, const std::vector<InputEntry> &entries) {
  const Fs::path temp_dir = prepare_temp_directory(options.output_path);
  const Fs::path alignment_path = temp_dir / "alignment.json";
  const int base_width = entries.front().width;
  const int base_height = entries.front().height;

  if (!options.auto_align && !options.global_auto_align) {
    dump_slices_no_alignment(entries, options.direction, temp_dir, options.parallelism);
    return merge_slices_from_temp(entries, options.direction, temp_dir, base_width, base_height);
  }

  if (options.auto_align) {
    std::vector<ImageInfo> inputs = load_selected_images(entries, options.parallelism);
    const AlignmentData alignment =
        compute_alignment_from_loaded_images(inputs, options.parallelism, "auto alignment");
    write_alignment_json(alignment_path, entries, alignment);
    dump_transformed_slices(entries, alignment.total_transforms, options.direction, false, CropRect{},
                            temp_dir, options.parallelism);
    return merge_slices_from_temp(entries, options.direction, temp_dir, base_width, base_height);
  }

  std::vector<ImageInfo> inputs = load_selected_images(entries, options.parallelism);
  const AlignmentData alignment =
      compute_alignment_from_loaded_images(inputs, options.parallelism, "global alignment");
  const CropRect crop = compute_crop_rect(base_width, base_height, alignment.total_transforms);
  write_alignment_json(alignment_path, entries, alignment);
  dump_transformed_slices(entries, alignment.total_transforms, options.direction, true, crop, temp_dir,
                          options.parallelism);
  return merge_slices_from_temp(entries, options.direction, temp_dir, crop.width, crop.height);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_args(argc, argv);
    const std::vector<InputEntry> entries = gather_inputs(options);
    const Image output = options.dump_intermediate
                             ? run_dump_pipeline(options, entries)
                             : run_in_memory_pipeline(options, entries);
    print_progress("writing output", 1, 1);
    write_output(options.output_path, output, options.jpeg_quality);
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }
}
