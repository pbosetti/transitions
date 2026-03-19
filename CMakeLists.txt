cmake_minimum_required(VERSION 3.16)

project(transitions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBJPEG REQUIRED IMPORTED_TARGET libjpeg)
pkg_check_modules(LIBRAW REQUIRED IMPORTED_TARGET libraw_r)
pkg_check_modules(LIBTIFF REQUIRED IMPORTED_TARGET libtiff-4)
pkg_check_modules(NLOHMANN_JSON REQUIRED IMPORTED_TARGET nlohmann_json)

# Use system-installed FFTW3 (single precision) and Eigen3 rather than
# downloading them via FetchContent.  On macOS install with:
#   brew install fftw eigen
# On Debian/Ubuntu:
#   apt-get install libfftw3-dev libeigen3-dev
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
find_package(Eigen3 REQUIRED)

add_executable(transitions
  src/main.cpp
  src/phase_corr.cpp
)

target_compile_options(transitions PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

target_link_libraries(transitions PRIVATE
  PkgConfig::LIBJPEG
  PkgConfig::LIBRAW
  PkgConfig::LIBTIFF
  PkgConfig::NLOHMANN_JSON
  Eigen3::Eigen
  PkgConfig::FFTW3F
)

# Approach 1: parallelized existing architecture
add_executable(transitions_parallel
  src/main_parallel.cpp
  src/phase_corr.cpp
)

target_compile_options(transitions_parallel PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

target_link_libraries(transitions_parallel PRIVATE
  PkgConfig::LIBJPEG
  PkgConfig::LIBRAW
  PkgConfig::LIBTIFF
  PkgConfig::NLOHMANN_JSON
  Eigen3::Eigen
  PkgConfig::FFTW3F
)

# Approach 2: concurrent pipeline architecture
add_executable(transitions_pipeline
  src/main_pipeline.cpp
  src/phase_corr.cpp
)

target_compile_options(transitions_pipeline PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

target_link_libraries(transitions_pipeline PRIVATE
  PkgConfig::LIBJPEG
  PkgConfig::LIBRAW
  PkgConfig::LIBTIFF
  PkgConfig::NLOHMANN_JSON
  Eigen3::Eigen
  PkgConfig::FFTW3F
)

add_executable(test
  src/main/test.cpp
  src/phase_corr.cpp
)
target_link_libraries(test PRIVATE
  Eigen3::Eigen
  PkgConfig::FFTW3F
)
