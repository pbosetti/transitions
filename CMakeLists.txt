cmake_minimum_required(VERSION 3.16)

project(transitions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include(FetchContent)

set(ENABLE_FLOAT ON CACHE BOOL "Build single-precision FFTW" FORCE)

FetchContent_Declare(
  FFTW3
  URL https://www.fftw.org/fftw-3.3.10.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(FFTW3)

FetchContent_Declare(
  Eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(Eigen)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBJPEG REQUIRED IMPORTED_TARGET libjpeg)
pkg_check_modules(LIBRAW REQUIRED IMPORTED_TARGET libraw_r)
pkg_check_modules(LIBTIFF REQUIRED IMPORTED_TARGET libtiff-4)

add_executable(transitions
  src/main.cpp
  src/phase_corr.cpp
)

target_include_directories(transitions PRIVATE
  ${fftw3_SOURCE_DIR}/api
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
  Eigen3::Eigen
  fftw3f
)

add_executable(test
  src/main/test.cpp
  src/phase_corr.cpp
)
target_include_directories(test PRIVATE
  ${fftw3_SOURCE_DIR}/api
)
target_link_libraries(test PRIVATE
  Eigen3::Eigen
  fftw3f
)
