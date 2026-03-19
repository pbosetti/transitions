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
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(EIGEN3 REQUIRED IMPORTED_TARGET eigen3)

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
  PkgConfig::EIGEN3
  PkgConfig::FFTW3F
)

add_executable(transitions_v1
  src/main_v1.cpp
  src/phase_corr.cpp
)

target_compile_options(transitions_v1 PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

target_link_libraries(transitions_v1 PRIVATE
  PkgConfig::LIBJPEG
  PkgConfig::LIBRAW
  PkgConfig::LIBTIFF
  PkgConfig::NLOHMANN_JSON
  PkgConfig::EIGEN3
  PkgConfig::FFTW3F
)

add_executable(test
  src/main/test.cpp
  src/phase_corr.cpp
)
target_link_libraries(test PRIVATE
  PkgConfig::EIGEN3
  PkgConfig::FFTW3F
)
