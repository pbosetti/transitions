cmake_minimum_required(VERSION 3.16)

project(transitions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBJPEG REQUIRED IMPORTED_TARGET libjpeg)
pkg_check_modules(LIBRAW REQUIRED IMPORTED_TARGET libraw_r)
pkg_check_modules(LIBTIFF REQUIRED IMPORTED_TARGET libtiff-4)

add_executable(transitions
  src/main.cpp
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
)
