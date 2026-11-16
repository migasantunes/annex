cmake_minimum_required(VERSION 3.24)

project(annex VERSION 0.1.0 LANGUAGES CXX)

# --- Global settings ----------------------------------------------------------

# Only set the standard when we are the top-level project, so a parent project
# embedding annex keeps control of its own settings. annex itself still
# requires C++20 through target_compile_features below.
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# A single-config generator with no build type produces an unoptimised build
# with no -O flag at all, which silently ruins benchmark numbers. Multi-config
# generators (Visual Studio, Ninja Multi-Config) choose the type at build time.
get_property(ANNEX_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT ANNEX_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  message(STATUS "annex: no build type given, defaulting to Release")
endif()

# --- Options --------------------------------------------------------------------

option(ANNEX_BUILD_TESTS "Build the annex unit tests" ${PROJECT_IS_TOP_LEVEL})
option(ANNEX_BUILD_BENCH "Build the annex micro-benchmarks" OFF)
option(ANNEX_BUILD_PYTHON "Build the annex Python extension module" OFF)
option(ANNEX_WERROR "Treat compiler warnings as errors" OFF)
option(ANNEX_NATIVE "Compile for the host CPU (-march=native); binaries are not portable" OFF)
set(ANNEX_SANITIZE "" CACHE STRING "Sanitizer: empty, 'address' (ASan+UBSan) or 'thread'")
set_property(CACHE ANNEX_SANITIZE PROPERTY STRINGS "" address thread)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(AnnexWarnings)
include(AnnexSanitizers)

# Sanitizers must instrument every object in the process (including GoogleTest
# and the Python extension), otherwise ASan container-overflow checks and TSan
# both report false positives. So they are applied globally, unlike warnings.
annex_enable_sanitizers("${ANNEX_SANITIZE}")

# --- Library --------------------------------------------------------------------

add_library(annex STATIC
  src/distance/scalar.cpp
  src/distance/dispatch.cpp
  # M3: src/distance/avx2.cpp goes here, with its own -mavx2 -mfma
  # set_source_files_properties() so only that file assumes AVX2.
  src/flat_index.cpp
)
add_library(annex::annex ALIAS annex)

target_include_directories(annex PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(annex PUBLIC cxx_std_20)

# The static library is linked into the Python extension, which is a shared
# object, so its code must be position-independent.
set_target_properties(annex PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)

annex_set_warnings(annex)

if(ANNEX_NATIVE)
  # PUBLIC so headers inlined into tests/benchmarks see the same ISA.
  target_compile_options(annex PUBLIC -march=native)
endif()

# --- Subprojects ----------------------------------------------------------------

if(ANNEX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(ANNEX_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(ANNEX_BUILD_PYTHON)
  add_subdirectory(python)
endif()
