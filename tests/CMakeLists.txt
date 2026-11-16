include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
  GIT_SHALLOW TRUE
)
# Don't install GoogleTest alongside annex, and on MSVC link the CRT the same
# way annex does.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(annex_tests
  distance_test.cpp
  flat_index_test.cpp
)
target_link_libraries(annex_tests PRIVATE annex::annex GTest::gtest_main)
annex_set_warnings(annex_tests)

include(GoogleTest)
# DISCOVERY_MODE PRE_TEST lists tests when ctest runs rather than at build
# time, so a sanitizer-instrumented binary isn't executed during the build.
# NO_PRETTY_VALUES keeps raw parameter bytes out of the ctest test names.
gtest_discover_tests(annex_tests DISCOVERY_MODE PRE_TEST NO_PRETTY_VALUES)
