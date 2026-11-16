# Python extension module annex._core.
#
# Reached from the root CMakeLists.txt when ANNEX_BUILD_PYTHON=ON. Normally
# driven by scikit-build-core (pip install .), which passes Python_EXECUTABLE
# and pybind11_DIR; a plain CMake build must pass them itself.

# Development.Module (not Development) because an extension module must not
# link libpython: the interpreter that imports it already provides the symbols,
# and manylinux images do not even ship libpython.so.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core src/bindings.cpp)

# PRIVATE: nothing links against the extension, so the core's usage
# requirements stop here.
target_link_libraries(_core PRIVATE annex::annex)

# The same strict warnings as the core, so the bindings are held to the same
# standard. -Wold-style-cast etc. fire only on our code: pybind11's headers come
# in as SYSTEM includes.
annex_set_warnings(_core)

# The version string lives in one place, project() in the root CMakeLists.txt,
# and reaches Python through this define.
target_compile_definitions(_core PRIVATE ANNEX_VERSION="${PROJECT_VERSION}")

# scikit-build-core installs into the wheel root, so this puts _core next to
# python/annex/__init__.py, i.e. it becomes annex._core.
install(TARGETS _core LIBRARY DESTINATION annex)
