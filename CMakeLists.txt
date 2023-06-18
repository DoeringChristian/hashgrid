cmake_minimum_required(VERSION 3.15)

project(hashgrid DESCRIPTION "Hash Grid Implementation")

if(NOT SKBUILD) # if not using scikit-build, locate the python interpreter using find_package
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
endif()

# Locate pybind11
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
          "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
find_package(pybind11 CONFIG REQUIRED)

include(cmake-defaults/CMakeLists.txt)

# Find the drjit package
if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  # When cross-compiling on MacOS, we cannot run drjit, but rather need to pass the relevant
  # path into cmake
  set(drjit_DIR ${DRJIT_CMAKE_DIR})
else()
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import drjit;print(drjit.get_cmake_dir())" OUTPUT_VARIABLE drjit_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(${drjit_DIR})

find_package(drjit)
if (NOT ${drjit_FOUND})
  message(FATAL_ERROR "Dr.Jit not found. Please install Dr.Jit using \"pip install drjit\"")
endif()

include_directories(${drjit_INCLUDE_DIR})

pybind11_add_module(_hashgrid_core MODULE src/main.cpp src/hashgrid.cpp)
target_link_libraries(_hashgrid_core PUBLIC nanothread drjit-core)
target_compile_features(_hashgrid_core PRIVATE cxx_std_17)
target_include_directories(_hashgrid_core PUBLIC ext/drjit/include)
target_compile_definitions(_hashgrid_core PRIVATE VERSION_INFO=${PROJECT_VERSION_INFO})

install(TARGETS _hashgrid_core DESTINATION .)
