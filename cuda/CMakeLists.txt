cmake_minimum_required(VERSION 3.20)
project(tessera_mps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Default to an optimised build if the caller didn't pick one (don't force /O2
# by hand — that conflicts with the debug runtime-check flag /RTC1).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  add_compile_options(/EHsc)
else()
  add_compile_options(-O3)
endif()

# Self-contained MPS tensor-network annealing engine (CPU). No external deps.
# The CUDA acceleration layer (cuSOLVER batched SVD) is the next step on top of
# this validated core.
add_library(tessera_mps STATIC
  src/linalg.cpp
  src/mps.cpp
)
target_include_directories(tessera_mps PUBLIC include)

# Shared library exposing the C ABI (tessera_qa_anneal) for the Rust crate
# (--features cuda). Links the MPS core.
add_library(tessera_qa SHARED src/ffi.cpp)
target_link_libraries(tessera_qa PRIVATE tessera_mps)
set_target_properties(tessera_qa PROPERTIES OUTPUT_NAME tessera_qa WINDOWS_EXPORT_ALL_SYMBOLS ON)

# --- tests (ctest) ---
enable_testing()

add_executable(test_linalg tests/test_linalg.cpp)
target_link_libraries(test_linalg PRIVATE tessera_mps)
add_test(NAME linalg COMMAND test_linalg)

add_executable(test_mps tests/test_mps.cpp)
target_link_libraries(test_mps PRIVATE tessera_mps)
add_test(NAME mps COMMAND test_mps)
