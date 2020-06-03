# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)

project(fpga-matrixTranspose VERSION 1.0.0 
  DESCRIPTION "Matrix Transpose for FPGA"
  LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Setup submodules that are required
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

# Include hlslib in CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/extern/hlslib/cmake)

# Find hlslib Intel OpenCL kernels
find_package(IntelFPGAOpenCL REQUIRED)

# OpenCL kernel targets generation
## setup cmake variables to generate header file
set(LOG_POINTS 3 CACHE STRING "Log of per sample data points")
math(EXPR POINTS "1 << ${LOG_POINTS}")
set(LOG_SIZE 6 CACHE STRING "Log of length of the square matrix")
math(EXPR SIZE "1 << ${LOG_SIZE}")
message("-- Log of length of matrix is ${LOG_SIZE}")

configure_file(
  "${CMAKE_SOURCE_DIR}/kernels/mtrans_config.h.in"
  "${CMAKE_BINARY_DIR}/kernels/mtrans_config.h"
)

# OpenCL host target generation
add_subdirectory(api/)

if (INTELFPGAOPENCL_FOUND)
  add_subdirectory(kernels)
else()
  message(WARNING, "Intel FPGA OpenCL SDK not found!")
endif()
