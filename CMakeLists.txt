# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)

project(fpga-matrixTranspose VERSION 1.0.0 
  DESCRIPTION "Matrix Transpose for FPGA"
  LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

option(USE_DEBUG "Debug Mode" OFF)

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

# Link argparse as static library
add_subdirectory(${CMAKE_SOURCE_DIR}/extern/argparse)

add_subdirectory(api/)
add_subdirectory(kernels)
