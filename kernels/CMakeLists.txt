# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)

##
# Get FPGA Board Name from env variable or use default
# Set it to CMakeCache
##
if(NOT DEFINED FPGA_BOARD_NAME)
  if(DEFINED $ENV{FPGA_BOARD_NAME})
    set(FPGA_BOARD_NAME $ENV{FPGA_BOARD_NAME} CACHE STRING "Target Board")
  else()
    set(FPGA_BOARD_NAME p520_hpc_sg280l CACHE STRING "Target Board")
  endif()
endif()


# OpenCL kernel targets generation
## setup cmake variables to generate header file
set(LOGPOINTS 3 CACHE STRING "Log of per sample data points")
math(EXPR POINTS "1 << ${LOGPOINTS}")

set(LOGSIZE 6 CACHE STRING "Log of length of the square matrix")
math(EXPR SIZE "1 << ${LOGSIZE}")
math(EXPR DEPTH "1 << (${LOGSIZE} + ${LOGSIZE} - ${LOGPOINTS})")

message("-- Log of length of matrix is ${LOGSIZE}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/common/mtrans_config.h.in"
  "${CMAKE_BINARY_DIR}/kernels/common/mtrans_config.h"
)

## Flags for different target options
set(AOC_FLAGS "-g -v -fp-relaxed -no-interleaving=default" CACHE STRING "AOC compiler flags")
separate_arguments(AOC_FLAGS)
set(EMU_FLAGS "-legacy-emulator -march=emulator" CACHE STRING "AOC emulation flags")
separate_arguments(EMU_FLAGS)
set(REP_FLAGS "-report -rtl" CACHE STRING "AOC report flags")
separate_arguments(REP_FLAGS)
set(PROF_FLAGS "-profile=all" CACHE STRING "AOC profile flags")
separate_arguments(PROF_FLAGS)

## 
# Call function to create custom build commands
# Generates targets:
#   - ${kernel_name}_emu: to generate emulation binary
#   - ${kernel_name}_rep: to generate report
#   - ${kernel_name}_syn: to generate synthesis binary
##
include(${CMAKE_SOURCE_DIR}/cmake/build_kernel.cmake)
set(kernels diagonal_bitrev diagonal simple_bitrev simple matrixTranspose)

if (INTELFPGAOPENCL_FOUND)
  build_mTranspose(${kernels})
endif()
