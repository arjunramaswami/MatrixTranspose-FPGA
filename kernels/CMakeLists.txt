# Arjun Ramaswami

cmake_minimum_required(VERSION 3.10)

##
# Get FPGA Board Name from env variable or use default
# Set it to CMakeCache
##
set(BOARD_NAME $ENV{FPGA_BOARD_NAME})
if(BOARD_NAME)
  set(FPGA_BOARD_NAME ${BOARD_NAME} CACHE STRING "Target Board")
else()
  set(FPGA_BOARD_NAME p520_hpc_sg280l CACHE STRING "Target Board")
endif()

## Flags for different target options
set(AOC_FLAGS "-g -v -fpc -fp-relaxed -cl-single-precision-constant -no-interleaving=default" CACHE STRING "AOC compiler flags")
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
set(kernels swi_bitrev_diag)

if (INTELFPGAOPENCL_FOUND)
  build_mTranspose(${kernels})
endif()