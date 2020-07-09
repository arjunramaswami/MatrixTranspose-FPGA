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
message("Selecting FPGA Board: ${FPGA_BOARD_NAME}")

## 
# Call function to create custom build commands
# Generates targets:
#   - ${kernel_name}_emu: to generate emulation binary
#   - ${kernel_name}_rep: to generate report
#   - ${kernel_name}_syn: to generate synthesis binary
##
include(${CMAKE_SOURCE_DIR}/cmake/build_kernel.cmake)
set(kernels diagonal_bitrev diagonal simple_bitrev)

if (INTELFPGAOPENCL_FOUND)
  build_mTranspose(${kernels})
endif()
