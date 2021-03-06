# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)

##
# Generate host executable that is required to call OpenCL kernel bitstreams
# Target: host
##

add_executable(host
  src/main.c
  src/transpose_fpga.c
  src/opencl_utils.c
  src/helper.c)

target_compile_options(host
  PRIVATE -Wall -Werror)

if(USE_DEBUG)
  target_compile_definitions(host PRIVATE DEBUG)
endif()

target_include_directories(host
  PRIVATE include
          "${IntelFPGAOpenCL_INCLUDE_DIRS}"
          "${CMAKE_SOURCE_DIR}/extern/argparse")

target_link_libraries(host
  PRIVATE "${IntelFPGAOpenCL_LIBRARIES}" argparse m)
