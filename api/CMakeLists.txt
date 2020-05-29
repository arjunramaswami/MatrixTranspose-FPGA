cmake_minimum_required(VERSION 3.10)

add_executable(host_mTranspose 
  src/main.c
  src/transpose_fpga.c
  src/opencl_utils.c
  common/helper.c
  ${CMAKE_SOURCE_DIR}/extern/argparse/argparse.c)   

target_compile_options(host_mTranspose
  PUBLIC -Wall -Werror)
  
target_include_directories(host_mTranspose
  PRIVATE include
  PUBLIC  common 
          ${IntelFPGAOpenCL_INCLUDE_DIRS}
          ${CMAKE_SOURCE_DIR}/extern/argparse)

target_link_libraries(host_mTranspose
  PUBLIC ${IntelFPGAOpenCL_LIBRARIES} m)