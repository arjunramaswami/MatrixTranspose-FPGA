cmake_minimum_required(VERSION 3.10)

## TODO: enumerate options to choose from
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
set(PROF_FLAGS "-profile=all" CACHE STRING "AOC profile flags")
separate_arguments(REP_FLAGS)
separate_arguments(PROF_FLAGS)

# OpenCL and related files
set(CL_FILES 
    "${CMAKE_SOURCE_DIR}/kernels/diag_SWI_bitrev.cl")
set(CL_INCLUDES "-I${CMAKE_SOURCE_DIR}/kernels/config.h")

# Command Output Names
set(bst_name ${SIZE}pt_mTranspose)
set(bitstream_emulate ${bst_name}_emulate.aocx)

add_custom_command(
  OUTPUT ${bitstream_emulate}
  COMMAND 
    ${IntelFPGAOpenCL_AOC} ${CL_FILES} ${CL_INCLUDES} ${AOC_FLAGS} ${EMU_FLAGS} -board=${FPGA_BOARD_NAME} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_emulate}
  MAIN_DEPENDENCY ${CL_FILES}
  COMMENT 
    "Building Matrix Transpose for emulation to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  mTranspose_emulate DEPENDS ${bitstream_emulate}
)