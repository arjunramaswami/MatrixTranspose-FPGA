# Arjun Ramaswami

##
# Creates Custom build targets for kernels passed as argument
# Targets:
#   - ${kernel_name}_emu: to generate emulation binary
#   - ${kernel_name}_rep: to generate report
#   - ${kernel_name}_syn: to generate synthesis binary
## 
## 
function(build_mTranspose)

  foreach(kernel_fname ${ARGN})

    set(CL_SRC "${CMAKE_SOURCE_DIR}/kernels/${kernel_fname}.cl")
    set(CL_INCLUDES "-I${CMAKE_BINARY_DIR}/kernels/mtrans_config.h")
    set(CL_HEADER "${CMAKE_BINARY_DIR}/kernels/mtrans_config.h")
    set(CL_INCL_DIR "-I${CMAKE_BINARY_DIR}/kernels" "-I${CMAKE_SOURCE_DIR}/kernels")

    set(EMU_BSTREAM 
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SIZE}pt_${kernel_fname}_emu.aocx")
    set(REP_BSTREAM 
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SIZE}pt_${kernel_fname}_rep.aocr")
    set(SYN_BSTREAM 
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SIZE}pt_${kernel_fname}.aocx")

    # Emulation Target
    add_custom_command(OUTPUT ${EMU_BSTREAM}
      COMMAND ${IntelFPGAOpenCL_AOC} ${CL_SRC} ${CL_INCL_DIR} ${AOC_FLAGS} ${EMU_FLAGS} -board=${FPGA_BOARD_NAME} -o ${EMU_BSTREAM}
      MAIN_DEPENDENCY ${CL_SRC}
      VERBATIM
    )
    
    add_custom_target(${kernel_fname}_emu
      DEPENDS ${EMU_BSTREAM} ${CL_SRC} ${CL_HEADER}
      COMMENT 
        "Building ${kernel_fname} for emulation to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )

    # Report Generation
    add_custom_command(OUTPUT ${REP_BSTREAM}
      COMMAND ${IntelFPGAOpenCL_AOC} ${CL_SRC} ${CL_INCL_DIR} ${AOC_FLAGS} ${REP_FLAGS} -board=${FPGA_BOARD_NAME} -o ${REP_BSTREAM}
      MAIN_DEPENDENCY ${CL_SRC} 
      VERBATIM
    )
    
    add_custom_target(${kernel_fname}_rep
      DEPENDS ${REP_BSTREAM} ${CL_SRC} ${CL_HEADER}
      COMMENT 
        "Building a report for ${kernel_fname} to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )

    # Synthesis Target
    add_custom_command(OUTPUT ${SYN_BSTREAM}
      COMMAND ${IntelFPGAOpenCL_AOC} ${CL_SRC} ${CL_INCL_DIR} ${AOC_FLAGS}  -board=${FPGA_BOARD_NAME} -o ${SYN_BSTREAM}
      MAIN_DEPENDENCY ${CL_SRC}
      VERBATIM
    )
    
    add_custom_target(${kernel_fname}_syn
      DEPENDS ${SYN_BSTREAM} ${CL_SRC} ${CL_HEADER}
      COMMENT 
        "Building a report for ${kernel_fname} to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )

  endforeach(kernel_fname)

endfunction()