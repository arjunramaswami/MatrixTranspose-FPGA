# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Multi Step Compilation for Intel OpenCL Kernel Code

# Check if Intel OpenCL SDK is installed and path set
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# Variables
LOGN := 6
BOARD := p520_hpc_sg280l     # Stratix 10

# Compilers
AOC := aoc

# Compiler Flags
AOCFLAGS   := -g -v -D LOGN=$(LOGN) -global-ring -duplicate-ring -cl-single-precision-constant  -no-interleaving=default
FPFLAGS    :=
EMUFLAGS   := -march=emulator -legacy-emulator
SIMFLAGS   := -march=simulator 
REPORTFLAGS := -report -rtl
PROFFLAGS  := -fp-relaxed -profile=all
SYNFLAGS   := -fp-relaxed -seed=7

# debug flags are purposely appended to emuflags instead of aocflags 
# therefore, DEBUG mode only for emulation
# mostly used to check if the preprocessor flags enable correct code compilation
ifeq ($(DEBUG),1)
	EMUFLAGS := $(EMUFLAGS) -Werror -DDEBUG
else
	EMUFLAGS := $(EMUFLAGS)
endif	

# Libraries 

# Paths to targets
TARGET_PATH    := fpgabitstream/transpose
TARGET_DIR  := $(TARGET_PATH)/synthesis_sp
FPFLAGS := $(FPFLAGS) 

TARGET_EMU_DIR   ?= $(TARGET_DIR)/emulation/emu$(LOGN)
TARGET_SIM_DIR   ?= $(TARGET_DIR)/simulation/sim$(LOGN)
TARGET_REPORT_DIR ?= $(TARGET_DIR)/report/rep$(LOGN)
TARGET_PROF_DIR  ?= $(TARGET_DIR)/profile/prof$(LOGN)
TARGET_SYN_DIR   ?= $(TARGET_DIR)/synthesis/syn$(LOGN)
	
# File
KERNEL_SRCS := ./transpose.cl

# Targets
TARGET_KERNEL := transpose.aocx
TARGET_INTERMEDIATE := transpose.aocr

default: help 
	$(error No target specified)

help: 
	$(info ***************************************************)
	$(info Choose one of the steps to compile the transposition kernel)
	$(info ***************************************************)
	$(info emulate  =  Compile kernels for emulation)
	$(info syn      =  Synthesize kernels  )
	$(info report   =  Intermediate Report )
	$(info profile  =  Profile kernels  )
	$(info sim      =  Simulate kernels )
	$(info clean    =  deletes syn folders and files)
	$(info emuclean =  deletes emulation folders and files)
	$(info fullclean = deletes all files )
	$(info help     =  Prints help )
	$(info ***************************************************)
	$(info Command Line Flags with Make:					  )
	$(info ***************************************************)
	$(info LOGN=(4:5:6) = select the size of the matrix to synthesize, default is 6. )
	$(info DEBUG=1     = enables debug messages for emulation)
	$(info ***************************************************)

# Kernel Emulation
emulate: info
	mkdir -p $(TARGET_EMU_DIR)
	aoc $(FPFLAGS) $(AOCFLAGS) $(EMUFLAGS) -board=$(BOARD) $(KERNEL_SRCS) -o $(TARGET_EMU_DIR)/$(TARGET_KERNEL)
#	ioc64 $(FPFLAGS) $(AOCFLAGS) $(EMUFLAGS) $(KERNEL_SRCS) -o $(TARGET_EMU_DIR)/$(TARGET_KERNEL)
	

# kernel intermediate compilation for report
report: info
	mkdir -p $(TARGET_REPORT_DIR)
	aoc $(FPFLAGS) $(AOCFLAGS) $(REPORTFLAGS) -board=$(BOARD) $(KERNEL_SRCS) -o $(TARGET_REPORT_DIR)/$(TARGET_INTERMEDIATE)

# Kernel Simulation - Only single precision floating point ops
sim: info 
	mkdir -p $(TARGET_SIM_DIR)
	aoc -D __FPGA_SP $(AOCFLAGS) $(SIMFLAGS) -board=$(BOARD) $(KERNEL_SRCS) -o $(TARGET_SIM_DIR)/$(TARGET_KERNEL)

# Kernel Profiling - Only single precision floating point ops
profile: info 
	mkdir -p $(TARGET_PROF_DIR)
	aoc -D __FPGA_SP $(AOCFLAGS) $(PROFFLAGS) -board=$(BOARD) $(KERNEL_SRCS) -o $(TARGET_PROF_DIR)/$(TARGET_KERNEL) 
	#@echo aoc -D __FPGA_SP $(AOCFLAGS) $(PROFFLAGS) --board $(BOARD) $(KERNEL_SRCS) -o $(TARGET_PROF_DIR)/$(TARGET_KERNEL) > profile_$$(date +%Y%m%d%H%M%S).log 2>&1

# Kernel Synthesis - Only single precision floating point ops
syn: info 
	mkdir -p $(TARGET_SYN_DIR)
	aoc -D __FPGA_SP $(AOCFLAGS) $(SYNFLAGS) -board=$(BOARD) $(KERNEL_SRCS) -o $(TARGET_SYN_DIR)/$(TARGET_KERNEL) 
	#@echo aoc -D __FPGA_SP $(AOCFLAGS) $(SYNFLAGS) --board $(BOARD) $(KERNEL_SRCS) -o $(TARGET_SYN_DIR)/$(TARGET_KERNEL) > syn_$$(date +%Y%m%d%H%M%S).log 2>&1 
	#nohup aoc -fp-relaxed -g -v $(KERNEL_SRCS) -o $@ > test.log &

clean:
	rm -rf d$(TARGET_SYN_DIR)/
emuclean: 
	rm -rf $(TARGET_EMU_DIR)/
fullclean: 
	rm -rf $(TARGET_PATH)/

info: 
	$(info -------------------------------)
	$(info QUARTUS VERSION    : $(QUARTUS_VERSION))
	$(info BOARD              : $(BOARD))
	$(info LOG MATRIX SIZE    : $(LOGN))
	$(info -------------------------------)

.PHONY : clean emuclean fullclean syn profile sim inter emulate help default info # ---------------
