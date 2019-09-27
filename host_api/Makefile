# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Compiles Host Code used to run FFT3d kernels 

# Check if Intel OpenCL SDK is installed and path set
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# Variables
DEBUG := 0

# Compilers
CC := gcc

# Compiler Flags
CFLAGS := -march=native -fno-omit-frame-pointer -ggdb3 -lm
# debug flag enables heap memory leak sanitization along with prints
ifeq ($(DEBUG),1)
	CFLAGS := $(CFLAGS) -O1 -DDEBUG
else
	CFLAGS := $(CFLAGS) -O3 -funroll-loops
endif
SRCS := $(wildcard main.c common/*.c api/*.c)
FPFLAGS := $(FPFLAGS)
CFLAGS := $(CFLAGS) $(SRCS) $(FPFLAGS)

# Include Directories 
AOCL_HEADER_DIRS := $(shell aocl compile-config)
INC := -Icommon -Iapi 
INC_DIRS := $(AOCL_HEADER_DIRS) $(INC)

# Linker Directories
AOCL_LINKER_DIRS = $(shell aocl ldflags) 
LIB_DIRS :=  $(AOCL_LINKER_DIRS)

# Libraries
AOCL_LIBS = $(shell aocl ldlibs)
LIBS := -lrt -lpthread
LIBS := $(LIBS) $(AOCL_LIBS)

# Target
TARGETDIR := .
TARGET := host

all : info
	mkdir -p $(TARGETDIR)
	$(CC) $(CFLAGS) $(INC_DIRS) $(LIB_DIRS)	$(LIBS) -o $(TARGETDIR)/$(TARGET)

clean : 
	rm -f $(TARGETDIR)/$(TARGET)

info :
	$(info -------------------------------)

.PHONY : all clean # ---------------