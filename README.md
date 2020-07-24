# Matrix Transposition

This repository contains OpenCL-based implementations of 2d matrix transposition for different scenarios. It also contains several experiments to understand ways to utilize BRAM in an FPGA.

## Folder Structure

Listed below:

- `api`    : host code required to setup and execute FPGA bitstreams
- `kernels`: several OpenCL matrix transpose kernels
- `extern` : external packages as submodules required to run the project
- `cmake`  : cmake modules used by the build system
- `scripts`: convenience slurm scripts
- `docs`   : describes models regarding performance and resource utilization
- `data`   : evaluation results and measurements

## Quick Build

```bash
mkdir build && cd build

cmake -DLOGSIZE=6 ..   // log of length of a side of the matrix

make            // build host, creates binary named `host`

make <kernelname>_emu     // emulation
make <kernelname>_rep     // report
make <kernelname>_syn     // synthesis

// Run emulation
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./host -n 64 -b 1 -p <path to aocx>

./host --help   // cmd line params
```

## Dependencies

- CMake >= 3.10
- Intel OpenCL FPGA SDK
- C Compiler with C11 support

Additional submodules used:

- [argparse](https://github.com/cofyc/argparse.git) for command line argument parsing
- [hlslib](https://github.com/definelicht/hlslib) for CMake Intel FPGA OpenCL find packages

## Runtime Parameters

```bash
Computing Matrix Transpose using FPGA

    -h, --help        show this help message and exit

Basic Options
    -n, --n=<int>     Length of Square Matrix
    -b, --b=<int>     Number of batched executions
    -p, --path=<str>  Path to bitstream
```

## Compile Definitions

- `LOGSIZE`: set the log of the length of the matrix. Example: `-DLOGSIZE=6`.
- `USE_DEBUG`: prints the fpga and cpu transpose outputs to compare.
  