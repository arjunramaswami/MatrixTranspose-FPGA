# Matrix Transposition

This repository consists of several experiments to optimize the area for matrix
transposition of 2d square matrices. The aim is to create an efficient in-situ
transposition logic.

Each experiment has a distinct branch:

### Master
Simple 2d matrix transposition that uses 8 replications and 3 private copies of
an NxN matrix

### Banked Transpose ND Range Kernel
Banking an NxN matrix into N banks of N elements each. Implemented using an 
ND range kernel. Uses 3 private copies of the NxN matrix.

### Banked Transpose Single Work Item Kernel
Banking an NxN matrix into N banks of N elements each. Implemented using an 
SingleWorkItem Kernel. Uses 4 private copies of the NxN matrix.

### Diagonal Matrix Transposition
2d transposition that stores data in diagonals of an NxN matrix, N banks of N 
elements each, banked row-wise. This creates an N wide read port and a 1-wide
write port. 

A diagram of the transposition is shown in the *transpose.cl* file in the 
branch. 

*Issues*:
- Couldn't bank column-wise that may lead to 1 wide read and write ports
- Need to reshuffle the row read from the buffer. 

#### Problems with Reshuffling 
- Reshuffling when writing into the 2d buffer spoils diagonalization
- Reshuffling when reading from the 2d buffer directly causes arbitration because the memory read pattern is not constant.
- Reshuffling by writing into a temporary buffer of N depth creates several errors:
  - Gets optimized out if written in the same loop as the write-channel
  - When written in a separate loop requires waiting for the N data items to output correct data
  - Reordering using a circular shift register requires skipping differing number of cycles based on the row, before outputting relevant data
- Reshuffling by reading and writing back to the same buffer, requires another BARRIER.

### Eklund Method using Shift registers

### Read Before Write 
Extension to the diagonal matrix transposition method that is still in progress.

## Modelling 2D Matrix Transposition


### Modelling BRAM Usage

### Modelling Performance

Best case realistic performance that can be obtained. 

Performance measured using the following metrics:

1. Throughput
2. Runtime
3. Bandwidth

Configurations used:

1. Single Bank
2. Interleaved Multiple Banks

#### Single Bank

Let's suppose the FPGA fetches data at maximum bandwidth from a single bank in global memory.

Maximum bandwidth that can be obtained is:

    bus_width_bytes = 512 / 8

    memory_controller_frequency = 300 MHz

    max_bandwidth = bus_width * memory_controller_frequency

                  = 19200.0 MB/s = 18.75 GB/s

Let's assume the following:

1. FPGA kernels are fully pipelined with an initiation interval of 1, meaning there are no pipeline stalls and every cycle provides data of a particular width through all stages of the pipeline.

2. Batch processing : transposing a large number of matrices to compute the effective performance of a single matrix transposition.

3. This implies an overlap in computation and memory access. Consider matrix transposition can be divided into 3 stages:
    - Fetch data from global mem to fpga
    - Transpose data in fpga
    - Store the transposed data back to global memory

    These 3 stages completely overlap.

The above assumptions imply that performance can be measured by computing the time taken for a specific stage considering this overlaps with other stages; each stage being large enough due to batch processing to compensate for negligible latencies.

Therefore, performance is modelled based on the effective time taken to fetch a single matrix from the global memory to the fpga.

    If N is the side of the matrix to be transposed and considering only square matrices,

    matrix_size_bytes = N * N / 8

    runtime = matrix_size_bytes / max_bandwidth

Throughput is nothing but the bandwidth, since it is the amount of data transposed per second:

    throughput =  max_bandwidth 

#### Interleaved Multiple Bank

If 4 banks feed 512 bits data @300 Mhz, there is a total of 2048 bits available per bank at an effective bandwidth of `18.75 GB/s * 4 = 75 GB/s`.

Therefore, the obtainable metrics would also scale by 4.

Runtime obtained
Throughput obtained

### TODO

1. Explain why the bus width is 512 along with specifications of the hardware (bus, board)
