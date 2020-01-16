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

## Modelling Performance

Assume the following:

1. FPGA kernels are fully pipelined with an initiation interval of 1, meaning there are no pipeline stalls and every cycle provides the same amount of data through all the stages of the pipeline.

2. Batch processing : transposing a large number of matrices to compute the effective performance of a single matrix transposition.

3. These imply an overlap in computation and memory access. Consider matrix transposition to be realized using 3 stages:
    - Fetch data from global mem to fpga
    - Transpose data in fpga
    - Store the transposed data back to global memory

    These 3 stages completely overlap.

The above assumptions imply that performance can be measured by computing the time taken for a specific stage, considering all stages overlap; each stage large enough, due to batch processing, to compensate for negligible latencies.

Performance are measured using the following metrics:

1. Throughput : Amount of bits transposed per second.
2. Runtime
3. Bandwidth

Let's consider the following configurations:

1. Single Bank
2. Multiple Banks

### Single Bank

Performance can be modelled based on the effective time taken to fetch a single matrix from the global memory to the fpga since all the stages overlap as mentioned previously. Therefore, the data fetched from a single bank of the global memory by the FPGA should match the width of the pipeline of 512 bits.

Let's suppose the FPGA fetches data at maximum bandwidth from a single bank in global memory. Therefore,

    If N is the side of the matrix to be transposed and considering only square matrices of complex single precision floating points,

    size_matrix = N * N * 8     # 8 bytes for a complex sp 
    eff_runtime = size_matrix / max_bandwidth

Maximum bandwidth that can be obtained is:

    bus_width_bytes = 512 / 8

    memory_controller_frequency = 300 MHz

    max_bandwidth = bus_width * memory_controller_frequency

                  = 19200.0 MB/s = 19.2 GB/s

Throughput is nothing but the bandwidth, since it is the amount of data transposed per second:

    throughput =  max_bandwidth = 19.2GB/sec

![Estimation of runtime for matrix transposition of different sizes](common/singlebank_transpose_est.png)

Throughput of the matrices would be the maximum obtainable of 19.2 GB/sec.

#### Conclusion

The performance of the kernel will be **memory bandwidth bound**, as seen in the throughput estimation. The throughput of all the sizes estimated is *19.2 GB/sec*. This is because of the following reason:

Kernel runs at a frequency larger than the frequency that the data is brought in to the FPGA (memory controller frequency). Considering the memory controller frequency is fixed, enabling more data per cycle will be the way to increase the performance. This is complemented by the fact that wider pipelines can be created to consume more data to obtain higher performance.

### Multiple Banks

As discussed previously, providing more data per cycle will alleviate the memory bandwidth bottleneck of the kernel's performance. This can be done by using more banks to provide more data per cycle to the kernel.

If 4 banks feed 512 bits data each @300 Mhz, there is a total of 2048 bits available at an effective bandwidth of `19.2 GB/s * 4 = 76 GB/s`.

If the kernel continues to consume the same width (of say 512 bits per cycle), the kernel is **compute-bound** i.e, there is sufficient data available but requires more computation. One can nevertheless, see a better performance because the frequency of the kernel is higher than the memory controller and hence will have a higher throughput.

If the kernel uses 4 bank equivalent width, the performance obtained will scale by 4 because of 4 times the bandwidth obtained and the kernel will again be **memory bandwidth bound** with a throughput of *76 GB/sec*.

![Estimation of runtime for matrix transposition of different sizes using 4 banks](common/4bank_transpose_est.png)

## TODO

1. Explain why the bus width is 512 along with specifications of the hardware (bus, board)

## Modelling BRAM Usage
