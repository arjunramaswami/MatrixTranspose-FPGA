# Matrix Transposition

This repository consists of several experiments to optimize the area for matrix
transposition of 2d square matrices. The aim is to create an efficient in-situ
transposition logic.

## Performance Model

### Theoretical Performance Estimate

Modelling the best performance of a matrix transposition is to model the number of matrices that can be transposed in a given period of time. This can be estimated using **Throughput** i.e., the amount of bits transposed per second.

Assume that there are no stalls in the pipeline and every cycle provides the same amount of data through all the stages of the pipeline. This implies an overlap in computation and memory access stages. Given a large input matrix, the latency of execution and storing data back to memory overlap with fetching the data. Therefore, the amount of bits transposed per second depends on the amount of bits fetched by the FPGA from the global memory i.e. the maximum bandwidth available.

The maximum bandwidth that can be obtained from a single bank of memory is:

    bus_width_bytes = 512 / 8

    memory_controller_frequency = 300 MHz

    max_bandwidth = bus_width_bytes * memory_controller_frequency

                  = 19200.0 MB/s = 19.2 GB/s

    throughput =  max_bandwidth = 19.2GB/sec

The performance of the kernel is now **memory bandwidth bound**. The throughput of all the sizes estimated is *19.2 GB/sec*. This is because of the following reason:

Kernel runs at a frequency larger than the frequency that the data is brought in to the FPGA (memory controller frequency). Considering the memory controller frequency is fixed, enabling more data per cycle to the kernel will increase the performance. This is complemented by the fact that wider pipelines can also be created to consume more data. Thereby, utilizing more banks to provide more data per cycle is an option to overcome the bottleneck.

If 4 banks feed 512 bits data each @300 Mhz, there is a total of 2048 bits available at an effective bandwidth of `19.2 GB/s * 4 = 76 GB/s`.

If the kernel continues to consume the same width (of say 512 bits per cycle), the kernel is **compute-bound** i.e, there is sufficient data available but requires more computation. One can nevertheless, see a better performance because the frequency of the kernel is higher than the memory controller and hence will have a higher throughput.

If the kernel uses 4 bank equivalent width, the performance obtained will scale by 4 because of 4 times the bandwidth obtained and the kernel will again be **memory bandwidth bound** with a throughput of *76 GB/sec*.

## TODO

1. Explain why the bus width is 512 along with specifications of the hardware (bus, board)

## Modelling BRAM Usage

Block RAMs in Intel FPGAs are made up of units called M20k blocks, which are  *20480* bits of memory each. The total number of M20k blocks available in each FPGA varies, with the Stratix 10 GX2800 FPGA containing *11721* M20k blocks. This results in the total BRAM memory of *229 Mbits* or *28.6 MB*.

    total_bram = 11721 * 20480 / ((2 ** 20) * 8) # MB

Each BRAM block has a programmable word width of a maximum of *40 bits*. Decreasing the word width increases the number of words that can be stored in the M20k block.  However, with a word width of 40 bits, a total of *512 words* can be stored in a single M20k block. This is also called the **Word Depth**.  

    word_depth = 20480 / 40

The width and thereof, the depth are both programmable.

Therefore, the estimation of BRAM usage is dependent on:

1. Width of data accessed per cycle.
2. Depth.
3. Private Copies
4. Replications.
5. Number of banks of local memory that are needed to access non-adjacent data simultaneously.

### Width of data accessed

The width of the adjacent data that are accessed defines the minimum number of M20ks required per bank of memory. N adjacent complex single precision floats accessed per cycle would require a minimum bank of width *N * 64 bits*. This bank will therefore consist of a minimum of *ceil(N * 64 / 40)* M20k blocks.

    data_type = 64  # bits; complex sp float
    width = N * 64
    m20_width = 40  # bits
    min_num_m20k_reqd_per_bank = ceil(width / m20_width)

If *N = 8*, this is a minimum of *13* M20k blocks required.

### Bank Depth

Assuming the maximum word width of 40 bits, a depth of minimum of 512 words can be stored in a single M20k block. If there are more words to be stored, there is a proportional increase in the M20k blocks used.

    num_words = data_size / num_data_per_cycle 
    min_num_m20k_reqd_per_bank = min_num_m20k_reqd_per_bank + (num_words / bank_depth)

**Note**: In some cases, there is a reduction of word width to have a deeper M20k block. Needs further research!

### Private Copies

Private copies are used for simultaneous access of memory by multiple loop iterations in different basic blocks. In a basic block, only one thread or iteration can exist at one time. even with ii or 1, each loop's basic block can have larger latency.  within the same basic block not the same cycle. Since only a thread or iteration is scheduled per cycle in a pipeline. But the access can take multiple cycles to complete (read has a latency of 4 cycles), this requires private copies of memory to enable simultaneous memory accesses.  Multiple loop iterations involve read and or write operations to different or same addresses in memory. For example, read and write simultaneously could occur in a cycle due to two separate ports, multiple reads as well. Therefore, these private copies can be in the same M20k block memory of clock multiplexing. These are multiple private copies of memory found within each replication.

If the word depth is less than 512, private copies are added to it, else more m20k blocks are added for them.

### Replications

Replications of memory allows multiple addresses to local memory by the same loop iteration. In order to access memory in different addresses in the same clock cycle, one needs to replicate memory. This is different from private copy because private copies are simultaneous access of memory in a cycle by different loop iterations meaning same basic block in different cycles.
TODO: scenario for replications

    repl = 
    min_num_m20k_reqd_per_bank = min_num_m20k_reqd_per_bank * repl

### Number of Banks required

Let's assume a scenario, such as a trivial matrix transposition, where N adjacent data are written in a cycle that leads to a minimum number of M20k blocks used, as discussed previously. If N non-adjacent addresses have to be read in a single cycle, this would require data to be replicated into banks such that multiple addresses that are not adjacent can be accessed in a single cycle. This proportionally increases the number of M20ks used.

Each bank has access to multiple ports, which the replications and private copies share. Each bank has different data in same addresses therefore, they occupy completely different m20ks.

    min_num_m20k_used = min_num_m20k_reqd_per_bank * num_banks
                      = 13 * 8 banks = 104 M20ks 

(aside): diagonal matrix transpose stores 8 words obtained per cycle directly in 8 separate banks, thereby utilizing only  8 * 2 = 16 M20k blocks, only 0.13% of BRAM usage.

    data_width = 512
    if depth < 512 words:
        m20k_width = 40 
        if private copies can fit within the 512 words:
            ceil(data_width / 40) * replications * banks
        else
            ceil(data_width / 32) * private copies * replications * banks
    
    if depth > 512 words
        m20k_width = 32
        ceil(data_width / 32) * private copies * replications * banks

## Design

explain the diagonal transpose!

## Implementation
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
