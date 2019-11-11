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