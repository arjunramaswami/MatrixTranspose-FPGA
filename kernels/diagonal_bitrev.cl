// Authors: Arjun Ramaswami, Tobias Kenter

/*
* This file performs the transpose of 2d square matrix based on the diagonal 
* transposition algorithm. 
* Inputs to transposition and outputs from transposition are in bit reversed *    order as required by the FFT kernels.
*
* Steps:
*  1. Host writes data in normal order to ddr
*  2. Fetch kernel reads from ddr, bitreverses and channels to transpose kernel
*  3. Transpose kernel bitreverses input back to normal order, rotates & stores *     in buffer
*       Then reads transposed values, rotates back and bitreverses as output
*  4. Store receives the bitreversed output from transpose,
*       bitreverses back to normal order and stores back to ddr
*/

#include "mtrans_config.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chanintrans[POINTS] __attribute__((depth(POINTS)));
channel float2 chanouttrans[POINTS] __attribute__((depth(POINTS)));

__attribute__((max_global_work_dim(0)))
kernel void fetch(global const volatile float2 * restrict src, int batch) {
  const unsigned N = (1 << LOGN);
  const unsigned STEPS = (1 << (LOGN - LOGPOINTS)); // N / 8

  for(unsigned k = 0; k < (batch * N); k++){ 
    float2 buf[N];

    #pragma unroll POINTS
    for(unsigned i = 0; i < N; i++){
      buf[i & ((1<<LOGN)-1)] = src[(k << LOGN) + i];    
    }

    for(unsigned j = 0; j < STEPS; j++){
      write_channel_intel(chanintrans[0], buf[j]);               // 0
      write_channel_intel(chanintrans[1], buf[(4 * (N / 8)) + j]);   // 32
      write_channel_intel(chanintrans[2], buf[(2 * (N / 8)) + j]);   // 16
      write_channel_intel(chanintrans[3], buf[(6 * (N / 8)) + j]);   // 48
      write_channel_intel(chanintrans[4], buf[(N / 8) + j]);       // 8
      write_channel_intel(chanintrans[5], buf[(5 * (N / 8)) + j]);   // 40
      write_channel_intel(chanintrans[6], buf[(3 * (N / 8)) + j]);   // 24
      write_channel_intel(chanintrans[7], buf[(7 * (N / 8)) + j]);   // 54
    }
  }
}

__attribute__((max_global_work_dim(0)))
kernel void transpose(int batch) {
  const unsigned N = (1 << LOGN);
  const unsigned STEPS = (1 << (LOGN - LOGPOINTS)); // N / 8

  // for use in expression to avoid type errors
  const unsigned NUM_POINTS = POINTS; 

  // iterate over batches of 2d matrices
  for(unsigned k = 0 ; k < batch; k++){

    // Buffer with width - 8 points, depth - (N*N / 8), banked column-wise
    float2 buf[DEPTH][POINTS];
      
    // iterate within a 2d matrix
    for(unsigned row = 0; row < N; row++){

      // temp buffer to store N elements in bitreverse order
      float2 bitrev[N];

      /* LOGN bit bitreversal. 
      *  Example of LOGN=6, N=64:
      *   orig_index   ->  0  1  2  3  4  5  6  7  
      *   bitrev_index ->  0 32 16 48  8 40 24 54
      *  
      *  The next 8 points are stored with a stride of 1 i.e.
      *   bitrev_index ->  1 33 17 49  9 41 25 55
      *
      * Bitrev: avoiding arbitration
      *   - writes in bitreverse addresses
      *   - reads sequential addresses
      */
      for(unsigned j = 0; j < STEPS; j++){
        bitrev[j] = read_channel_intel(chanintrans[0]);               // 0
        bitrev[(4 * (N / 8))+ j] = read_channel_intel(chanintrans[1]);   // 32
        bitrev[(2 * (N / 8))+ j] = read_channel_intel(chanintrans[2]);   // 16
        bitrev[(6 * (N / 8))+ j] = read_channel_intel(chanintrans[3]);   // 48
        bitrev[(N / 8) + j] = read_channel_intel(chanintrans[4]);       // 8
        bitrev[(5 * (N / 8)) + j] = read_channel_intel(chanintrans[5]);   // 40
        bitrev[(3 * (N / 8)) + j] = read_channel_intel(chanintrans[6]);   // 24
        bitrev[(7 * (N / 8)) + j] = read_channel_intel(chanintrans[7]);   // 54
      }

      /* For each outer loop iteration, N data items are processed.
       * Considering BRAM is 512bits (8 points) wide,
       * rotations should wrap around.
       *
       * These N data items should reside in N/8 rows in buf.
       * Each of this N/8 rows are rotated by 1
       * 
       * Example, N = 64, matrix = 64x64
       *  --------------------------------
       *    0  1  2  3  4  5  6  7 
       *    8  9 10 11 12 13 14 15 
       *   16 17 18 .. 
       *   24 25 26 ..
       *   39 32 33 ..     <- rotated by 1 
       *   48 40 41 ..
       *   ...
       *   ...
       *   70 71 64 65 ..  <- rotated by 2
       *   78 79 80 81 .. 
       *   ...
       *  --------------------------------
       *
       * Therefore, every N / 8 rows are rotated by 1
       *  i.e. same as row 
       */

      // fill the POINTS wide row of the buffer each iteration
      // N/8 rows filled with the same rotation
      for(unsigned j = 0; j < STEPS; j++){

        // Temporary buffer of POINTS length to rotate before filling the matrix
        // rotate_in: avoiding arbitration
        //  - writes sequential addresses
        //  - reads rotated addresses
        float2 rotate_in[POINTS]__attribute__((memory("MLAB")));
        #pragma unroll POINTS 
        for(unsigned i = 0; i < POINTS; i++){
          rotate_in[i] = bitrev[(j * NUM_POINTS) + i];
        }

        // where: index of rotation, domain of values [0,7]
        // buf_row: row id
        #pragma unroll POINTS 
        for(unsigned i = 0; i < POINTS; i++){
          unsigned rot = (row & (NUM_POINTS - 1));
          unsigned where = ((i + NUM_POINTS) - rot) & (NUM_POINTS - 1);
          //unsigned buf_row = (row * (N / 8)) + j;
          unsigned buf_row = (row << (LOGN - LOGPOINTS)) + j;

          buf[buf_row][i] = rotate_in[where];
        }
      }
    } // row

    for(unsigned row = 0; row < N; row++){

      // rotate_out: N point buffer required to bitreverse
      // writes: sequential addresses
      // reads: rotation + bitreverse addresses
      float2 rotate_out[N];
      unsigned offset = 0;            

      #pragma unroll POINTS
      for(unsigned j = 0; j < N; j++){

        /* 
        *  row_rotate selects rows with strides of N / 8 
        *     j << (LOGN << LOGPOINTS) - N/8, 2N/8 ..
        * 
        *  Every j or N points, rotate
        *   : first N points:
        *     buf[0][0], buf[N/8][1], buf[2N/8][2] ..
        *
        *   : second N points:
        *     buf[N-1*N/8][0], buf[0][1], buf[N/8][2] ..
        *
        *  Therefore, (DEPTH + j - row)
        */
        unsigned rot = ((DEPTH + j - row) << (LOGN - LOGPOINTS)) & (DEPTH -1);

        /* 
        * offset: every 8 points, increase by 1 i.e., shift to the next row.
        *    0  1  2  3  4  5  6  7 
        *    8  9 10 11 12 13 14 15 <- offset
        */
        unsigned offset = (row >> LOGPOINTS);

        unsigned row_rotate = offset + rot;


        /* col_rotate selects elements within each row
        *   - values between [0, 7]
        */
        unsigned col_rotate = j & (NUM_POINTS - 1);

        /* Contents of rotate_out:
        * j=0:  0 64 128 192 256 ...
        * j=1:  1 65 129 193 257 ...
        *      ...
        * j=4: 452 4 68 132 196 ..   
        *
        * hence, needs to be rotated and bitreversed
        */
        rotate_out[j] = buf[row_rotate][col_rotate];
      }

      for(unsigned j = 0; j < STEPS; j++){
        unsigned rot_out = row & (N - 1);
        
        unsigned chan0 = (rot_out + j) & (N - 1);                 // 0
        unsigned chan1 = ((4 * N / 8) + rot_out + j) & (N - 1);  // 32
        unsigned chan2 = ((2 * N / 8) + rot_out + j) & (N - 1);  // 16
        unsigned chan3 = ((6 * N / 8) + rot_out + j) & (N - 1);  // 48
        unsigned chan4 = ((N / 8) + rot_out + j) & (N - 1);       // 8
        unsigned chan5 = ((5 * N / 8) + rot_out + j) & (N - 1);  // 40
        unsigned chan6 = ((3 * N / 8) + rot_out + j) & (N - 1);  // 24
        unsigned chan7 = ((7 * N / 8) + rot_out + j) & (N - 1);  // 56

        write_channel_intel(chanouttrans[0], rotate_out[chan0]); 
        write_channel_intel(chanouttrans[1], rotate_out[chan1]); 
        write_channel_intel(chanouttrans[2], rotate_out[chan2]); 
        write_channel_intel(chanouttrans[3], rotate_out[chan3]); 
        write_channel_intel(chanouttrans[4], rotate_out[chan4]); 
        write_channel_intel(chanouttrans[5], rotate_out[chan5]); 
        write_channel_intel(chanouttrans[6], rotate_out[chan6]); 
        write_channel_intel(chanouttrans[7], rotate_out[chan7]); 
      }
    } // row
  } // iter matrices
}

__attribute__((max_global_work_dim(0)))
kernel void store(global float2 * restrict dest, int batch) {
  const int N = (1 << LOGN);
  const unsigned STEPS = (1 << (LOGN - LOGPOINTS)); // N / 8

  for(unsigned i = 0; i < batch; i++){

    for(unsigned j = 0; j < N; j++){

      float2 buf[N];
      for(unsigned k = 0; k < STEPS; k++){

        buf[k] = read_channel_intel(chanouttrans[0]);
        buf[4 * N / 8 + k] = read_channel_intel(chanouttrans[1]);
        buf[2 * N / 8 + k] = read_channel_intel(chanouttrans[2]);
        buf[6 * N / 8 + k] = read_channel_intel(chanouttrans[3]);
        buf[N / 8 + k] = read_channel_intel(chanouttrans[4]);
        buf[5 * N / 8 + k] = read_channel_intel(chanouttrans[5]);
        buf[3 * N / 8 + k] = read_channel_intel(chanouttrans[6]);
        buf[7 * N / 8 + k] = read_channel_intel(chanouttrans[7]);
      }

      for(unsigned k = 0; k < STEPS; k++){
        unsigned where = (i * N * N) + (j * N) + (k * POINTS);

        #pragma unroll POINTS
        for(unsigned l = 0; l < POINTS; l++){
          dest[where + l] = buf[(k * POINTS) + l];
        }
      }
      
    }
  }
}
