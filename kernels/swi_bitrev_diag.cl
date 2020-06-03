//  Author: Arjun Ramaswami

#include "mtrans_config.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chanintrans[POINTS] __attribute__((depth(POINTS)));
channel float2 chanouttrans[POINTS] __attribute__((depth(POINTS)));

int bit_reversed(int x, int bits) {
  int y = 0;
  #pragma unroll 
  for (int i = 0; i < bits; i++) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  return y;
}


__attribute__((max_global_work_dim(0)))
kernel void fetch(global const volatile float2 * restrict src, int batch) {
  const unsigned N = (1 << LOGN);

  for(unsigned k = 0; k < (batch * N); k++){ 
    float2 buf[N];

    #pragma unroll 8
    for(unsigned i = 0; i < N; i++){
      buf[i & ((1<<LOGN)-1)] = src[(k << LOGN) + i];    
    }

    for(unsigned j = 0; j < (N / 8); j++){
      write_channel_intel(chanintrans[0], buf[j]);               // 0
      write_channel_intel(chanintrans[1], buf[4 * N / 8 + j]);   // 32
      write_channel_intel(chanintrans[2], buf[2 * N / 8 + j]);   // 16
      write_channel_intel(chanintrans[3], buf[6 * N / 8 + j]);   // 48
      write_channel_intel(chanintrans[4], buf[N / 8 + j]);       // 8
      write_channel_intel(chanintrans[5], buf[5 * N / 8 + j]);   // 40
      write_channel_intel(chanintrans[6], buf[3 * N / 8 + j]);   // 24
      write_channel_intel(chanintrans[7], buf[7 * N / 8 + j]);   // 54
    }
  }
}

__attribute__((max_global_work_dim(0)))
kernel void transpose(int batch) {
  const unsigned N = (1 << LOGN);
  const unsigned DEPTH = (1 << (LOGN + LOGN - LOGPOINTS));

  // iterate over N 2d matrices
  for(unsigned k = 0 ; k < batch; k++){

    // Buffer with width - 8 points, depth - (N*N / 8), banked column-wise
    float2 buf[DEPTH][POINTS];
      
    // iterate within a 2d matrix
    for(unsigned row = 0; row < N; row++){

      // Temporary buffer to rotate before filling the matrix
      //float2 rotate_in[POINTS];
      float2 rotate_in[N];

      // bit-reversed ordered input stored in normal order
      for(unsigned j = 0; j < (N / 8); j++){
        rotate_in[j] = read_channel_intel(chanintrans[0]);               // 0
        rotate_in[4 * N / 8 + j] = read_channel_intel(chanintrans[1]);   // 32
        rotate_in[2 * N / 8 + j] = read_channel_intel(chanintrans[2]);   // 16
        rotate_in[6 * N / 8 + j] = read_channel_intel(chanintrans[3]);   // 48
        rotate_in[N / 8 + j] = read_channel_intel(chanintrans[4]);       // 8
        rotate_in[5 * N / 8 + j] = read_channel_intel(chanintrans[5]);   // 40
        rotate_in[3 * N / 8 + j] = read_channel_intel(chanintrans[6]);   // 24
        rotate_in[7 * N / 8 + j] = read_channel_intel(chanintrans[7]);   // 54
      }

      /* For each outer loop iteration, N data items are processed.
       * These N data items should reside in N/8 rows in buf.
       * Each of this N/8 rows are rotated by 1
       * Considering BRAM is POINTS wide, rotations should wrap around at POINTS
       * row & (POINTS - 1)
       */
      unsigned rot = row & (POINTS - 1);

      // fill the POINTS wide row of the buffer each iteration
      // N/8 rows filled with the same rotation
      for(unsigned j = 0; j < N / 8; j++){

        #pragma unroll 8
        for(unsigned i = 0; i < 8; i++){
            unsigned where = ((i + POINTS) - rot) & (POINTS - 1);
            unsigned buf_row = (row * (N / 8)) + j;
            buf[buf_row][i] = rotate_in[(j * POINTS) + where];
        }
      }
    }

    for(unsigned row = 0; row < N; row++){

      float2 rotate_out[N];
      unsigned offset = 0;            

      #pragma unroll 8
      for(unsigned j = 0; j < N; j++){
        unsigned rot = (DEPTH + j - row) << (LOGN - LOGPOINTS) & (DEPTH -1);
        unsigned offset = row >> LOGPOINTS;
        unsigned row_rotate = offset + rot;
        unsigned col_rotate = j & (POINTS - 1);

        rotate_out[j] = buf[row_rotate][col_rotate];
      }

      for(unsigned j = 0; j < N / 8; j++){
        unsigned rev = bit_reversed((j * POINTS), LOGN);
        unsigned rot_out = row & (N - 1);
        
        unsigned chan0 = (rot_out + rev) & (N - 1);                 // 0
        unsigned chan1 = ((4 * N / 8) + rot_out + rev) & (N - 1);  // 32
        unsigned chan2 = ((2 * N / 8) + rot_out + rev) & (N - 1);  // 16
        unsigned chan3 = ((6 * N / 8) + rot_out + rev) & (N - 1);  // 48
        unsigned chan4 = ((N / 8) + rot_out + rev) & (N - 1);       // 8
        unsigned chan5 = ((5 * N / 8) + rot_out + rev) & (N - 1);  // 40
        unsigned chan6 = ((3 * N / 8) + rot_out + rev) & (N - 1);  // 24
        unsigned chan7 = ((7 * N / 8) + rot_out + rev) & (N - 1);  // 56

        write_channel_intel(chanouttrans[0], rotate_out[chan0]);    // 0
        write_channel_intel(chanouttrans[1], rotate_out[chan1]);   // 32
        write_channel_intel(chanouttrans[2], rotate_out[chan2]);   // 16
        write_channel_intel(chanouttrans[3], rotate_out[chan3]);   // 48
        write_channel_intel(chanouttrans[4], rotate_out[chan4]);   // 8
        write_channel_intel(chanouttrans[5], rotate_out[chan5]);   // 40
        write_channel_intel(chanouttrans[6], rotate_out[chan6]);   // 24
        write_channel_intel(chanouttrans[7], rotate_out[chan7]);   // 54
      }
    } // row

  } // iter matrices
        
}

__attribute__((max_global_work_dim(0)))
kernel void store(global float2 * restrict dest, int batch) {
  const int N = (1 << LOGN);
  for(unsigned j = 0; j < (batch * N); j++){
    float2 buf[N];
    for(unsigned k = 0; k < (N / 8); k++){

      unsigned where = (k * 8);
      
      buf[where + 0] = read_channel_intel(chanouttrans[0]);
      buf[where + 1] = read_channel_intel(chanouttrans[1]);
      buf[where + 2] = read_channel_intel(chanouttrans[2]);
      buf[where + 3] = read_channel_intel(chanouttrans[3]);
      buf[where + 4] = read_channel_intel(chanouttrans[4]);
      buf[where + 5] = read_channel_intel(chanouttrans[5]);
      buf[where + 6] = read_channel_intel(chanouttrans[6]);
      buf[where + 7] = read_channel_intel(chanouttrans[7]);
    }

    for(unsigned k = 0; k < (N / 8); k++){
      unsigned where = (j * N) + (k * 8);
      unsigned rev = bit_reversed((k * POINTS), LOGN);
      //unsigned rev = k;

      dest[where + 0] = buf[rev];
      dest[where + 1] = buf[4 * N / 8 + rev];   // 32
      dest[where + 2] = buf[2 * N / 8 + rev];   // 16
      dest[where + 3] = buf[6 * N / 8 + rev];   // 48
      dest[where + 4] = buf[N / 8 + rev];       // 8
      dest[where + 5] = buf[5 * N / 8 + rev];   // 40
      dest[where + 6] = buf[3 * N / 8 + rev];   // 24
      dest[where + 7] = buf[7 * N / 8 + rev];   // 54
    }
  }


}