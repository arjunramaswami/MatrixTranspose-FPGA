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
      write_channel_intel(chanintrans[7], buf[7 * N / 8 + j]);   // 56
    }
  }
}

__attribute__((max_global_work_dim(0)))
kernel void transpose(int batch) {

  const unsigned N = (1 << LOGN);
  unsigned revcolt, where, where_write;


  // Perform N times N*N transpositions and transfers
  for(unsigned p = 0; p < batch; p++){

    float2 buf[N * N];
    for(unsigned i = 0; i < N; i++){
      for(unsigned k = 0; k < (N / 8); k++){
        where = ((i << LOGN) + (k << LOGPOINTS));

        buf[where + 0] = read_channel_intel(chanintrans[0]);
        buf[where + 1] = read_channel_intel(chanintrans[1]);
        buf[where + 2] = read_channel_intel(chanintrans[2]);
        buf[where + 3] = read_channel_intel(chanintrans[3]);
        buf[where + 4] = read_channel_intel(chanintrans[4]);
        buf[where + 5] = read_channel_intel(chanintrans[5]);
        buf[where + 6] = read_channel_intel(chanintrans[6]);
        buf[where + 7] = read_channel_intel(chanintrans[7]);
      }
    }

    for(unsigned i = 0; i < N; i++){
      for(unsigned k = 0; k < N; k++){
        printf("%f ", buf[(i*N) + k].x);
      }
      printf("\n");
    }
    printf("\n\n");

    for(unsigned i = 0; i < N; i++){
      revcolt = bit_reversed(i, LOGN);

      for(unsigned k = 0; k < (N / 8); k++){
        where_write = ((k * N) + revcolt);

        write_channel_intel(chanouttrans[0], buf[where_write]);               // 0
        write_channel_intel(chanouttrans[1], buf[where_write + 4 * (N / 8) * N]);   // 32
        write_channel_intel(chanouttrans[2], buf[where_write + 2 * (N / 8) * N]);   // 16
        write_channel_intel(chanouttrans[3], buf[where_write + 6 * (N / 8) * N]);   // 48
        write_channel_intel(chanouttrans[4], buf[where_write + (N / 8) * N]);       // 8
        write_channel_intel(chanouttrans[5], buf[where_write + 5 * (N / 8) * N]);   // 40
        write_channel_intel(chanouttrans[6], buf[where_write + 3 * (N / 8) * N]);   // 24
        write_channel_intel(chanouttrans[7], buf[where_write + 7 * (N / 8) * N]);   // 54
      }
    }
  }
        
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

    for(unsigned k = 0; k < N; k++){
      printf("%d: %f\n", k, buf[k].x);
    }
    printf("\n\n");
    for(unsigned k = 0; k < (N / 8); k++){
      unsigned where = (j * N) + (k * 8);
      //unsigned rev = bit_reversed((k * POINTS), LOGN);
      unsigned rev = k;

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