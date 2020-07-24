// Author: Arjun Ramaswami

/*
* Transpose of a 2d square matrix by accessing data column wise
* Input and output are in normal order
*/

#include "mtrans_config.h"

// Log of the number of replications of the pipeline
#define LOGREPL 2            // 4 replications 
#define REPL (1 << LOGREPL)  // 4 replications 
#define UNROLL_FACTOR 8 

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chaninTranspose[8] __attribute__((depth(8)));
channel float2 chanoutTranspose[8] __attribute__((depth(8)));

__attribute__((max_global_work_dim(0)))
kernel void fetch(global const volatile float2 * restrict src, int batch) {
  const unsigned N = (1 << LOGN);

  for(unsigned i = 0; i < (batch * N * (N / 8)); i++){

    write_channel_intel(chaninTranspose[0], src[(i * 8) + 0]);
    write_channel_intel(chaninTranspose[1], src[(i * 8) + 1]);
    write_channel_intel(chaninTranspose[2], src[(i * 8) + 2]);
    write_channel_intel(chaninTranspose[3], src[(i * 8) + 3]);
    write_channel_intel(chaninTranspose[4], src[(i * 8) + 4]);
    write_channel_intel(chaninTranspose[5], src[(i * 8) + 5]);
    write_channel_intel(chaninTranspose[6], src[(i * 8) + 6]);
    write_channel_intel(chaninTranspose[7], src[(i * 8) + 7]);
  }
}

__attribute__((max_global_work_dim(0)))
kernel void transpose(int iter) {
  const unsigned N = (1 << LOGN);

  local float2 buf[N][N];  // buf[N][N] banked on column 

  for(unsigned j = 0; j < iter; j++){

    #pragma loop_coalesce
    for(unsigned i = 0; i < N; i++){
      for(unsigned k = 0; k < (N / 8); k++){
        unsigned where_read = k * 8;

        buf[i][where_read + 0] = read_channel_intel(chaninTranspose[0]);
        buf[i][where_read + 1] = read_channel_intel(chaninTranspose[1]);
        buf[i][where_read + 2] = read_channel_intel(chaninTranspose[2]);
        buf[i][where_read + 3] = read_channel_intel(chaninTranspose[3]);
        buf[i][where_read + 4] = read_channel_intel(chaninTranspose[4]);
        buf[i][where_read + 5] = read_channel_intel(chaninTranspose[5]);
        buf[i][where_read + 6] = read_channel_intel(chaninTranspose[6]);
        buf[i][where_read + 7] = read_channel_intel(chaninTranspose[7]);
      }
    }

    #pragma loop_coalesce
    for(unsigned i = 0; i < N; i++){
      for(unsigned k = 0; k < (N / 8); k++){
        unsigned where_write = k * 8;

        write_channel_intel(chanoutTranspose[0], buf[where_write + 0][i]);         
        write_channel_intel(chanoutTranspose[1], buf[where_write + 1][i]);   
        write_channel_intel(chanoutTranspose[2], buf[where_write + 2][i]);   
        write_channel_intel(chanoutTranspose[3], buf[where_write + 3][i]);   
        write_channel_intel(chanoutTranspose[4], buf[where_write + 4][i]);
        write_channel_intel(chanoutTranspose[5], buf[where_write + 5][i]);   
        write_channel_intel(chanoutTranspose[6], buf[where_write + 6][i]);   
        write_channel_intel(chanoutTranspose[7], buf[where_write + 7][i]);   
      }
    }
  }

}

__attribute__((max_global_work_dim(0)))
kernel void store(global float2 * restrict dest, int batch) {
  const int N = (1 << LOGN);

  for(unsigned i = 0; i < (batch * N * (N / 8)); i++){
    dest[(i * 8) + 0] = read_channel_intel(chanoutTranspose[0]);
    dest[(i * 8) + 1] = read_channel_intel(chanoutTranspose[1]);
    dest[(i * 8) + 2] = read_channel_intel(chanoutTranspose[2]);
    dest[(i * 8) + 3] = read_channel_intel(chanoutTranspose[3]);
    dest[(i * 8) + 4] = read_channel_intel(chanoutTranspose[4]);
    dest[(i * 8) + 5] = read_channel_intel(chanoutTranspose[5]);
    dest[(i * 8) + 6] = read_channel_intel(chanoutTranspose[6]);
    dest[(i * 8) + 7] = read_channel_intel(chanoutTranspose[7]);
  }
}