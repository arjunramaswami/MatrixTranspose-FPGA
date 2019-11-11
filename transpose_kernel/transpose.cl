/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef LOGN
#define LOGN 6
#endif

// Macros for the 8 point 1d FFT
#define LOGPOINTS 3
#define POINTS (1 << LOGPOINTS)

// Log of the number of replications of the pipeline
#define LOGREPL 2            // 4 replications 
#define REPL (1 << LOGREPL)  // 4 replications 
#define UNROLL_FACTOR 8 

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chaninTranspose[8] __attribute__((depth(8)));
channel float2 chanoutTranspose[8] __attribute__((depth(8)));

// --- CODE -------------------------------------------------------------------
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


__attribute__((reqd_work_group_size((1 << LOGN), 1, 1)))
kernel void fetch(global const volatile float2 * restrict src) {
  const int N = ( 1 << LOGN);
  local float2 buf[8 * N];

  unsigned where_global = get_global_id(0) << LOGPOINTS;
  unsigned where_local= get_local_id(0) << LOGPOINTS;

  buf[(where_local & ((1 << (LOGN + LOGPOINTS)) - 1))] = src[where_global];
  buf[(where_local & ((1 << (LOGN + LOGPOINTS)) - 1)) + 1] = src[where_global + 1];
  buf[(where_local & ((1 << (LOGN + LOGPOINTS)) - 1)) + 2] = src[where_global + 2];
  buf[(where_local & ((1 << (LOGN + LOGPOINTS)) - 1)) + 3] = src[where_global + 3];
  buf[(where_local & ((1 << (LOGN + LOGPOINTS)) - 1)) + 4] = src[where_global + 4];
  buf[(where_local & ((1 << (LOGN + LOGPOINTS)) - 1)) + 5] = src[where_global + 5];
  buf[(where_local & ((1 << (LOGN + LOGPOINTS)) - 1)) + 6] = src[where_global + 6];
  buf[(where_local & ((1 << (LOGN + LOGPOINTS)) - 1)) + 7] = src[where_global + 7];

  barrier(CLK_LOCAL_MEM_FENCE);

  unsigned where_buf = get_local_id(0) << LOGPOINTS;

  // Stream fetched data over 8 channels to the FFT engine
  write_channel_intel(chaninTranspose[0], buf[where_buf + 0]);
  write_channel_intel(chaninTranspose[1], buf[where_buf + 1]);
  write_channel_intel(chaninTranspose[2], buf[where_buf + 2]);
  write_channel_intel(chaninTranspose[3], buf[where_buf + 3]);
  write_channel_intel(chaninTranspose[4], buf[where_buf + 4]);
  write_channel_intel(chaninTranspose[5], buf[where_buf + 5]);
  write_channel_intel(chaninTranspose[6], buf[where_buf + 6]);
  write_channel_intel(chaninTranspose[7], buf[where_buf + 7]);
}

__attribute__((max_global_work_dim(0)))
kernel void transpose(int iter) {
  const unsigned N = (1 << LOGN);

  // TODO: 8 bytes for float2 but 16 for double2
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

__attribute__((reqd_work_group_size((1 << LOGN), 1, 1)))
kernel void store(global float2 * restrict dest) {
  const int N = (1 << LOGN);
  local float2 buf[8 * N];

  unsigned where_buf = get_local_id(0) << LOGPOINTS;

  buf[(where_buf + 0) & ((1 << (LOGN + LOGPOINTS)) - 1)] = read_channel_intel(chanoutTranspose[0]);
  buf[(where_buf + 1) & ((1 << (LOGN + LOGPOINTS)) - 1)] = read_channel_intel(chanoutTranspose[1]);
  buf[(where_buf + 2) & ((1 << (LOGN + LOGPOINTS)) - 1)] = read_channel_intel(chanoutTranspose[2]);
  buf[(where_buf + 3) & ((1 << (LOGN + LOGPOINTS)) - 1)] = read_channel_intel(chanoutTranspose[3]);
  buf[(where_buf + 4) & ((1 << (LOGN + LOGPOINTS)) - 1)] = read_channel_intel(chanoutTranspose[4]);
  buf[(where_buf + 5) & ((1 << (LOGN + LOGPOINTS)) - 1)] = read_channel_intel(chanoutTranspose[5]);
  buf[(where_buf + 6) & ((1 << (LOGN + LOGPOINTS)) - 1)] = read_channel_intel(chanoutTranspose[6]);
  buf[(where_buf + 7) & ((1 << (LOGN + LOGPOINTS)) - 1)] = read_channel_intel(chanoutTranspose[7]);

  barrier(CLK_LOCAL_MEM_FENCE);

  unsigned where_global = get_global_id(0) << LOGPOINTS;
  unsigned where = get_local_id(0) << LOGPOINTS;

  dest[where_global]     = buf[where + 0];
  dest[where_global + 1] = buf[where + 1];
  dest[where_global + 2] = buf[where + 2];
  dest[where_global + 3] = buf[where + 3];
  dest[where_global + 4] = buf[where + 4];
  dest[where_global + 5] = buf[where + 5];
  dest[where_global + 6] = buf[where + 6];
  dest[where_global + 7] = buf[where + 7];
}