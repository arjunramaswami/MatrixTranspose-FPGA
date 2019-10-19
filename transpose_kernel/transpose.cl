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

  unsigned where = get_global_id(0) << LOGPOINTS;
  unsigned where_global = where;

  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1))] = src[where_global];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 1] = src[where_global + 1];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 2] = src[where_global + 2];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 3] = src[where_global + 3];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 4] = src[where_global + 4];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 5] = src[where_global + 5];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 6] = src[where_global + 6];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 7] = src[where_global + 7];

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

// Kernel that fetches data from global memory 
/*
kernel void fetch(global const volatile float2 * restrict src) {
  const unsigned N = (1 << LOGN); 
  const unsigned buf_len = 8 * 16;

  for(unsigned k = 0; k < N * N / (8 * 16); k++){ //  N * N * N / REPL
    float28 buf[16];

    for(unsigned i = 0; i < 16; i++){
      unsigned where = (k * buf_len) + (i * 8);
      buf[i].i0 =  src[where];    
      buf[i].i1 =  src[where + 1];    
      buf[i].i2 =  src[where + 2];    
      buf[i].i3 =  src[where + 3];    
      buf[i].i4 =  src[where + 4];    
      buf[i].i5 =  src[where + 5];    
      buf[i].i6 =  src[where + 6];    
      buf[i].i7 =  src[where + 7];    
    }

    for(unsigned i = 0; i < 16; i++){
      write_channel_intel(chaninTranspose[0], buf[i].i0);
      write_channel_intel(chaninTranspose[1], buf[i].i1);
      write_channel_intel(chaninTranspose[2], buf[i].i2);
      write_channel_intel(chaninTranspose[3], buf[i].i3);
      write_channel_intel(chaninTranspose[4], buf[i].i4);
      write_channel_intel(chaninTranspose[5], buf[i].i5);
      write_channel_intel(chaninTranspose[6], buf[i].i6);
      write_channel_intel(chaninTranspose[7], buf[i].i7);
    }
  }

  for(unsigned k = 0; k < N * N / 8; k++){ //  N * N * N / REPL
    float2 buf[8];
    #pragma unroll 8
    for(unsigned i = 0; i < 8; i++){
      buf[i] =  src[(k << LOGN) + i];    
      write_channel_intel(chaninTranspose[i], buf[i]);
    }

  }
  for(unsigned k = 0; k < N; k++){ //  N * N * N / REPL

    float2 buf[N * 8];
    #pragma unroll 
    for(unsigned i = 0; i < N; i++){
      buf[i & ((1<<LOGN)-1)] =  src[(k << LOGN) + i];    
    }

    for (unsigned i = 0; i < (N / 8); i++){
      #pragma unroll
      for(unsigned j = 0; j < 8; j++){
       // printf(" Fetch : %d - %f %f \n", (i * 8) + j, buf[(i * 8) + j].x, buf[(i * 8) + j].y);
        write_channel_intel(chaninTranspose[j], buf[ (i * 8) + j]);           
      }
    }
    // // printf(" fetching %d rows\n", k);
  }
}
*/

// Transposes fetched data; stores them to global memory
__attribute__((reqd_work_group_size((1 << (LOGN + LOGN - LOGPOINTS)), 1, 1)) )
kernel void transpose(int iter) {
  const unsigned N = (1 << LOGN);
  unsigned row, bank;

  // TODO: 8 bytes for float2 but 16 for double2
  local float2 __attribute__((numbanks(N))) buf[N][N];  // buf[N][N] banked on column 
  unsigned lid = get_local_id(0);

  //              col | 0| 1| 2| 3| 4| 5| 6| 7 | 
  //                  --------------------------
  // bank / row 0 -> |  0  8 16 24 32 40 48 56 |   
  //        row 1 -> | 57  1  9 17 25 33 41 49 |  
  //        row 2 -> | 50 58  2 10 18 26 34 42 |  
  //        row 3 -> | 43 51 59  3 11 19 27 35 |  
  //        row 4 -> | 36 44 52 60  4 12 20 28 |  
  //        row 5 -> | 29 37 45 53 61  5 13 21 |  
  //        row 6 -> | 22 30 38 46 54 62  6 14 |  
  //        row 7 -> | 15 23 31 39 47 55 63  7 |  
  //
  //  This means the writes are into different banks, loads is from one bank with a larger READ port size, leading to negligible increase in memory


  #pragma unroll
  for(unsigned i = 0; i < 8; i++){                
    unsigned x = (lid * 8) + i;
    row = (x & ((1 << LOGN) - 1));                    // Every work item is written to the next row. Values range from [0,...,N-1], wraps around N. 
    bank = ((x >> LOGN) + x) & ((1 << LOGN) - 1);     // Decides the column in each successive row. Every row, writes into the successive column, wrapping around after N columns
    buf[row][bank] = read_channel_intel(chaninTranspose[i]);
    //printf(" %d, %d - %f, %f \n ", row, bank, buf[row][bank].x, buf[row][bank].y);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  unsigned local_id = (get_local_id(0) * 8);
  unsigned bnk = local_id & ((1 << LOGN) - 1);
  unsigned rows = (local_id >> LOGN);             // lid / N

  write_channel_intel(chanoutTranspose[0], buf[rows][bnk + 0]);
  write_channel_intel(chanoutTranspose[1], buf[rows][bnk + 1]);
  write_channel_intel(chanoutTranspose[2], buf[rows][bnk + 2]);
  write_channel_intel(chanoutTranspose[3], buf[rows][bnk + 3]);
  write_channel_intel(chanoutTranspose[4], buf[rows][bnk + 4]);
  write_channel_intel(chanoutTranspose[5], buf[rows][bnk + 5]);
  write_channel_intel(chanoutTranspose[6], buf[rows][bnk + 6]);
  write_channel_intel(chanoutTranspose[7], buf[rows][bnk + 7]);

}
/*
  for (unsigned i = 0; i < N; i++){
    for (unsigned j = 0; j < N; j++){
      unsigned index = ((j % N ) * N) + ((i + j) % N);
      buf[index] = read_channel_intel(chaninTranspose[]);
    }
  }

  const unsigned buf_len = (N * N * 2) - 16;
  float2 transpose_buf[buf_len];
  unsigned exit_condition = 2 * (buf_len / 8);

  for (unsigned step = 0; step < iter * exit_condition; step++) {
    float28 data;

    // push data to register
    if( step < ( N * N / 8)){
      data.i0 =  read_channel_intel(chaninTranspose[0]);
      data.i1 =  read_channel_intel(chaninTranspose[1]);
      data.i2 =  read_channel_intel(chaninTranspose[2]);
      data.i3 =  read_channel_intel(chaninTranspose[3]);
      data.i4 =  read_channel_intel(chaninTranspose[4]);
      data.i5 =  read_channel_intel(chaninTranspose[5]);
      data.i6 =  read_channel_intel(chaninTranspose[6]);
      data.i7 =  read_channel_intel(chaninTranspose[7]);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    data = transpose_step(data, step, transpose_buf, buf_len, LOGN);
    
    if (step >= ((buf_len / 8) - 1)){
      write_channel_intel(chanoutTranspose[0], data.i0);
      write_channel_intel(chanoutTranspose[1], data.i1);
      write_channel_intel(chanoutTranspose[2], data.i2);
      write_channel_intel(chanoutTranspose[3], data.i3);
      write_channel_intel(chanoutTranspose[4], data.i4);
      write_channel_intel(chanoutTranspose[5], data.i5);
      write_channel_intel(chanoutTranspose[6], data.i6);
      write_channel_intel(chanoutTranspose[7], data.i7);
    }
  }
  */

/*
  #pragma unroll
  for (unsigned i = 0; i < N; i++){
    #pragma unroll
    for(unsigned j = i; j < N; j++){
      unsigned src = i;
      unsigned dest = (col * N) + row;
      float2 tmp = transpose_buf[src];
      transpose_buf[src] = transpose_buf[dest];
      transpose_buf[dest] = tmp;
    }
  }
  */

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

/*
kernel void store(global float2 * restrict dest){
  const unsigned N = (1 << LOGN);

  for(unsigned k = 0; k < N; k++){ 
    // printf("Storing %d rows\n", k);

    float2 buf[N];
    for (unsigned i = 0; i < (N / 8); i++){
      #pragma unroll
      for(unsigned j = 0; j < 8; j++){
        buf[(i * 8) + j] = read_channel_intel(chanoutTranspose[j]);
        // printf("%d - %f %f \n", (i * 8) + j, buf[(i * 8) + j].x, buf[(i * 8) + j].y);
      }
    }

    #pragma unroll 
    for(unsigned i = 0; i < N; i++){
      dest[(k << LOGN) + i] = buf[i];
    }
  }
}
*/