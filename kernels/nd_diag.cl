//  Authors: Tobias Kenter, Arjun Ramaswami

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

// Transposes fetched data; stores them to global memory
__attribute__((reqd_work_group_size((1 << (LOGN + LOGN - LOGPOINTS)), 1, 1)) )
kernel void transpose(int iter) {
  const unsigned N = (1 << LOGN);

  // TODO: 8 bytes for float2 but 16 for double2
  //#define DEPTH = (N*N/POINTS)
  const unsigned DEPTH = (1 << (LOGN + LOGN - LOGPOINTS));
  local float2 buf1[DEPTH][POINTS];
  unsigned lid = get_local_id(0);

  //        layout following modification of earlier MaxJ design: rotate at new input row
  //        illustration: POINTS 4, NxN 8x8
  //
  //        POINTS    |  0| 1| 2| 3 |
  //                  --------------------------
  //        dep 0  -> |  0  1  2  3 |
  //        dep 1  -> |  4  5  6  7 |
  //        dep 2  -> | 11  8  9 10 |
  //        dep 3  -> | 15 12 13 14 |
  //        dep 4  -> | 18 19 16 17 |
  //        dep 5  -> | 22 23 20 21 |
  //        dep 6  -> | 25 26 27 24 |
  //        dep 7  -> | 29 30 31 28 |
  //        dep 8  -> | 32 33 34 35 |
  //        dep 9  -> | 36 37 38 39 |
  //        dep 10 -> | 43 40 41 42 |
  //        dep 11 -> | 47 44 45 46 |
  //        dep 12 -> | 50 51 48 49 |
  //        dep 13 -> | 54 55 52 53 |
  //        dep 14 -> | 57 58 59 56 |
  //        dep 15 -> | 61 62 63 60 |
  // read: lid 0; vals 00,08,16,24; dep 00,02,04,06; rot 0
  // read: lid 1; vals 32,40,48,56; dep 08,10,12,14; rot 0
  // read: lid 2; vals 01,09,17,25; dep 06,00,02,04; rot 1
  // read: lid 3; vals 33,41,49,57; dep 14,08,10,12; rot 1
  // read: lid 4; vals 02,10,18,26; dep 04,06,00,02; rot 2
  // ...
  // read: lid 8; vals 04,12,20,28; dep 01,03,05,07; rot 0
  // ...
  // read: lid 15; vals 07,15,23,31;
  // formula:
  // block: N * (lid % (N/POINTS))
  // rotate: [(POINTS+i-(lid/(N/POINTS))) * (N/POINTS)] % N
  // offset: (lid / N)

  unsigned rot = (lid >> (LOGN - LOGPOINTS)) & (POINTS-1); //0...POINTS-1

  float2 rotate_in[POINTS];
  #pragma unroll
  for(unsigned i = 0; i < POINTS; i++){
    rotate_in[i] = read_channel_intel(chaninTranspose[i]);
  }

  #pragma unroll
  for(unsigned i = 0; i < POINTS; i++){
    buf1[lid][i] = rotate_in[(i+POINTS-rot) & (POINTS-1)];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  /*
  if(lid==0){
    for(int d=0; d<N*N/POINTS; d++){
      for(int i=0; i<POINTS; i++){
        printf("%f, ",buf1[d][i].s0);
      }
      printf("\n");
    }
  }*/

  float2 rotate_out[POINTS];
  unsigned row_base = (lid & (N/POINTS-1)) << LOGN; // 0, N, 2N, ...
  unsigned row_offset = lid >> LOGN; //0... N/POINTS
  #pragma unroll
  for(unsigned i = 0; i < POINTS; i++){
    unsigned row_rotate = ((POINTS+i-(lid>>(LOGN-LOGPOINTS))) << (LOGN-LOGPOINTS)) & (N-1); // 0, N/POINTS, 2N/POINTS, ... << rotated
    unsigned row = row_base + row_rotate + row_offset;
    /*if(i==0)
      printf("lid %u, row_base %u, row_rotate %u, row_offset %u, row %u\n", lid, row_base, row_rotate, row_offset, row);*/
    rotate_out[i] = buf1[row][i];
  }

  #pragma unroll
  for(unsigned i = 0; i < POINTS; i++){
    write_channel_intel(chanoutTranspose[i], rotate_out[(i+rot) & (POINTS-1)]);
  }
}

/*
  #pragma unroll
  for(unsigned i = 0; i < 8; i++){
    unsigned loc_id = (get_local_id(0) * 8) + i;
    unsigned order = ((loc_id >> LOGN) + loc_id) & ((1 << LOGN) - 1);
    //unsigned col = loc_id & ((1 << LOGN) - 1);
    write_channel_intel(chanoutTranspose[i], tmp[order]);
  }
}
*/
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
