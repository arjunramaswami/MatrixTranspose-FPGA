// Authors: Tobias Kenter, Arjun Ramaswami 

/*
* This file performs the transpose of 2d square matrix based on the diagonal 
* transposition algorithm. 
* Inputs to transposition and outputs from transposition are in normal order as * required by the FFT kernels.
*/

#include "mtrans_config.h"
#include "diagonal_bitrev.cl" 

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chaninTranspose[8] __attribute__((depth(8)));
channel float2 chanoutTranspose[8] __attribute__((depth(8)));

__attribute__((max_global_work_dim(0)))
kernel void fetch(global const volatile float2 * restrict src, int batch) {
  const unsigned STEPS = (1 << (LOGN - LOGPOINTS)); // N / 8

  for(unsigned k = 0; k < (batch * N); k++){ 
    float2 buf[N];

    #pragma unroll POINTS
    for(unsigned i = 0; i < N; i++){
      buf[i & ((1<<LOGN)-1)] = src[(k << LOGN) + i];    
    }

    for(unsigned j = 0; j < STEPS; j++){
      write_channel_intel(chaninTranspose[0], buf[j]);               // 0
      write_channel_intel(chaninTranspose[1], buf[(4 * (N / 8)) + j]);   // 32
      write_channel_intel(chaninTranspose[2], buf[(2 * (N / 8)) + j]);   // 16
      write_channel_intel(chaninTranspose[3], buf[(6 * (N / 8)) + j]);   // 48
      write_channel_intel(chaninTranspose[4], buf[(N / 8) + j]);       // 8
      write_channel_intel(chaninTranspose[5], buf[(5 * (N / 8)) + j]);   // 40
      write_channel_intel(chaninTranspose[6], buf[(3 * (N / 8)) + j]);   // 24
      write_channel_intel(chaninTranspose[7], buf[(7 * (N / 8)) + j]);   // 54
    }
  }

}
/*
__attribute__((max_global_work_dim(0)))
kernel void fetch(global const volatile float2 * restrict src, int batch) {
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
*/
__attribute__((max_global_work_dim(0)))
kernel void transpose(int batch) {
  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  //float2 __attribute__((memory, numbanks(8))) bitrev_in[2][N];
  float2 bitrev_in[2][N];
  float2 __attribute__((memory, numbanks(8))) bitrev_out[2][N];
  
  int initial_delay = DELAY + DELAY; // for each of the bitrev buffer

  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((batch * DEPTH) + DEPTH); step++){

    float2x8 data, data_out;
    if (step < ((batch * DEPTH) - initial_delay)) {
      data.i0 = read_channel_intel(chaninTranspose[0]);
      data.i1 = read_channel_intel(chaninTranspose[1]);
      data.i2 = read_channel_intel(chaninTranspose[2]);
      data.i3 = read_channel_intel(chaninTranspose[3]);
      data.i4 = read_channel_intel(chaninTranspose[4]);
      data.i5 = read_channel_intel(chaninTranspose[5]);
      data.i6 = read_channel_intel(chaninTranspose[6]);
      data.i7 = read_channel_intel(chaninTranspose[7]);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    // Swap buffers every N*N/8 iterations 
    // starting from the additional delay of N/8 iterations
    is_bufA = (( (step + DELAY) & (DEPTH - 1)) == 0) ? !is_bufA: is_bufA;

    // Swap bitrev buffers every N/8 iterations
    is_bitrevA = ( (step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

    unsigned row = step & (DEPTH - 1);
    data = bitreverse_in(data,
      is_bitrevA ? bitrev_in[0] : bitrev_in[1], 
      is_bitrevA ? bitrev_in[1] : bitrev_in[0], 
      row);

    writeBuf(data,
      is_bufA ? buf[0] : buf[1],
      step);

    data_out = readBuf(
      is_bufA ? buf[1] : buf[0], 
      step);

    unsigned start_row = (step + DELAY) & (DEPTH -1);
    data_out = bitreverse_out(
      is_bitrevA ? bitrev_out[0] : bitrev_out[1],
      is_bitrevA ? bitrev_out[1] : bitrev_out[0],
      data_out, start_row);

    if (step >= (DEPTH)) {
      write_channel_intel(chanoutTranspose[0], data_out.i0);
      write_channel_intel(chanoutTranspose[1], data_out.i1);
      write_channel_intel(chanoutTranspose[2], data_out.i2);
      write_channel_intel(chanoutTranspose[3], data_out.i3);
      write_channel_intel(chanoutTranspose[4], data_out.i4);
      write_channel_intel(chanoutTranspose[5], data_out.i5);
      write_channel_intel(chanoutTranspose[6], data_out.i6);
      write_channel_intel(chanoutTranspose[7], data_out.i7);
    }
  }
}

/*
__attribute__((max_global_work_dim(0)))
kernel void store(global float2 * restrict dest, int batch) {
  const unsigned STEPS = (1 << (LOGN - LOGPOINTS)); // N / 8

  for(unsigned i = 0; i < batch; i++){
    for(unsigned j = 0; j < N; j++){

      float2 buf[N];
      
      for(unsigned k = 0; k < STEPS; k++){
        buf[k] = read_channel_intel(chanoutTranspose[0]);
        buf[4 * N / 8 + k] = read_channel_intel(chanoutTranspose[1]);
        buf[2 * N / 8 + k] = read_channel_intel(chanoutTranspose[2]);
        buf[6 * N / 8 + k] = read_channel_intel(chanoutTranspose[3]);
        buf[N / 8 + k] = read_channel_intel(chanoutTranspose[4]);
        buf[5 * N / 8 + k] = read_channel_intel(chanoutTranspose[5]);
        buf[3 * N / 8 + k] = read_channel_intel(chanoutTranspose[6]);
        buf[7 * N / 8 + k] = read_channel_intel(chanoutTranspose[7]);
      }
      

      for(unsigned k = 0; k < STEPS; k++){
        
        unsigned index_out = k * 8;
        unsigned index0 = bit_reversed(index_out + 0, LOGN);
        unsigned index1 = bit_reversed(index_out + 1, LOGN);
        unsigned index2 = bit_reversed(index_out + 2, LOGN);
        unsigned index3 = bit_reversed(index_out + 3, LOGN);
        unsigned index4 = bit_reversed(index_out + 4, LOGN);
        unsigned index5 = bit_reversed(index_out + 5, LOGN);
        unsigned index6 = bit_reversed(index_out + 6, LOGN);
        unsigned index7 = bit_reversed(index_out + 7, LOGN);

        buf[index0] = read_channel_intel(chanoutTranspose[0]);
        buf[index1] = read_channel_intel(chanoutTranspose[1]);
        buf[index2] = read_channel_intel(chanoutTranspose[2]);
        buf[index3] = read_channel_intel(chanoutTranspose[3]);
        buf[index4] = read_channel_intel(chanoutTranspose[4]);
        buf[index5] = read_channel_intel(chanoutTranspose[5]);
        buf[index6] = read_channel_intel(chanoutTranspose[6]);
        buf[index7] = read_channel_intel(chanoutTranspose[7]);
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
*/

__attribute__((max_global_work_dim(0)))
kernel void store(global float2 * restrict dest, int batch) {

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
