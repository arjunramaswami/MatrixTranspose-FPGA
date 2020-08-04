// Authors: Tobias Kenter, Arjun Ramaswami 

/*
* This file performs the transpose of 2d square matrix based on the diagonal 
* transposition algorithm. 
* Inputs to transposition and outputs from transposition are in normal order as * required by the FFT kernels.
*/

#include "mtrans_config.h"
#include "diagonal_bitrev.cl" 

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chaninTranspose[POINTS] __attribute__((depth(POINTS)));
channel float2 chanoutTranspose[POINTS] __attribute__((depth(POINTS)));

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

__attribute__((max_global_work_dim(0)))
kernel void transpose(int batch) {
  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  float2 bitrev_in[2][N], bitrev_out[2][N] ;
  //float2 bitrev_in[2][N] __attribute__((memory("MLAB")));
  
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