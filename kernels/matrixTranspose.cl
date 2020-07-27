// Authors: Arjun Ramaswami, Tobias Kenter

/*
* This file performs the transpose of 2d square matrix based on the diagonal 
* transposition algorithm. 
* Inputs to transposition and outputs from transposition are in normal order as * required by the FFT kernels.
*/

#include "mtrans_config.h"
#include "diagonal_opt.cl" 

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chaninTranspose[POINTS] __attribute__((depth(POINTS)));
channel float2 chanoutTranspose[POINTS] __attribute__((depth(POINTS)));

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
kernel void transpose(int batch) {
  const unsigned N = (1 << LOGN);
  bool is_bufA = false;

  float2 bufA[2][DEPTH][POINTS];

  for(unsigned step = 0; step < ((batch * DEPTH) + DEPTH); step++){

    float2 data[POINTS];
    float2x8 data_out;

    if (step < (batch * DEPTH) ) {
      data[0] = read_channel_intel(chaninTranspose[0]);
      data[1] = read_channel_intel(chaninTranspose[1]);
      data[2] = read_channel_intel(chaninTranspose[2]);
      data[3] = read_channel_intel(chaninTranspose[3]);
      data[4] = read_channel_intel(chaninTranspose[4]);
      data[5] = read_channel_intel(chaninTranspose[5]);
      data[6] = read_channel_intel(chaninTranspose[6]);
      data[7] = read_channel_intel(chaninTranspose[7]);
    } else {
      data[0] = data[1] = data[2] = data[3] = 
                data[4] = data[5] = data[6] = data[7] = 0;
    }

    is_bufA = ((step & (DEPTH - 1)) == 0) ? !is_bufA : is_bufA;

    data_out = readBuf(is_bufA ? bufA[1] : bufA[0], step);

    writeBuf(data, is_bufA ? bufA[0] : bufA[1], step);

    if (step >= DEPTH) {
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
  const unsigned N = (1 << LOGN);

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
