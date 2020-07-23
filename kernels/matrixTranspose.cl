// Authors: Tobias Kenter, Arjun Ramaswami

/*
* This file performs the transpose of 2d square matrix based on the diagonal 
* transposition algorithm. 
* Inputs to transposition and outputs from transposition are in normal order as * required by the FFT kernels.
*/

#define LOGPOINTS 3
#define POINTS (1 << LOGPOINTS)
#define UNROLL_FACTOR POINTS

#include "diagonal_opt.cl" 
#include "mtrans_config.h"

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
kernel void transpose(int batch) {
  const unsigned N = (1 << LOGN);
  const unsigned DEPTH = (1 << (LOGN + LOGN - LOGPOINTS)); // (N * N / 8)
  bool is_bufA = false;

  float2 bufA[DEPTH][POINTS], bufB[DEPTH][POINTS];

  for(unsigned step = 0; step < ((batch * DEPTH) + DEPTH); step++){

    float2 data[POINTS];
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

    transpose_step(data, 
      is_bufA ? bufA : bufB, 
      is_bufA ? bufB : bufA, 
      step, LOGN);

    if (step >= DEPTH) {
      write_channel_intel(chanoutTranspose[0], data[0]);
      write_channel_intel(chanoutTranspose[1], data[1]);
      write_channel_intel(chanoutTranspose[2], data[2]);
      write_channel_intel(chanoutTranspose[3], data[3]);
      write_channel_intel(chanoutTranspose[4], data[4]);
      write_channel_intel(chanoutTranspose[5], data[5]);
      write_channel_intel(chanoutTranspose[6], data[6]);
      write_channel_intel(chanoutTranspose[7], data[7]);
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