// Authors: Tobias Kenter, Arjun Ramaswami

/*
* This file performs the transpose of 2d square matrix based on the diagonal 
* transposition algorithm. 
* Inputs to transposition and outputs from transposition are in normal order as * required by the FFT kernels.
*/

#define LOGPOINTS 3
#define POINTS (1 << LOGPOINTS)

#include "diagonal_opt.cl" 

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
  const unsigned DEPTH = (1 << (LOGN + LOGN - LOGPOINTS));
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
  
  //printf("Finished Fetch\n");
}

__attribute__((max_global_work_dim(0)))
kernel void transpose(int batch) {
  const unsigned N = (1 << LOGN);
  const unsigned DEPTH = (1 << (LOGN + LOGN - LOGPOINTS));
  const unsigned DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8

  // swap every matrix
  float2 bufA[DEPTH][POINTS], bufB[DEPTH][POINTS];
  // swap every step
  float2 bitrev_inA[N], bitrev_inB[N]; 
  float2 bitrev_outA[N], bitrev_outB[N]; 
  
  bool is_bufA = false, is_bitrevA = false;

  // additional iterations to fill the buffers
  for(int step = 0; step < ((batch * DEPTH) + DEPTH + DELAY); step++){
    float2 data[POINTS];
    //printf("Step - %d \n", step);
    // Read data from channels
    if (step < (batch * DEPTH)) {
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

    // Swap buffers every N*N/8 iterations 
    // starting from the additional delay of N/8 iterations
    // TODO: const unsigned DELAY = N /8
    is_bufA = (( (step - DELAY) & (DEPTH - 1)) == 0) ? !is_bufA: is_bufA;

    // Swap bitrev buffers every N/8 iterations
    is_bitrevA = ((step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

    /*
    printf("%d: isbufA - %s ", step, is_bufA ? "true" : "false");
    printf(" is_bitrevA - %s \n", is_bitrevA ? "true" : "false");
    */
    /*
    if(step == DEPTH + DELAY){
      printf("Buf A at step %d\n", step);
      for(unsigned i = 0; i < DEPTH; i++){
        printf("%d: ", i);
        for(unsigned j = 0; j < POINTS; j++){
          printf("(%f, %f) ", bufA[i][j].x, bufA[i][j].y);
        }
        printf("\n");
      }
      printf("\n\n");
    }
    */

    unsigned row = step & (DEPTH - 1);

    transpose_step(data, 
      is_bufA ? bufA : bufB, 
      is_bufA ? bufB : bufA, 
      is_bitrevA ? bitrev_inA : bitrev_inB, 
      is_bitrevA ? bitrev_inB : bitrev_inA, 
      is_bitrevA ? bitrev_outA : bitrev_outB, 
      is_bitrevA ? bitrev_outA : bitrev_outA,  
      step, row, LOGN);

    // Write result to channels
    // TODO: >= or >
    if (step >= (DEPTH + DELAY)) {

      /*
      printf("%d: ", step);
      for(unsigned j = 0; j < POINTS; j++){
        printf("(%f, %f) ", data[j].x, data[j].y);
      }
      printf("\n\n");
      */

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
  //printf("Finished \n");
}

__attribute__((max_global_work_dim(0)))
kernel void store(global float2 * restrict dest, int batch) {
  const int N = (1 << LOGN);
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
        unsigned where = (i * N * N) + (j * N) + (k * POINTS);

        #pragma unroll POINTS
        for(unsigned l = 0; l < POINTS; l++){
          dest[where + l] = buf[(k * POINTS) + l];
        }
      }
      
    }
  }

}