//  Author: Arjun Ramaswami

#define _POSIX_C_SOURCE 199309L  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define _USE_MATH_DEFINES

#include "transpose_fpga.h"

/*
 * \brief  Fill matrix with index as data
 * \param  transpose_data: pointer to square matrix of size N * N 
 * \param  cpu_transpose_data: pointer to square matrix of size N * N that has the same data as the input matrix
 * \param  N: length of square matrix
 * \param  iter: number of iterations of square matrix of size N*N
 */
void get_input_data(float2 *transpose_data, float2 *cpu_transpose_data, unsigned N, unsigned iter){

  // Else randomly generate values and write to a file 
  printf("Creating data \n");
  for (size_t j = 0; j < iter; j++){
    for (size_t i = 0; i < (N * N); i++) {
      unsigned disp = j * N * N;
      cpu_transpose_data[disp + i].x = transpose_data[disp + i].x = (float)i;
      cpu_transpose_data[disp + i].y = transpose_data[disp + i].y = (float)i;
    }
  }
  /*
  for (size_t i = 0; i < iter * (N * N); i++) {
    printf(" %d : transpose[%d] = (%f, %f)\n", i, i, transpose_data[i].x, transpose_data[i].y);
  }
  */
}

/*
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 */
double getTimeinMilliSec(){
   struct timespec a;
   clock_gettime(CLOCK_MONOTONIC, &a);
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0e3;
}

/*
 * \brief compute matrix transpose in CPU to verify FPGA implementation
 */
void cpu_mTranspose(float2 *verify_data, int N, unsigned batch){
  float2 *temp = (float2 *)malloc(sizeof(float2) * batch * N * N);

  for(size_t k = 0; k < batch; k++){
    for(size_t i = 0; i < N; i++){
      for(size_t j = 0; j < N; j++){
        temp[(k * N * N) + (j * N) + i].x = verify_data[(k * N * N) + (i * N) + j].x;
        temp[(k * N * N) + (j * N) + i].y = verify_data[(k * N * N) + (i * N) + j].y;
      }
    }
  }

  for(size_t i = 0; i < (batch * N * N); i++){
    verify_data[i].x = temp[i].x;
    verify_data[i].y = temp[i].y;
  }

  /*
  for (size_t i = 0; i < batch * N * N; i++) {
    printf(" %d : transpose[%d] = (%f, %f)\n", i, i, verify_data[i].x, verify_data[i].y);
  }
  */

  free(temp);
}

/**
 * \brief  verify fpga computed matrix transpose with cpu
 * \param  fpga_out: pointer to fpga Matrix Transpose output
 * \param  cpu_out: pointer to cpu Matrix Transpose output
 * \param  N: length of square matrix
 * \param  batch: number of batched transposes
 */
void verify_mTranspose(float2 *fpga_out, float2 *cpu_out, int N, int batch){
  float mag_sum = 0, noise_sum = 0;

  for (size_t i = 0; i < batch * N * N; i++){
    float magnitude = cpu_out[i].x * cpu_out[i].x + \
                      cpu_out[i].y * cpu_out[i].y;
    float noise = (cpu_out[i].x - fpga_out[i].x) * (cpu_out[i].x - fpga_out[i].x) + (cpu_out[i].y - fpga_out[i].y) * (cpu_out[i].y - fpga_out[i].y);
    mag_sum += magnitude;
    noise_sum += noise;

#ifdef DEBUG
    printf("%zu: fpga - (%f %f) cpu - (%f %f)\n", i, fpga_out[i].x, fpga_out[i].y, cpu_out[i].x, cpu_out[i].y);
#endif

  }
  float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  printf("-> Signal to noise ratio on output sample: %f --> %s\n\n", db, db > 120 ? "PASSED" : "FAILED");
}

/**
 * \brief  print configuration of matrix transposition
 * \param  n: length of matrix
 * \param  batch: number of batched transpositions
 * \param  use_svm: svm based PCIe transposes
 * \param  path: path to bitstream
 * \param  isND: kernel implemented as ND range or single work item
 */
void print_config(int N, int batch, int use_svm, char *path, int isND){
  printf("\n------------------------------------------\n");
  printf("Matrix Transpose Configuration: \n");
  printf("--------------------------------------------\n");
  printf("Type               = Single Precision Complex 2d Matrix Transpose\n");
  printf("Points             = [%d x %d] \n", N, N);
  printf("# Batched Transposes  = %d \n", batch);
  printf("%s PCIe Transfer \n", use_svm ? "SVM based":"");
  printf("Path to %skernel = %s \n", isND ? "ND Range ": "Single Work Item ", path);
  printf("--------------------------------------------\n\n");
}

/**
 * \brief  print time taken for fpga execution of matrix transpose
 * \param  batched kernel execution time
 * \param  pcie read time
 * \param  pcie write time
 * \param  n: length of square matrix
 * \param  batch: number of batched transpositions
 */
void display_measures(double b_exec, double pcie_rd_t, double pcie_wr_t, int N, int batch){

  double exec = b_exec / batch;
  double gpoints_per_sec = (N * N  / (exec * 1e-3)) * 1e-9;
  double gBytes_per_sec = 0.0;

  gBytes_per_sec =  gpoints_per_sec * 8; // bytes

  printf("\n------------------------------------------\n");
  printf("Average Measurements of Matrix Transpose\n");
  printf("--------------------------------------------\n");
  printf("Points             = [%d x %d]\n", N, N);
  printf("PCIe Write         = %.2lfms\n", pcie_wr_t);
  printf("Batch Kernel Execution  = %.2lfms\n", b_exec);
  printf("Kernel Execution   = %.2lfms\n", exec);
  printf("PCIe Write         = %.2lfms\n", pcie_rd_t);
  printf("Throughput         = %.2lf GB/s\n", gBytes_per_sec);
}