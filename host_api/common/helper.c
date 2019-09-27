/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#define _POSIX_C_SOURCE 199309L  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define _USE_MATH_DEFINES

// common dependencies
#include "transpose_api.h"


// --- CODE ------------------------------------------------------------------

/******************************************************************************
 * \brief  create random single precision floating point values for FFT 
 *         computation or read existing ones if already saved in a file
 * \param  fft_data  : pointer to fft3d sized allocation of sp complex data for fpga
 * \param  fftw_data : pointer to fft3d sized allocation of sp complex data for fftw cpu computation
 * \param  N : 3 element integer array containing the size of FFT3d  
 * \param  fname : path to input file to read from or write into
 *****************************************************************************/
void get_input_data(float2 *transpose_data, float2 *cpu_transpose_data, unsigned N[2], unsigned iterations){
  unsigned i = 0, j = 0;

  // Else randomly generate values and write to a file 
  printf("Creating data \n");
  for (j = 0; j < iterations; j++){
    for (i = 0; i < (N[0] * N[1]); i++) {
      cpu_transpose_data[i].x = transpose_data[i].x = (float)i;
      cpu_transpose_data[i].y = transpose_data[i].y = (float)i;
  #ifdef DEBUG
      printf(" %d : transpose[%d] = (%f, %f)\n", i, i, transpose_data[i].x, transpose_data[i].y);
  #endif
    }
  }
}

/******************************************************************************
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 *****************************************************************************/
double getTimeinMilliSec(){
   struct timespec a;
   clock_gettime(CLOCK_MONOTONIC, &a);
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0e3;
}

void compute_matrix_transpose(float2 *verify_data, int N[2]){
  unsigned i, j;
  float2 *temp = (float2 *)malloc(sizeof(float2) * N[0] * N[1]);

  for(i = 0; i < N[0]; i++){
    for(j = 0; j < N[1]; j++){
      temp[j * N[1] + i].x = verify_data[(i * N[1]) + j].x;
      temp[j * N[1] + i].y = verify_data[(i * N[1]) + j].y;
    }
  }

  for(i = 0; i < (N[0] * N[1]); i++){
    verify_data[i].x = temp[i].x;
    verify_data[i].y = temp[i].y;
  }

  free(temp);
}

/******************************************************************************
 * \brief  verify computed fft3d with FFTW fft3d
 * \param  fft_data  : pointer to fft3d sized allocation of sp complex data for fpga
 * \param  fftw_data : pointer to fft3d sized allocation of sp complex data for fftw cpu computation
 * \param  N : 3 element integer array containing the size of FFT3d  
 *****************************************************************************/
void verify_transpose(float2 *transpose_data, float2 *cpu_transpose_data, int N[2]){
  unsigned where, i, j, k;
  float mag_sum = 0, noise_sum = 0, magnitude, noise;

  for( i = 0; i < (N[0] * N[1]); i++){
#ifdef DEBUG
    printf("%d : fpga - (%e %e) cpu - (%e %e)\n", i, transpose_data[i].x, transpose_data[i].y, cpu_transpose_data[i].x, cpu_transpose_data[i].y);
#endif            
  }

  /*
  for (i = 0; i < (N[0] * N[1]); i++) {
    float magnitude = fftw_data[where][0] * fftw_data[where][0] + \
                      fftw_data[where][1] * fftw_data[where][1];
    float noise = (fftw_data[where][0] - fft_data[where].x) \
        * (fftw_data[where][0] - fft_data[where].x) + 
        (fftw_data[where][1] - fft_data[where].y) * (fftw_data[where][1] - fft_data[where].y);

    mag_sum += magnitude;
    noise_sum += noise;
#ifdef DEBUG
    printf("%d : fpga - (%e %e) cpu - (%e %e)\n", where, fft_data[where].x, fft_data[where].y, fftw_data[where][0], fftw_data[where][1]);
#endif            
  }

  float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  printf("-> Signal to noise ratio on output sample: %f --> %s\n\n", db, db > 120 ? "PASSED" : "FAILED");
  */
}

/******************************************************************************
 * \brief  print time taken for fpga and fftw runs to a file
 * \param  fftw_time, fpga_time: double
 * \param  iter - number of iterations of each
 * \param  fname - filename given through cmd line arg
 * \param  N - fft size
 *****************************************************************************/
void compute_metrics( double fpga_runtime, double fpga_computetime, double fftw_runtime, unsigned iter, int N[2]){
  char filename[] = "../outputfiles/output.csv";
  printf("Printing metrics to %s\n", filename);

  FILE *fp = fopen(filename,"r");
  if(fp == NULL){
    fp = fopen(filename,"w");
    if(fp == NULL){
      printf("Unable to create file - %s\n", filename);
      exit(1);
    }
    fprintf(fp,"device, N, runtime, computetime, throughput\n");
  }
  else{
    fp = fopen(filename,"a");
  }

  printf("\nNumber of runs: %d\n\n", iter);
  printf("\tFFT Size\tRuntime(ms)\tComputetime(ms)\tThroughput(GFLOPS/sec)\t\n");
  printf("fpga:");
  fprintf(fp, "fpga,");

  if(fpga_runtime != 0.0 || fpga_computetime != 0.0){
    fpga_runtime = fpga_runtime / iter;
    fpga_computetime = fpga_computetime / iter;
    double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fpga_computetime * 1E-3)) * 1E-9;
    double gflops = 3 * 5 * N[0] * N[1] * N[2] * (log((double)N[0])/log((double)2))/(fpga_computetime * 1E-3 * 1E9);
    printf("\t  %d³ \t\t %.4f \t %.4f \t  %.4f \n", N[0], fpga_runtime, fpga_computetime, gflops);
    fprintf(fp, "%d,%.4f,%.4f,%.4f\n", N[0], fpga_runtime, fpga_computetime, gflops);
  }
  else{
    printf("ERROR in FFT3d \n");
  }

  printf("fftw:"); 
  fprintf(fp, "fftw,"); 
  if(fftw_runtime != 0.0){
    fftw_runtime = fftw_runtime / iter;
    double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fftw_runtime * 1E-3)) * 1E-9;
    double gflops = 3 * 5 * N[0] * N[1] * N[2]* (log((double)N[0])/log((double)2))/(fftw_runtime * 1E-3 * 1E9);
    printf("\t  %d³ \t\t\t\t %.4f \t  %.4f \t\n", N[0], fftw_runtime, gflops);
    fprintf(fp, "%d,%.4f,%.4f\n", N[0], fftw_runtime, gflops);
  }
  else{
    printf("ERROR in FFT3d\n");
  }

  fclose(fp);
}

