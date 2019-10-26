/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#include <stdio.h>

// common dependencies
#include "CL/opencl.h"
#include "api/transpose_api.h"  // Common declarations and API
#include "api/transpose_fpga.h"  // Common declarations and API

// local dependencies
#include "common/argparse.h"  // Cmd-line Args to set some global vars
#include "common/helper.h"  // Cmd-line Args to set some global vars

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

void main(int argc, const char **argv) {
  unsigned i = 0, j = 0, iter = 1000;
  double fpga_runtime = 0.0, fpga_computetime = 0.0, cpu_runtime = 0.0;
  char *bitstream_path = "../transpose_kernel/fpgabitstream";
  float2 *matrix_data, *verify_data;
  int N[2] = {64,64};

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('m',"n1", &N[0], "Matrix 1st Dim"),
    OPT_INTEGER('n',"n2", &N[1], "Matrix 2nd Dim"),
    OPT_INTEGER('i',"iter", &iter, "Number of Iterations"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing Matrix Transposition using FPGA", "Matrix size is mandatory");
  argc = argparse_parse(&argparse, argc, argv);

  printf("------------------------------\n");
  printf("Matrix Size : %d %d\n", N[0], N[1]);
  printf("Number of matrices : %d \n", iter);
  printf("------------------------------\n\n");
  // Allocate mem for input buffer and verification buffer
  matrix_data = (float2 *)malloc(sizeof(float2) * iter * N[0] * N[1]);
  verify_data = (float2 *)malloc(sizeof(float2) * iter * N[0] * N[1]);

  get_input_data(matrix_data, verify_data, N, iter);
  printf("Computing Matrix Transposition\n");

  // initialize FPGA
  if (fpga_initialize_()){
    printf("Error initializing FPGA. Exiting\n");
    if(matrix_data)
      free(matrix_data);

    exit(1);
  }

  // check if required bitstream exists
  if(!fpga_check_bitstream_(bitstream_path, N, iter)){
    printf("Bitstream not found. Exiting\n");
    if(matrix_data)
      free(matrix_data);
    if(verify_data)
      free(verify_data);

    exit(1);
  }

  // execute fpga matrix transpose
  double start = getTimeinMilliSec();
  fpga_computetime += fpga_matrix_transpose_(N, matrix_data, iter);
  double stop = getTimeinMilliSec();
  fpga_runtime += stop - start;
  
  // Else randomly generate values and write to a file 
  /*
  printf("Transposed data \n");
  for (j = 0; j < iter; j++){
   for (i = 0; i < (N[0] * N[1]); i++) {
     printf(" %d : transpose[%d] = (%f, %f)\n", i, i, matrix_data[i].x, matrix_data[i].y);
   }
  }
  */

  printf("FPGA Runtime = %.4fms\n", fpga_computetime);

  printf("\nComputing Matrix Transposition\n");
  compute_matrix_transpose(verify_data, N, iter);

  printf("\nChecking Correctness\n");
  verify_transpose(matrix_data, verify_data, N, iter);

  // Print performance metrics
  //compute_metrics(fpga_runtime, fpga_computetime, cpu_runtime, N);

  // Free the resources allocated
  printf("\nCleaning up\n\n");
  if(matrix_data)
    free(matrix_data);
  if(verify_data)
    free(verify_data);
}
