//  Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "CL/opencl.h"

#include "argparse.h"
#include "transpose_fpga.h"
#include "helper.h"

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

int main(int argc, const char **argv) {

  int N = 64, batch = 0, isND = 0;
  char *path;
  int use_svm = 0, use_emulator = 0;
  bool bitreverse = false;

  const char *platform = "Intel(R) FPGA";

  fpga_t timing = {0.0, 0.0, 0.0, 0};

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('n',"n", &N, "Length of Square Matrix"),
    OPT_INTEGER('b',"b", &batch, "Number of batched executions"),
    OPT_BOOLEAN('v',"svm", &use_svm, "Use SVM"),
    OPT_STRING('p', "path", &path, "Path to bitstream"),
    OPT_BOOLEAN('r', "bitreverse", &bitreverse, "Bitreverse i/o"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing Matrix Transpose using FPGA", "Dimension of the matrix is mandatory, default batchation is 1");
  argc = argparse_parse(&argparse, argc, argv);

  print_config(N, batch, use_svm, path, isND);

  if(fpga_initialize(platform, path, use_svm, use_emulator)){
    return 1;
  }

  size_t inp_sz = sizeof(float2) * N * N * batch;

  float2 *inp = (float2*)fpgaf_complex_malloc(inp_sz, use_svm);
  float2 *verify = (float2*)fpgaf_complex_malloc(inp_sz, use_svm);
  float2 *out = (float2*)fpgaf_complex_malloc(inp_sz, use_svm);

  get_input_data(inp, verify, N, batch, bitreverse);

  printf("Transposing Matrix\n");
  timing = mTranspose(N, inp, out, batch, use_svm, isND);

  printf("\nComputing Matrix Transposition\n");
  cpu_mTranspose(verify, N, batch);

  printf("\nChecking Correctness\n");
  verify_mTranspose(out, verify, N, batch, bitreverse);

  // destroy data
  fpga_final();

  if(timing.valid == 1){

    if(timing.exec_t == 0.0){
      fprintf(stderr, "Measurement invalid\n");
      return 1;
    }

    display_measures(timing.exec_t, timing.pcie_read_t, timing.pcie_write_t, N, batch);
  }

  free(inp);
  free(verify);
  free(out);

  return 0;
}