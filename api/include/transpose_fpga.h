//  Authors: Arjun Ramaswami

#ifndef TRANSPOSE_FPGA_H
#define TRANSPOSE_FPGA_H

typedef struct {
  float x;
  float y;
} float2;

typedef struct {
  double x;
  double y;
} double2;

typedef struct fpga_timing {
  double pcie_read_t;
  double pcie_write_t;
  double exec_t;
  int valid;
} fpga_t;

// Initialize FPGA
extern int fpga_initialize(const char *platform_name, const char *path, int use_svm, int use_emulator);

// Finalize FPGA
extern void fpga_final();

// Single precision complex memory allocation
extern void* fpgaf_complex_malloc(size_t sz, int svm);;

// Single precision Matrix Transpose
fpga_t mTranspose(int N, float2 *inp, float2 *out, int batch, int use_svm, int isND);

#endif
