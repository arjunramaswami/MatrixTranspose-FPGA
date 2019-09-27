/******************************************************************************
 *  Authors: Arjun Ramaswami
 *****************************************************************************/

#ifndef FFT_FPGA_H
#define FFT_FPGA_H

// Initialize FPGA
int fpga_initialize_();

// Finalize FPGA
void fpga_final_();

// Single precision FFT3d procedure
double fpga_matrix_transpose_(int N[2], float2 *din, unsigned iter);

// Check fpga bitstream present in directory
int fpga_check_bitstream_(char *data_path, int N[2], unsigned iter);
#endif
