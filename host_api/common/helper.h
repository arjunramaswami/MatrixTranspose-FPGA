/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef HELPER_H
#define HELPER_H

void get_input_data(float2 *transpose_data, float2 *cpu_transpose_data, unsigned N[2], unsigned iter);

void compute_matrix_transpose(float2 *verify_data, int N[2], unsigned iter);

void verify_transpose(float2 *transpose_data, float2 *cpu_transpose_data, int N[2], int iter);

void compute_metrics( double fpga_runtime, double fpga_computetime, double fftw_runtime, unsigned iter, int N[2]);

double getTimeinMilliSec();


#endif // HELPER_H
