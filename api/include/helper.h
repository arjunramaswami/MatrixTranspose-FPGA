//  Author: Arjun Ramaswami

#ifndef HELPER_H
#define HELPER_H

#include <stdbool.h>

void get_input_data(float2 *transpose_data, float2 *cpu_transpose_data, unsigned N, unsigned iter, bool bitreverse);

double getTimeinMilliSec();

void cpu_mTranspose(float2 *verify_data, int N, unsigned batch);

void verify_mTranspose(float2 *fpga_out, float2 *cpu_out, int N, int batch, bool bitreverse);

void print_config(int n, int batch, int use_svm, char *path, int isND);

void display_measures(double b_exec, double pcie_rd_t, double pcie_wr_t, int N, int batch);

#endif // HELPER_H
