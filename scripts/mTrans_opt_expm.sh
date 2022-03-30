#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J mTrans64
#SBATCH -p fpgasyn
#SBATCH --mem=90000MB 
#SBATCH --time=24:00:00

module load intelFPGA_pro/20.2.0 nalla_pcie/19.4.0_hpc

cd ../build_test

cmake -DLOGSIZE=5 ..
make
make matrixTranspose_bitrev_opt_syn
