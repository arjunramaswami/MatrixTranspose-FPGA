#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J mTrans_make
#SBATCH -p fpgasyn
#SBATCH --mem=90000MB 
#SBATCH --time=24:00:00

module load intelFPGA_pro/20.1.0 nalla_pcie/19.4.0_hpc

cd ../kernels
make VERBOSE=1 syn
