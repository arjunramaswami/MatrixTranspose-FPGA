#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J mTrans
#SBATCH -p all
#SBATCH --time=24:00:00

module load intelFPGA_pro/20.4.0 nalla_pcie/19.4.0_hpc

cd ../build

make matrixTranspose_syn
