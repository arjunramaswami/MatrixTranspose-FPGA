#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J cmake_mTrans_opt
#SBATCH -p fpgasyn
#SBATCH --mem=90000MB 
#SBATCH --time=24:00:00

module load intelFPGA_pro/20.1.0 nalla_pcie/19.4.0_hpc

cd ../build

cmake -DLOG_SIZE=5 ..
make
make diagonal_bitrev_syn
