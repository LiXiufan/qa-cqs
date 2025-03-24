#pbs file for submission to HPC

#!/bin/bash
#PBS -P Proj_para
#PBS -q parallel
#PBS -N cqs_simulation_hpc
#PBS -o output_logs/output_$PBS_ARRAYID.out
#PBS -e error_logs/error_$PBS_ARRAYID.err
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=12:mpiprocs=12
#PBS -j oe

cd $PBS_O_WORKDIR

module load python/3.8

K_max=5

L=$((PBS_ARRAYID/K_max+3))
K=$((PBS_ARRAYID % K_max))

echo "Running test.py with L=$L, K=$K"

python3 test.py --L $L --K $K
