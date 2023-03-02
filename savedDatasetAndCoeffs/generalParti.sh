#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -c 24  # number of processor cores (i.e. threads)
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00   # walltime
#SBATCH -J "v_low_Ori"   # job name

#module purge                                 # purge if you already have modules loaded
/home/ramana44/.conda/envs/myenv/bin/python3.9 /home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/parallelizeRK_coeffsUsingMap.py

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE=""

exit 0