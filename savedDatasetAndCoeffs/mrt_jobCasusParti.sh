#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -c 20  # number of processor cores (i.e. threads)
#SBATCH -p casus
#SBATCH -A casus
#SBATCH --gres=gpu:1

#SBATCH --time=48:00:00   # walltime
#SBATCH -J "SmartHalf"   # job name

#module purge                                 # purge if you already have modules loaded
/home/ramana44/.conda/envs/myenv/bin/python3.9 /home/ramana44/oasis_mri_2d_slices_hybridAutoencodingSmartGridAlphaHalf_75_RK/savedDatasetAndCoeffs/parallelizeRK_coeffsUsingMap.py

exit 0