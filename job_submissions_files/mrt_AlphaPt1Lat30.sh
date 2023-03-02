#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -c 16  # number of processor cores (i.e. threads)
#SBATCH -p casus
#SBATCH -A casus
#SBATCH --gres=gpu:1

#SBATCH --time=48:00:00   # walltime
#SBATCH -J "HA1L30"   # job name

#module purge                                 # purge if you already have modules loaded
/home/ramana44/.conda/envs/myenv/bin/python3.9 /home/ramana44/topological-analysis-of-curved-spaces-and-hybridization-of-autoencoders/training_call_files/mrt_train_testAlphaPt1Lat30.py

exit 0