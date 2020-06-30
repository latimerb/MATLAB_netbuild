#!/bin/bash

#SBATCH --partition debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -A TG-DBS180005
#SBATCH --mem-per-cpu=5000
#SBATCH --job-name=ca1
#SBATCH --output=ca1%j.out
#SBATCH --time 0-00:30
#SBATCH -L matlab:24


module load matlab/2019b
matlab -nodesktop -nodisplay -nosplash < BuildNetwork2.m


