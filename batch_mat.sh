#!/bin/bash

#SBATCH --partition compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -A TG-DBS180005
#SBATCH --mem-per-cpu=5000
#SBATCH --job-name=ca1
#SBATCH --output=ca1%j.out
#SBATCH --time 0-12:00
#SBATCH --qos=normal
#SBATCH -L matlab:24


module load matlab/2019b
matlab -nodesktop -nodisplay -nosplash < BuildNetwork.m


