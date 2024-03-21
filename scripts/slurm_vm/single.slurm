#!/bin/bash -l
#SBATCH --job-name single
#SBATCH --output=/home/unimelb.edu.au/adidisheim/out/single.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/out/single.err
#SBATCH --chdir=/home/unimelb.edu.au/adidisheim/

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --cpus-per-task 8
#SBATCH --time 24:00:00

python3 wip.py 0
