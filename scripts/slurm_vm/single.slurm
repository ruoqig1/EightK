#!/bin/bash -l
#SBATCH --job-name single
#SBATCH --output=/home/adidishe/EightK/out/single.out
#SBATCH --error=/home/adidishe/EightK/out/single.err
#SBATCH --chdir=/home/adidishe/EightK

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 30G
#SBATCH --cpus-per-task 8
#SBATCH --time 24:00:00

source /home/adidishe/EightK/venv/bin/activate

python3 marina.py