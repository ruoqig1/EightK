#!/bin/bash -l

#SBATCH --job-name wsj_pre_1
#SBATCH --output=/home/adidishe/EightK/out/untar_1.out
#SBATCH --error=/home/adidishe/EightK/out/untar_1.err
#SBATCH --chdir=/home/adidishe/EightK

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --cpus-per-task 8
#SBATCH --time 10:00:00

module load gcccore/10.2.0 python/3.8.6
source /data/gpfs/projects/punim2039/envs/nlp/bin/activate

python3 /scratch/adidishe/EightK/data.py -1

