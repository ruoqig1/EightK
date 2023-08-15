#!/bin/bash -l
#SBATCH --job-name wsj_pre_$1
#SBATCH --output=/home/adidishe/EightK/out/preprocess_$1.out
#SBATCH --error=/home/adidishe/EightK/out/preprocess_$1.err
#SBATCH --chdir=/home/adidishe/EightK

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 8G
#SBATCH --cpus-per-task 2
#SBATCH --time 1:00:00

module load gcccore/10.2.0 python/3.8.6
source /data/gpfs/projects/punim2039/envs/nlp/bin/activate

python3 data.py $1
