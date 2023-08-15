#!/bin/bash -l

#SBATCH --job-name=data_merge
#SBATCH --output=/home/adidishe/EightK/out/data_merge.out
#SBATCH --error=/home/adidishe/EightK/out/data_merge.rr
#SBATCH --chdir=/home/adidishe/EightK

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00

module load fosscuda/2020b
module load gcc/10.2.0
module load cuda/11.1.1
module load openmpi/4.0.5
module load python/3.8.6
module load tensorflow/2.6.0

source /data/gpfs/projects/punim2039/envs/nlp_gpu/bin/activate

python3 /scratch/adidishe/EightK/data.py -2

