#!/bin/bash -l

#SBATCH --job-name=data_merge
#SBATCH --output=/home/adidishe/EightK/out/merge_embeddings.out
#SBATCH --error=/home/adidishe/EightK/out/merge_embeddings.err
#SBATCH --chdir=/home/adidishe/EightK

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00

module load fosscuda/2020b
module load gcc/10.2.0
module load cuda/11.1.1
module load openmpi/4.0.5
module load python/3.8.6
module load tensorflow/2.6.0

source /data/gpfs/projects/punim2039/envs/nlp_gpu/bin/activate

python3 /home/adidishe/EightK/merge_embeddings.py 0 5

