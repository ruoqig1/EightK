#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --output=/home/adidishe/EightK/out/test.out
#SBATCH --error=/home/adidishe/EightK/out/test.err
#SBATCH --chdir=/home/adidishe/EightK

#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

module load fosscuda/2020b
module load gcc/10.2.0
module load cuda/11.1.1
module load openmpi/4.0.5
module load python/3.8.6
module load tensorflow/2.6.0

source /data/gpfs/projects/punim2039/envs/nlp_gpu/bin/activate

# Run nvidia-smi to log GPU usage
while sleep 60; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> /home/adidishe/EightK/out/gpu_usage.csv; done &

python3 wip.py 0 5
