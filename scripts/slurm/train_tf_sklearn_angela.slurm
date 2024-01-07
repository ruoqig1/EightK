#!/bin/bash -l
#SBATCH --job-name tf_sklearn
#SBATCH -o "/data/projects/punim2119/EightK/outputs/%j.out"
#SBATCH --chdir=/data/projects/punim2119/EightK/

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=ruoqig@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# Partition for the job:
#SBATCH --account="punim2119"

#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=72G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00


module load GCCcore/11.3.0
module load foss/2022a Python/3.10.4 SciPy-bundle/2022.05
module load CUDA/11.7.0 UCX-CUDA/1.13.1-CUDA-11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0

source /data/projects/punim2119/EightK/venv/bin/activate

python3 train_tf_lr.py --cpu=0
