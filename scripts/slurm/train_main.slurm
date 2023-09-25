#!/bin/bash -l
#SBATCH --job-name train_main
#SBATCH --output=/home/adidishe/EightK/out/train_main_%a.out
#SBATCH --error=/home/adidishe/EightK/out/train_main_%a.err
#SBATCH --chdir=/home/adidishe/EightK
#SBATCH --array=0-89

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 64G
#SBATCH --cpus-per-task 4
#SBATCH --time 24:00:00


module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0
source  ~/venvs/nlp_gpu/bin/activate

python3 train_main.py ${SLURM_ARRAY_TASK_ID}
