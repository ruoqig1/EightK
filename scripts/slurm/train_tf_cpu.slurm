#!/bin/bash -l
#SBATCH --job-name tf_no_norm
#SBATCH --output=/home/adidishe/EightK/out/tf_no_norm_%a.out
#SBATCH --error=/home/adidishe/EightK/out/tf_no_norm_%a.err
#SBATCH --chdir=/home/adidishe/EightK
#SBATCH --array=0-19

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.autr

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 8G
#SBATCH --cpus-per-task 8
#SBATCH --time 3-0:00:00


module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
source  ~/venvs/nlp_gpu/bin/activate

python3 train_tf.py ${SLURM_ARRAY_TASK_ID} --cpu=1
