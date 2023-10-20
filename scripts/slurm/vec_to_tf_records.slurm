#!/bin/bash -l
#SBATCH --job-name="vec_to_tf_records"
#SBATCH --output=/home/adidishe/EightK/out/vec_to_tf_%a.out
#SBATCH --error=/home/adidishe/EightK/out/vec_to_tf_%a.err
#SBATCH --chdir=/home/adidishe/EightK
#SBATCH --array=0-18

# Partition for the job:
#SBATCH --account="punim2039"

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-20:0:00

#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0
source  ~/venvs/nlp_gpu/bin/activate

python3 vec_to_tf_records.py ${SLURM_ARRAY_TASK_ID} --news_on_eight=1
