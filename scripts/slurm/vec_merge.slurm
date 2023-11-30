#!/bin/bash -l
#SBATCH --job-name="bec_merge"
#SBATCH --output=/home/adidishe/EightK/out/bec_merge.out
#SBATCH --error=/home/adidishe/EightK/out/bec_merge.err
#SBATCH --chdir=/home/adidishe/EightK

# Partition for the job:
#SBATCH --account="punim2039"

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:0:00

#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0
source  ~/venvs/nlp_gpu/bin/activate

python3 vec_merge.py --legal=1 --eight=1 --news_on_eight=0 --ati=2