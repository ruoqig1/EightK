#!/bin/bash -l
#SBATCH --job-name="ticker_relevant"
#SBATCH --output=/home/adidishe/EightK/out/ticker_relevant%a.out
#SBATCH --error=/home/adidishe/EightK/out/ticker_relevant%a.err
#SBATCH --chdir=/home/adidishe/EightK
#SBATCH --array=0-1

# Partition for the job:
#SBATCH --partition=cascade
#SBATCH --account="punim2039"

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-6:00:00

module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0
source  ~/venvs/nlp_gpu/bin/activate

python3 news_05_select_and_merge_tickers_relevant.py ${SLURM_ARRAY_TASK_ID} --thirdparty=1
